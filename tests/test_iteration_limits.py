"""Iteration budgets are incomplete searches, never infeasibility certificates."""
import unittest
from unittest.mock import patch

import numpy as np

from lpmip import branch_bound as bb
from lpmip.model import Model
from lpmip.simplex import LPResult, solve_standard


class PhaseOneBudgetTests(unittest.TestCase):
    def solve(self, limit, **kwargs):
        # Unique feasible x=(1,1), objective 2. No model, random benchmark or simulator.
        return solve_standard(np.eye(2), np.ones(2), np.ones(2), max_iter=limit, **kwargs)

    def test_feasible_lp_dantzig_and_bland_exhaustion_not_infeasible(self):
        for rule, auto in [("dantzig", False), ("bland", False), ("bland", True)]:
            with self.subTest(rule=rule, auto=auto):
                result = self.solve(1, rule=rule, auto_bland=auto)
                self.assertEqual(result.status, "iteration_limit")
                self.assertIsNone(result.x)
                self.assertTrue(np.isnan(result.obj))
                self.assertEqual(result.iterations, 1)
                self.assertIn("phase I", result.message)

    def test_default_bland_restart_exhaustion_remains_incomplete(self):
        result = self.solve(1)
        self.assertEqual(result.status, "iteration_limit")
        self.assertTrue(result.restarted_with_bland)
        self.assertEqual(result.pivot_rule, "bland")
        self.assertIn("phase I", result.message)
        self.assertIsNone(result.x)

    def test_sufficient_budget_preserves_the_known_optimum(self):
        for rule in ["dantzig", "bland"]:
            with self.subTest(rule=rule):
                result = self.solve(10, rule=rule)
                self.assertEqual(result.status, "optimal")
                np.testing.assert_array_equal(result.x, np.ones(2))
                self.assertEqual(result.obj, 2.0)

    def test_truly_infeasible_lp_requires_converged_phase_one(self):
        # The same x cannot equal both 1 and 2. This is a genuine certificate.
        result = solve_standard(np.ones((2, 1)), np.array([1., 2.]), np.array([1.]), max_iter=10)
        self.assertEqual(result.status, "infeasible")
        self.assertIsNone(result.x)
        self.assertTrue(np.isposinf(result.obj))


class BranchBoundLimitTests(unittest.TestCase):
    def model(self):
        model = Model(sense="min")
        x = model.add_var("x", 0, 1, integer=True)
        model.set_objective(x)
        return model

    def limited(self, status):
        return LPResult(status, None, np.nan, 1, "bland")

    def test_unresolved_root_is_not_pruned_as_infeasible(self):
        for status in ["iteration_limit", "cycling"]:
            with self.subTest(status=status), patch.object(bb, "solve_model", return_value=self.limited(status)):
                result = bb.solve(self.model())
                self.assertEqual(result.status, "lp_limit")
                self.assertIsNone(result.x)
                self.assertTrue(np.isnan(result.obj))
                self.assertTrue(np.isnan(result.gap))
                self.assertEqual(result.nodes, 1)
                self.assertEqual(result.log[-1]["lp_status"], status)

    def test_lp_only_entry_keeps_limit_and_unknown_gap(self):
        model = Model(sense="min")
        x = model.add_var("x", 0, 1)
        model.set_objective(x)
        for status in ["iteration_limit", "cycling"]:
            with self.subTest(status=status), patch.object(bb, "solve_model", return_value=self.limited(status)):
                result = bb.solve(model)
                self.assertEqual(result.status, "lp_limit")
                self.assertIsNone(result.x)
                self.assertTrue(np.isnan(result.obj))
                self.assertTrue(np.isnan(result.gap))
                self.assertEqual(result.log[-1]["lp_status"], status)

    def test_incumbent_survives_without_false_optimality_or_gap(self):
        # Scripted relaxations isolate B&B propagation, not a measured solver result.
        rows = [LPResult("optimal", np.array([.5]), .5, 1, "bland"),
                LPResult("optimal", np.array([0.]), 0., 1, "bland"), self.limited("iteration_limit")]
        with patch.object(bb, "solve_model", side_effect=rows):
            result = bb.solve(self.model())
        self.assertEqual(result.status, "lp_limit")
        np.testing.assert_array_equal(result.x, np.array([0.]))
        self.assertEqual(result.obj, 0.)
        self.assertTrue(np.isnan(result.gap))
        self.assertEqual(result.nodes, 3)


if __name__ == "__main__":
    unittest.main()
