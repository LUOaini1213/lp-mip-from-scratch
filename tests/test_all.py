# -*- coding: utf-8 -*-
"""每条声明都对着 HiGHS 验一遍。全部离线、无网络。"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.bench_all import highs_lp, highs_mip                  # noqa: E402
from lpmip import branch_bound as bb                             # noqa: E402
from lpmip.model import INF, Model                               # noqa: E402
from lpmip.simplex import solve_model, solve_standard            # noqa: E402
from problems import classic, transit                            # noqa: E402


class Reduction(unittest.TestCase):
    """归约到标准型的四件事，逐件验。"""

    def _roundtrip(self, m: Model):
        """我的解 与 HiGHS 解同一个标准型，目标值必须一致。"""
        mine = solve_model(m)
        st, obj = highs_lp(m)
        self.assertEqual(mine.status, st, m.name)
        if st == "optimal":
            self.assertAlmostEqual(mine.obj, obj, places=6, msg=m.name)
        return mine

    def test_slack_surplus_and_equality(self):
        m = Model("mixed", sense="min")
        a, b, c = m.add_var("a"), m.add_var("b"), m.add_var("c")
        m.add_constraint(a + b + c == 10)
        m.add_constraint(a - b >= 2)
        m.add_constraint(c <= 4)
        m.set_objective(2 * a + 3 * b + c)
        self._roundtrip(m)

    def test_negative_rhs_is_normalised(self):
        m = Model("negrhs", sense="min")
        x, y = m.add_var("x"), m.add_var("y")
        m.add_constraint(-x - y <= -5)
        m.set_objective(x + 2 * y)
        sf = m.to_standard_form()
        self.assertTrue(np.all(sf.b >= -1e-12), "标准型要求 b >= 0")
        self._roundtrip(m)

    def test_shifted_and_bounded_variables(self):
        m = Model("bounds", sense="max")
        x = m.add_var("x", lb=1, ub=5)
        y = m.add_var("y", lb=-2, ub=3)
        m.add_constraint(x + 2 * y <= 8)
        m.set_objective(x + y)
        self._roundtrip(m)

    def test_free_variable_is_split(self):
        m = Model("free", sense="min")
        x = m.add_var("x", lb=-INF, ub=INF)
        y = m.add_var("y")
        m.add_constraint(x + y >= 3)
        m.add_constraint(x <= 5)
        m.set_objective(x + y)
        self._roundtrip(m)

    def test_free_variable_with_upper_bound(self):
        m = Model("free_ub", sense="max")
        x = m.add_var("x", lb=-INF, ub=4)
        y = m.add_var("y", lb=0, ub=6)
        m.add_constraint(x + y <= 8)
        m.add_constraint(x - y >= -5)
        m.set_objective(2 * x + y)
        r = self._roundtrip(m)
        self.assertLessEqual(r.x[0], 4 + 1e-9, "上界必须真的生效")


class Status(unittest.TestCase):
    def test_infeasible_is_detected(self):
        m = Model("infeas", sense="min")
        x = m.add_var("x")
        m.add_constraint(x >= 5)
        m.add_constraint(x <= 2)
        m.set_objective(x)
        self.assertEqual(solve_model(m).status, "infeasible")
        self.assertEqual(highs_lp(m)[0], "infeasible")

    def test_unbounded_is_detected(self):
        m = Model("unb", sense="max")
        x, y = m.add_var("x"), m.add_var("y")
        m.add_constraint(x - y <= 1)
        m.set_objective(x + y)
        self.assertEqual(solve_model(m).status, "unbounded")
        self.assertEqual(highs_lp(m)[0], "unbounded")


class PivotRules(unittest.TestCase):
    """主元规则的差别是真的，不是教科书上的警告。"""

    def test_dantzig_cycles_on_beale_and_bland_does_not(self):
        A, b, c = classic.beale_cycling()
        d = solve_standard(A, b, c, rule="dantzig", max_iter=200, auto_bland=False)
        bl = solve_standard(A, b, c, rule="bland", max_iter=200, auto_bland=False)
        self.assertEqual(d.status, "cycling", "Dantzig 规则在 Beale 例子上应当循环")
        self.assertEqual(bl.status, "optimal")
        self.assertAlmostEqual(bl.obj, -0.05, places=9)
        ref = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        self.assertAlmostEqual(bl.obj, ref.fun, places=9)

    def test_auto_restart_rescues_the_default_rule(self):
        """默认配置遇到循环会自动换 Bland 重启，不该把循环甩给调用方。"""
        A, b, c = classic.beale_cycling()
        r = solve_standard(A, b, c, rule="dantzig", max_iter=200, auto_bland=True)
        self.assertEqual(r.status, "optimal")
        self.assertTrue(r.restarted_with_bland)

    def test_klee_minty_hits_the_exponential_worst_case(self):
        for n in (3, 4, 5, 6):
            m = classic.klee_minty(n)
            r = solve_model(m, rule="dantzig", max_iter=5000)
            self.assertEqual(r.iterations, 2 ** n - 1,
                             "n=%d 应当正好走 2^n-1 步" % n)


class BranchAndBound(unittest.TestCase):
    def test_knapsack_matches_highs(self):
        for n in (8, 10, 12, 14):
            m, *_ = classic.knapsack(seed=3, n=n)
            r = bb.solve(m)
            st, obj = highs_mip(m)
            self.assertEqual(st, "optimal")
            self.assertAlmostEqual(r.obj, obj, places=6, msg="n=%d" % n)

    def test_solution_is_actually_integral_and_feasible(self):
        m, w, v, cap = classic.knapsack(seed=3, n=12)
        r = bb.solve(m)
        for i in m.integer_vars():
            self.assertAlmostEqual(r.x[i], round(r.x[i]), places=6)
        self.assertLessEqual(sum(w[i] * r.x[i] for i in range(len(w))), cap + 1e-6)

    def test_cuts_flag_refuses_instead_of_lying(self):
        """割平面没实现。传 True 必须报错，**不许静默当成没传**。

        第一版这里有一个"割"，它会切掉可行整数解——那会让求解器返回错误的最优值。
        删掉之后留一个显式的 NotImplementedError，比留一个静默降级安全。
        """
        m, *_ = classic.knapsack(seed=3, n=8)
        with self.assertRaises(NotImplementedError):
            bb.solve(m, cuts=True)


class Formulations(unittest.TestCase):
    """建模比换求解器重要——这是本仓的主张，得写成断言。"""

    def test_assignment_lp_relaxation_is_integral(self):
        """指派型约束矩阵全幺模，线性松弛天然整数，分支定界只用一个节点。"""
        for n in (6, 8, 10, 12):
            trips = transit.instance(n, seed=1)
            m = transit.model_assignment(trips)
            lp = solve_model(m)
            self.assertEqual(lp.status, "optimal")
            for v in lp.x:
                self.assertAlmostEqual(v, round(v), places=6, msg="n=%d 松弛解应当已整数" % n)
            self.assertEqual(bb.solve(m).nodes, 1, "n=%d 不该分支" % n)

    def test_bigM_gives_the_same_answer_but_a_useless_bound(self):
        for n in (4, 5, 6):
            trips = transit.instance(n, seed=1)
            good = transit.model_assignment(trips)
            fleet = transit.fleet_from_assignment(trips, bb.solve(good).obj)
            big = transit.model_bigM(trips, n_buses=max(fleet, 1))
            lp = solve_model(big)
            r = bb.solve(big, max_nodes=5000)
            self.assertAlmostEqual(r.obj, fleet, places=6, msg="两种写法答案必须一致")
            self.assertLess(lp.obj, fleet - 0.5, "big-M 的松弛界应当明显更松")
            self.assertGreater(r.nodes, 5, "big-M 应当真的要分支")

    def test_tight_big_m_needs_fewer_nodes(self):
        cap, profit, fixed, total = classic.production_mip(seed=5, n=7)
        tight = bb.solve(classic.build_production(cap, profit, fixed, total, big_m=None),
                         max_nodes=8000)
        loose = bb.solve(classic.build_production(cap, profit, fixed, total, big_m=10000),
                         max_nodes=8000)
        self.assertAlmostEqual(tight.obj, loose.obj, places=6, msg="最优值必须一样")
        self.assertLess(tight.nodes, loose.nodes, "M 收紧应当少探节点")


class CommittedResults(unittest.TestCase):
    """results/bench.json 里的结论必须还成立，而且是与新跑比对、不是只读旧值。"""

    def test_bench_check_passes(self):
        r = subprocess.run([sys.executable, str(ROOT / "bench" / "bench_all.py"), "--check"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_every_instance_agrees_with_highs(self):
        d = json.loads((ROOT / "results" / "bench.json").read_text(encoding="utf-8"))
        self.assertTrue(d["correctness"]["all_agree"])
        self.assertGreaterEqual(d["correctness"]["n_cases"], 10)

    def test_the_speed_claim_is_stated_honestly(self):
        """小实例上我并不慢——这一点不许在 README 里被说成"慢几个数量级"。"""
        rows = {r["case"]: r for r in json.loads(
            (ROOT / "results" / "bench.json").read_text(encoding="utf-8"))["cost"]["rows"]}
        self.assertLess(rows["klee_minty_3"]["speed_ratio"], 2.0,
                        "小 LP 上 HiGHS 的固定开销占主导，我不该被说成慢")
        self.assertGreater(rows["dense_lp_80x120"]["speed_ratio"], 10.0,
                           "规模拉起来之后差距才显出来")


if __name__ == "__main__":
    unittest.main()
