# -*- coding: utf-8 -*-
"""每条声明都对着 HiGHS 验一遍。全部离线、无网络。"""
from __future__ import annotations

import json
import os
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
from lpmip.cuts import tighten, validate_pure_integer           # noqa: E402
from problems import classic, transit, vsp                      # noqa: E402


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

class Cuts(unittest.TestCase):
    """Gomory 分数割。**第一版这里放过一个会切掉可行整数解的假货**，所以这一组
    不满足于"跑出来对"，要按定义把有效性验一遍。"""

    def test_cuts_never_change_the_optimum(self):
        for n in (8, 10, 12, 14):
            m, *_ = classic.knapsack(seed=3, n=n)
            plain, cut = bb.solve(m), bb.solve(m, cuts=True)
            st, ref = highs_mip(m)
            self.assertEqual(st, "optimal")
            self.assertAlmostEqual(plain.obj, ref, places=6, msg="n=%d" % n)
            self.assertAlmostEqual(cut.obj, ref, places=6,
                                   msg="加割不许改最优值 n=%d" % n)

    def test_no_feasible_integer_point_is_ever_cut_off(self):
        """按定义验有效性：枚举全部 2^n 个 0/1 点，原模型可行的必须仍然可行。

        这正是第一版那个假割（`x <= floor(v)`）会当场失败的地方。
        """
        for n in (8, 10, 12):
            m, *_ = classic.knapsack(seed=3, n=n)
            tightened, stats = tighten(m)
            self.assertGreater(stats["cuts"], 0, "n=%d 一条割都没加，这测试就没意义" % n)
            kept = 0
            for mask in range(1 << n):
                x = [float((mask >> i) & 1) for i in range(n)]
                if not m.is_feasible(x):
                    continue
                kept += 1
                self.assertTrue(tightened.is_feasible(x),
                                "n=%d 割切掉了可行整数点 %s" % (n, x))
            self.assertGreater(kept, 0)

    def test_cuts_only_tighten_the_bound_never_past_the_optimum(self):
        for n in (8, 10, 12, 14):
            m, *_ = classic.knapsack(seed=3, n=n)
            _, st = tighten(m)
            _, ref = highs_mip(m)
            self.assertLess(st["bound_after"], st["bound_before"] - 1e-9,
                            "n=%d 保留下来的割必须抬高界" % n)
            self.assertGreaterEqual(st["bound_after"], ref - 1e-6,
                                    "n=%d 界被压到最优值以下 = 切掉了整数解" % n)

    def test_mixed_integer_is_refused_not_silently_skipped(self):
        """分数割要求行里所有变量（含松弛变量）取整数。混合整数不满足，必须拒绝。"""
        cap, profit, fixed, total = classic.production_mip(seed=5, n=7)
        mixed = classic.build_production(cap, profit, fixed, total, big_m=None)
        ok, why = validate_pure_integer(mixed)
        self.assertFalse(ok)
        self.assertIn("连续变量", why)
        with self.assertRaises(ValueError):
            bb.solve(mixed, cuts=True)

    def test_cuts_can_make_things_worse_and_that_stays_in_the_bench(self):
        """n=12 上割抬高了界，节点数却从 29 涨到 31。反例留着，不许被悄悄挑掉。"""
        d = json.loads((ROOT / "results" / "bench.json").read_text(encoding="utf-8"))["cuts"]
        self.assertTrue(d["all_same_optimum"])
        self.assertTrue(d["mixed_integer_is_refused"])
        self.assertGreaterEqual(d["n_hurt"], 1,
                                "README 里写着有反例；如果它消失了，README 也要改")


class MultiDepot(unittest.TestCase):
    """MDVSP。**这一组存在的理由是我在这里写错过一个结论。**"""

    def test_single_depot_relaxation_is_integral(self):
        """单场站是纯网络流，矩阵全幺模，松弛天然整数。"""
        for n in (6, 8):
            trips = transit.instance(n, seed=1)
            r = solve_model(vsp.model_single_depot(trips))
            self.assertEqual(r.status, "optimal")
            for v in r.x:
                self.assertAlmostEqual(v, round(v), places=6, msg="n=%d" % n)

    def test_adding_a_depot_alone_does_not_make_it_fractional(self):
        """**被推翻的那条结论**：只加场站不产生分数松弛。

        这是负面结果，写成断言是为了防止有人（包括我）以后又把
        「多一个场站难度就换一个量级」写回 README。
        """
        d = json.loads((ROOT / "results" / "vsp.json").read_text(encoding="utf-8"))["ladder"]
        self.assertGreaterEqual(d["n_rows"], 10)
        self.assertEqual(d["n_rows_with_fractional_lp"], 0,
                         "阶梯里出现了分数松弛——结论要重写，不是改断言")
        self.assertTrue(d["all_agree_with_highs"])

    def test_depot_eligibility_is_what_actually_makes_it_fractional(self):
        """加上场站准入限制之后才出分数松弛，6 个冻结种子逐个复现。"""
        for seed in vsp.FRACTIONAL_SEEDS:
            m = vsp.model_from_random(seed)
            lp = solve_model(m)
            self.assertEqual(lp.status, "optimal", "seed=%d" % seed)
            n_frac = sum(1 for v in lp.x if abs(v - round(v)) > 1e-6)
            self.assertGreater(n_frac, 0, "seed=%d 不再出分数解了" % seed)
            mip = bb.solve(m, max_nodes=20000)
            st, ref = highs_mip(m)
            self.assertEqual(st, "optimal")
            self.assertAlmostEqual(mip.obj, ref, places=6, msg="seed=%d" % seed)
            self.assertGreater(mip.nodes, 1, "seed=%d 出了分数解就该分支" % seed)

    def test_random_instances_are_reproducible_across_processes(self):
        """站名先排序再进循环——集合迭代顺序按哈希来，跨进程会变。

        第一版就在这里踩过：同一个种子两次跑出不同实例，分数变量数 16 变 22。
        """
        code = chr(10).join([
            "import sys; sys.path.insert(0, %r)" % str(ROOT),
            "from problems import vsp",
            "i = vsp.random_instance(17)",
            "print(i['n'], ','.join(i['depots']), i['vehicle_cost'],",
            "      len(i['dh']), sum(i['dh'].values()))",
        ])
        outs = set()
        for hs in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            outs.add(r.stdout.strip())
        self.assertEqual(len(outs), 1, "换 PYTHONHASHSEED 就变了：%s" % outs)

    def test_a_fractional_relaxation_does_not_imply_a_loose_bound(self):
        """有一个反例的松弛是分数的，界间隙却是 0.00%。

        分数解与整数解目标值相同——松弛解"不整数"和界"不紧"是两回事，
        README 里把这条单独写出来了。
        """
        d = json.loads((ROOT / "results" / "vsp.json").read_text(
            encoding="utf-8"))["counterexamples"]
        self.assertTrue(d["all_fractional"])
        self.assertTrue(d["all_agree_with_highs"])
        self.assertGreaterEqual(d["n_with_zero_gap"], 1)

    def test_vsp_bench_check_passes(self):
        r = subprocess.run([sys.executable, str(ROOT / "bench" / "bench_vsp.py"), "--check"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


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
