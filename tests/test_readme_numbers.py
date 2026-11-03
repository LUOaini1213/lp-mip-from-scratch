# -*- coding: utf-8 -*-
"""README 里的数字必须来自已提交的 results/，不是手打的。

速度表由 `bench/bench_all.py` 渲染、`--check` 逐字比对（见 test_all.py）；
这里管其余所有表格和英文首屏 / 速览表里的数字。改错 README 里任何一格，
或者 results/ 重算后数字变了而 README 没跟着改，这里都会红。

纯读文件，不跑求解器，几毫秒。
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems import vsp                                          # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
BENCH = json.loads((ROOT / "results" / "bench.json").read_text(encoding="utf-8"))
VSPJ = json.loads((ROOT / "results" / "vsp.json").read_text(encoding="utf-8"))


def _cell(c: str) -> str:
    return c.replace("**", "").strip()


def table(header_start: str) -> list[list[str]]:
    """README 里表头以 header_start 开头的那张 Markdown 表，返回数据行（已去掉 ** 与空白）。"""
    lines = README.splitlines()
    hits = [i for i, ln in enumerate(lines) if ln.startswith(header_start)]
    assert len(hits) == 1, "README 里找不到（或不止一张）表头为 %r 的表" % header_start
    rows = []
    for ln in lines[hits[0] + 2:]:
        if not ln.startswith("|"):
            break
        rows.append([_cell(c) for c in ln.strip().strip("|").split("|")])
    return rows


def yes(b: bool) -> str:
    return "是" if b else "否"


def num(x: float) -> float:
    """README 用 U+2212 写负号。"""
    return float(x.replace("−", "-"))


class ChineseTables(unittest.TestCase):

    def test_formulations_table(self):
        rows = table("| 写法 | 班次 |")
        data = BENCH["formulations"]["transit"]
        self.assertEqual(len(rows), len(data))
        for got, r in zip(rows, data):
            self.assertEqual(got, [r["formulation"], str(r["trips"]), str(r["n_vars"]),
                                   str(r["n_cons"]), "%.3f" % r["lp_bound"],
                                   "%.3f" % r["mip_obj"], str(r["nodes"]),
                                   yes(r["lp_already_integral"])])

    def test_big_m_table(self):
        rows = table("| M 的取值 |")
        data = BENCH["formulations"]["big_m_sweep"]
        self.assertEqual(len(rows), len(data))
        for got, r in zip(rows, data):
            if r["big_m"] != "tight(cap_i)":
                self.assertEqual(got[0], str(r["big_m"]))
            self.assertEqual(got[1:], ["%.2f" % r["lp_bound"], "%.2f" % r["mip_obj"],
                                       str(r["nodes"]), yes(r["agrees_with_highs"])])

    def test_beale_table(self):
        rows = {r[0]: r for r in table("| 规则 | 状态 |")}
        b = BENCH["pivot"]["beale"]
        self.assertTrue(rows["Dantzig"][1].startswith(b["dantzig"]["status"]))
        self.assertEqual(rows["Dantzig"][2], str(b["dantzig"]["iterations"]))
        self.assertEqual(rows["Bland"][1], b["bland"]["status"])
        self.assertEqual(rows["Bland"][2], str(b["bland"]["iterations"]))
        self.assertAlmostEqual(num(rows["Bland"][3]), b["bland"]["obj"], places=9)
        self.assertAlmostEqual(num(rows["HiGHS（参照）"][3]), b["highs_obj"], places=9)

    def test_klee_minty_table(self):
        rows = table("| n | 我的迭代数 |")
        data = BENCH["pivot"]["klee_minty"]
        self.assertEqual(len(rows), len(data))
        for got, r in zip(rows, data):
            self.assertEqual(got, [str(r["n"]), str(r["iterations"]),
                                   str(r["two_pow_n_minus_1"]), yes(r["agrees_with_highs"])])

    def test_cuts_table(self):
        rows = table("| 实例 | 松弛界（割前） |")
        data = BENCH["cuts"]["rows"]
        self.assertEqual(len(rows), len(data))
        for got, r in zip(rows, data):
            self.assertEqual(got, [r["case"], "%.3f" % r["lp_bound_before"],
                                   "%.3f" % r["lp_bound_after"], "%g" % r["mip_obj"],
                                   str(r["cuts_added"]), str(r["rounds"]),
                                   "%d → %d" % (r["nodes_without_cuts"], r["nodes_with_cuts"]),
                                   yes(r["agrees_with_highs"])])

    def test_vsp_ladder_table(self):
        rows = table("| 配置 | 变量 |")
        names = ["single_depot_n8", "two_depots_n8", "three_depots_n8",
                 "two_depots_asym_n8", "two_depots_capped_n8", "two_depots_cheapveh_n8"]
        by = {r["case"]: r for r in VSPJ["ladder"]["rows"]}
        self.assertEqual(len(rows), len(names))
        for got, name in zip(rows, names):
            r = by[name]
            self.assertEqual(got[1:], [str(r["n_vars"]), "%.2f" % r["lp_bound"],
                                       str(r["n_fractional"]), str(r["nodes"]),
                                       yes(r["agrees_with_highs"])], name)
        # 「n=6 的六档同样全是 0」
        n6 = [r for r in VSPJ["ladder"]["rows"] if r["case"].endswith("_n6")]
        self.assertEqual(len(n6), 6)
        self.assertTrue(all(r["n_fractional"] == 0 for r in n6))

    def test_counterexample_table(self):
        rows = table("| 种子 | 班次 |")
        data = VSPJ["counterexamples"]["rows"]
        self.assertEqual(len(rows), len(data))
        for got, r in zip(rows, data):
            seed = int(r["case"].rsplit("seed", 1)[1])
            self.assertEqual(got, [str(seed), str(r["n_trips"]), str(r["n_depots"]),
                                   "%g" % r["vehicle_cost"], str(r["n_fractional"]),
                                   "%.2f" % r["lp_bound"], "%.2f" % r["mip_obj"],
                                   "%.2f%%" % r["gap_pct"], str(r["nodes"])])


class EnglishSummary(unittest.TestCase):
    """英文首屏与速览表里的每个数字，都从 results/ 现算出来再到 README 里找。"""

    def setUp(self):
        self.flat = re.sub(r"\s+", " ", README)

    def assertSays(self, text: str):
        self.assertIn(text, self.flat, "README 里应当写着：%s" % text)

    def _node_range(self):
        t = BENCH["formulations"]["transit"]
        a = {r["nodes"] for r in t if r["formulation"] == "assignment"}
        b = [r["nodes"] for r in t if r["formulation"] == "bigM"]
        self.assertEqual(len(a), 1)
        return a.pop(), min(b), max(b)

    def test_formulation_numbers(self):
        a, lo, hi = self._node_range()
        self.assertSays("%d node vs %d–%d for the same fleet problem" % (a, lo, hi))
        self.assertSays("| %d node vs %d–%d nodes |" % (a, lo, hi))

    def test_big_m_numbers(self):
        sweep = BENCH["formulations"]["big_m_sweep"]
        tight = [r["nodes"] for r in sweep if r["big_m"] == "tight(cap_i)"]
        loose = {r["nodes"] for r in sweep if r["big_m"] != "tight(cap_i)"}
        self.assertEqual(len(loose), 1)
        self.assertSays("| %d nodes vs %d |" % (tight[0], loose.pop()))

    def test_pivot_numbers(self):
        b = BENCH["pivot"]["beale"]
        self.assertSays("stopped at %d iterations vs optimal in %d"
                        % (b["dantzig"]["iterations"], b["bland"]["iterations"]))
        km = BENCH["pivot"]["klee_minty"]
        self.assertTrue(all(r["matches_worst_case"] for r in km))
        self.assertSays("2ⁿ−1 iterations for n = %d..%d" % (km[0]["n"], km[-1]["n"]))

    def test_speed_numbers(self):
        cost = {r["case"]: r for r in BENCH["cost"]["rows"]}
        big = cost["dense_lp_80x120"]["speed_ratio"]
        self.assertSays("ratio %.1f on klee_minty_3, %.1f× on dense 80×120"
                        % (cost["klee_minty_3"]["speed_ratio"], big))
        self.assertSays("%.1f× slower than HiGHS at 80×120" % big)
        # 首屏写的「最大实例」必须真是 bench 里最大的那几个
        dense = sorted(c for c in cost if c.startswith("dense_lp_"))
        self.assertIn("dense_lp_80x120", dense)
        knap = max(int(c.rsplit("_", 1)[1]) for c in cost if c.startswith("knapsack_"))
        trips = {r["n_trips"] for r in VSPJ["counterexamples"]["rows"]}
        self.assertSays("80×120 dense LP, %d-item knapsack, %d-trip MDVSP"
                        % (knap, max(trips)))

    def test_cut_numbers(self):
        c = BENCH["cuts"]
        root = [r for r in c["rows"] if r["nodes_with_cuts"] == 1]
        worse = [r for r in c["rows"] if r["nodes_with_cuts"] > r["nodes_without_cuts"]]
        self.assertEqual(len(worse), 1)
        w = worse[0]
        self.assertSays("cuts closed %d of %d knapsacks at the root" % (len(root), c["n_cases"]))
        self.assertSays("(%d → %d nodes)" % (w["nodes_without_cuts"], w["nodes_with_cuts"]))
        self.assertSays("%d of %d knapsacks solved at the root; %s grows %d → %d nodes"
                        % (len(root), c["n_cases"], w["case"],
                           w["nodes_without_cuts"], w["nodes_with_cuts"]))

    def test_vsp_numbers(self):
        lad, base, hunt = VSPJ["ladder"], VSPJ["baseline_hunt"], VSPJ["hunt"]
        lad_gap = sum(1 for r in lad["rows"] if abs(r["gap_pct"]) > 1e-9)
        self.assertSays("%d of %d random multi-depot instances had a bound gap"
                        % (base["n_with_gap"], base["tried"]))
        self.assertSays("eligibility only %d of %d did" % (len(hunt["gap_seeds"]), hunt["tried"]))
        self.assertSays("%d of %d ladder configs and %d of %d random instances with a bound gap; "
                        "with eligibility, %d of %d fractional LP vertices from this simplex, "
                        "%d of %d with a positive gap"
                        % (lad_gap, lad["n_rows"], base["n_with_gap"], base["tried"],
                           hunt["n_hits"], hunt["tried"], len(hunt["gap_seeds"]), hunt["tried"]))

    def test_frozen_constants_match_the_committed_json(self):
        """常量、JSON、README 三处说的是同一组种子。"""
        self.assertEqual(VSPJ["hunt"]["hits"], list(vsp.FRACTIONAL_SEEDS))
        self.assertEqual(VSPJ["hunt"]["gap_seeds"], list(vsp.FRACTIONAL_SEEDS_WITH_GAP))
        self.assertEqual(VSPJ["hunt"]["tried"], vsp.HUNT_TRIED)
        self.assertEqual(VSPJ["baseline_hunt"]["tried"], vsp.BASELINE_TRIED)
        self.assertSays("6000 个随机实例里 %d 个的松弛顶点是分数的（%.2f%%）， 其中 %d 个界间隙 > 0"
                        % (VSPJ["hunt"]["n_hits"], VSPJ["hunt"]["hit_rate_pct"],
                           len(VSPJ["hunt"]["gap_seeds"])))


if __name__ == "__main__":
    unittest.main()
