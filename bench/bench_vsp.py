# -*- coding: utf-8 -*-
"""时空网络 VSP / MDVSP 的测量——**主要用途是推翻我自己写下的一条结论**。

    python bench/bench_vsp.py             # 重算并写 results/vsp.json
    python bench/bench_vsp.py --check     # 只比对
    python bench/bench_vsp.py --hunt 6000 # 重跑那次随机搜索（慢，约 8 分钟，不进 CI）

三组：

  1. **阶梯**：单场站 -> 两场站 -> 三场站 -> 非对称空驶 -> 场站容量上限。
     每一档都记「松弛里有几个分数变量」「分支定界几个节点」「与 HiGHS 是否一致」。
     我原本预期分数变量会在"两场站"这一档出现。**它没有。**
  2. **冻结的 6 个反例**：`vsp.FRACTIONAL_SEEDS`，加上场站准入限制之后真出分数松弛的实例。
  3. **搜索统计**：6000 个随机实例里命中 6 个（0.10%），6 个全在 n=8。
     这个数字是 `--hunt 6000` 跑出来的，默认不重跑（太慢），但可以复现。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.bench_all import compare, highs_mip                          # noqa: E402
from lpmip import branch_bound as bb                                    # noqa: E402
from lpmip.simplex import solve_model                                   # noqa: E402
from problems import transit, vsp                                       # noqa: E402

RESULTS = ROOT / "results"
FRAC_TOL = 1e-6


def _n_fractional(x) -> int:
    return sum(1 for v in x if abs(v - round(v)) > FRAC_TOL)


def _row(label: str, m, max_nodes: int = 20000) -> dict:
    lp = solve_model(m)
    if lp.status != "optimal":
        return {"case": label, "lp_status": lp.status}
    mip = bb.solve(m, max_nodes=max_nodes)
    st, ref = highs_mip(m)
    agrees = (st == "optimal" and mip.status == "optimal"
              and abs(mip.obj - ref) < 1e-6 * max(1.0, abs(ref)))
    return {"case": label,
            "n_vars": len(m.vars), "n_cons": len(m.cons),
            "lp_bound": round(lp.obj, 6),
            "n_fractional": _n_fractional(lp.x),
            "mip_obj": round(mip.obj, 6),
            "gap_pct": round((mip.obj - lp.obj) / max(abs(mip.obj), 1e-9) * 100, 4),
            "nodes": mip.nodes,
            "agrees_with_highs": bool(agrees)}


def _asymmetric_table(trips, depots, seed: int) -> dict:
    """一张非对称空驶表。纯标准库随机，站名排过序 —— 跨进程一致。"""
    import random
    rng = random.Random(seed)
    names = sorted({t.frm for t in trips} | {t.to for t in trips}) + list(depots)
    return {(a, b): rng.choice([2, 4, 7, 11, 16, 23, 31])
            for a in names for b in names if a != b}


def bench_ladder() -> dict:
    """一档一档加耦合，看分数松弛什么时候出现。**答案是：这五档都没出现。**"""
    rows = []
    for n in (6, 8):
        trips = transit.instance(n, seed=1)
        rows.append(_row("single_depot_n%d" % n, vsp.model_single_depot(trips)))
        rows.append(_row("two_depots_n%d" % n,
                         vsp.model_multi_depot(trips, ["dA", "dB"])))
        rows.append(_row("three_depots_n%d" % n,
                         vsp.model_multi_depot(trips, ["dA", "dB", "dC"])))

        tb = _asymmetric_table(trips, ["dA", "dB"], seed=n)
        rows.append(_row("two_depots_asym_n%d" % n,
                         vsp.model_multi_depot(trips, ["dA", "dB"], dh_table=tb)))

        m = vsp.model_multi_depot(trips, ["dA", "dB"], dh_table=tb)
        _, arcs = vsp.build_network(trips, ["dA", "dB"], 5.0, tb)
        vsp.add_depot_capacity(m, arcs, {"dA": 2, "dB": 2})
        rows.append(_row("two_depots_capped_n%d" % n, m))

        # 车辆固定成本压到与空驶同量级：万一整数性只是被 1000 这个大常数"压"出来的
        rows.append(_row("two_depots_cheapveh_n%d" % n,
                         vsp.model_multi_depot(trips, ["dA", "dB"], dh_table=tb,
                                               vehicle_cost=40.0)))
    frac = [r for r in rows if r.get("n_fractional")]
    return {"rows": rows,
            "n_rows": len(rows),
            "n_rows_with_fractional_lp": len(frac),
            "all_agree_with_highs": all(r.get("agrees_with_highs") for r in rows),
            "note": ("**只加场站不产生分数松弛。** 非对称空驶、场站容量上限、"
                     "把车辆固定成本压到与空驶同量级，都试过，%d 档全部松弛整数、"
                     "一个节点解完。原先写在 vsp.py docstring 里的"
                     "「多一个场站，难度就换一个量级」是错的。" % len(rows))}


def bench_frozen_counterexamples() -> dict:
    """加上场站准入限制之后，真出分数松弛的 6 个实例。"""
    rows = []
    for s in vsp.FRACTIONAL_SEEDS:
        ins = vsp.random_instance(s)
        r = _row("eligibility_seed%d" % s, vsp.model_from_random(s))
        r["n_trips"] = ins["n"]
        r["n_depots"] = len(ins["depots"])
        r["vehicle_cost"] = ins["vehicle_cost"]
        rows.append(r)
    return {"rows": rows,
            "all_fractional": all(r["n_fractional"] > 0 for r in rows),
            "all_agree_with_highs": all(r["agrees_with_highs"] for r in rows),
            "all_n_trips_8": all(r["n_trips"] == 8 for r in rows),
            "n_with_zero_gap": sum(1 for r in rows if abs(r["gap_pct"]) < 1e-9),
            "note": ("6 个反例全部落在 n=8（搜索范围 n=4..8 的最大档）。"
                     "其中 %d 个的界间隙是 0.00%% —— **松弛出分数解不等于界变松**，"
                     "存在一个分数最优解和一个整数最优解目标值相同的情形。"
                     % sum(1 for r in rows if abs(r["gap_pct"]) < 1e-9))}


def hunt(n_seeds: int) -> dict:
    """重跑随机搜索。慢：6000 个约 8 分钟。"""
    hits, tried = [], 0
    for seed in range(1, n_seeds + 1):
        try:
            m = vsp.model_from_random(seed)
        except Exception:
            continue
        tried += 1
        r = solve_model(m)
        if r.status != "optimal":
            continue
        if _n_fractional(r.x):
            hits.append(seed)
            print("  命中 seed=%d" % seed, flush=True)
        if seed % 500 == 0:
            print("  ... %d/%d 命中 %d" % (seed, n_seeds, len(hits)), flush=True)
    return {"tried": tried, "hits": hits, "n_hits": len(hits),
            "hit_rate_pct": round(100.0 * len(hits) / max(tried, 1), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--hunt", type=int, default=0,
                    help="重跑随机搜索的实例数（慢，不进 CI）")
    args = ap.parse_args()

    if args.hunt:
        h = hunt(args.hunt)
        print("试 %d，命中 %d（%.4f%%）：%s" % (h["tried"], h["n_hits"],
                                              h["hit_rate_pct"], h["hits"]))
        # 只比对搜索范围内的种子：--hunt 300 不该因为没跑到 seed=3845 就算失败。
        expect = [s for s in vsp.FRACTIONAL_SEEDS if s <= args.hunt]
        got = [s for s in h["hits"] if s <= args.hunt]
        if got != expect:
            print("不一致：期望 %s，实得 %s" % (expect, got))
            return 1
        print("ok: 命中的种子与 vsp.FRACTIONAL_SEEDS 在 seed<=%d 范围内完全一致" % args.hunt)
        return 0

    fresh = {
        "ladder": bench_ladder(),
        "counterexamples": bench_frozen_counterexamples(),
        "hunt": {"tried": vsp.HUNT_TRIED, "hits": list(vsp.FRACTIONAL_SEEDS),
                 "n_hits": len(vsp.FRACTIONAL_SEEDS),
                 "hit_rate_pct": round(100.0 * len(vsp.FRACTIONAL_SEEDS)
                                       / vsp.HUNT_TRIED, 4),
                 "note": ("这一组不在默认路径上重跑（6000 个实例约 8 分钟）。"
                          "用 `--hunt 6000` 复现；命中的种子必须包含 FRACTIONAL_SEEDS。")},
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "vsp.json"
    if args.check:
        if not out.exists():
            print("missing", out)
            return 1
        drift = compare(json.loads(out.read_text(encoding="utf-8")), fresh)
        if drift:
            print("DRIFT (%d 处)：" % len(drift))
            for line in drift[:20]:
                print("  " + line)
            return 1
        print("ok: results/vsp.json 与新跑一致")
        return 0
    out.write_text(json.dumps(fresh, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    print("阶梯：%d 档，其中松弛出分数解的 %d 档"
          % (fresh["ladder"]["n_rows"], fresh["ladder"]["n_rows_with_fractional_lp"]))
    print("反例：%d 个，全部分数 = %s，全部与 HiGHS 一致 = %s"
          % (len(fresh["counterexamples"]["rows"]),
             fresh["counterexamples"]["all_fractional"],
             fresh["counterexamples"]["all_agree_with_highs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
