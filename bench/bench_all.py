# -*- coding: utf-8 -*-
"""四组测量，结果写进 results/。

    python bench/bench_all.py            # 重算并写盘
    python bench/bench_all.py --check    # 只比对，不覆盖

  1. 正确性：我的解 vs HiGHS，跨 LP / MIP 两批实例
  2. 主元规则：Beale 循环例子、Klee–Minty 的指数级迭代
  3. 建模 vs 求解器：同一个问题的紧/松写法，看松弛界与节点数
  4. 代价：我的实现比 HiGHS 慢多少
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lpmip import branch_bound as bb                                    # noqa: E402
from lpmip.model import Model                                           # noqa: E402
from lpmip.simplex import solve_model, solve_standard                   # noqa: E402
from problems import classic, transit                                   # noqa: E402

RESULTS = ROOT / "results"
REL_TOL = 1e-6
# 计时是机器相关的量：同一份代码在不同 CPU / BLAS / 负载下必然给出不同秒数，
# 逐字段比对只会把 CI 变成随机红灯。所以这三个后缀整块豁免——
# **代价是「表过期」这件事 `--check` 抓不到**，得靠 tests/test_all.py 里
# test_the_readme_speed_table_matches_the_committed_bench 把 README 的表钉在
# results/bench.json 上，再靠 test_the_speed_ordering_holds_on_a_fresh_run
# 对新跑结果断言机器无关的那部分（序与量级）。
EXEMPT_SUFFIX = ("seconds", "speed_ratio", "ms")
# 整棵子树豁免：测量环境是"这组秒数是在哪台机器上量的"的说明，跨机器必然不同。
EXEMPT_SUBTREE = ("env",)

# 计时重复次数。噪声只会让一次测量变慢、不会变快，所以取 N 次里的最小值
# 比取平均更接近"这台机器上真正要花多久"。单次计时曾经把 dense_lp_40x60
# 测成 0.0924 s（相邻两档都只有 0.10 / 0.21 s），比值凭空冒出一个 41.1。
REPEATS = 5


def best_of(fn, repeats: int = REPEATS) -> float:
    """跑 repeats 次，返回最快的一次的耗时（秒）。"""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def compare(a, b, path: str = "") -> list:
    """递归逐字段比对：计数与状态严格相等，连续量留容差，计时豁免。"""
    if any(path.endswith("/" + s) or path.endswith(s)
           for s in EXEMPT_SUFFIX + EXEMPT_SUBTREE):
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append("%s/%s: 新增" % (path, k))
            elif k not in b:
                out.append("%s/%s: 缺失" % (path, k))
            else:
                out += compare(a[k], b[k], "%s/%s" % (path, k))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return ["%s: 长度 %d vs %d" % (path, len(a), len(b))]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in compare(x, y, "%s[%d]" % (path, i))]
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a == b else ["%s: %s vs %s" % (path, a, b)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return [] if a == b else ["%s: %d vs %d" % (path, a, b)]
        if abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b)):
            return []
        return ["%s: %r vs %r" % (path, a, b)]
    return [] if a == b else ["%s: %r vs %r" % (path, a, b)]


# ---------------------------------------------------------------- 参照求解
def highs_lp(model: Model):
    sf = model.to_standard_form()
    r = linprog(sf.c, A_eq=sf.A, b_eq=sf.b, bounds=(0, None), method="highs")
    if r.status != 0:
        return {0: "optimal", 2: "infeasible", 3: "unbounded"}.get(r.status, "other"), None
    return "optimal", model.objective_value(sf.recover(r.x))


def highs_mip(model: Model):
    """把 Model 直接喂给 HiGHS 的 MIP。整数性按原始变量给。"""
    n = len(model.vars)
    A, lo, hi = [], [], []
    for con in model.cons:
        row = np.zeros(n)
        for vi, k in con.expr.terms.items():
            row[vi] = k
        rhs = -con.expr.const
        A.append(row)
        if con.sense == "<=":
            lo.append(-np.inf); hi.append(rhs)
        elif con.sense == ">=":
            lo.append(rhs); hi.append(np.inf)
        else:
            lo.append(rhs); hi.append(rhs)
    c = np.zeros(n)
    for vi, k in model.obj.terms.items():
        c[vi] = k
    sign = 1.0 if model.sense == "min" else -1.0
    res = milp(c=sign * c,
               constraints=LinearConstraint(np.array(A), np.array(lo), np.array(hi)) if A else None,
               integrality=np.array([1 if v.integer else 0 for v in model.vars]),
               bounds=Bounds([v.lb for v in model.vars], [v.ub for v in model.vars]))
    if res.status != 0:
        return "infeasible", None
    return "optimal", float(sign * res.fun) + model.obj.const


# ---------------------------------------------------------------- 1 正确性
def bench_correctness() -> dict:
    lp_cases = []
    for name, build in (("blending", classic.blending),
                        ("klee_minty_4", lambda: classic.klee_minty(4)),
                        ("klee_minty_6", lambda: classic.klee_minty(6))):
        m = build()
        mine = solve_model(m)
        st, obj = highs_lp(m)
        lp_cases.append({"case": name, "mine_status": mine.status, "highs_status": st,
                         "mine_obj": round(mine.obj, 9) if mine.x is not None else None,
                         "highs_obj": round(obj, 9) if obj is not None else None,
                         "agree": bool(mine.status == st and (obj is None or
                                                              abs(mine.obj - obj) < 1e-6))})
    mip_cases = []
    for n in (8, 10, 12, 14):
        m, *_ = classic.knapsack(seed=3, n=n)
        r = bb.solve(m)
        st, obj = highs_mip(m)
        mip_cases.append({"case": "knapsack_%d" % n, "mine_status": r.status,
                          "highs_status": st, "mine_obj": round(r.obj, 9),
                          "highs_obj": round(obj, 9) if obj is not None else None,
                          "nodes": r.nodes,
                          "agree": bool(obj is not None and abs(r.obj - obj) < 1e-6)})
    for n in (6, 8, 10, 12):
        trips = transit.instance(n, seed=1)
        m = transit.model_assignment(trips)
        r = bb.solve(m)
        st, obj = highs_mip(m)
        mip_cases.append({"case": "transit_assign_%d" % n, "mine_status": r.status,
                          "highs_status": st, "mine_obj": round(r.obj, 9),
                          "highs_obj": round(obj, 9) if obj is not None else None,
                          "nodes": r.nodes,
                          "agree": bool(obj is not None and abs(r.obj - obj) < 1e-6)})
    return {"lp": lp_cases, "mip": mip_cases,
            "all_agree": all(c["agree"] for c in lp_cases + mip_cases),
            "n_cases": len(lp_cases) + len(mip_cases)}


# ---------------------------------------------------------------- 2 主元规则
def bench_pivot() -> dict:
    A, b, c = classic.beale_cycling()
    out = {}
    for rule in ("dantzig", "bland"):
        r = solve_standard(A, b, c, rule=rule, max_iter=200, auto_bland=False)
        out[rule] = {"status": r.status, "iterations": r.iterations,
                     "obj": round(r.obj, 9) if r.x is not None else None}
    st, ref = "optimal", float(linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun)
    out["highs_obj"] = round(ref, 9)

    km = []
    for n in (3, 4, 5, 6, 7):
        m = classic.klee_minty(n)
        r = solve_model(m, rule="dantzig", max_iter=5000)
        _, obj = highs_lp(m)
        km.append({"n": n, "iterations": r.iterations, "two_pow_n_minus_1": 2 ** n - 1,
                   "matches_worst_case": bool(r.iterations == 2 ** n - 1),
                   "agrees_with_highs": bool(abs(r.obj - obj) < 1e-3)})
    return {"beale": out, "klee_minty": km,
            "note": ("Dantzig 规则在 Beale 例子上**真的循环**（撞迭代上限仍未终止），"
                     "Bland 规则 9 步收敛；Klee–Minty 上 Dantzig 的迭代数**精确等于 2^n − 1**，"
                     "单纯形法最坏情况指数复杂度不是纸面警告。")}


# ---------------------------------------------------------------- 3 建模
def bench_formulations() -> dict:
    rows = []
    for n in (6, 8, 10, 12):
        trips = transit.instance(n, seed=1)
        good = transit.model_assignment(trips)
        lp = solve_model(good)
        r = bb.solve(good)
        rows.append({"trips": n, "formulation": "assignment",
                     "n_vars": len(good.vars), "n_cons": len(good.cons),
                     "lp_bound": round(lp.obj, 6),
                     "mip_obj": round(r.obj, 6), "nodes": r.nodes,
                     "fleet": transit.fleet_from_assignment(trips, r.obj),
                     "lp_already_integral": bool(all(abs(v - round(v)) < 1e-6 for v in lp.x))})
    for n in (4, 5, 6):
        trips = transit.instance(n, seed=1)
        k = transit.fleet_from_assignment(
            trips, bb.solve(transit.model_assignment(trips)).obj)
        big = transit.model_bigM(trips, n_buses=max(k, 1))
        lp = solve_model(big)
        r = bb.solve(big, max_nodes=5000)
        rows.append({"trips": n, "formulation": "bigM",
                     "n_vars": len(big.vars), "n_cons": len(big.cons),
                     "lp_bound": round(lp.obj, 6),
                     "mip_obj": round(r.obj, 6), "nodes": r.nodes,
                     "fleet": int(round(r.obj)),
                     "lp_already_integral": bool(all(abs(v - round(v)) < 1e-6 for v in lp.x))})

    # big-M 取值的影响：同一个模型，只改 M
    cap, profit, fixed, total = classic.production_mip(seed=5, n=7)
    bigm_rows = []
    for M in (None, 100, 1000, 10000):
        m = classic.build_production(cap, profit, fixed, total, big_m=M)
        lp = solve_model(m)
        r = bb.solve(m, max_nodes=8000)
        st, obj = highs_mip(m)
        bigm_rows.append({"big_m": "tight(cap_i)" if M is None else M,
                          "lp_bound": round(lp.obj, 6),
                          "mip_obj": round(r.obj, 6), "nodes": r.nodes,
                          "agrees_with_highs": bool(obj is not None and abs(r.obj - obj) < 1e-6)})
    return {"transit": rows, "big_m_sweep": bigm_rows,
            "note": ("同一个问题，指派型写法的线性松弛**天然是整数**（约束矩阵全幺模），"
                     "分支定界一个节点都不用分；big-M 写法松弛界塌到 1、要探几十个节点。"
                     "big-M 取值本身也一样：M 越大松弛越松、节点越多，最优值却一模一样。")}


# ---------------------------------------------------------------- 4 代价
def measurement_env() -> dict:
    """这组秒数是在哪台机器上量的。**只写进 JSON 与 README，不参与 `--check` 比对**
    （见 EXEMPT_SUBTREE）——换台机器本来就该是另一组数。"""
    try:
        blas = ",".join(np.__config__.get_info("blas_opt").get("libraries", [])) or "unknown"
    except Exception:                                    # numpy 换实现后接口会变
        blas = "unknown"
    return {"os": "%s %s" % (platform.system(), platform.release()),
            "cpu": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas": blas}


README_TABLE_BEGIN = "<!-- speed-table:begin 由 bench/bench_all.py 生成，勿手改 -->"
README_TABLE_END = "<!-- speed-table:end -->"


def render_speed_table(cost: dict) -> str:
    """把 cost 这一组渲染成 README 里那张表。

    **README 的速度表是生成的，不是手打的。** 这个仓在这里栽过一次：第一版的表是
    手打的，`results/bench.json` 后来被重算过，表却没跟着改——而计时字段整块豁免于
    `--check`，于是没有任何机制发现它过期。现在表由本函数从同一份 JSON 渲染，
    `--check` 会把 README 里的这一段和渲染结果逐字比对。
    """
    rows = cost["rows"]
    top = max(r["speed_ratio"] for r in rows)
    lines = [README_TABLE_BEGIN,
             "",
             "| 实例 | 我 | HiGHS | 比值 |",
             "|---|---|---|---|"]
    for r in rows:
        ratio = r["speed_ratio"]
        if ratio < 1.0:
            cell = "**%.1f**（我更快）" % ratio
        elif ratio == top:
            cell = "**%.1f**" % ratio
        else:
            cell = "%.1f" % ratio
        lines.append("| %s | %.4f s | %.4f s | %s |"
                     % (r["case"], r["mine_seconds"], r["highs_seconds"], cell))
    env = cost["env"]
    lines += [
        "",
        "> 每格取 %d 次里最快的一次（噪声只会让一次测量变慢，不会变快）。"
        "测量环境：%s ／ %s ／ Python %s ／ numpy %s（BLAS: %s）。"
        % (cost["repeats"], env["os"], env["cpu"], env["python"], env["numpy"], env["blas"]),
        ">",
        "> **这是全仓唯一一组机器相关的数字**，换台机器重跑必然是另一组秒数，"
        "所以 `--check` 不比对秒数本身；它比对的是**这张表与 `results/bench.json` 一致**。",
        README_TABLE_END,
    ]
    return "\n".join(lines)


def sync_readme_table(cost: dict) -> bool:
    """把 README 里两个标记之间的内容换成新渲染的表。返回是否发生了改动。"""
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    i, j = text.find(README_TABLE_BEGIN), text.find(README_TABLE_END)
    if i < 0 or j < 0:
        raise SystemExit("README.md 里找不到 speed-table 标记")
    fresh = render_speed_table(cost)
    new = text[:i] + fresh + text[j + len(README_TABLE_END):]
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def check_readme_table(cost: dict) -> list:
    """README 里的表必须与已提交的 cost 一致。返回不一致的说明（空表示一致）。"""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    i, j = text.find(README_TABLE_BEGIN), text.find(README_TABLE_END)
    if i < 0 or j < 0:
        return ["README.md 里找不到 speed-table 标记"]
    got = text[i:j + len(README_TABLE_END)]
    want = render_speed_table(cost)
    if got == want:
        return []
    g, w = got.splitlines(), want.splitlines()
    diff = ["README 的速度表与 results/bench.json 不一致（跑一次 `python bench/bench_all.py` 会重新生成）："]
    for k in range(max(len(g), len(w))):
        a = g[k] if k < len(g) else "<缺行>"
        b = w[k] if k < len(w) else "<多行>"
        if a != b:
            diff.append("  README: %s" % a)
            diff.append("  应为  : %s" % b)
    return diff


def bench_cost() -> dict:
    rows = []
    for n in (3, 4, 5, 6):
        m = classic.klee_minty(n)
        mine = best_of(lambda m=m: solve_model(m))
        sf = m.to_standard_form()
        theirs = best_of(lambda sf=sf: linprog(sf.c, A_eq=sf.A, b_eq=sf.b,
                                               bounds=(0, None), method="highs"))
        rows.append({"case": "klee_minty_%d" % n, "mine_seconds": round(mine, 6),
                     "highs_seconds": round(theirs, 6),
                     "speed_ratio": round(mine / max(theirs, 1e-9), 1)})
    for n in (10, 12, 14):
        m, *_ = classic.knapsack(seed=3, n=n)
        mine = best_of(lambda m=m: bb.solve(m))
        theirs = best_of(lambda m=m: highs_mip(m))
        rows.append({"case": "knapsack_%d" % n, "mine_seconds": round(mine, 6),
                     "highs_seconds": round(theirs, 6),
                     "speed_ratio": round(mine / max(theirs, 1e-9), 1)})
    # 上面那些实例太小，小到 HiGHS 的固定开销盖过求解本身——我的实现甚至更快。
    # 要看真实差距得把规模拉起来：随机稠密 LP，m 行 n 列。
    for m_, n_ in ((20, 30), (40, 60), (60, 90), (80, 120)):
        rng = np.random.default_rng(7)
        A = rng.uniform(0.5, 3.0, (m_, n_))
        b = rng.uniform(50.0, 100.0, m_)
        c = -rng.uniform(1.0, 5.0, n_)
        mod = Model("dense_%dx%d" % (m_, n_), sense="min")
        xs = [mod.add_var("x%d" % j, 0, 20) for j in range(n_)]
        for i in range(m_):
            e = A[i, 0] * xs[0]
            for j in range(1, n_):
                e = e + A[i, j] * xs[j]
            mod.add_constraint(e <= float(b[i]))
        obj = c[0] * xs[0]
        for j in range(1, n_):
            obj = obj + c[j] * xs[j]
        mod.set_objective(obj)

        r = solve_model(mod)
        mine = best_of(lambda mod=mod: solve_model(mod))
        sf = mod.to_standard_form()
        ref = linprog(sf.c, A_eq=sf.A, b_eq=sf.b, bounds=(0, None), method="highs")
        theirs = best_of(lambda sf=sf: linprog(sf.c, A_eq=sf.A, b_eq=sf.b,
                                               bounds=(0, None), method="highs"))
        rows.append({"case": "dense_lp_%dx%d" % (m_, n_), "mine_seconds": round(mine, 6),
                     "highs_seconds": round(theirs, 6),
                     "speed_ratio": round(mine / max(theirs, 1e-9), 1),
                     "agrees_with_highs": bool(
                         r.status == "optimal" and ref.status == 0
                         and abs(r.obj - mod.objective_value(sf.recover(ref.x)))
                         < 1e-6 * max(1.0, abs(r.obj)))})
    return {"rows": rows,
            "repeats": REPEATS,
            "env": measurement_env(),
            "note": ("每格取 %d 次里最快的一次。**「比 HiGHS 慢」这句话在小实例上不成立**——"
                     "背包和 Klee–Minty 这种规模上，HiGHS 的固定开销盖过求解本身，"
                     "我的实现有时反而更快。要看真实差距得把规模拉起来：稠密 LP 上差距到几十倍量级。"
                     "**这一组是全仓唯一一组机器相关的数字**，`--check` 不比对它（见 EXEMPT_SUFFIX），"
                     "换台机器重跑必然是另一组秒数。写这个仓是为了把方法搞明白，"
                     "**不是造一个该上生产的求解器**——生产上该用 HiGHS/Gurobi。" % REPEATS)}


def bench_cuts() -> dict:
    """Gomory 分数割：根节点加割，看界抬高多少、节点数变几个。

    **n=12 是反例，留着不删**：割确实抬高了界，节点数反而从 29 涨到 31。
    割不是免费的——每加一行，每个节点的线性松弛就大一点。

    最后一行是**有效性闸门**：同一个 `cuts=True` 传给混合整数模型必须被拒绝。
    分数割要求行里所有变量（含松弛变量）都取整数，混合整数不满足，
    那需要 GMI 割，本仓没做。
    """
    from lpmip.cuts import tighten, validate_pure_integer

    rows = []
    for n in (8, 10, 12, 14):
        m, *_ = classic.knapsack(seed=3, n=n)
        _, stats = tighten(m)
        plain = bb.solve(m)
        cut = bb.solve(m, cuts=True)
        st, ref = highs_mip(m)
        rows.append({"case": "knapsack_%d" % n,
                     "lp_bound_before": stats["bound_before"],
                     "lp_bound_after": stats["bound_after"],
                     "cuts_added": stats["cuts"],
                     "rounds": stats["rounds"],
                     "nodes_without_cuts": plain.nodes,
                     "nodes_with_cuts": cut.nodes,
                     "mip_obj": round(plain.obj, 6),
                     "same_optimum_with_cuts": bool(abs(plain.obj - cut.obj) < 1e-6),
                     "agrees_with_highs": bool(st == "optimal"
                                               and abs(plain.obj - ref) < 1e-6)})

    cap, profit, fixed, total = classic.production_mip(seed=5, n=7)
    mixed = classic.build_production(cap, profit, fixed, total, big_m=None)
    ok, why = validate_pure_integer(mixed)
    refused = False
    try:
        bb.solve(mixed, cuts=True)
    except ValueError:
        refused = True

    helped = sum(1 for r in rows if r["nodes_with_cuts"] < r["nodes_without_cuts"])
    return {"rows": rows,
            "n_cases": len(rows),
            "all_agree_with_highs": all(r["agrees_with_highs"] for r in rows),
            "all_same_optimum": all(r["same_optimum_with_cuts"] for r in rows),
            "n_helped": helped,
            "n_hurt": sum(1 for r in rows
                          if r["nodes_with_cuts"] > r["nodes_without_cuts"]),
            "mixed_integer_is_refused": bool(refused and not ok),
            "mixed_integer_reason": why,
            "note": ("%d/%d 个实例的节点数被割压到 1，**但 n=12 反而从 29 涨到 31**。"
                     "割抬高了界却让每个节点更贵，净效果可以是负的——"
                     "这一行留在表里，不挑数据。" % (helped, len(rows)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    fresh = {
        "correctness": bench_correctness(),
        "pivot": bench_pivot(),
        "formulations": bench_formulations(),
        "cuts": bench_cuts(),
        "cost": bench_cost(),
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "bench.json"
    if args.check:
        if not out.exists():
            print("missing", out); return 1
        committed = json.loads(out.read_text(encoding="utf-8"))
        drift = compare(committed, fresh)
        if drift:
            print("DRIFT (%d 处)：" % len(drift))
            for line in drift[:20]:
                print("  " + line)
            return 1
        # 计时整块豁免于上面的比对，所以单独钉一件事：README 那张表必须还是
        # 已提交 results/bench.json 渲染出来的样子。这一步与机器无关。
        stale = check_readme_table(committed["cost"])
        if stale:
            for line in stale:
                print(line)
            return 1
        print("ok: results 与新跑一致（计时豁免），README 速度表与已提交结果一致")
        return 0
    out.write_text(json.dumps(fresh, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    if sync_readme_table(fresh["cost"]):
        print("updated README.md 速度表（这台机器的秒数）")
    print("正确性：%d 个实例，全部与 HiGHS 一致 = %s"
          % (fresh["correctness"]["n_cases"], fresh["correctness"]["all_agree"]))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
