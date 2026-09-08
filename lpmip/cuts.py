# -*- coding: utf-8 -*-
"""Gomory 分数割：真的那种。

第一版这里放过一个假货——对分数变量加 `x <= floor(v)`，那会**切掉可行整数解**，
是一次分支伪装成割。这一版按定义做，并且**每个实例都拿 HiGHS 验最优值**。

## 推导

设线性松弛的最终单纯形表里，基变量 x_k 取了分数值。该行读作

    x_k + Σ_{j 非基} α_j · x_j = β        （β = x_k 的取值）

对任何可行整数点，把每个系数下取整得到有效不等式
`x_k + Σ ⌊α_j⌋ x_j ≤ ⌊β⌋`，与原式相减即得 **Gomory 分数割**

    Σ_{j 非基} {α_j} · x_j  ≥  {β}          其中 {r} = r − ⌊r⌋

## 有效性前提（这条最容易被忽略）

上面那步"下取整仍有效"要求**行里所有变量都取整数值**——**包括松弛变量**。
所以分数割只对**纯整数规划、且约束数据为整数**成立：那时 `Ax + s = b` 里
A、b 皆整、x 整，s 才整。混合整数要用 Gomory 混合整数割（GMI），本模块**不做**，
`validate_pure_integer()` 会直接拒绝。

## 把松弛变量代回原变量

割写出来带松弛变量：`e'x + r's ≥ d`。标准型里每行至多一个松弛/剩余列，
系数 σ = ±1，于是 `σ·s_i = b_i − Σ_{j∈结构列} A[i,j]·x_j`，代回去就只剩结构列；
再经 `StandardForm.mapping` 折回原始变量（平移与正负部）。

参考：Gomory (1958)；Cordone《Gomory cuts》讲义；Letchford & Lodi (2002)
"Strengthening Chvátal–Gomory cuts and Gomory fractional cuts", ORL 30:74–82。
"""
from __future__ import annotations

import math

import numpy as np

from .model import INF, Constraint, LinExpr, Model
from .simplex import solve_model

EPS = 1e-9
FRAC_TOL = 1e-6
KEEP_TOL = 1e-7          # 一条割要被保留，界至少得抬高这个相对量（见 tighten）


def _frac(v: float) -> float:
    f = v - math.floor(v + EPS)
    return 0.0 if f < FRAC_TOL or f > 1 - FRAC_TOL else f


def validate_pure_integer(model: Model) -> tuple[bool, str]:
    """分数割只对纯整数、整数数据的模型有效。不满足就说清哪一条不满足。"""
    if not model.integer_vars():
        return False, "模型没有整数变量"
    cont = [v.name for v in model.vars if not v.integer]
    if cont:
        return False, ("存在连续变量 %s —— 分数割不适用于混合整数，"
                       "那需要 Gomory 混合整数割（本模块未实现）" % cont[:3])
    for con in model.cons:
        for k in con.expr.terms.values():
            if abs(k - round(k)) > 1e-9:
                return False, "约束系数 %r 不是整数，松弛变量因此不必取整" % k
        if abs(con.expr.const - round(con.expr.const)) > 1e-9:
            return False, "右端 %r 不是整数" % (-con.expr.const)
    for v in model.vars:
        for bnd in (v.lb, v.ub):
            if bnd not in (-INF, INF) and abs(bnd - round(bnd)) > 1e-9:
                return False, "变量界 %r 不是整数" % bnd
    return True, "纯整数且数据为整数，分数割有效"


def gomory_cut_rows(sf, tableau: np.ndarray, basis: list, max_cuts: int = 5):
    """从最终单纯形表导出若干条割，返回 [(标准型列上的系数 g, 右端 r)]，语义为 g·y >= r。"""
    m = tableau.shape[0] - 1
    n = sf.A.shape[1]
    out = []
    for i in range(m):
        if len(out) >= max_cuts:
            break
        beta = tableau[i, -1]
        fb = _frac(beta)
        if fb <= 0.0:
            continue
        g = np.zeros(n)
        for j in range(n):
            if j == basis[i]:
                continue
            fj = _frac(tableau[i, j])
            if fj > 0.0:
                g[j] = fj
        if g.sum() > EPS:
            out.append((g, fb))
    return out


def cut_to_constraint(sf, model: Model, g: np.ndarray, rhs: float) -> Constraint | None:
    """把标准型上的割 `g·y >= rhs` translate 成原始变量上的约束。

    先用每行的松弛列把 s 消成结构列，再用 mapping 折回原始变量。
    """
    A, b = sf.A, sf.b
    n_struct = len(sf.mapping) and max(
        max(p for p, _, _ in sf.mapping),
        max((q for _, q, _ in sf.mapping if q is not None), default=-1)) + 1

    coef = np.array(g[:n_struct], dtype=float)      # 结构列上的系数
    const = 0.0                                     # 移到左边的常数
    for j in range(n_struct, A.shape[1]):
        if abs(g[j]) <= EPS:
            continue
        rows = np.flatnonzero(np.abs(A[:, j]) > EPS)
        if len(rows) != 1:
            return None                             # 不是"每行一个松弛"的形状，放弃
        i = int(rows[0])
        sigma = A[i, j]                             # ±1
        # sigma·s = b_i − Σ_struct A[i,k]·y_k  =>  s = (b_i − Σ...)/sigma
        w = g[j] / sigma
        const += w * b[i]
        coef[:n_struct] -= w * A[i, :n_struct]

    # 结构列 -> 原始变量：y_pos − y_neg = x − shift
    terms: dict[int, float] = {}
    shift_const = 0.0
    for vi, (pos, neg, shift) in enumerate(sf.mapping):
        k = coef[pos] if pos is not None else 0.0
        if neg is not None:
            if abs(coef[pos] + coef[neg]) > 1e-7:
                return None                         # 正负部系数不对称，无法折回
            k = coef[pos]
        if abs(k) > EPS:
            terms[vi] = terms.get(vi, 0.0) + k
            shift_const += k * shift
    if not terms:
        return None
    # Σ k·(x − shift) + const >= rhs   =>   Σ k·x >= rhs − const + Σ k·shift
    expr = LinExpr(terms, 0.0)
    return Constraint(expr - (rhs - const + shift_const), ">=")


def tighten(model: Model, rounds: int = 3, max_cuts_per_round: int = 5,
            verbose: bool = False) -> tuple[Model, dict]:
    """在根节点反复加割，返回 (加过割的模型, 统计)。

    每轮：解松弛 -> 取分数基变量的行 -> 导出割 -> 折回原变量 -> 加进模型。
    **只在割真的抬高了松弛界时才保留**——加了不涨的割纯属浪费，还会拖慢每个节点。
    """
    ok, why = validate_pure_integer(model)
    if not ok:
        raise ValueError("Gomory 分数割不适用于该模型：%s" % why)

    cur = model
    base = solve_model(cur, want_tableau=True)
    if base.status != "optimal":
        return cur, {"rounds": 0, "cuts": 0, "bound_before": None, "bound_after": None,
                     "reason": base.status}
    bound0 = base.obj
    total = 0
    history = [round(bound0, 9)]
    for _ in range(rounds):
        r = solve_model(cur, want_tableau=True)
        if r.status != "optimal" or r.tableau is None:
            break
        sf = cur.to_standard_form()
        rows = gomory_cut_rows(sf, r.tableau, r.basis, max_cuts_per_round)
        if not rows:
            break
        added = 0
        for g, rhs in rows:
            con = cut_to_constraint(sf, cur, g, rhs)
            if con is None:
                continue
            trial = Model(cur.name, cur.sense)
            for v in cur.vars:
                trial.add_var(v.name, v.lb, v.ub, v.integer)
            trial.cons = list(cur.cons) + [con]
            trial.obj = cur.obj
            t = solve_model(trial)
            if t.status != "optimal":
                continue                                    # 割把 LP 弄不可行了，丢弃
            # **相对阈值，不是 1e-9 绝对阈值。** 保留与否是个离散决定：
            # 若某条割只把界抬高 1e-9 量级，那么换一台机器（BLAS 不同、最后几位有出入）
            # 就可能保留/丢弃不同的割，`cuts_added` 与节点数随之漂移，
            # 提交进 results/ 的数字就复现不了。eda-copilot 正是栽在这类地方。
            need = KEEP_TOL * max(1.0, abs(r.obj))
            improves = (t.obj > r.obj + need) if cur.sense == "min" else (t.obj < r.obj - need)
            if improves:
                cur, added = trial, added + 1
        total += added
        history.append(round(solve_model(cur).obj, 9))
        if added == 0:
            break
    after = solve_model(cur)
    return cur, {"rounds": len(history) - 1, "cuts": total,
                 "bound_before": round(bound0, 9),
                 "bound_after": round(after.obj, 9),
                 "bound_trace": history,
                 "n_cons_before": len(model.cons), "n_cons_after": len(cur.cons)}
