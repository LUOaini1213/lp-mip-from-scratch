# -*- coding: utf-8 -*-
"""两阶段表格单纯形法，两种主元规则。

    min c'x   s.t.  Ax = b (b >= 0),  x >= 0

**为什么要写两种主元规则**：Dantzig 规则（选最负的检验数）是教科书默认，跑得快，
但**会循环**——在退化顶点上原地打转永不终止。Bland 规则（进基与离基都取最小下标）
可证明有限终止，代价是通常多迭代几步。这不是纸上谈兵：`problems/cycling.py` 里的
Beale 例子在 Dantzig 规则下真的会转圈，`bench/bench_pivot.py` 把两者摆在一起。

默认用 Dantzig + 迭代上限，撞上限就自动重启为 Bland——生产求解器也大多这么做
（先快后稳），而不是一上来就用最慢但最稳的那个。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9


@dataclass
class LPResult:
    status: str                 # optimal / infeasible / unbounded / cycling / iteration_limit
    x: np.ndarray | None
    obj: float
    iterations: int
    pivot_rule: str
    restarted_with_bland: bool = False
    message: str = ""
    # 最终单纯形表与基（仅当 want_tableau=True 时填）。割平面要从这里取行。
    tableau: np.ndarray | None = None
    basis: list | None = None


def _choose_entering(reduced: np.ndarray, rule: str) -> int:
    """选进基列。返回 -1 表示已最优。"""
    if rule == "bland":
        for j in range(len(reduced)):            # 最小下标
            if reduced[j] < -EPS:
                return j
        return -1
    j = int(np.argmin(reduced))                  # Dantzig：最负
    return j if reduced[j] < -EPS else -1


def _choose_leaving(T: np.ndarray, basis: list, col: int, rule: str) -> int:
    """最小比值检验。返回 -1 表示无界。

    平局怎么破，正是循环与否的关键：Bland 规则要求在**基变量下标最小**的那行离基。
    """
    m = T.shape[0] - 1
    best_ratio, cands = np.inf, []
    for i in range(m):
        a = T[i, col]
        if a > EPS:
            r = T[i, -1] / a
            if r < best_ratio - EPS:
                best_ratio, cands = r, [i]
            elif abs(r - best_ratio) <= EPS:
                cands.append(i)
    if not cands:
        return -1
    if rule == "bland":
        return min(cands, key=lambda i: basis[i])
    return cands[0]


def _pivot(T: np.ndarray, basis: list, row: int, col: int) -> None:
    T[row, :] /= T[row, col]
    for i in range(T.shape[0]):
        if i != row and abs(T[i, col]) > 0:
            T[i, :] -= T[i, col] * T[row, :]
    basis[row] = col


def _run(T: np.ndarray, basis: list, rule: str, max_iter: int) -> tuple[str, int]:
    it = 0
    while it < max_iter:
        j = _choose_entering(T[-1, :-1], rule)
        if j < 0:
            return "optimal", it
        i = _choose_leaving(T, basis, j, rule)
        if i < 0:
            return "unbounded", it
        _pivot(T, basis, i, j)
        it += 1
    return "iteration_limit", it


def solve_standard(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                   rule: str = "dantzig", max_iter: int = 10000,
                   auto_bland: bool = True, want_tableau: bool = False) -> LPResult:
    """解 min c'x, Ax=b, x>=0。两阶段法。"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    m, n = A.shape
    assert np.all(b >= -EPS), "标准型要求 b >= 0，归约时应已处理"

    # ---- 第一阶段：加人工变量，最小化它们的和，找一个可行基 ----
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    basis = list(range(n, n + m))
    T[-1, n:n + m] = 0.0
    T[-1, :n] = -A.sum(axis=0)                  # 把人工变量从目标里消掉
    T[-1, -1] = -b.sum()

    status, it1 = _run(T, basis, rule, max_iter)
    restarted = False
    if status == "iteration_limit":
        if auto_bland and rule != "bland":
            return _restart_bland(A, b, c, max_iter)
        # A positive artificial objective proves infeasibility only at a
        # converged phase-I optimum. Budget exhaustion is no such certificate.
        return LPResult("iteration_limit", None, np.nan, it1, rule, False,
                        "phase I did not converge within %d iterations (%s)" % (max_iter, rule))
    if -T[-1, -1] > 1e-7:
        return LPResult("infeasible", None, np.inf, it1, rule, False,
                        "第一阶段目标 %.3g > 0" % (-T[-1, -1]))

    # 人工变量若还留在基里，尝试换出去（退化时会发生）
    for i in range(m):
        if basis[i] >= n:
            for j in range(n):
                if abs(T[i, j]) > EPS:
                    _pivot(T, basis, i, j)
                    break

    # ---- 第二阶段：换回真实目标 ----
    T2 = np.zeros((m + 1, n + 1))
    T2[:m, :n] = T[:m, :n]
    T2[:m, -1] = T[:m, -1]
    T2[-1, :n] = c
    T2[-1, -1] = 0.0
    for i in range(m):
        if basis[i] < n:
            T2[-1, :] -= T2[-1, basis[i]] * T2[i, :]

    status, it2 = _run(T2, basis, rule, max_iter)
    if status == "iteration_limit":
        if auto_bland and rule != "bland":
            return _restart_bland(A, b, c, max_iter)
        return LPResult("cycling", None, np.nan, it1 + it2, rule, False,
                        "%s 规则在 %d 次迭代内未终止" % (rule, max_iter))
    if status == "unbounded":
        return LPResult("unbounded", None, -np.inf, it1 + it2, rule)

    x = np.zeros(n)
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T2[i, -1]
    res = LPResult("optimal", x, float(c @ x), it1 + it2, rule, restarted)
    if want_tableau:
        res.tableau, res.basis = T2, list(basis)
    return res


def _restart_bland(A, b, c, max_iter) -> LPResult:
    r = solve_standard(A, b, c, rule="bland", max_iter=max_iter, auto_bland=False)
    r.restarted_with_bland = True
    r.message = "Dantzig 规则撞上迭代上限，自动重启为 Bland 规则" + (": " + r.message if r.message else "")
    return r


def solve_model(model, rule: str = "dantzig", max_iter: int = 10000,
                want_tableau: bool = False) -> LPResult:
    """解一个 `lpmip.model.Model` 的**线性松弛**（忽略整数约束）。"""
    sf = model.to_standard_form()
    r = solve_standard(sf.A, sf.b, sf.c, rule=rule, max_iter=max_iter,
                       want_tableau=want_tableau)
    if r.x is not None:
        xr = sf.recover(r.x)
        r.x = xr
        r.obj = model.objective_value(xr)
    return r
