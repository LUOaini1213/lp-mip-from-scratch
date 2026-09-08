# -*- coding: utf-8 -*-
"""分支定界求 MIP。

骨架是教科书的：解线性松弛 -> 若整数解则更新上界 -> 否则挑一个分数变量分两支。
真正决定快慢的是三个选择，每个都做成了开关，好在 bench 里量出各自值多少：

  branching   分支变量怎么挑：most_fractional / first
  search      节点怎么取：best_first（先看界最好的）/ dfs（先扎到底拿可行解）
  cuts        **未实现**，传 True 会直接抛错——见 `_add_gomory_cuts` 的 docstring

**这两个开关加起来，都不如把模型写对重要。** `bench/bench_formulations.py` 里同一个
问题换个写法，节点数差两个数量级，而上面三个开关全开也只是几倍。这是这个仓最想讲的一条。

界的处理按最小化写死（`Model` 若是 max，`to_standard_form` 已经取过负）。
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .model import INF, Model
from .simplex import solve_model

EPS = 1e-7


@dataclass
class MIPResult:
    status: str                  # optimal / infeasible / node_limit
    x: np.ndarray | None
    obj: float
    nodes: int                   # 解过线性松弛的节点数
    lp_iterations: int
    cuts_added: int = 0
    gap: float = 0.0
    log: list = field(default_factory=list)


@dataclass(order=True)
class _Node:
    key: float                                  # 堆序：界 或 深度
    seq: int
    lo: dict = field(compare=False, default_factory=dict)
    hi: dict = field(compare=False, default_factory=dict)
    depth: int = field(compare=False, default=0)


def _is_integral(v: float, tol: float = 1e-6) -> bool:
    return abs(v - round(v)) <= tol


def _clone_with_bounds(model: Model, lo: dict, hi: dict) -> Model:
    """按分支加紧变量界后的子问题。只改界，不复制约束对象。"""
    sub = Model(model.name, model.sense)
    for v in model.vars:
        sub.add_var(v.name,
                    max(v.lb, lo.get(v.idx, -INF)),
                    min(v.ub, hi.get(v.idx, INF)),
                    v.integer)
    sub.cons = model.cons
    sub.obj = model.obj
    return sub


def _pick_branch_var(x: np.ndarray, int_idx: list, strategy: str) -> int:
    frac = [(i, abs(x[i] - round(x[i]))) for i in int_idx if not _is_integral(x[i])]
    if not frac:
        return -1
    if strategy == "first":
        return frac[0][0]
    return max(frac, key=lambda t: t[1])[0]         # most_fractional


def solve(model: Model, branching: str = "most_fractional", search: str = "best_first",
          cuts: bool = False, max_nodes: int = 20000,
          verbose: bool = False) -> MIPResult:
    int_idx = model.integer_vars()
    if not int_idx:
        r = solve_model(model)
        return MIPResult(r.status, r.x, r.obj, 1, r.iterations)

    minimise = (model.sense == "min")
    best_obj = INF if minimise else -INF
    best_x = None
    nodes = lp_iters = n_cuts = 0
    log: list = []

    root = _Node(0.0, 0)
    heap = [root]
    seq = 1

    cut_model = model
    if cuts:
        _add_gomory_cuts(model)          # 直接抛 NotImplementedError，不静默降级

    while heap and nodes < max_nodes:
        node = heapq.heappop(heap)
        sub = _clone_with_bounds(cut_model, node.lo, node.hi)
        r = solve_model(sub)
        nodes += 1
        lp_iters += r.iterations

        if r.status != "optimal":
            continue                                     # 不可行或无界，剪掉
        bound = r.obj
        if best_x is not None:                           # 界剪枝
            if minimise and bound >= best_obj - 1e-9:
                continue
            if not minimise and bound <= best_obj + 1e-9:
                continue

        j = _pick_branch_var(r.x, int_idx, branching)
        if j < 0:                                        # 松弛解已整数 -> 可行解
            if (minimise and bound < best_obj) or (not minimise and bound > best_obj):
                best_obj, best_x = bound, r.x.copy()
                log.append({"node": nodes, "incumbent": round(bound, 9)})
            continue

        v = r.x[j]
        for lo, hi in ((dict(node.lo), {**node.hi, j: math.floor(v)}),
                       ({**node.lo, j: math.ceil(v)}, dict(node.hi))):
            key = bound if search == "best_first" else -(node.depth + 1)
            if not minimise and search == "best_first":
                key = -bound
            heapq.heappush(heap, _Node(key, seq, lo, hi, node.depth + 1))
            seq += 1

    status = "optimal" if best_x is not None else "infeasible"
    if nodes >= max_nodes and heap:
        status = "node_limit"
    return MIPResult(status, best_x, best_obj if best_x is not None else np.nan,
                     nodes, lp_iters, n_cuts, 0.0, log)


def _add_gomory_cuts(model: Model, rounds: int = 1, max_cuts: int = 20):
    """**没有实现。** 这里原本有一个"割平面"，但它是错的，删掉了。

    第一版写的是：对取分数值 v 的整数变量，加一条 `x <= floor(v)` 或 `x >= ceil(v)`。
    那不是割——**它会切掉可行的整数解**。有效割的定义是"不切掉任何可行整数点，
    但切掉当前分数解"；上面那条把整数点也切了，等于把一次分支伪装成一条割，
    会让求解器返回错误的最优值。

    真正的 Gomory 分数割要从最终单纯形表里取分数基变量所在的行，
    导出 `sum(frac(a_ij)·x_j) >= frac(b_i)`（j 取非基列），而且这里的列包含松弛变量，
    要表达回原始变量还得把 `s_i = b_i - A_i x` 代回去。这些都做得到，但没做——
    与其上一个自己没把握验对的割，不如把这块留白写在明处。

    **能替代它的东西已经有了**：`bench/bench_formulations.py` 说明**把模型写紧**
    （手工加上有效不等式、收紧 big-M）带来的节点数下降，比任何搜索开关都大——
    而那些收紧本身就是人工加的割，只不过是有效的。
    """
    raise NotImplementedError(
        "割平面未实现；第一版写的那个会切掉可行整数解，已删除。见本函数 docstring。")
