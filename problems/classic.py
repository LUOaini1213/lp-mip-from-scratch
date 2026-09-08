# -*- coding: utf-8 -*-
"""几个用来验求解器本身的经典实例。

`beale_cycling`  Beale 1955 的循环例子。Dantzig 规则（选最负检验数）在它上面
                 会在退化顶点之间原地打转、永不终止；Bland 规则可证明有限终止。
                 这不是理论警告——`bench/bench_pivot.py` 跑出来就是这个结果。
`klee_minty`     Klee–Minty 立方体：Dantzig 规则要经过 2^n − 1 个顶点才到最优，
                 单纯形法最坏情况指数复杂度的标准反例。用来量"最坏情况有多坏"。
`blending`       配料 LP，纯粹的正确性对照基线。
`knapsack`       0/1 背包，B&B 的基础对照。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpmip.model import Model                                    # noqa: E402


def beale_cycling():
    """Beale 的例子，直接给标准型 (A, b, c)。

    min -0.75 x4 + 150 x5 - 0.02 x6 + 6 x7
    s.t. 0.25 x4 - 60 x5 - 0.04 x6 + 9 x7 + x1 = 0
         0.5  x4 - 90 x5 - 0.02 x6 + 3 x7 + x2 = 0
                                x6        + x3 = 1
    全部变量 >= 0。最优值 -0.05。
    """
    A = np.array([
        [1., 0., 0., 0.25, -60., -0.04, 9.],
        [0., 1., 0., 0.50, -90., -0.02, 3.],
        [0., 0., 1., 0.00,   0.,  1.00, 0.],
    ])
    b = np.array([0., 0., 1.])
    c = np.array([0., 0., 0., -0.75, 150., -0.02, 6.])
    return A, b, c


def klee_minty(n: int = 6):
    """Klee–Minty 立方体（标准 max 形式，这里转成 min）。

    max sum(10^(n-j) x_j)
    s.t. 2*sum_{j<i} 10^(i-j) x_j + x_i <= 100^(i-1),  x >= 0
    """
    m = Model("klee_minty_%d" % n, sense="max")
    x = [m.add_var("x%d" % (j + 1)) for j in range(n)]
    for i in range(1, n + 1):
        expr = x[i - 1].as_expr()
        for j in range(1, i):
            expr = expr + 2 * (10 ** (i - j)) * x[j - 1]
        m.add_constraint(expr <= float(100 ** (i - 1)), "c%d" % i)
    obj = x[n - 1].as_expr() * 1.0
    for j in range(1, n + 1):
        if j != n:
            obj = obj + (10 ** (n - j)) * x[j - 1]
    m.set_objective(obj)
    return m


def blending():
    """配料 LP：两种原料配出满足营养下限的最便宜混合。"""
    m = Model("blending", sense="min")
    a = m.add_var("corn", lb=0, ub=10)
    b = m.add_var("soy", lb=0, ub=10)
    m.add_constraint(2 * a + 5 * b >= 12, "protein")
    m.add_constraint(4 * a + 1 * b >= 8, "energy")
    m.add_constraint(a + b <= 15, "volume")
    m.set_objective(2.5 * a + 3.2 * b)
    return m


def knapsack(seed: int = 3, n: int = 14):
    """0/1 背包。容量取总重的一半，保证不平凡。"""
    import random
    rng = random.Random(seed)
    w = [rng.randint(3, 20) for _ in range(n)]
    v = [rng.randint(5, 30) for _ in range(n)]
    cap = sum(w) // 2
    m = Model("knapsack_%d" % n, sense="max")
    x = [m.add_var("x%d" % i, 0, 1, integer=True) for i in range(n)]
    expr = w[0] * x[0]
    for i in range(1, n):
        expr = expr + w[i] * x[i]
    m.add_constraint(expr <= cap, "cap")
    obj = v[0] * x[0]
    for i in range(1, n):
        obj = obj + v[i] * x[i]
    m.set_objective(obj)
    return m, w, v, cap


def production_mip(seed: int = 5, n: int = 8):
    """带固定成本的生产计划：经典的 big-M 结构，用来演示 M 的取值有多要命。

    每种产品要么不生产，要么至少生产一个批量；生产就付固定成本。
    `x_i <= M_i * y_i` 里的 M_i 取多大，直接决定线性松弛有多松。
    """
    import random
    rng = random.Random(seed)
    cap = [rng.randint(20, 60) for _ in range(n)]
    profit = [rng.randint(3, 12) for _ in range(n)]
    fixed = [rng.randint(20, 60) for _ in range(n)]
    total = int(sum(cap) * 0.45)
    return cap, profit, fixed, total


def build_production(cap, profit, fixed, total, big_m=None) -> Model:
    """`big_m=None` 表示用最紧的 M（= 该产品自身产能上限）。"""
    n = len(cap)
    m = Model("production", sense="max")
    x = [m.add_var("x%d" % i, 0, cap[i]) for i in range(n)]
    y = [m.add_var("y%d" % i, 0, 1, integer=True) for i in range(n)]
    for i in range(n):
        M = cap[i] if big_m is None else big_m
        m.add_constraint(x[i] <= M * y[i], "link_%d" % i)
    expr = x[0].as_expr()
    for i in range(1, n):
        expr = expr + x[i]
    m.add_constraint(expr <= float(total), "capacity")
    obj = profit[0] * x[0] - fixed[0] * y[0]
    for i in range(1, n):
        obj = obj + profit[i] * x[i] - fixed[i] * y[i]
    m.set_objective(obj)
    return m
