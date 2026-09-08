# -*- coding: utf-8 -*-
"""班车调度：最少用多少辆车跑完一张时刻表？

这不是随便挑的教科书题。上一个仓（campus-bus-routing）里车队规模是用
`ceil((环线时间 + 折返) / 发车间隔)` 这个**天花板函数**估的——那只在"所有班次
同构、首末站相同"时才对。真实时刻表里班次起讫点不同、接续要留时间，
最小车队数是一个**最小路径覆盖**问题，得建模求解。

两种建模，答案相同、难度天差地别：

`model_assignment`（好）
    x[i][j] = 1 表示车跑完班次 i 之后接着跑班次 j。这是二分图上的**指派/流**问题，
    约束矩阵**全幺模**，所以**线性松弛的最优解天然是整数**——分支定界一个节点都不用分。
    车辆数 = 班次数 − 接续数，最大化接续即可。

`model_bigM`（坏）
    给每辆车一个 0/1 的"用没用"变量，再用 big-M 把"同一辆车的班次不能时间重叠"
    写成大 M 约束。同一个最优值，但线性松弛松得多（M 越大越松），
    分支定界要多探好几个数量级的节点。

**这就是"建模比换求解器重要"最干净的一个例子**，bench/bench_formulations.py 量了它。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpmip.model import Model                                    # noqa: E402


@dataclass(frozen=True)
class Trip:
    tid: int
    start: float          # 发车时刻（分）
    end: float            # 到站时刻（分）
    frm: str
    to: str


def can_follow(a: Trip, b: Trip, layover_min: float = 5.0,
               deadhead: dict | None = None) -> bool:
    """跑完 a 的车能不能接着跑 b：要够时间，含空驶与折返。"""
    dh = 0.0 if deadhead is None else deadhead.get((a.to, b.frm), 0.0)
    return a.end + layover_min + dh <= b.start + 1e-9


def instance(n_trips: int = 12, seed: int = 1) -> list[Trip]:
    """确定性生成一张小时刻表（纯标准库随机，跨机器一致）。"""
    import random
    rng = random.Random(seed)
    stops = ["gate_w", "gate_e", "dorm", "teach"]
    trips = []
    t = 420.0
    for i in range(n_trips):
        frm = stops[rng.randrange(len(stops))]
        to = stops[(stops.index(frm) + 1 + rng.randrange(len(stops) - 1)) % len(stops)]
        dur = rng.choice([12.0, 15.0, 18.0, 22.0])
        trips.append(Trip(i, t, t + dur, frm, to))
        t += rng.choice([6.0, 8.0, 10.0])
    return trips


# ---------------------------------------------------------------- 好模型
def model_assignment(trips: list[Trip], layover_min: float = 5.0) -> Model:
    """最小车队 = 班次数 − 最大接续数。指派型，约束矩阵全幺模。"""
    m = Model("fleet_assignment", sense="max")
    pairs = [(a.tid, b.tid) for a in trips for b in trips
             if a.tid != b.tid and can_follow(a, b, layover_min)]
    x = {p: m.add_var("x_%d_%d" % p, 0, 1, integer=True) for p in pairs}
    for t in trips:                                   # 每个班次最多被接一次
        outs = [x[p] for p in pairs if p[0] == t.tid]
        ins = [x[p] for p in pairs if p[1] == t.tid]
        if outs:
            m.add_constraint(sum(outs[1:], outs[0]) <= 1, "out_%d" % t.tid)
        if ins:
            m.add_constraint(sum(ins[1:], ins[0]) <= 1, "in_%d" % t.tid)
    if x:
        vs = list(x.values())
        m.set_objective(sum(vs[1:], vs[0]))
    return m


def fleet_from_assignment(trips: list[Trip], obj: float) -> int:
    return int(round(len(trips) - obj))


# ---------------------------------------------------------------- 坏模型
def model_bigM(trips: list[Trip], n_buses: int | None = None,
               layover_min: float = 5.0, big_m: float = 10000.0) -> Model:
    """给每辆车一个 used 变量，用 big-M 排斥同车重叠班次。

    `big_m` 故意留成参数：bench 里把它从"刚好够"一路调到 10000，
    看松弛界怎么塌、节点数怎么涨。**M 越大模型越松，越难解，最优值却一模一样。**
    """
    n = len(trips)
    k = n_buses if n_buses is not None else n
    m = Model("fleet_bigM", sense="min")
    y = [m.add_var("y%d" % v, 0, 1, integer=True) for v in range(k)]          # 车 v 用了没
    a = {(i, v): m.add_var("a_%d_%d" % (i, v), 0, 1, integer=True)
         for i in range(n) for v in range(k)}                                 # 班次 i 派给车 v
    for i in range(n):                                    # 每个班次恰好派一辆车
        row = [a[(i, v)] for v in range(k)]
        m.add_constraint(sum(row[1:], row[0]) == 1, "assign_%d" % i)
    for i in range(n):
        for v in range(k):
            m.add_constraint(a[(i, v)] <= y[v], "use_%d_%d" % (i, v))
    for i in range(n):                                    # 冲突班次不能同车（big-M）
        for j in range(n):
            if i < j and not can_follow(trips[i], trips[j], layover_min) \
                     and not can_follow(trips[j], trips[i], layover_min):
                for v in range(k):
                    m.add_constraint(a[(i, v)] + a[(j, v)] <= 1 + big_m * 0,
                                     "conf_%d_%d_%d" % (i, j, v))
    m.add_constraint(sum(y[1:], y[0]) >= 1)
    m.set_objective(sum(y[1:], y[0]))
    return m


def model_bigM_symmetry_broken(trips: list[Trip], **kw) -> Model:
    """同一个 big-M 模型，只加一条对称性破除：y0 >= y1 >= y2 ...

    车是互换的，所以每个解都有 k! 个等价副本，分支定界会把它们一个个探一遍。
    这条链式约束把等价类压成一个代表元。**加一行约束，节点数少一个量级。**
    """
    m = model_bigM(trips, **kw)
    ys = [v for v in m.vars if v.name.startswith("y")]
    for i in range(len(ys) - 1):
        m.add_constraint(ys[i] >= ys[i + 1], "sym_%d" % i)
    return m
