# -*- coding: utf-8 -*-
"""时空网络上的车辆调度（VSP / MDVSP），外加一条被自己的数据推翻的猜想。

`problems/transit.py` 那个"指派型"模型是按接续关系拼的，能用，但不是这一类问题的
标准建模。文献（Kliewer et al. 2006；Xu et al. 2018）用**时空网络流**：把"某站某时刻"
做成节点，班次、空驶、停站等待、出场入场各做成弧，求最小费用流。

节点：(站点, 时刻)。弧：

  trip      班次弧，**必须恰好被一辆车覆盖**（流量下界 = 上界 = 1）
  deadhead  空驶弧：a 站放空开到 b 站，赶得上才有
  wait      同站相邻时刻之间的等待弧
  pull_out  场站 -> 首站（出场，**每用一条就是多一辆车**）
  pull_in   末站 -> 场站（入场）

目标：最小化 车辆数 × 固定成本 + 空驶时间 × 单位成本。

--------------------------------------------------------------------------------
## 一条被自己的数据推翻的猜想

这个文件的 docstring 里，我原本写死过一句结论：

    「同一张时刻表，只因为多了一个场站，难度就换了一个量级。」

推理链看着无懈可击：单场站是纯网络流，约束矩阵全幺模，松弛天然整数；多场站是
多商品流，MDVSP 是 NP 难的（Bertossi, Carraresi & Gallo 1987），所以松弛该出分数解、
该分支。**推理没错，结论是错的。** `bench/bench_vsp.py` 把它测了：

  * 只把场站从 1 个加到 2、3 个 —— 换非对称空驶表、换车辆成本量级、加场站容量上限，
    **松弛全部整数、全部一个节点解完、全部与 HiGHS 一致**。
  * 再随机撒 1500 个多场站实例（n=4..8，2~3 个场站，随机成本），**0 个出分数松弛**。

错在哪：**NP 难是关于最坏实例的陈述，不是关于随机实例的承诺。**
"矩阵不再全幺模"只说明"整数性不再被保证"，不等于"分数解会出现"。
我把一个必要条件当成了充分条件——而且是在没测之前就写进了结论里。

## 那到底要什么才出分数解

看模型就知道：`model_multi_depot` 里场站之间**只在出场/入场两端有区别**，
中段的空驶弧、等待弧对所有场站完全相同。耦合太弱，各商品几乎可以独立取整。

真实 MDVSP 还有一条本文件原先漏掉的耦合：**班次的场站准入限制**——
车型、里程、驾驶员归属，都会让某个班次只能由部分场站承担。补上这一条
（`model_multi_depot(..., eligible=...)`）之后，分数松弛才出现：

    6000 个随机实例（n=4..8，2~3 场站，随机准入），**6 个出分数松弛，命中率 0.10%**，
    而且 **6 个全部落在 n=8**，也就是搜索范围里最大的那一档。

`FRACTIONAL_SEEDS` 冻结了这 6 个种子，`random_instance(seed)` 逐位可重建，
`tests/test_all.py` 拿它们当回归。所以本文件现在的结论是：

    **让 MDVSP 变难的不是"多了一个场站"，是"场站服务不了所有班次"；
    即便加上这条，在 n<=8 的规模上也要一千多个实例才碰上一个分数松弛。**

## 与 campus-bus-routing 的关系

上一个仓用 `ceil((环线时间 + 折返) / 发车间隔)` 估车队规模，只在"班次同构、
首末站相同"时成立。本文件是把那个天花板函数换成真模型之后的样子：起讫点可以不同、
接续要留时间、空驶要算钱、车要回自己的场站。**两个仓的车队数不可直接比较**，
口径不是一回事。
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpmip.model import Model                                    # noqa: E402
from problems import transit                                     # noqa: E402
from problems.transit import Trip                                # noqa: E402

VEHICLE_COST = 1000.0        # 一辆车的固定成本，默认远大于任何空驶费用
DEADHEAD_COST = 1.0          # 每分钟空驶的成本

# `bench/bench_vsp.py --hunt 6000` 在 6000 个随机实例里找到的、线性松弛真出分数解的种子。
# 命中率 6/6000 = 0.10%，且 6 个全部落在 n=8。改动 `random_instance` 会让它们失效——
# `tests/test_all.py::MultiDepot` 会立刻发现。
FRACTIONAL_SEEDS = (17, 670, 3845, 4189, 4764, 5948)
HUNT_TRIED = 6000


@dataclass(frozen=True)
class Arc:
    kind: str                # trip / deadhead / wait / pull_out / pull_in
    tail: tuple              # (站点, 时刻)
    head: tuple
    cost: float
    trip_id: int = -1
    depot: str = ""


def deadhead_minutes(a: str, b: str, table: dict | None = None) -> float:
    """站间空驶时间。默认给一张对称的小表，缺省 8 分钟。"""
    if a == b:
        return 0.0
    if table and (a, b) in table:
        return table[(a, b)]
    return 8.0


def build_network(trips: list[Trip], depots: list[str], layover: float = 5.0,
                  dh_table: dict | None = None,
                  vehicle_cost: float = VEHICLE_COST) -> tuple[list, list]:
    """造时空网络，返回 (节点表, 弧表)。"""
    nodes = set()
    arcs: list[Arc] = []

    for t in trips:                                          # 班次弧
        u, v = (t.frm, t.start), (t.to, t.end)
        nodes |= {u, v}
        arcs.append(Arc("trip", u, v, 0.0, t.tid))

    for a in trips:                                          # 空驶弧
        for b in trips:
            if a.tid == b.tid:
                continue
            dh = deadhead_minutes(a.to, b.frm, dh_table)
            if a.end + layover + dh <= b.start + 1e-9:
                u, v = (a.to, a.end), (b.frm, b.start)
                if u != v:
                    nodes |= {u, v}
                    arcs.append(Arc("deadhead", u, v, DEADHEAD_COST * dh))

    by_stop: dict[str, list[float]] = {}                     # 等待弧
    for (s, tm) in nodes:
        by_stop.setdefault(s, []).append(tm)
    for s, times in by_stop.items():
        for t1, t2 in zip(sorted(times), sorted(times)[1:]):
            arcs.append(Arc("wait", (s, t1), (s, t2), 0.0))

    for d in depots:                                         # 出场 / 入场
        src, snk = ("%s_src" % d, -1.0), ("%s_snk" % d, 1e9)
        nodes |= {src, snk}
        for t in trips:
            dh = deadhead_minutes(d, t.frm, dh_table)
            arcs.append(Arc("pull_out", src, (t.frm, t.start),
                            vehicle_cost + DEADHEAD_COST * dh, depot=d))
            dh2 = deadhead_minutes(t.to, d, dh_table)
            arcs.append(Arc("pull_in", (t.to, t.end), snk,
                            DEADHEAD_COST * dh2, depot=d))
    return sorted(nodes), arcs


def _set_objective(m: Model, arcs, f) -> None:
    keys = sorted(f)
    obj = arcs[keys[0][-1]].cost * f[keys[0]]
    for k in keys[1:]:
        obj = obj + arcs[k[-1]].cost * f[k]
    m.set_objective(obj)


def model_single_depot(trips: list[Trip], depot: str = "depot",
                       layover: float = 5.0, dh_table: dict | None = None,
                       vehicle_cost: float = VEHICLE_COST) -> Model:
    """单场站最小费用流。**约束矩阵是网络流矩阵，全幺模，松弛天然整数。**"""
    nodes, arcs = build_network(trips, [depot], layover, dh_table, vehicle_cost)
    m = Model("vsp_single", sense="min")
    f = {(i,): m.add_var("f%d" % i, 0, 1 if a.kind == "trip" else len(trips),
                         integer=True)
         for i, a in enumerate(arcs)}

    for i, a in enumerate(arcs):                             # 班次必须被覆盖
        if a.kind == "trip":
            m.add_constraint(f[(i,)] == 1, "cover_%d" % a.trip_id)

    src, snk = ("%s_src" % depot, -1.0), ("%s_snk" % depot, 1e9)
    for nd in nodes:                                         # 流量守恒
        if nd in (src, snk):
            continue
        ins = [f[(i,)] for i, a in enumerate(arcs) if a.head == nd]
        outs = [f[(i,)] for i, a in enumerate(arcs) if a.tail == nd]
        if ins and outs:
            m.add_constraint(sum(ins[1:], ins[0]) - sum(outs[1:], outs[0]) == 0,
                             "flow_%s_%s" % nd)
    _set_objective(m, arcs, f)
    return m


def model_multi_depot(trips: list[Trip], depots: list[str], layover: float = 5.0,
                      dh_table: dict | None = None, eligible: dict | None = None,
                      vehicle_cost: float = VEHICLE_COST) -> Model:
    """多场站：车必须回到自己出发的场站，于是变成多商品流，不再全幺模。

    每条弧按场站复制一份（商品 = 场站），班次弧要求"所有场站的流量合计 = 1"。

    `eligible`: `{trip_id: {可以承担这个班次的场站}}`。**这一条才是难度的真来源**——
    不给它的话（默认全部场站都能跑全部班次），1500 个随机实例里 0 个出分数松弛；
    给了之后 6000 个里出 6 个。见模块 docstring。
    """
    nodes, arcs = build_network(trips, depots, layover, dh_table, vehicle_cost)
    m = Model("vsp_multi", sense="min")
    f = {}
    for d in depots:
        for i, a in enumerate(arcs):
            if a.kind in ("pull_out", "pull_in") and a.depot != d:
                continue
            if a.kind == "trip" and eligible is not None and d not in eligible[a.trip_id]:
                continue
            f[(d, i)] = m.add_var("f_%s_%d" % (d, i), 0,
                                  1 if a.kind == "trip" else len(trips), integer=True)

    for i, a in enumerate(arcs):                             # 覆盖：跨场站合计为 1
        if a.kind == "trip":
            row = [f[(d, i)] for d in depots if (d, i) in f]
            if not row:
                raise ValueError("班次 %d 没有任何场站可以承担" % a.trip_id)
            m.add_constraint(sum(row[1:], row[0]) == 1, "cover_%d" % a.trip_id)

    for d in depots:                                         # 每个商品各自守恒
        for nd in nodes:
            if nd[0].endswith("_src") or nd[0].endswith("_snk"):
                continue
            ins = [f[(d, i)] for i, a in enumerate(arcs) if a.head == nd and (d, i) in f]
            outs = [f[(d, i)] for i, a in enumerate(arcs) if a.tail == nd and (d, i) in f]
            if ins and outs:
                m.add_constraint(sum(ins[1:], ins[0]) - sum(outs[1:], outs[0]) == 0,
                                 "flow_%s_%s_%s" % (d, nd[0], nd[1]))
    _set_objective(m, arcs, f)
    return m


def add_depot_capacity(m: Model, arcs, caps: dict) -> Model:
    """给多场站模型加"车库放得下几辆"的上限：该场站出场弧流量之和 <= 容量。

    测下来**这条也不产生分数松弛**（5 个实例全整数），所以它在 README 里是反面证据，
    不是卖点。
    """
    for d, cap in sorted(caps.items()):
        outs = [v for v in m.vars if v.name.startswith("f_%s_" % d)
                and arcs[int(v.name.rsplit("_", 1)[1])].kind == "pull_out"]
        if outs:
            m.add_constraint(sum(outs[1:], outs[0]) <= cap, "depotcap_%s" % d)
    return m


def random_instance(seed: int) -> dict:
    """确定性生成一个带场站准入限制的随机 MDVSP 实例。

    **纯标准库随机，且站名先 `sorted` 再进循环**——集合的迭代顺序按哈希来，
    跨进程会变（PYTHONHASHSEED），第一版就在这里踩过：同一个种子两次跑出不同实例，
    分数变量数从 16 变成 22。
    """
    rng = random.Random(seed)
    n = rng.choice([4, 5, 6, 7, 8])
    trip_seed = rng.randrange(1, 999)
    trips = transit.instance(n, seed=trip_seed)
    depots = ["dA", "dB"] if rng.random() < 0.5 else ["dA", "dB", "dC"]
    stops = {t.frm for t in trips} | {t.to for t in trips}
    names = sorted(stops) + depots                      # 排序过，跨进程一致
    dh = {(a, b): rng.choice([1, 3, 6, 9, 14, 20, 28, 40])
          for a in names for b in names if a != b}
    eligible = {}
    for t in trips:
        picked = [d for d in depots if rng.random() < 0.65] or [rng.choice(depots)]
        eligible[t.tid] = set(picked)
    vehicle_cost = rng.choice([10.0, 30.0, 100.0, 1000.0])
    return {"trips": trips, "depots": depots, "dh": dh, "eligible": eligible,
            "vehicle_cost": vehicle_cost, "n": n, "trip_seed": trip_seed}


def model_from_random(seed: int) -> Model:
    """把 `random_instance(seed)` 直接建成模型。"""
    ins = random_instance(seed)
    return model_multi_depot(ins["trips"], ins["depots"], dh_table=ins["dh"],
                             eligible=ins["eligible"],
                             vehicle_cost=ins["vehicle_cost"])


def fleet_from_flow(arcs, x, var_index) -> int:
    """车辆数 = 出场弧上的总流量。"""
    return int(round(sum(x[var_index[k]] for k in var_index if k[0] == "pull_out")))
