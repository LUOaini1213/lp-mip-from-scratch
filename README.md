# lp-mip-from-scratch

[![ci](https://github.com/LUOaini1213/lp-mip-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/LUOaini1213/lp-mip-from-scratch/actions/workflows/ci.yml)

**In one paragraph (EN)** — a linear- and mixed-integer-programming stack written from
scratch: an algebraic modelling layer that reduces any LP to standard form, a two-phase
tableau simplex with two pivot rules, and branch-and-bound over the LP relaxation.
Every answer is cross-checked against HiGHS (`scipy.optimize.linprog` / `milp`) on
**11 instances**, and the interesting results are not about the solver at all — they are
about **modelling**: the same scheduling problem written two ways differs by 1 node
versus 30, and loosening one big-M constant triples the search tree while leaving the
optimum untouched. What the solver work does buy is a concrete answer to
"how do you know it's right", and one cut section that says *not implemented* because
the first attempt was wrong.

自己写的 LP / MIP 建模层 + 单纯形法 + 分支定界，每个答案都拿 HiGHS 对过。
**结论的重点不在求解器，在建模**：同一个班车调度问题换个写法，分支定界从 30 个节点
降到 1 个；big-M 常数放大 100 倍，节点数翻三倍而最优值一动不动。

---

## 这是什么，不是什么

**是**：把 LP/MIP 从建模到求解走通一遍，并且每一步都能被独立验证。
**不是**：一个该上生产的求解器。生产上该用 HiGHS / Gurobi ——
本仓的 `bench/bench_all.py` 恰恰在量自己比它们慢多少。

这是 2026-09-08 新写的项目，**不对应任何过往经历**，README 里没有任何"当年做过"的数字。

---

## 30 秒跑通

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v   # 19 个测试
python bench/bench_all.py                 # 四组测量 -> results/bench.json
python bench/bench_all.py --check         # 与已提交结果比对
```

```python
from lpmip.model import Model
from lpmip import branch_bound as bb

m = Model(sense="max")
x = [m.add_var("x%d" % i, 0, 1, integer=True) for i in range(3)]
m.add_constraint(2*x[0] + 3*x[1] + x[2] <= 5)
m.set_objective(5*x[0] + 4*x[1] + 3*x[2])
r = bb.solve(m)          # -> status='optimal', obj=9.0, nodes=5
```

---

## 五个结论

### 1. 建模比换求解器重要，而且差距是数量级的

同一个问题：**给定班次时刻表，最少要几辆车**。两种写法：

- **指派型**：`x[i][j]=1` 表示一辆车跑完班次 i 接着跑 j。这是二分图上的流问题，
  约束矩阵**全幺模**，线性松弛的最优解**天然是整数**。
- **big-M 型**：给每辆车一个"用没用"的 0/1 变量，用大 M 约束排斥同车重叠班次。

| 写法 | 班次 | 变量 | 约束 | 线性松弛界 | MIP 最优 | **节点数** | 松弛已整数 |
|---|---|---|---|---|---|---|---|
| assignment | 6 | 6 | 6 | 3.000 | 3.000 | **1** | 是 |
| assignment | 8 | 14 | 10 | 4.000 | 4.000 | **1** | 是 |
| assignment | 10 | 28 | 14 | 6.000 | 6.000 | **1** | 是 |
| assignment | 12 | 45 | 18 | 8.000 | 8.000 | **1** | 是 |
| bigM | 4 | 15 | 32 | **1.000** | 3.000 | 29 | 否 |
| bigM | 5 | 18 | 42 | **1.000** | 3.000 | 31 | 否 |
| bigM | 6 | 21 | 52 | **1.000** | 3.000 | 21 | 否 |

**两种写法答案一模一样，代价差一个数量级。** big-M 的松弛界是 1.0，而真值是 3——
这个界毫无信息量，分支定界只能靠穷举补上。指派型压根不用分支。

### 2. big-M 该取多大，本身就是一个可测的决定

固定成本生产计划，`x_i <= M_i · y_i`。只改 M，模型的可行整数解集合完全不变：

| M 的取值 | 线性松弛界 | MIP 最优 | 节点数 | 与 HiGHS 一致 |
|---|---|---|---|---|
| **紧（= 该产品产能上限）** | **970.51** | 953.00 | **5** | 是 |
| 100 | 1040.56 | 953.00 | 19 | 是 |
| 1000 | 1082.36 | 953.00 | 19 | 是 |
| 10000 | 1086.54 | 953.00 | 19 | 是 |

**把 M 从"随手写个大数"改成"该变量自己的上界"，节点数少了近四倍**，一行代码的事。
这是最便宜的优化，也是最容易被忽略的——很多人会先去调求解器参数。

### 3. 主元规则的差别不是教科书警告，是真的会挂

Dantzig 规则（选最负的检验数）跑得快，但**在退化顶点上会循环**。
Beale 1955 的例子上：

| 规则 | 状态 | 迭代 | 目标值 |
|---|---|---|---|
| Dantzig | **cycling（200 步仍未终止）** | 203 | — |
| Bland | optimal | **9** | −0.05 |
| HiGHS（参照） | optimal | — | −0.05 |

所以默认配置是 **Dantzig + 迭代上限，撞上限自动重启为 Bland**——先快后稳，
而不是一上来就用最慢但最稳的那个。`test_auto_restart_rescues_the_default_rule` 锁住这个行为。

### 4. 单纯形法的指数最坏情况，能精确复现出来

Klee–Minty 立方体上，Dantzig 规则要走遍 2ⁿ−1 个顶点：

| n | 我的迭代数 | 2ⁿ−1 | 与 HiGHS 目标值一致 |
|---|---|---|---|
| 3 | 7 | 7 | 是 |
| 4 | 15 | 15 | 是 |
| 5 | 31 | 31 | 是 |
| 6 | 63 | 63 | 是 |
| 7 | 127 | 127 | 是 |

**一步不多一步不少。** 这既验证了实现，也说明"单纯形法实践中很快"是关于实例分布的
经验断言，不是关于算法的定理。

### 5. 「比 HiGHS 慢」这句话，在小实例上是假的

我原本打算写"我的实现慢几个数量级"。测出来不是：

| 实例 | 我 | HiGHS | 比值 |
|---|---|---|---|
| klee_minty_3 | 0.0002 s | 0.0010 s | **0.2**（我更快） |
| klee_minty_6 | 0.0011 s | 0.0008 s | 1.4 |
| knapsack_14 | 0.0164 s | 0.0021 s | 7.9 |
| dense_lp_20×30 | 0.0102 s | 0.0018 s | 5.6 |
| dense_lp_40×60 | 0.0386 s | 0.0029 s | 13.5 |
| dense_lp_60×90 | 0.1122 s | 0.0047 s | 23.9 |
| dense_lp_80×120 | 0.2214 s | 0.0069 s | **32.1** |

**在教科书规模上，HiGHS 的固定开销盖过求解本身，我的实现甚至更快**——
拿那种实例说"我的求解器不慢"是自欺欺人。规模拉到 80×120，差距才显出来（32 倍），
而且随规模持续拉开。`test_the_speed_claim_is_stated_honestly` 把这两头都写成了断言，
防止 README 只挑一头讲。

---

## 割平面：没做，以及为什么把它删了

第一版有一个 `_add_gomory_cuts`，做法是：对取分数值 v 的整数变量，加一条
`x <= floor(v)` 或 `x >= ceil(v)`。

**那不是割。** 有效割的定义是"不切掉任何可行整数点，但切掉当前分数解"；
上面那条把整数点也切了——它是**一次分支伪装成一条割**，会让求解器返回错误的最优值。

真正的 Gomory 分数割要从最终单纯形表里取分数基变量所在的行，导出
`sum(frac(a_ij)·x_j) >= frac(b_i)`，而且那些列包含松弛变量，要表达回原始变量
还得把 `s_i = b_i - A_i x` 代回去。做得到，但没做。
`bb.solve(m, cuts=True)` 现在**直接抛 NotImplementedError**，不静默降级，
`test_cuts_flag_refuses_instead_of_lying` 锁住这个行为。

替代它的东西已经有了：第 1、2 条里那些**手工收紧模型**的做法，本质就是人工加的有效割，
而且效果比我能写出来的通用割更大。

---

## 正确性怎么保证

不是"我读了一遍觉得对"，是**每个实例都有第二个求解器给答案**：

- LP：`scipy.optimize.linprog`（HiGHS）
- MIP：`scipy.optimize.milp`（HiGHS 的分支定界）
- **11 个实例全部一致**，含最优值、不可行、无界三种状态

归约到标准型的四件事各有独立测试：松弛/剩余变量、右端取正、变量平移与上界、
自由变量拆分。第五条测试专门验"自由变量带上界"这个最容易写错的组合——
第一版就在这里写岔过（多了一个"镜像"分支，系数符号处理不一致），已删。

---

## 仓库里有什么 / 没有什么

| 有 | 没有 |
|---|---|
| 建模层：变量、线性表达式、三种关系、到标准型的归约 | 二次/非线性、锥优化 |
| 两阶段表格单纯形，Dantzig / Bland 两种主元规则 | 修正单纯形、稀疏 LU、对偶单纯形 |
| 分支定界，两种分支变量选择、两种搜索顺序 | **割平面（见上一节）**、预处理、启发式找可行解 |
| 与 HiGHS 的逐实例交叉验证 | 大规模实例（最大 80×120） |
| 班车调度的两种建模对照 | 列生成、分支定价 |

```
lpmip/model.py         建模层 + 到标准型的归约
lpmip/simplex.py       两阶段单纯形，两种主元规则
lpmip/branch_bound.py  分支定界
problems/transit.py    班车最小车队：指派型 vs big-M 型
problems/classic.py    Beale 循环、Klee–Minty、背包、配料、固定成本生产
bench/bench_all.py     四组测量，--check 比对已提交结果
tests/test_all.py      19 个测试，全部对着 HiGHS 验
```

---

## 诚实边界

1. **实例都很小**，最大 80×120 的稠密 LP、14 个 0/1 变量的背包。结论不可外推到工业规模。
2. **割平面没实现**，理由见上。
3. **单纯形是稠密表格法**，不是生产求解器用的修正单纯形 + 稀疏分解，所以规模一大就吃力。
4. **数值稳定性没做认真处理**：容差是固定的 1e-9，没有比例化、没有 Harris 比值检验。
5. 一句话：**可以说**「自己实现了 LP/MIP 的建模与求解，并用 HiGHS 逐实例验证」；
   **不可以说**「写了一个可用的求解器」。
