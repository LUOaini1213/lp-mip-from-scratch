# lp-mip-from-scratch

[![ci](https://github.com/LUOaini1213/lp-mip-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/LUOaini1213/lp-mip-from-scratch/actions/workflows/ci.yml)

**Built:** an LP/MIP stack from scratch on NumPy (modelling layer, two-phase tableau simplex
with Dantzig and Bland pivoting, branch-and-bound, pure-integer Gomory fractional cuts), every
answer cross-checked against HiGHS via SciPy. **Found:** the formulation mattered more than any
solver switch (1 node vs 21–31 for the same fleet problem); cuts closed 3 of 4 knapsacks at the
root but grew the 4th tree (29 → 31 nodes); my own claim that a second depot makes bus scheduling
far harder was false: 0 of 1500 random multi-depot instances had a bound gap, and even with depot
eligibility only 5 of 6000 did. **Limits:** toy sizes (80×120 dense LP, 14-item knapsack, 8-trip
MDVSP), dense tableau, no GMI cuts or column generation, 43.5× slower than HiGHS at 80×120 (one machine).

自己写的 LP / MIP 建模层 + 单纯形法 + 分支定界 + **真的 Gomory 分数割**，
每个答案都拿 HiGHS 对过，割则**按定义**枚举全部 2ⁿ 个整数点验有效性。
**这个仓最值钱的两条结论都是负面的**：割不是免费的（四个背包里有一个越割越慢）；
以及我写下的「多一个场站，难度就换一个量级」被自己的数据推翻了。

### Findings at a glance (EN)

Every number below is read from `results/bench.json` or `results/vsp.json` by
`tests/test_readme_numbers.py`; editing one by hand turns CI red. The detailed write-up
further down is in Chinese.

| # | Finding | Number |
|---|---|---|
| 1 | Formulation beats solver tuning: same optimal fleet, assignment vs big-M model | 1 node vs 21–31 nodes |
| 2 | Big-M set to each variable's own bound instead of an arbitrary large M | 5 nodes vs 19 |
| 3 | Dantzig's rule really cycles on Beale's 1955 example; Bland's rule terminates | stopped at 203 iterations vs optimal in 9 |
| 4 | Klee–Minty reproduces the simplex worst case exactly | 2ⁿ−1 iterations for n = 3..7 |
| 5 | "Slower than HiGHS" is false on the smallest LP and true at size (one machine, see §5) | ratio 0.3 on klee_minty_3, 43.5× on dense 80×120 |
| 6 | Gomory cuts help, but not always | 3 of 4 knapsacks solved at the root; knapsack_12 grows 29 → 31 nodes |
| 7 | A second depot alone does not make MDVSP harder; depot eligibility does, rarely | 0 of 12 ladder configs and 0 of 1500 random instances with a bound gap; with eligibility, 6 of 6000 fractional LP vertices from this simplex, 5 of 6000 with a positive gap |

**Main limits.** Instances are small, so nothing here extrapolates to industrial sizes.
Only pure-integer Gomory cuts (a model with continuous variables is refused, not silently
skipped). Dense tableau with fixed 1e-9 tolerances, no presolve, no column generation.
Timings come from one Windows machine; `--check` compares counts, statuses and bounds,
not seconds. "Fractional relaxation" in §7 means the vertex this simplex returns, which
depends on column order and pivot rule; the bound gap is the property of the instance.

---

## 这是什么，不是什么

**是**：把 LP/MIP 从建模到求解走通一遍，并且每一步都能被独立验证。
**不是**：一个该上生产的求解器。生产上该用 HiGHS / Gurobi ——
本仓的 `bench/bench_all.py` 恰恰在量自己比它们慢多少。

这是 2026-09-08 新写的项目，**不对应任何过往经历**，README 里没有任何"当年做过"的数字。

---

## 30 秒跑通

```bash
# 需要 Python 3.10 或 3.11。requirements 把 numpy 钉在 1.24.3，而它没有 cp312 的
# wheel，3.12 上 pip 会转去源码构建并失败——首次推送时 CI 正是在这里红的。
# 为什么宁可放弃 3.12 也要钉版本，见文末「诚实边界」第 6 条。
# 可选：先 python -m venv .venv 建个虚拟环境并激活，省得污染系统环境。
pip install -r requirements.txt

python -m unittest discover -s tests -v        # 58 个测试（32 个原有 + 12 个预算/状态回归 + 14 个 README 数字核对）
python bench/bench_all.py                      # 五组测量 -> results/bench.json
                                               #（顺带重新生成下面第 5 条那张速度表）
python bench/bench_all.py --check              # 与已提交结果比对
python bench/bench_vsp.py                      # 车辆调度 -> results/vsp.json
python bench/bench_vsp.py --hunt 6000          # 重搜那 6 个反例（本机约 1 分钟）
python bench/bench_vsp.py --hunt-baseline 1500 # 重跑对照组（本机约 40 秒）
```

```python
from lpmip.model import Model
from lpmip import branch_bound as bb

m = Model(sense="max")
x = [m.add_var("x%d" % i, 0, 1, integer=True) for i in range(3)]
m.add_constraint(2*x[0] + 3*x[1] + x[2] <= 5)
m.set_objective(5*x[0] + 4*x[1] + 3*x[2])
r = bb.solve(m)             # -> status='optimal', obj=9.0, nodes=5
r = bb.solve(m, cuts=True)  # 根节点加 Gomory 分数割
```

调用方要先看 `status` 再用解：迭代预算耗尽返回 `iteration_limit` / `lp_limit`，
无界松弛返回 `lp_unbounded`，都不是不可行证书，gap 记为未知（NaN）而不是 0。
来龙去脉见 [CHANGELOG.md](CHANGELOG.md) 的 2026-10-12 一条。

---

## 七个结论

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

<!-- speed-table:begin 由 bench/bench_all.py 生成，勿手改 -->

| 实例 | 我 | HiGHS | 比值 |
|---|---|---|---|
| klee_minty_3 | 0.0001 s | 0.0005 s | **0.3**（我更快） |
| klee_minty_4 | 0.0002 s | 0.0004 s | **0.5**（我更快） |
| klee_minty_5 | 0.0005 s | 0.0005 s | 1.0 |
| klee_minty_6 | 0.0010 s | 0.0005 s | 2.2 |
| knapsack_10 | 0.0073 s | 0.0007 s | 11.0 |
| knapsack_12 | 0.0108 s | 0.0007 s | 15.7 |
| knapsack_14 | 0.0155 s | 0.0010 s | 14.9 |
| dense_lp_20x30 | 0.0098 s | 0.0009 s | 10.6 |
| dense_lp_40x60 | 0.0372 s | 0.0015 s | 25.2 |
| dense_lp_60x90 | 0.1015 s | 0.0030 s | 33.6 |
| dense_lp_80x120 | 0.2195 s | 0.0051 s | **43.5** |

> 每格取 5 次里最快的一次（噪声只会让一次测量变慢，不会变快）。测量环境：Windows 10 ／ Intel64 Family 6 Model 140 Stepping 1, GenuineIntel ／ Python 3.11.5 ／ numpy 1.24.3（BLAS: mkl_rt）。
>
> **这是全仓唯一一组机器相关的数字**，换台机器重跑必然是另一组秒数，所以 `--check` 不比对秒数本身；它比对的是**这张表与 `results/bench.json` 一致**。
<!-- speed-table:end -->

**在最小的那几档上，HiGHS 的固定开销盖过求解本身，我的实现甚至更快**——
拿那种实例说"我的求解器不慢"是自欺欺人。规模一拉起来就拉开了：
连 14 个 0/1 变量的背包都已经拉开，稠密 LP 上到几十倍量级。

**这张表只说得到"几十倍"这个精度，别的都别读出来。** 秒数是机器相关的量：
`--check` 把 `seconds` / `speed_ratio` 整块豁免（`bench_all.py` 的 `EXEMPT_SUFFIX`），
换台机器重跑必然是另一组数。这个豁免曾经真的漏过东西：第一版的表是手打的，
`results/bench.json` 后来被重算过、表却没跟着改，而 `--check` 照样绿。
现在这张表由 `bench/bench_all.py` 从同一份 JSON **渲染**出来，
`--check` 会把 README 里的这一段和渲染结果逐字比对，
`test_the_readme_speed_table_matches_the_committed_bench` 再验一遍——
故意把表里任何一个数字改错一位，两者都会红。

跨机器还站得住的是序，不是秒数：最小的那档 LP 上我更快（比值 < 1），
稠密 LP 拉到 80×120 我慢一个数量级以上，而且比值随规模显著变大。
`test_the_speed_ordering_holds_on_a_fresh_run` 对**新跑**结果断言这三条（阈值留得比
本机实测宽，共享 runner 上 HiGHS 的固定开销占比会变），
`test_the_speed_claim_is_stated_honestly` 继续钉住冻结值，防止 README 只挑一头讲。

顺带一个教训：这张表原先每格只测一次，`dense_lp_40x60` 曾被测成 0.0924 s，
把比值推到 41 倍，而前后两次提交里同一个实例的耗时都落在 0.037–0.039 s。
实例是 `default_rng(7)` 确定性生成的、算法一行没改，迭代次数不可能变——
那一格只能是测量被干扰了，不是什么"形状效应"。现在每格取 5 次里最快的一次：
噪声只会让一次测量变慢、不会变快，所以 min 比 mean 更接近"这台机器上真正要花多久"。

### 6. 割平面是真的，但**不是免费的**

根节点反复加 Gomory 分数割，**只保留真的抬高了松弛界的那些**：

| 实例 | 松弛界（割前） | 松弛界（割后） | MIP 最优 | 割数 | 轮数 | **节点数** | 与 HiGHS 一致 |
|---|---|---|---|---|---|---|---|
| knapsack_8 | 98.889 | **94.000** | 94 | 12 | 3 | 17 → **1** | 是 |
| knapsack_10 | 142.000 | **136.000** | 136 | 7 | 2 | 27 → **1** | 是 |
| knapsack_12 | 163.444 | 160.333 | 159 | 10 | 3 | 29 → **31** | 是 |
| knapsack_14 | 194.889 | **188.000** | 188 | 12 | 3 | 35 → **1** | 是 |

三个实例的界被割到与最优值完全相等，**一个节点解完**。
**但 n=12 是反例：界确实抬高了（163.44 → 160.33），节点数反而从 29 涨到 31。**
割抬高界的同时让每个节点的线性松弛更大更贵，净效果可以是负的。
这一行留在表里，`test_cuts_can_make_things_worse_and_that_stays_in_the_bench` 锁住它——
**如果哪天它消失了，README 也必须跟着改。**

### 7. 我写下的一条结论，被自己的数据推翻了

`problems/vsp.py` 是时空网络上的车辆调度。写它的时候我在 docstring 里下了结论：

> 「同一张时刻表，只因为多了一个场站，难度就换了一个量级。」

推理链看着无懈可击：单场站是纯网络流，矩阵全幺模，松弛天然整数；多场站是多商品流，
MDVSP 是 NP 难的，所以松弛该出分数解、该分支。**然后我去测了。**

| 配置 | 变量 | 线性松弛 | **分数变量** | 节点 | 与 HiGHS 一致 |
|---|---|---|---|---|---|
| 单场站 n=8 | 49 | 4072.00 | **0** | 1 | 是 |
| 两场站 n=8 | 98 | 4072.00 | **0** | 1 | 是 |
| 三场站 n=8 | 147 | 4072.00 | **0** | 1 | 是 |
| 两场站 + 非对称空驶 n=8 | 98 | 4077.00 | **0** | 1 | 是 |
| 两场站 + 场站容量上限 n=8 | 98 | 4077.00 | **0** | 1 | 是 |
| 两场站 + 车辆成本压到 40 n=8 | 98 | 237.00 | **0** | 1 | 是 |

n=6 的六档同样全是 0。再随机撒 **1500 个**多场站实例（n=4..8，2~3 场站，随机成本），
本仓单纯形返回的松弛顶点**依然 0 个是分数的，也 0 个有界间隙**
（逐个解了 MIP：线性松弛的最优值都等于整数最优值）。

这 1500 个是**对照组**：`bench/bench_vsp.py --hunt-baseline 1500` 与下面那个
`--hunt 6000` 跑的是**同一批实例**，逐位相同的时刻表、场站、空驶表、车辆成本，
唯一的差别是这一组不加场站准入限制（`eligible=None`）。所以"难的是准入不是场站数"
这句话是被对照出来的，不是猜出来的。两组 CI 每次推送都重跑，
数字落在 `results/vsp.json` 的 `baseline_hunt` 与 `hunt` 两块。

**错在哪：NP 难是关于最坏实例的陈述，不是关于随机实例的承诺。**
"矩阵不再全幺模"只说明整数性不再被*保证*，不等于分数解会*出现*。
我把一个必要条件当成了充分条件——而且是在没测之前就写进了结论。

**那到底要什么？** 看模型就知道：上面那个写法里，场站之间只在出场/入场两端有区别，
中段的空驶弧、等待弧对所有场站完全相同，耦合太弱。真实 MDVSP 还有一条我漏掉的耦合：
**班次的场站准入限制**（车型、里程、驾驶员归属，让某个班次只能由部分场站承担）。
补上之后，分数松弛才出现——**6000 个随机实例里 6 个的松弛顶点是分数的（0.10%），
其中 5 个界间隙 > 0**。「顶点分数」是本仓单纯形返回的那个最优顶点的性质
（模型 + 列序 + 主元规则，见下文「集合的迭代顺序」），界间隙才只属于实例；
其余 5994 个的松弛顶点本身就是整数解，间隙必为 0。`--hunt 6000` 两组种子都重验：

| 种子 | 班次 | 场站 | 车辆成本 | 分数变量 | 线性松弛 | MIP 最优 | 界间隙 | 节点 |
|---|---|---|---|---|---|---|---|---|
| 17 | 8 | 3 | 100 | 16 | 412.50 | 454.00 | 9.14% | 5 |
| 670 | 8 | 3 | 100 | 19 | 382.00 | 382.00 | **0.00%** | 3 |
| 3845 | 8 | 2 | 30 | 13 | 188.00 | 198.00 | 5.05% | 3 |
| 4189 | 8 | 3 | 30 | 21 | 183.00 | 184.00 | 0.54% | 5 |
| 4764 | 8 | 3 | 10 | 12 | 129.50 | 133.00 | 2.63% | 5 |
| 5948 | 8 | 3 | 1000 | 22 | 4646.00 | 5140.00 | 9.61% | 7 |

两个附带发现：

- **6 个全部落在 n=8**，也就是搜索范围（n=4..8）里最大的那一档。
- **seed=670 的松弛是分数的，界间隙却是 0.00%。** 分数最优解与整数最优解目标值相同——
  **"松弛解不整数"和"界不紧"是两回事**，这两件事我原先也是混着说的。

所以本仓现在的结论是：**让 MDVSP 变难的不是"多了一个场站"，是"场站服务不了所有班次"；
即便加上这条，在 n≤8 的规模上也要一千多个实例才碰上一个分数松弛顶点
（6000 个里 6 个，界真正变松的 5 个）。**

这 6 个种子冻结在 `vsp.FRACTIONAL_SEEDS` 里，`random_instance(seed)` 逐位可重建，
`tests/test_all.py::MultiDepot` 拿它们当回归，并且**把那条被推翻的结论也写成了断言**
（`test_adding_a_depot_alone_does_not_make_it_fractional`），
防止将来有人——包括我自己——又把它写回来。

---

## 割平面：从"删掉的假货"到"按定义实现"

第一版这里有一个 `_add_gomory_cuts`，做法是：对取分数值 v 的整数变量，加一条
`x <= floor(v)` 或 `x >= ceil(v)`。**那不是割。** 有效割的定义是"不切掉任何可行整数点，
但切掉当前分数解"；上面那条把整数点也切了——它是**一次分支伪装成一条割**，
会让求解器返回错误的最优值。删掉之后，`cuts=True` 曾经**直接抛 NotImplementedError**。

现在按定义做出来了，在 `lpmip/cuts.py`：

**推导。** 最终单纯形表里，取分数值的基变量所在行读作
`x_k + Σ_{j 非基} α_j·x_j = β`。对任何可行整数点，把系数下取整得到有效不等式
`x_k + Σ ⌊α_j⌋ x_j ≤ ⌊β⌋`，与原式相减即得 **Gomory 分数割**：

    Σ_{j 非基} {α_j}·x_j ≥ {β}          其中 {r} = r − ⌊r⌋

**有效性前提（最容易被忽略的一条）。** 上面"下取整仍有效"那一步要求
**行里所有变量都取整数值——包括松弛变量**。所以分数割只对**纯整数、且约束数据为整数**
的模型成立：那时 `Ax + s = b` 里 A、b 皆整、x 整，s 才整。混合整数要用 GMI 割，
本仓**没做**。`cuts.validate_pure_integer()` 会带着原因拒绝：

```
ValueError: Gomory 分数割不适用于该模型：存在连续变量 ['x0', 'x1', 'x2'] ——
分数割不适用于混合整数，那需要 Gomory 混合整数割（本模块未实现）
```

**不是静默跳过。** `test_mixed_integer_is_refused_not_silently_skipped` 锁住这个行为。

**把松弛变量代回原变量。** 割写出来带松弛变量：`e'x + r's ≥ d`。标准型里每行至多一个
松弛/剩余列、系数 σ = ±1，于是 `σ·s_i = b_i − Σ A[i,j]·x_j`，代回去就只剩结构列；
再经 `StandardForm.mapping` 折回原始变量（平移与正负部）。

**怎么知道这次是对的。** 不靠"读一遍觉得对"：

- `test_no_feasible_integer_point_is_ever_cut_off` **枚举全部 2ⁿ 个 0/1 点**
  （n = 8、10、12，共 5376 个点），原模型可行的必须在加割后仍然可行。
  **第一版那个假割会在这里当场失败。**
- `test_cuts_only_tighten_the_bound_never_past_the_optimum`：保留下来的割必须抬高界，
  且界不得被压到真最优值以下。
- `test_cuts_never_change_the_optimum`：四个背包加割前后都与 HiGHS 一致。

「保留哪些割」是个**离散决定**，所以阈值用的是相对量（`KEEP_TOL = 1e-7`）而不是 1e-9：
若某条割只把界抬高 1e-9 量级，换一台机器（BLAS 不同、最后几位有出入）就可能留下
不同的割，`cuts_added` 与节点数随之漂移，提交进 `results/` 的数字就复现不了。
改成相对阈值后重跑，上表四行**一个数字都没变**——说明现在保留的割都不是靠边际留下来的。

---

## 正确性怎么保证

不是"我读了一遍觉得对"，是**每个结论都有第二个来源**：

| 声明 | 独立验证方式 |
|---|---|
| LP 最优值 | `scipy.optimize.linprog`（HiGHS） |
| MIP 最优值 | `scipy.optimize.milp`（HiGHS 的分支定界） |
| **割的有效性** | **枚举全部 2ⁿ 个整数点，按定义验"没切掉可行点"** |
| 归约到标准型 | 四件事各有独立测试，外加"自由变量带上界"这个易错组合 |
| 已提交的 results | `--check` 重跑一遍再逐字段比对，不是只读旧值；**计时字段整块豁免**（机器相关），改由下一行兜住 |
| 计时表不过期 | README 的速度表由 `bench_all.py` 从 `results/bench.json` 渲染，`--check` 与一条测试各比对一次 |
| README 其余数字 | `tests/test_readme_numbers.py` 把其余每张表和英文首屏的数字从 `results/` 现算再比对 |
| 6/6000、5/6000 与 1500/0 | CI 每次推送把两次随机搜索**整个重跑**（`--hunt 6000`、`--hunt-baseline 1500`），分数顶点与界间隙两组种子都比对，不是读冻结值 |
| **模型构造可复现** | 换三个 `PYTHONHASHSEED` 起子进程，**弧表与实例字段必须逐位相同** |
| lpmip/ 不依赖 SciPy | CI 里卸掉 SciPy 再跑一遍，免得"自己实现"悄悄靠在参照求解器上 |

### 集合的迭代顺序，在这个仓咬了两次

**第一次**：`random_instance` 用集合迭代顺序建空驶表，同一个种子跨进程跑出不同实例，
分数变量数从 16 变成 22。改成先 `sorted` 才稳，并且**写了条测试**。

**第二次**：那条测试只盖了 `random_instance`，**没盖 `build_network` 本身**。
那里等待弧是按 `for (s, tm) in nodes` 追加的，而 `nodes` 是集合。顺序一变，
模型的**列序**就变，单纯形走的路径跟着变。

**推上去，CI 第一次跑就红了**：

```
DRIFT (3 处)：
  /ladder/n_rows_with_fractional_lp: 0 vs 1
  /ladder/rows[10]/n_fractional: 0 vs 18
  /ladder/rows[10]/nodes: 1 vs 3
```

注意 **`lp_bound` 没有出现在漂移列表里**——目标值两边都是 4077.000000，
分岔的只是"落在哪个最优顶点上"。这个 LP 有多个最优顶点，其中至少一个是分数的，
ubuntu 那次正好落在了它上面。所以：

> **「分数变量个数」是（模型 + 列序 + 主元规则）三者的性质，不是模型单独的性质。**
> 只有把前两者钉死，这个指标才有意义。

本机怎么跑都是绿的，因为本机也只是碰巧。修法是 `sorted(nodes)`，
测试改成比对**整张弧表**而不是某个数值，并且**先把 `sorted` 摘掉验证它确实会失败**
（3 个哈希种子给出 3 份不同输出），再装回去。

---

## 联网 review：查了什么，改了什么

写完 VSP 之后去查了文献，主要是想确认时空网络的标准写法，以及自己那条结论站不站得住。

| 查的东西 | 结果 |
|---|---|
| 时空网络 VSP/MDVSP 的标准建模（节点 = (站, 时刻)，弧 = 班次/空驶/等待/出入场） | 与我的写法一致，Kliewer 等人的做法 |
| MDVSP 是不是 NP 难 | 是（Bertossi, Carraresi & Gallo 1987） |
| **NP 难能不能推出"随机小实例的松弛会出分数解"** | **不能。** 文献给的是最坏情况；实际算法（列生成、分支定价、Lagrangian）能把大实例解到很小的间隙，恰恰说明这类实例的界很紧。**我那条结论就死在这一步** |
| Gomory 分数割的精确形式与松弛变量代回 | Cordone 讲义、Letchford & Lodi (2002) |
| 列生成 / 集合划分 | 读了，**没实现**，见「诚实边界」 |

需要说清的一点：**文献里我没有找到一句可直接引用的"MDVSP 松弛通常整数"**，
所以 README 里那张表的依据是**本仓自己跑出来的 12 档 + 1500 个对照实例**
（`--hunt-baseline 1500`，CI 每次重算），不是转述别人的结论。找不到的东西就说找不到。

- [A time–space network based exact optimization model for multi-depot bus scheduling (Kliewer et al., EJOR 2006)](https://www.sciencedirect.com/science/article/abs/pii/S0377221705002274)
- [Solving a Large-Scale Multi-Depot Vehicle Scheduling Problem in Urban Bus Systems (2018)](https://onlinelibrary.wiley.com/doi/10.1155/2018/4868906)
- [A branch and bound algorithm for the multiple depot vehicle scheduling problem (Carpaneto, Dell'Amico, Fischetti & Toth, Networks 1989)](https://www.semanticscholar.org/paper/A-branch-and-bound-algorithm-for-the-multiple-depot-Carpaneto-Dell'Amico/6dc720c774402c23bb893952a738b882032640f4)
- [An overview of optimization approaches for scheduling and rostering resources in public transportation (arXiv 2310.13425)](https://arxiv.org/pdf/2310.13425)
- [Gomory cuts — Cordone lecture notes](https://homes.di.unimi.it/cordone/courses/2014-for/Lez18-gomory.pdf)
- [Strengthening Chvátal–Gomory cuts and Gomory fractional cuts (Letchford & Lodi, ORL 2002)](https://www.lancaster.ac.uk/staff/letchfoa/articles/2002-Gomory.pdf)

---

## 仓库里有什么 / 没有什么

| 有 | 没有 |
|---|---|
| 建模层：变量、线性表达式、三种关系、到标准型的归约 | 二次/非线性、锥优化 |
| 两阶段表格单纯形，Dantzig / Bland 两种主元规则 | 修正单纯形、稀疏 LU、对偶单纯形 |
| 分支定界，两种分支变量选择、两种搜索顺序 | 预处理、启发式找可行解 |
| **Gomory 分数割**（纯整数），按定义验过有效性 | **GMI 割**（混合整数）——会被显式拒绝，不静默降级 |
| 班车调度：指派型 vs big-M 型 | **列生成 / 分支定价**（读了，没写） |
| 时空网络 VSP / MDVSP，含场站准入限制与场站容量 | 时间窗、车队异构、乘务排班 |
| 与 HiGHS 的逐实例交叉验证 | 大规模实例（最大 80×120 稠密 LP） |

```
lpmip/model.py         建模层 + 到标准型的归约 + is_feasible（验割用）
lpmip/simplex.py       两阶段单纯形，两种主元规则，可导出最终表与基
lpmip/branch_bound.py  分支定界，cuts=True 走 Gomory 割
lpmip/cuts.py          Gomory 分数割：推导、有效性闸门、松弛变量代回
problems/transit.py    班车最小车队：指派型 vs big-M 型
problems/classic.py    Beale 循环、Klee–Minty、背包、配料、固定成本生产
problems/vsp.py        时空网络 VSP / MDVSP，以及那条被推翻的结论
bench/bench_all.py     五组测量（含割），--check 比对已提交结果并核对 README 的速度表
bench/bench_vsp.py     车辆调度阶梯 + 6 个冻结反例，--hunt N / --hunt-baseline N 重跑两次搜索
tests/test_all.py      32 个测试，全部对着 HiGHS 或按定义验
tests/test_readme_numbers.py  14 个测试：README 每张表、英文首屏每个数字都对 results/ 现算
```

---

## 诚实边界

1. **实例都很小**，最大 80×120 的稠密 LP、14 个 0/1 变量的背包、8 个班次的 MDVSP。
   结论不可外推到工业规模——**第 7 条尤其如此**：分数松弛只在搜索范围最大的那一档出现，
   很可能规模再大一点就常见了，我没能力测。
2. **只做了纯整数的 Gomory 分数割**，混合整数的 GMI 割没做，传进去会报错拒绝。
3. **单纯形是稠密表格法**，不是生产求解器用的修正单纯形 + 稀疏分解，所以规模一大就吃力。
4. **数值稳定性没做认真处理**：容差是固定的 1e-9，没有比例化、没有 Harris 比值检验。
5. **列生成没实现。** 查资料时读了集合划分 + 列生成的做法，也知道那才是解大规模
   MDVSP 的主流路线，但本仓没写。**读过不等于做过。**
6. **钉死的版本是参照环境，不是唯一能复现的环境。** `requirements.txt` 把 numpy 钉在
   1.24.3、scipy 钉在 1.11.1，这个 numpy 没有 cp312 的 wheel，所以钉版本的安装只支持
   Python 3.10 / 3.11（首次推送时 CI 正是在 3.12 上红的）。这里原先写的是「放宽版本，
   `--check` 比对的字段就没了保证，两者只能选一个」，那句话没有证据：本机 Python 3.11.5
   + numpy 2.4.6 / scipy 1.17.1 下全部测试与两个 `--check` 都过；CI 的 `unpinned` 作业
   每次推送在 Python 3.12、3.13 + 当时最新的 numpy/scipy 上重跑测试、两个 `--check`
   和两次随机搜索。哪天新版本让计数或界值漂了，先红的是那个作业。
   **注意这句话不覆盖计时**：秒数从来就不跨机器复现，`--check` 也不比对它。
7. **计时是全仓唯一一组机器相关的数字。** 第 5 条那张表只在生成它的那台机器上成立
   （表下面写了是哪台）。跨机器成立的是它的序，不是它的秒数。
8. 一句话：**可以说**「自己实现了 LP/MIP 的建模、单纯形、分支定界与 Gomory 分数割，
   并用 HiGHS 逐实例验证，还测出并推翻了自己写下的一条结论」；
   **不可以说**「写了一个可用的求解器」。
