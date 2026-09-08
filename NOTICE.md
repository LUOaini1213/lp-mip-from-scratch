# NOTICE

## 数据

所有实例都是**确定性生成**或来自公开教科书：Beale (1955) 的循环例子、
Klee–Minty (1972) 立方体、随机背包与随机稠密 LP（固定种子）。
**不含任何真实企业数据或第三方数据集。**

## 第三方依赖

| 包 | 用途 | 许可证 |
|---|---|---|
| [NumPy](https://numpy.org/) | 矩阵运算 | BSD-3-Clause |
| [SciPy](https://scipy.org/) | **参照求解器**：`optimize.linprog` / `optimize.milp`（内置 HiGHS） | BSD-3-Clause |
| [HiGHS](https://highs.dev/) | 经 SciPy 间接使用，本仓所有答案的独立校验来源 | MIT |

本仓自己的求解器不依赖 SciPy——`lpmip/` 只用 NumPy。SciPy 只出现在 `bench/` 与
`tests/` 里，作为交叉验证的第二意见。
