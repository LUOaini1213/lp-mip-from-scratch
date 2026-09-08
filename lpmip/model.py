# -*- coding: utf-8 -*-
"""建模层：一个够用的代数建模接口，外加到标准型的归约。

写这一层不是为了跟 PuLP 比功能，是因为**"建模"这件事本身有内容**：
同一个问题写成不同的模型，求解难度能差几个数量级（见 bench/bench_formulations.py），
而这个差别只有在你自己掌管从模型到矩阵那一步时才看得清。

对外是 `Var / LinExpr / Constraint / Model`；对内 `Model.to_standard_form()`
把任意 LP 归约成单纯形法要的标准型：

    min c'x   s.t.  Ax = b,  x >= 0

归约要处理四件事，每件都会在测试里被单独验证：
  1. `<=` 加松弛变量、`>=` 减剩余变量、`==` 不动；
  2. b 必须非负——为负就整行取反（连同不等号方向）；
  3. 下界非 0 的变量做平移 x = lb + x'；有上界的再加一行 x' <= ub - lb；
  4. 自由变量拆成 x = x⁺ - x⁻。

拆完之后原始变量的取值要能还原回去，`Solution.value()` 负责这件事。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INF = float("inf")


class LinExpr:
    """线性表达式 sum(coef * var) + const。"""

    __slots__ = ("terms", "const")

    def __init__(self, terms: dict | None = None, const: float = 0.0):
        self.terms: dict[int, float] = dict(terms or {})
        self.const = float(const)

    # --- 运算符：让模型写起来像数学式子 -----------------------------------
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return LinExpr(self.terms, self.const + other)
        other = other.as_expr()
        t = dict(self.terms)
        for k, v in other.terms.items():
            t[k] = t.get(k, 0.0) + v
            if t[k] == 0.0:
                del t[k]
        return LinExpr(t, self.const + other.const)

    __radd__ = __add__

    def __neg__(self):
        return LinExpr({k: -v for k, v in self.terms.items()}, -self.const)

    def __sub__(self, other):
        return self + (-(other if isinstance(other, LinExpr) else
                         (other if isinstance(other, (int, float)) else other.as_expr())))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, k: float):
        if not isinstance(k, (int, float)):
            raise TypeError("只支持线性模型：表达式不能乘表达式")
        return LinExpr({a: b * k for a, b in self.terms.items()}, self.const * k)

    __rmul__ = __mul__

    def as_expr(self) -> "LinExpr":
        return self

    # --- 比较运算符直接产生约束 -------------------------------------------
    def __le__(self, rhs):
        return Constraint(self - rhs, "<=")

    def __ge__(self, rhs):
        return Constraint(self - rhs, ">=")

    def __eq__(self, rhs):                                   # noqa: A003
        return Constraint(self - rhs, "==")

    def __repr__(self):
        return "LinExpr(%r, %r)" % (self.terms, self.const)


class Var:
    """决策变量。`lb`/`ub` 是变量界，`integer` 交给分支定界。"""

    __slots__ = ("idx", "name", "lb", "ub", "integer")

    def __init__(self, idx: int, name: str, lb: float, ub: float, integer: bool):
        self.idx, self.name = idx, name
        self.lb, self.ub, self.integer = float(lb), float(ub), bool(integer)

    def as_expr(self) -> LinExpr:
        return LinExpr({self.idx: 1.0})

    def __add__(self, o):
        return self.as_expr() + o

    __radd__ = __add__

    def __sub__(self, o):
        return self.as_expr() - o

    def __rsub__(self, o):
        return (-self.as_expr()) + o

    def __neg__(self):
        return -self.as_expr()

    def __mul__(self, k):
        return self.as_expr() * k

    __rmul__ = __mul__

    def __le__(self, o):
        return self.as_expr() <= o

    def __ge__(self, o):
        return self.as_expr() >= o

    def __eq__(self, o):                                     # noqa: A003
        return self.as_expr() == o

    def __hash__(self):
        return hash(self.idx)

    def __repr__(self):
        return "Var(%s)" % self.name


@dataclass
class Constraint:
    expr: LinExpr                    # expr <sense> 0
    sense: str
    name: str = ""


@dataclass
class StandardForm:
    """min c'x, Ax = b, x >= 0，外加把解映射回原始变量所需的信息。"""
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    n_original: int
    # 每个原始变量 -> (正部列, 负部列 or None, 平移量)
    mapping: list = field(default_factory=list)
    col_names: list = field(default_factory=list)

    def recover(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_original)
        for i, (pos, neg, shift) in enumerate(self.mapping):
            v = x[pos] if pos is not None else 0.0
            if neg is not None:
                v -= x[neg]
            out[i] = v + shift
        return out


class Model:
    """一个 LP / MIP 模型。"""

    def __init__(self, name: str = "model", sense: str = "min"):
        assert sense in ("min", "max")
        self.name, self.sense = name, sense
        self.vars: list[Var] = []
        self.cons: list[Constraint] = []
        self.obj = LinExpr()

    def add_var(self, name: str | None = None, lb: float = 0.0, ub: float = INF,
                integer: bool = False) -> Var:
        v = Var(len(self.vars), name or ("x%d" % len(self.vars)), lb, ub, integer)
        self.vars.append(v)
        return v

    def add_vars(self, n: int, prefix: str = "x", **kw) -> list[Var]:
        return [self.add_var("%s%d" % (prefix, i), **kw) for i in range(n)]

    def add_constraint(self, con: Constraint, name: str = "") -> Constraint:
        if not isinstance(con, Constraint):
            raise TypeError("约束要用 <= / >= / == 写出来")
        con.name = name or ("c%d" % len(self.cons))
        self.cons.append(con)
        return con

    def set_objective(self, expr, sense: str | None = None) -> None:
        self.obj = expr.as_expr() if not isinstance(expr, LinExpr) else expr
        if sense:
            assert sense in ("min", "max")
            self.sense = sense

    # ---------------------------------------------------------------- 归约
    def to_standard_form(self) -> StandardForm:
        """归约到 min c'x, Ax = b, x >= 0。四步都在 docstring 里写明了。"""
        cols: list[str] = []
        mapping: list[tuple] = []
        extra_rows: list[tuple] = []          # (系数字典, 关系, 右端) —— 变量上界行

        # 步骤 3/4：变量平移与自由变量拆分。
        # 下界有限 -> 平移 x = lb + x'；下界无限 -> 拆成 x = x⁺ - x⁻。
        # 上界一律用一行约束表达，不做"镜像"那种特例——特例是第一版写错的地方。
        for v in self.vars:
            if v.lb == -INF:
                pos, neg, shift = len(cols), len(cols) + 1, 0.0
                cols += ["%s+" % v.name, "%s-" % v.name]
                if v.ub < INF:
                    extra_rows.append(({pos: 1.0, neg: -1.0}, "<=", v.ub))
            else:
                pos, neg, shift = len(cols), None, v.lb
                cols.append("%s'" % v.name if v.lb != 0.0 else v.name)
                if v.ub < INF:
                    extra_rows.append(({pos: 1.0}, "<=", v.ub - v.lb))
            mapping.append((pos, neg, shift))

        n_struct = len(cols)
        rows: list[tuple[dict, str, float]] = []
        for con in self.cons:
            # sum(k·x) + const <sense> 0，代入 x = shift + x'（或 x⁺ - x⁻）后
            # 常数并入右端：sum(k·x') <sense> -const - sum(k·shift)
            coef: dict[int, float] = {}
            for vi, k in con.expr.terms.items():
                p, ng, _ = mapping[vi]
                coef[p] = coef.get(p, 0.0) + k
                if ng is not None:
                    coef[ng] = coef.get(ng, 0.0) - k
            rhs = -con.expr.const - sum(k * mapping[vi][2] for vi, k in con.expr.terms.items())
            rows.append((coef, con.sense, rhs))
        rows += extra_rows

        m = len(rows)
        # 步骤 1：加松弛 / 剩余
        n_slack = sum(1 for _, s, _ in rows if s != "==")
        n_total = n_struct + n_slack
        A = np.zeros((m, n_total))
        b = np.zeros(m)
        sl = n_struct
        for i, (coef, sense, rhs) in enumerate(rows):
            for j, k in coef.items():
                A[i, j] = k
            b[i] = rhs
            if sense == "<=":
                A[i, sl] = 1.0
                cols.append("s%d" % i)
                sl += 1
            elif sense == ">=":
                A[i, sl] = -1.0
                cols.append("e%d" % i)
                sl += 1
        # 步骤 2：b 必须非负
        for i in range(m):
            if b[i] < 0:
                A[i, :] *= -1.0
                b[i] *= -1.0

        c = np.zeros(n_total)
        for vi, k in self.obj.terms.items():
            p, ng, _ = mapping[vi]
            c[p] += k
            if ng is not None:
                c[ng] -= k
        if self.sense == "max":
            c = -c                          # 内部一律求最小
        return StandardForm(A, b, c, len(self.vars), mapping, cols)

    def objective_value(self, x: np.ndarray) -> float:
        return float(sum(k * x[i] for i, k in self.obj.terms.items()) + self.obj.const)

    def integer_vars(self) -> list[int]:
        return [v.idx for v in self.vars if v.integer]

    def is_feasible(self, x, tol: float = 1e-6) -> bool:
        """给定一个点，判断它是否满足全部约束与变量界。

        写它是为了**直接按定义验割的有效性**：枚举小背包的全部 0/1 点，
        原模型可行的点，加割之后必须仍然可行。第一版那个假割正是在这里会被抓住。
        """
        for v in self.vars:
            if x[v.idx] < v.lb - tol or x[v.idx] > v.ub + tol:
                return False
        for con in self.cons:
            val = sum(k * x[i] for i, k in con.expr.terms.items()) + con.expr.const
            if con.sense == "<=" and val > tol:
                return False
            if con.sense == ">=" and val < -tol:
                return False
            if con.sense == "==" and abs(val) > tol:
                return False
        return True
