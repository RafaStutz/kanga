from __future__ import annotations

"""
Symbolic primitive library used by kanga. It contains the set of primitive
functions that are available in pykan <https://github.com/KindXiaoming/pykan/blob/master/kan/utils.py>.

Each Primitive bundles three back‑end implementations (NumPy, PyTorch, SymPy),
a complexity cost integer, and a safe PyTorch callable that guards against
singularities when the network is training.

The module exposes:

- PRIMS: an immutable tuple of Primitive objects.
- NP_FUNCS: NumPy‑only vectorised functions (same order as PRIMS) so
  that MI/mRMR can run without importing torch.
"""

import numpy as np
import torch
import sympy
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Primitive:
    name: str
    np_fn: Callable[[np.ndarray], np.ndarray]
    torch_fn: Callable[[torch.Tensor], torch.Tensor]
    sym_fn: Callable[[sympy.Expr], sympy.Expr]
    cost: int 
    safe_torch: Callable[[torch.Tensor, torch.Tensor], tuple[tuple, torch.Tensor]]

    def __iter__(self):
        yield self.name
        yield self.np_fn
        yield self.torch_fn
        yield self.sym_fn
        yield self.cost
        yield self.safe_torch


f_inv = lambda x, y_th: (
    (x_th := 1 / y_th),
    y_th / x_th * x * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x) * (torch.abs(x) >= x_th),
)

f_inv2 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 2)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x ** 2) * (torch.abs(x) >= x_th),
)

f_inv3 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 3)),
    y_th / x_th * x * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x ** 3) * (torch.abs(x) >= x_th),
)

f_inv4 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 4)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x ** 4) * (torch.abs(x) >= x_th),
)

f_inv5 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 5)),
    y_th / x_th * x * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x ** 5) * (torch.abs(x) >= x_th),
)

f_sqrt = lambda x, y_th: (
    (x_th := 1 / y_th ** 2),
    x_th / y_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.sqrt(torch.abs(x)) * torch.sign(x)) * (torch.abs(x) >= x_th),
)

f_power1d5 = lambda x, _y_th: torch.abs(x) ** 1.5

f_invsqrt = lambda x, y_th: (
    (x_th := 1 / y_th ** 2),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / torch.sqrt(torch.abs(x))) * (torch.abs(x) >= x_th),
)

f_log = lambda x, y_th: (
    (x_th := torch.e ** (-y_th)),
    -y_th * (torch.abs(x) < x_th) + torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th),
)

f_tan = lambda x, y_th: (
    (clip := x % torch.pi),
    (delta := torch.pi / 2 - torch.arctan(y_th)),
    -y_th / delta * (clip - torch.pi / 2) * (torch.abs(clip - torch.pi / 2) < delta)
    + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi / 2) >= delta),
)

f_arctanh = lambda x, y_th: (
    (delta := 1 - torch.tanh(y_th) + 1e-4),
    y_th * torch.sign(x) * (torch.abs(x) > 1 - delta)
    + torch.nan_to_num(torch.arctanh(x)) * (torch.abs(x) <= 1 - delta),
)

f_arcsin = lambda x, _y_th: (
    (),
    torch.pi / 2 * torch.sign(x) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1),
)

f_arccos = lambda x, _y_th: (
    (),
    torch.pi / 2 * (1 - torch.sign(x)) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1),
)

f_exp = lambda x, y_th: (
    (x_th := torch.log(y_th)),
    y_th * (x > x_th) + torch.exp(x) * (x <= x_th),
)


PRIMS: tuple[Primitive, ...] = (
    Primitive("0", lambda x: np.zeros_like(x), lambda x: x * 0, lambda x: x * 0, 0, lambda x, y: ((), x * 0)),
    Primitive("x", lambda x: x, lambda x: x, lambda x: x, 1, lambda x, y: ((), x)),
    Primitive("x^2", lambda x: x ** 2, lambda x: x ** 2, lambda x: x ** 2, 2, lambda x, y: ((), x ** 2)),
    Primitive("x^3", lambda x: x ** 3, lambda x: x ** 3, lambda x: x ** 3, 3, lambda x, y: ((), x ** 3)),
    Primitive("x^4", lambda x: x ** 4, lambda x: x ** 4, lambda x: x ** 4, 3, lambda x, y: ((), x ** 4)),
    Primitive("x^5", lambda x: x ** 5, lambda x: x ** 5, lambda x: x ** 5, 3, lambda x, y: ((), x ** 5)),
    Primitive("1/x", lambda x: 1 / x, lambda x: 1 / x, lambda x: 1 / x, 2, f_inv),
    Primitive("1/x^2", lambda x: 1 / x ** 2, lambda x: 1 / x ** 2, lambda x: 1 / x ** 2, 2, f_inv2),
    Primitive("1/x^3", lambda x: 1 / x ** 3, lambda x: 1 / x ** 3, lambda x: 1 / x ** 3, 3, f_inv3),
    Primitive("1/x^4", lambda x: 1 / x ** 4, lambda x: 1 / x ** 4, lambda x: 1 / x ** 4, 4, f_inv4),
    Primitive("1/x^5", lambda x: 1 / x ** 5, lambda x: 1 / x ** 5, lambda x: 1 / x ** 5, 5, f_inv5),
    Primitive("sqrt", np.sqrt, torch.sqrt, sympy.sqrt, 2, f_sqrt),
    Primitive("x^0.5", np.sqrt, torch.sqrt, sympy.sqrt, 2, f_sqrt),
    Primitive("x^1.5", lambda x: np.abs(x) ** 1.5, lambda x: torch.abs(x) ** 1.5, lambda x: sympy.Abs(x) ** 1.5, 4, f_power1d5),
    Primitive("1/sqrt(x)", lambda x: 1 / np.sqrt(x), lambda x: 1 / torch.sqrt(x), lambda x: 1 / sympy.sqrt(x), 2, f_invsqrt),
    Primitive("1/x^0.5", lambda x: 1 / np.sqrt(x), lambda x: 1 / torch.sqrt(x), lambda x: 1 / sympy.sqrt(x), 2, f_invsqrt),
    Primitive("exp", np.exp, torch.exp, sympy.exp, 2, f_exp),
    Primitive("log", np.log, torch.log, sympy.log, 2, f_log),
    Primitive("abs", np.abs, torch.abs, sympy.Abs, 3, lambda x, y: ((), torch.abs(x))),
    Primitive("sin", np.sin, torch.sin, sympy.sin, 2, lambda x, y: ((), torch.sin(x))),
    Primitive("cos", np.cos, torch.cos, sympy.cos, 2, lambda x, y: ((), torch.cos(x))),
    Primitive("tan", np.tan, torch.tan, sympy.tan, 3, f_tan),
    Primitive("tanh", np.tanh, torch.tanh, sympy.tanh, 3, lambda x, y: ((), torch.tanh(x))),
    Primitive("sgn", np.sign, torch.sign, sympy.sign, 3, lambda x, y: ((), torch.sign(x))),
    Primitive("arcsin", np.arcsin, torch.arcsin, sympy.asin, 4, f_arcsin),
    Primitive("arccos", np.arccos, torch.arccos, sympy.acos, 4, f_arccos),
    Primitive("arctan", np.arctan, torch.arctan, sympy.atan, 4, lambda x, y: ((), torch.arctan(x))),
    Primitive("arctanh", np.arctanh, torch.arctanh, sympy.atanh, 4, f_arctanh),
    Primitive("gaussian", lambda x: np.exp(-x ** 2), lambda x: torch.exp(-x ** 2), lambda x: sympy.exp(-x ** 2), 3, lambda x, y: ((), torch.exp(-x ** 2))),
)


NP_FUNCS: tuple[Callable[[np.ndarray], np.ndarray], ...] = tuple(p.np_fn for p in PRIMS)

__all__ = ["Primitive", "PRIMS", "NP_FUNCS"]
