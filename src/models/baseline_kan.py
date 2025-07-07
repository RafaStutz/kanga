from __future__ import annotations

"""Baseline KAN training.

For each depth in {2, 4, 6} we train a KAN network with:

* in_dim=2, out_dim=1
* hidden layers = 5 neurons per layer
* SiLU as residual function
* GRID = 5, spline order k = 3

The main function returns a dictionary of metrics by depth and a dictionary of trained models.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import sympy as sp 
import numpy.typing as npt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
from kan import KAN

__all__ = ["SymbolicKANConfig", "train_symbolic_kan"]


@dataclass(slots=True)
class SymbolicKANConfig:
    """Hyper‑parameters shared across depths."""

    test_size: float = 0.2
    random_state: int = 10

    lamb: float = 0.01
    lamb_entropy: float = 1.0

    steps_phase1: int = 120

    device: str = "cpu"
    seed: int = 0


def _train_one_depth(
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.floating],
    X_test: npt.NDArray[np.floating],
    y_test: npt.NDArray[np.floating],
    depth: int,
    cfg: SymbolicKANConfig,
) -> tuple[dict[str, float], Any]:
    """Trains a KAN for a specific depth and returns (metrics, model)."""

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_input = torch.from_numpy(X_train.astype(np.float32, copy=False)).float()
    train_label = torch.from_numpy(y_train.astype(np.float32, copy=False)).float()
    test_input = torch.from_numpy(X_test.astype(np.float32, copy=False)).float()
    test_label = torch.from_numpy(y_test.astype(np.float32, copy=False)).float()

    dataset = {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": test_input,
        "test_label": test_label
    }

    hidden = [5] * depth
    layer_sizes = [2, *hidden, 1]

    model = KAN(
        width=layer_sizes,
        grid=5,
        k=3,
        device=cfg.device,
        seed=cfg.seed,
        symbolic_enabled=False,
    )

    model.to(cfg.device)

    t0 = perf_counter()

    model.fit(
        dataset,
        opt="Adam",
        steps=cfg.steps_phase1,
        lamb=cfg.lamb,
        lamb_entropy=cfg.lamb_entropy,
    )

    train_time = perf_counter() - t0

    t1 = perf_counter()
    model.prune()
    prune_time = perf_counter() - t1
    
    t2 = perf_counter()
    model.auto_symbolic()
    sym_time = perf_counter() - t2

    formula = model.symbolic_formula()[0][0]
    print("Recovered formula:")
    print(formula)

    with torch.no_grad():
        # numeric evaluation
        y_pred_tr = model(train_input).cpu().numpy()
        y_pred_te = model(test_input).cpu().numpy()

    # Symbolic evaluation
    symb_cols: list[tuple[int, sp.Symbol]] = []
    for s in formula.free_symbols:
        name = str(s)
        if name.startswith("x_") and name[2:].isdigit():
            symb_cols.append((int(name[2:]), s))
        elif name.startswith("x") and name[1:].isdigit():
            symb_cols.append((int(name[1:]), s))
        else:
            continue

    if not symb_cols:
        def _eval_sym(X: np.ndarray) -> np.ndarray:
            const = float(formula.evalf())
            return np.full(X.shape[0], const, dtype=float)
    else:
        symb_cols.sort()                           
        cols, syms = zip(*symb_cols)                 
        f_lam = sp.lambdify(syms, formula, "numpy")

        def _eval_sym(X: np.ndarray) -> np.ndarray:
            try:
                val = f_lam(*[X[:, j] for j in cols])
            except Exception:
                return np.full(X.shape[0], np.nan, dtype=float)
            if np.isscalar(val):
                return np.full(X.shape[0], float(val), dtype=float)
            return np.asarray(val, dtype=float).reshape(-1)

    y_sym_tr = _eval_sym(X_train)
    y_sym_te = _eval_sym(X_test)

    # metrics
    def _safe_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MSE que devolve NaN se y_pred contiver NaN/Inf."""
        if not np.isfinite(y_pred).all():
            return np.nan
        return mean_squared_error(y_true, y_pred)

    mse_tr       = mean_squared_error(y_train, y_pred_tr)
    mse_te       = mean_squared_error(y_test , y_pred_te)
    mse_sym_tr   = _safe_mse(y_train, y_sym_tr)
    mse_sym_te   = _safe_mse(y_test , y_sym_te)
    r2           = r2_score         (y_test , y_pred_te)

    metrics = {
        "mse_tr":      mse_tr,
        "mse_te":      mse_te,
        "mse_sym_tr":  mse_sym_tr,
        "mse_sym_te":  mse_sym_te,
        "r2":          r2,
        "train_time":    train_time,
        "prune_time":    prune_time,
        "sym_time":      sym_time,
        "kan_time":      train_time + prune_time + sym_time,
        "depth":       depth,
        "formula":     str(formula),
    }
    

    return metrics, model


def train_symbolic_kan(
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.floating],
    X_test: npt.NDArray[np.floating],
    y_test: npt.NDArray[np.floating],
    *,
    cfg: SymbolicKANConfig | None = None,
) -> tuple[dict[int, dict[str, float]], dict[int, Any]]:
    """Train KANs for depths in dict {}.

    Parameters
    ----------
    X_train, y_train : arrays
        Training data already split.
    X_test, y_test : arrays
        Test data already split.
    cfg
        Global config (seed, steps, etc.).

    Returns
    -------
    metrics_per_depth : dict
        ``{2: {...}, 4: {...}, 6: {...}}``
    models_per_depth : dict
        ``{2: model2, 4: model4, 6: model6}``
    """

    cfg = cfg or SymbolicKANConfig()
    depths = (2,)

    metrics: dict[int, dict[str, float]] = {}
    models: dict[int, Any] = {}

    for d in depths:
        m, mdl = _train_one_depth(X_train, y_train, X_test, y_test, d, cfg)
        metrics[d] = m
        models[d] = mdl

    return metrics, models