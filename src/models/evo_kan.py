from __future__ import annotations

"""Baseline KAN training.

For each depth in {2, 4, 6} we train a KAN network with:

* in_dim=2, out_dim=1
* hidden layers = 5 neurons per layer
* SiLU as residual function
* GRID = 5, spline order k = 3
* Global penalty lamb = 1e-4

The main function returns a dictionary of metrics by depth and a dictionary of trained models.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
from kan import KAN

__all__ = ["SymbolicKANConfig", "train_symbolic_kan_ga"]

import pdb

@dataclass(slots=True)
class SymbolicKANConfig:
    """Hyper‑parameters shared across depths."""

    test_size: float = 0.2
    random_state: int = 10

    lamb: float = 0.001
    lamb_entropy: float = 1.0

    steps_phase1: int = 100

    device: str = "cpu"
    seed: int = 0


def _train_one_depth(
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.floating],
    X_test: npt.NDArray[np.floating],
    y_test: npt.NDArray[np.floating],
    depth: int,
    cfg: SymbolicKANConfig,
    lib: list[str],
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
        symbolic_enabled=True,
    )

    model.to(cfg.device)

    tic = perf_counter()

    model.fit(
        dataset,
        opt="Adam",
        steps=cfg.steps_phase1,
        #lamb=cfg.lamb,
        #lamb_entropy=cfg.lamb_entropy,
    )

    #model.prune()
    model.auto_symbolic(lib=lib)

    formula = model.symbolic_formula()[0][0]
    print("Recovered formula:")
    print(formula)

    opt_time = perf_counter() - tic
    
    with torch.no_grad():
        y_pred = model(test_input)          
        y_pred = y_pred.cpu().numpy()

        mse  = mean_squared_error(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "opt_time": opt_time,
            "depth": depth,
            "formula": str(formula)
        }
    

    return metrics, model


def train_symbolic_kan_ga(
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.floating],
    X_test: npt.NDArray[np.floating],
    y_test: npt.NDArray[np.floating],
    lib: list[str],
    *,
    cfg: SymbolicKANConfig | None = None,
) -> tuple[dict[int, dict[str, float]], dict[int, Any]]:
    """Treina KANs para profundidades {2,3,4}.

    Parameters
    ----------
    X_train, y_train : arrays
        Dados de treinamento já separados.
    X_test, y_test : arrays  
        Dados de teste já separados.
    cfg
        Config global (seed, steps, etc.).

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
        m, mdl = _train_one_depth(X_train, y_train, X_test, y_test, d, cfg, lib=lib)
        metrics[d] = m
        models[d] = mdl

    return metrics, models