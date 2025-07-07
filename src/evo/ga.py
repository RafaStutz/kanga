from __future__ import annotations

"""Evolutionary feature‑subset search.
A GA that maximises the mRMR score with a very light cost penalty.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from functools import partial
import random

import numpy as np
import numpy.typing as npt
from deap import base, creator, tools, algorithms
from sklearn.feature_selection import mutual_info_regression

from kanga.src.utils.feat_sel import mrmr_score
from kanga.src.utils.logger import save_curve
from kanga.src.utils.feat_sel import prim_mask_to_feat_mask
from kanga.src.utils.symbolic_lib import PRIMS
import pandas as pd
from scipy.spatial.distance import pdist


@dataclass(slots=True, frozen=True)
class GAConfig:
    pop_size: int = 100
    generations: int = 50
    cxpb: float = 0.8
    mutpb: float = 0.2
    tourn_size: int = 3
    seed: int = 25


def _build_toolbox(
    n_bits: int,
    fitness_fn: Callable[[npt.NDArray[np.bool_]], float],
    cfg: GAConfig,
) -> base.Toolbox:
    rng_np = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", rng_np.choice, [False, True], p=[0.5, 0.5])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_bits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_bits)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tourn_size)
    return toolbox


# -----------------------------------------------------------------------------
# helpers for statistics -------------------------------------------------------


def _size(pop):
    return float(np.mean([ind.sum() for ind in pop]))


def _diversity(pop):
    if len(pop) < 2:
        return 0.0
    bin_pop = np.vstack(pop).astype(int)
    d = pdist(bin_pop, metric="hamming")
    return float(d.mean())


# -----------------------------------------------------------------------------
# main GA ----------------------------------------------------------------------

def _log_best_individual(gen, hof, n_prims):
    """Log the best individual's selected primitives"""
    if len(hof) > 0:
        best = hof[0]
        selected = [PRIMS[i].name for i, val in enumerate(best) if val]
        print(f"Gen {gen}: Best fitness={best.fitness.values[0]:.4f}, Primitives={selected}")


def run_ga(
    fcq_cache: dict[str, npt.NDArray[np.floating]],
    X_feat: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    *,
    cfg: GAConfig | None = None,
    cache_dir: str | Path | None = None,
    logger_path: str | Path | None = None,
) -> tuple[npt.NDArray[np.bool_], list[float]]:
    cfg = cfg or GAConfig()
    n_features = fcq_cache["xy"].shape[0]
    n_prims = len(PRIMS)
    n_vars = n_features // n_prims

    print("Pre-computing MI scores...")
    all_mi_scores = mutual_info_regression(X_feat, y, random_state=12)
    print("MI computation done.")

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        tag = f"ga_{n_prims}_{cfg.pop_size}_{cfg.generations}_{cfg.seed}.npz"
        fpath = cache_dir / tag
        if fpath.exists():
            data = np.load(fpath)
            return data["mask"].astype(bool), data["curve"].tolist()

    fitness_fn = partial(_fitness_wrapper, cache=fcq_cache, n_vars=n_vars, X_feat=X_feat, y=y, mi_scores=all_mi_scores)
    toolbox = _build_toolbox(n_prims, fitness_fn, cfg)

    pop = toolbox.population(n=cfg.pop_size)
    hof = tools.HallOfFame(1, similar=lambda a, b: np.array_equal(a, b))

    stats = tools.Statistics(lambda ind: ind)
    stats.register("best", lambda pop: max(ind.fitness.values[0] for ind in pop))
    stats.register("mean", lambda pop: np.mean([ind.fitness.values[0] for ind in pop]))
    stats.register("median", lambda pop: np.median([ind.fitness.values[0] for ind in pop]))
    stats.register("worst", lambda pop: min(ind.fitness.values[0] for ind in pop))
    stats.register("size", _size)
    stats.register("div", _diversity)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=cfg.cxpb,
        mutpb=cfg.mutpb,
        ngen=cfg.generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    best = hof[0]
    print(f"Best individual: {best}, Fitness: {best.fitness.values[0]:.4f}")
    best_mask = np.asarray(best, dtype=bool)
    raw_best = np.asarray(logbook.select("best"), dtype=np.float32)
    curve: list[float] = np.maximum.accumulate(raw_best).tolist()

    if cache_dir:
        np.savez_compressed(fpath, mask=best_mask.astype(bool), curve=np.asarray(curve))

    if logger_path is not None:
        logger_path = Path(logger_path)
        logger_path.parent.mkdir(parents=True, exist_ok=True)
        save_curve(curve, logger_path)
        pd.DataFrame(logbook).to_csv(logger_path.with_suffix(".csv"), index=False)

    return best_mask, curve

def _fitness_wrapper(individual: npt.NDArray[np.bool_], *, cache, n_vars: int, X_feat, y, mi_scores) -> float:
    """
    Fitness for fixed-size primitive selection (exactly K primitives)
    """
    K = 4  # Target number of primitives
    
    # Penalize if not exactly K primitives
    n_selected = individual.sum()
    if n_selected < K:
        return (-np.inf,)
    
    feat_mask = prim_mask_to_feat_mask(individual, n_vars)
    idx = np.flatnonzero(feat_mask)
    
    total_mi = mi_scores[idx].sum()

    # Model complexity
    n_features = len(idx)
    selected_prims = [p for p, selected in zip(PRIMS, individual) if selected]

    # Complexity includes both number and cost of primitives
    primitive_costs = sum(p.cost for p in selected_prims)
    complexity = n_features + primitive_costs

    # Ratio: information per unit of complexity
    score = total_mi / complexity
    
    return (float(score),)

__all__ = ["GAConfig", "run_ga"]
