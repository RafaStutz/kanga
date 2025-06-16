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

from kanga.src.utils.feat_sel import mrmr_score
from kanga.src.utils.logger import save_curve
from kanga.src.utils.feat_sel import prim_mask_to_feat_mask
from kanga.src.utils.symbolic_lib import PRIMS
import pandas as pd
from scipy.spatial.distance import pdist


_LAMBDA_COST: float = 100000.0

PRIM_COSTS = np.array([p.cost for p in PRIMS], dtype=np.int16)


@dataclass(slots=True, frozen=True)
class GAConfig:
    pop_size: int = 1000
    generations: int = 100
    cxpb: float = 0.8
    mutpb: float = 0.2
    tourn_size: int = 3
    seed: int = 25


# -----------------------------------------------------------------------------
# toolbox ---------------------------------------------------------------------


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
    toolbox.register("attr_bool", rng_np.choice, [False, True], p=[0.1, 0.9])
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


def run_ga(
    fcq_cache: dict[str, npt.NDArray[np.floating]],
    *,
    cfg: GAConfig | None = None,
    cache_dir: str | Path | None = None,
    logger_path: str | Path | None = None,
) -> tuple[npt.NDArray[np.bool_], list[float]]:
    cfg = cfg or GAConfig()
    n_features = fcq_cache["xy"].shape[0]
    n_prims = len(PRIMS)
    n_vars = n_features // n_prims

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        tag = f"ga_{n_prims}_{cfg.pop_size}_{cfg.generations}_{cfg.seed}.npz"
        fpath = cache_dir / tag
        if fpath.exists():
            data = np.load(fpath)
            return data["mask"].astype(bool), data["curve"].tolist()

    fitness_fn = partial(_fitness_wrapper, cache=fcq_cache, n_vars=n_vars)
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
        verbose=False,
    )

    best = hof[0]
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


# -----------------------------------------------------------------------------
# fitness ----------------------------------------------------------------------


def _fitness_wrapper(individual: npt.NDArray[np.bool_], *, cache, n_vars: int) -> float:
    feat_mask = prim_mask_to_feat_mask(individual, n_vars)
    score = mrmr_score(feat_mask, cache)

    sel     = individual.astype(int)
    k       = sel.sum()                         
    if k == 0:                                  
        return (-np.inf,)

    avg_cost = (sel @ PRIM_COSTS) / k           

    fitness  = score \
              - _LAMBDA_COST * avg_cost        

    return (float(fitness),)

__all__ = ["GAConfig", "run_ga"]
