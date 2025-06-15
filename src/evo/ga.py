from __future__ import annotations

"""Evolutionary feature‑subset search .

Implements a minimal GA using DEAP to optimise the
mRMR‑FCQ fitness.
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
from kanga.src.utils.feat_sel import get_feature_names



@dataclass(slots=True, frozen=True)
class GAConfig:
    pop_size: int = 100
    generations: int = 100
    cxpb: float = 0.9            
    mutpb: float = 0.1          
    tourn_size: int = 3         
    seed: int = 25                      


def _build_toolbox(
    n_features: int,
    fitness_fn: Callable[[npt.NDArray[np.bool_]], float],
    cfg: GAConfig,
) -> base.Toolbox:
    """Create and return a DEAP toolbox fully configured."""

    rng_np = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", rng_np.choice, [False, True])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_features)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tourn_size)

    return toolbox

def run_ga(
    fcq_cache: dict[str, npt.NDArray[np.floating]],
    *,
    cfg: GAConfig | None = None,
    cache_dir: str | Path | None = None,
    logger_path: str | Path | None = None,
) -> tuple[npt.NDArray[np.bool_], list[float]]:
    """Execute GA and return (best_mask, fitness_curve)."""

    cfg = cfg or GAConfig()
    n_features = fcq_cache["xy"].shape[0]

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        tag = f"ga_{n_features}_{cfg.pop_size}_{cfg.generations}_{cfg.seed}.npz"
        fpath = cache_dir / tag
        if fpath.exists():
            data = np.load(fpath)
            return data["mask"].astype(bool), data["curve"].tolist()

    fitness_fn = partial(_fitness_wrapper, cache=fcq_cache)
    toolbox = _build_toolbox(n_features, fitness_fn, cfg)

    pop = toolbox.population(n=cfg.pop_size)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("best", np.max)

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

    best_mask: npt.NDArray[np.bool_] = np.asarray(hof[0], dtype=bool)
    raw_best = np.asarray(logbook.select("best"), dtype=np.float32)
    curve: list[float] = np.maximum.accumulate(raw_best).tolist()

    if cache_dir:
        np.savez_compressed(fpath, mask=best_mask.astype(bool), curve=np.asarray(curve))

    if logger_path is not None:
        save_curve(curve, logger_path)

    return best_mask, curve


def _fitness_wrapper(individual: npt.NDArray[np.bool_], *, cache) -> tuple[float]:
    """DEAP expects a tuple fitness."""
    score = mrmr_score(individual, cache)
    if not np.isfinite(score):      # pega nan ou ±inf
        score = -1e9               # penaliza forte
    return (score,)

__all__ = ["GAConfig", "run_ga"]