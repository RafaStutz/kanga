import numpy as np
from kanga.src.utils.feat_sel import build_feature_matrix, compute_fcq_cache
from kanga.src.evo.ga import GAConfig, run_ga
import pandas as pd
import pytest
from pathlib import Path


def fake_dataset(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] ** 2 - X[:, 1]).astype(np.float32)
    return X, y


def test_ga_runs():
    X, y = fake_dataset()
    Xf = build_feature_matrix(X, cache_dir=None)
    cache = compute_fcq_cache(Xf, y, cache_dir=None)

    cfg = GAConfig(
        pop_size=30,     
        generations=10,
        seed=123,
    )

    best_mask, curve = run_ga(cache, cfg=cfg, cache_dir=None, logger_path=None,)

    assert best_mask.dtype == bool and best_mask.size == Xf.shape[1]

    assert best_mask.sum() >= 1

    assert len(curve) == cfg.generations + 1

    assert all(curve[i] <= curve[i + 1] for i in range(len(curve) - 1))


def test_ga_runs_with_logger(tmp_path: Path):
    X, y = fake_dataset()
    Xf = build_feature_matrix(X, cache_dir=None)
    cache = compute_fcq_cache(Xf, y, cache_dir=None)

    cfg = GAConfig(pop_size=20, generations=5, seed=123)

    csv_path = tmp_path / "curve.csv"

    best_mask, curve = run_ga(
        cache,
        cfg=cfg,
        cache_dir=None,
        logger_path=csv_path, 
    )

    assert best_mask.dtype == bool and best_mask.size == Xf.shape[1]
    assert best_mask.sum() >= 1

    assert len(curve) == cfg.generations + 1
    assert all(c1 <= c2 for c1, c2 in zip(curve, curve[1:]))

    assert csv_path.exists() and csv_path.stat().st_size > 0

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["gen", "best"]
    assert len(df) == len(curve)
    assert np.allclose(df["best"].values, curve)