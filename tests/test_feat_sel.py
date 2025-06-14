import numpy as np
from kanga.src.utils.symbolic_lib import PRIMS
from kanga.src.utils.feat_sel import (
    build_feature_matrix,
    compute_fcq_cache,
    mrmr_score,
    get_feature_names,
)


def fake_dataset(n: int = 20, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] ** 2 - X[:, 1]).astype(np.float32)  # qualquer relação
    return X, y


def test_build_shape():
    X, _ = fake_dataset()
    Xf = build_feature_matrix(X, cache_dir=None)
    assert Xf.shape == (X.shape[0], X.shape[1] * len(PRIMS))
    assert np.isfinite(Xf).all()


def test_fcq_cache_consistency():
    X, y = fake_dataset()
    Xf = build_feature_matrix(X, cache_dir=None)
    cache = compute_fcq_cache(Xf, y, cache_dir=None)
    m = Xf.shape[1]
    assert cache["xy"].shape == (m,)
    assert cache["xx"].shape == (m, m)


def test_mrmr_score_monotonic():
    """
    Em FCQ a pontuação é rel / red.
    Se o conjunto crescer, a relevância média aumenta
    e a redundância média também — não há garantia
    estrita de monotonicidade.  Aqui testamos apenas
    que o score é finito e >= 0.
    """
    X, y = fake_dataset()
    Xf = build_feature_matrix(X, cache_dir=None)
    cache = compute_fcq_cache(Xf, y, cache_dir=None)
    m = Xf.shape[1]

    mask = np.zeros(m, dtype=bool)
    mask[:5] = True
    score = mrmr_score(mask, cache)

    assert np.isfinite(score)
    assert score >= 0.0


def test_feature_names_order():
    names = get_feature_names(2)
    assert names[1].startswith("x0_") and names[len(PRIMS)].startswith("x1_")