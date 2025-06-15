"""
Unit-tests for the dataio.py loader.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import List

import importlib
import numpy as np
import pytest

# ---------- IMPORTAÇÃO CORRETA DO MÓDULO ----------
# ajuste aqui se seu pacote estiver noutra pasta
dataio = importlib.import_module("kanga.src.data.dataio")
# --------------------------------------------------

# funções usadas diretamente nos testes (opcional)
_row_to_xy  = dataio._row_to_xy
_pad_block  = dataio._pad_block
load_feynman = dataio.load_feynman


@pytest.fixture()
def fake_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture()
def fake_meta(monkeypatch) -> List[dict]:
    """Patch _load_meta to return 3 toy equations with 2 real variables."""
    meta = [
        {"symbols": ["x1", "x2", "y"]},
        {"symbols": ["x1", "x2", "y"]},
        {"symbols": ["x1", "x2", "y"]},
    ]
    monkeypatch.setattr(dataio, "_load_meta", lambda cache, tier: meta)
    return meta


@pytest.fixture()
def fake_dataset(monkeypatch):
    """HF-like Dataset com acesso via [] e atributos."""
    rows = ["1 2 9 10", "3 4 8 12", "5 6 7 14"]

    class FakeDS(dict):
        def __init__(self):
            super().__init__(
                text=rows,
                eq_index=[0, 1, 2],
                features={"eq_index": None},
            )

        def __getattr__(self, item):
            return self[item]

    ds = FakeDS()
    monkeypatch.setattr(dataio, "_fetch_dataset", lambda *_, **__: ds)
    return ds


##############################################################################
# unit tests
##############################################################################
def test_row_to_xy():
    x, y = dataio._row_to_xy("1 2 3 4")
    assert np.allclose(x, [1, 2, 3])
    assert y == 4


@pytest.mark.parametrize("d_max", [2, 4])
def test_pad_block_shapes(d_max: int):
    rows = [ (np.arange(i+1, i+3, dtype=np.float32), float(i))
             for i in range(3) ]
    X, y = dataio._pad_block(rows, d_max)
    assert X.shape == (3, d_max)
    assert y.shape == (3,)
    # padding zeros on the right
    assert np.all(X[:, -1] == 0) if d_max > 2 else True


def test_load_feynman_builds_cache(fake_cache, fake_dataset, fake_meta):
    X, y, idx, meta = dataio.load_feynman(
        tier="easy",
        split="train",
        include_dummy=False,
        cache_dir=fake_cache,
    )
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    assert idx.tolist() == [0, 1, 2]
    assert meta == fake_meta
    expected_npz = fake_cache / "easy_train_nodummy.npz"
    assert expected_npz.exists()


def test_cache_hit_skips_download(monkeypatch, fake_cache, fake_dataset,
                                  fake_meta):
    """Second call must read .npz and NOT call _fetch_dataset."""
    _ = dataio.load_feynman("easy", "train",
                            cache_dir=fake_cache, include_dummy=False)

    monkeypatch.setattr(dataio, "_fetch_dataset",
                        lambda *_: (_ for _ in ()).throw(
                            RuntimeError("cache not used")))

    X, y, _, _ = dataio.load_feynman("easy", "train",
                                     cache_dir=fake_cache, include_dummy=False)
    assert X.shape[0] == 3 and y.size == 3
