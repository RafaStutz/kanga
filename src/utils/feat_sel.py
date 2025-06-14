import numpy as np
import numpy.typing as npt
from pathlib import Path
from hashlib import blake2b
from tqdm import tqdm
from sklearn.feature_selection import f_regression

from .symbolic_lib import PRIMS, NP_FUNCS

def build_feature_matrix(
    X: npt.NDArray[np.floating],
    *,
    dtype: np.dtype = np.float32,
    cache_dir: str | Path | None = ".cache",
    force: bool = False,
) -> npt.NDArray[np.floating]:
    """Returns X_feat (n x d x |PRIMS|).

    Result is cached in ``cache_dir`` using a hash.
    """
    X = np.asarray(X, dtype=dtype, order="C")
    n, d = X.shape

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        digest = blake2b(X.tobytes(), digest_size=8).hexdigest()
        fpath = cache_dir / f"feat_{digest}_{d}_{len(PRIMS)}.npz"
        if fpath.exists() and not force:
            return np.load(fpath)["arr_0"]

    blocks: list[npt.NDArray[np.floating]] = []
    for col in tqdm(X.T, desc="Expanding primitives"):
        for fn in NP_FUNCS:
            blocks.append(fn(col))
    X_feat = np.stack(blocks, axis=1).astype(dtype, copy=False)

    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e9, neginf=-1e9)

    if cache_dir:
        np.savez_compressed(fpath, X_feat)

    return X_feat


def _relevance_f_stat(
    X_feat: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Calculates univariate F‑statistic for each column vs y."""
    fvals, _ = f_regression(X_feat, y)
    fvals = np.nan_to_num(fvals, nan=0.0, posinf=0.0)
    return fvals.astype(np.float32, copy=False)


def _redundancy_corr(
    X_feat: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Pearson correlation matrix between columns."""
    corr = np.corrcoef(X_feat, rowvar=False).astype(np.float32, copy=False)
    np.abs(corr, out=corr)
    np.nan_to_num(corr, nan=0.0, copy=False)
    return corr


def compute_fcq_cache(
    X_feat: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    *,
    cache_dir: str | Path | None = ".cache",
    force: bool = False,
):
    """Calculates and redundancy for mRMR‑FCQ."""
    y = np.asarray(y, dtype=X_feat.dtype).ravel()

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        h = blake2b(X_feat.tobytes() + y.tobytes(), digest_size=8).hexdigest()
        fpath = cache_dir / f"fcq_{h}.npz"
        if fpath.exists() and not force:
            data = np.load(fpath)
            return {"xy": data["xy"], "xx": data["xx"]}

    rel = _relevance_f_stat(X_feat, y)
    red = _redundancy_corr(X_feat)

    if cache_dir:
        np.savez_compressed(fpath, xy=rel, xx=red)

    return {"xy": rel, "xx": red}


def mrmr_score(mask: npt.NDArray[np.bool_], cache: dict[str, npt.NDArray[np.floating]]) -> float:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return -np.inf

    rel = cache["xy"][idx].mean(dtype=np.float32)
    sub = cache["xx"][np.ix_(idx, idx)]
    red = sub[np.triu_indices_from(sub, k=1)].mean(dtype=np.float32)

    return float(rel / (red + 1e-9))


def get_feature_names(d_original: int) -> list[str]:
    names: list[str] = []
    for j in range(d_original):
        for prim in PRIMS:
            names.append(f"x{j}_{prim.name}")
    return names
