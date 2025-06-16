from __future__ import annotations
"""Feynman CSV data loader.

Assume that each CSV has columns `[var1, var2, ..., target]`,
where the right‑most column target is the value to predict.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "load_feynman_csv",
    "load_equation",
]


def _read_csv(csv_name: str) -> pd.DataFrame:
    current_dir = Path(__file__).parent
    csv_path = current_dir / "feynman_dataframes" / f"{csv_name}.csv"
    return pd.read_csv(csv_path)


def _split(df: pd.DataFrame, *, test_size: float, random_state: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in CSV – expected format [*, target]")

    X = df.drop(columns="target").to_numpy(dtype=np.float32, copy=False)
    y = df["target"].to_numpy(dtype=np.float32, copy=False)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def load_feynman_csv(csv_name: str, *, test_size: float = 0.2, random_state: int | None = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads a single Feynman CSV and splits it.

    Parameters
    ----------
    csv_name : str
    test_size : float, default 0.2
        Proportion of the test set (80/20 by default).
    random_state : int | None, default 0
        Seed for reproducibility (``None`` = random).

    Returns
    -------
    X_tr, X_te, y_tr, y_te : np.ndarray
        Arrays ready for KAN.fit (dtype ``float32``).
    """
    df = _read_csv(csv_name)
    return _split(df, test_size=test_size, random_state=random_state)


def load_equation(equation_key: str, *, root_dir: str | Path = "./feynman_dataframes", test_size: float = 0.2, random_state: int | None = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    csv_path = Path(root_dir) / f"{equation_key}.csv"
    return load_feynman_csv(csv_path, test_size=test_size, random_state=random_state)