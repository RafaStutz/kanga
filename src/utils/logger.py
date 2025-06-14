from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_curve(curve: list[float], path: str | Path) -> None:
    df = pd.DataFrame({"gen": range(len(curve)), "best": curve})
    df.to_csv(path, index=False)

def plot_convergence(curve: list[float], *, show: bool = False, save: str | Path | None = None):
    plt.figure()
    plt.plot(curve, linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("GA convergence")
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()