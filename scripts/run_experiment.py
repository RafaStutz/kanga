from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import sympy as sp
from sklearn.metrics import mean_squared_error, r2_score

from kanga.src.data.dataio import load_feynman_csv                        
from kanga.src.utils.feat_sel import build_feature_matrix, compute_fcq_cache, get_feature_names  
from kanga.src.evo.ga import run_ga                                   
from kanga.src.models.baseline_kan import SymbolicKANConfig, train_symbolic_kan 
from kanga.src.models.evo_kan import train_symbolic_kan_ga   
from kanga.src.utils.symbolic_lib import PRIMS


def main() -> None:
    parser = argparse.ArgumentParser(description="End‑to‑end experiment runner")
    parser.add_argument("--csv", type=Path, required=True, help="Path to dataset CSV")
    parser.add_argument("--out", type=Path, default="metrics.json", help="Output JSON")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # 1 Load & split
    csv_name = args.csv.stem
    X_tr, X_te, y_tr, y_te = load_feynman_csv(csv_name, random_state=args.seed)

    # 2. Feature expansion (symbolic lib primitives)
    X_feat_tr = build_feature_matrix(X_tr)
    X_feat_te = build_feature_matrix(X_te)

    # 3. mRMR cache
    mi_cache = compute_fcq_cache(X_feat_tr, y_tr)

    # 4. KAN baseline training
    base_metrics, base_models = train_symbolic_kan(X_feat_tr, y_tr, X_feat_te, y_te)
    baseline_kan_time = base_metrics[2]["kan_time"]
    baseline_sym_time = base_metrics[2]["sym_time"]
    base_report = {
        "depth":     2,
        "metrics":   base_metrics[2],
        "formula":   str(base_models[2].symbolic_formula()[0][0]),
        "total_time":  baseline_kan_time,
        "sym_time":  baseline_sym_time,
    }

    # 5.  Genetic Algorithm
    tic_ga = perf_counter()
    best_mask, fitness_curve = run_ga(
        mi_cache, 
        X_feat_tr,
        y_tr,      
        logger_path=f"runs/ga_{csv_name}.png"
    )
    ga_time = perf_counter() - tic_ga
    feat_names = np.array(get_feature_names(X_tr.shape[1]))

    lib_ga = [prim.name for prim, flag in zip(PRIMS, best_mask) if flag]

    print(f"GA library: {lib_ga}")
    print(f"GA best mask: {best_mask}")

    # 6. KAN with GA library
    evo_metrics, evo_models = train_symbolic_kan_ga(
        X_feat_tr, y_tr, X_feat_te, y_te, lib=lib_ga)

    evo_kan_time       = evo_metrics[2]["kan_time"]
    total_pipeline_s   = ga_time + evo_kan_time

    evo_report = {                      
        "library":    lib_ga,
        "depth":      2,
        "metrics":    evo_metrics[2],
        "formula":    str(evo_models[2].symbolic_formula()[0][0]),
        "kan_time":   evo_kan_time,
        "sym_time":   evo_metrics[2]["sym_time"],
        "ga_time":    ga_time,
        "total_evo_pipeline": total_pipeline_s,
    }

    # 7. Assemble and dump JSON
    report: dict[str, Any] = {
        "dataset": args.csv.name,
        "n_train": int(X_tr.shape[0]),
        "n_test":  int(X_te.shape[0]),
        "baseline": base_report,
        "evokan":  evo_report,
        "ga": {
            "generations": len(fitness_curve),
            "best_fitness": float(fitness_curve[-1]),
            "time_s": ga_time,
            "mask_size": int(best_mask.sum()),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Metrics saved: {args.out}")


if __name__ == "__main__":
    main()