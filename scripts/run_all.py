#!/usr/bin/env python
"""Runs run_experiment.py for all datasets and saves a JSON for each dataset.

Usage:
    python run_all.py \
        --data-dir kanga/src/data/srsd_dataframes \
        --summary   summary.json \
        --out-dir   results \
        --seed      0
"""
from __future__ import annotations
import argparse, json, subprocess, traceback
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Pasta onde estão os .csv")
    p.add_argument("--summary", type=Path, required=True,
                   help="Arquivo summary.json (para listar os datasets)")
    p.add_argument("--out-dir",  type=Path, default=Path("results"),
                   help="Onde gravar os *.json individuais")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.summary.open() as f:
        datasets = list(json.load(f)["equations"].keys())

    print(f"=== Executando {len(datasets)} datasets ===")

    for name in datasets:
        csv_path = args.data_dir / f"{name}.csv"
        out_json = args.out_dir / f"{name}_metrics.json"

        cmd = [
            "python", "run_experiment.py",
            "--csv",  str(csv_path),
            "--out",  str(out_json),
            "--seed", str(args.seed),
        ]
        print(f"\n  {name} …")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"‼  {name} failed: {e}\n{traceback.format_exc()}")
            continue

    print("Done.")

if __name__ == "__main__":
    main()