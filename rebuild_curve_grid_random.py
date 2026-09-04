"""
rebuild_curves_grid_random.py  --  Order-independent grid curves (Reviewer R2.6).

WHY
---
The baseline anytime_curves.csv was produced with grid search traversing its lattice
in a fixed row-major sweep. The traversal-order experiment showed that this choice
alone changes grid's efficiency by an order of magnitude, so any efficiency number
derived from those curves is partly an ordering artefact.

This script regenerates GRID's anytime curves using a RANDOM traversal order (a
different permutation per repetition), then splices them into a copy of the baseline
curves file, leaving the other eight optimizers untouched. The result is a curves file
in which grid's efficiency is order-independent (averaged over random orderings),
suitable for re-running the common-target and dataset-level analyses.

USAGE
-----
    python rebuild_curves_grid_random.py \
        --curves results/baseline/anytime_curves.csv \
        --out results/baseline_gridrandom \
        --n_reps 30

Then re-run the efficiency pipeline on the new curves:

    python analyze_common_target.py \
        --curves results/baseline_gridrandom/anytime_curves.csv \
        --out results/efficiency_gridrandom --targets 0.90,0.95,0.98

    python analyze_efficiency_datasetlevel.py \
        --by_run results/efficiency_gridrandom/common_target_by_run.csv \
        --out results/efficiency_gridrandom --target 0.95

NOTE
----
Only grid is recomputed (1 optimizer x 10 datasets x n_reps), so this is far lighter
than a full re-run. The other optimizers' curves are copied through unchanged, so the
comparison stays budget-matched and internally consistent.
"""

import argparse
import importlib.util
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_bench(path):
    spec = importlib.util.spec_from_file_location("hb", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hpo_benchmark_v3.py")
    ap.add_argument("--curves", required=True,
                    help="baseline anytime_curves.csv (grid rows will be replaced)")
    ap.add_argument("--out", default="results/baseline_gridrandom")
    ap.add_argument("--n_reps", type=int, default=30)
    args = ap.parse_args()

    hb = load_bench(args.bench)
    os.makedirs(args.out, exist_ok=True)

    old = pd.read_csv(args.curves)
    for c in ["dataset", "optimizer", "eval", "best_so_far"]:
        if c not in old.columns:
            sys.exit(f"[ERROR] curves file missing column: {c}")
    seedcol = "rep" if "rep" in old.columns else "seed"
    budget = int(old["eval"].max())

    n_grid_old = int((old["optimizer"] == "grid").sum())
    print(f"Baseline curves: {len(old)} rows; grid rows to be replaced: {n_grid_old}")
    print(f"Budget = {budget}; regenerating grid with RANDOM traversal order.\n")

    rows = []
    for oid, name in hb.CONFIG["datasets"]:
        print(f"--- {name} ---", flush=True)
        X, y = hb.load_dataset(oid)
        for rep in range(args.n_reps):
            cfg = dict(hb.CONFIG)
            cfg["grid_order"] = "random"          # the whole point
            cfg["cv_n_jobs"] = cfg.get("cv_n_jobs", 1)
            seed_split, seed_cv, seed_opt = hb._resolve_seeds(rep, cfg)

            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=cfg["test_size"],
                random_state=seed_split, stratify=y)
            Xtr, ytr = hb.maybe_subsample(Xtr, ytr, cfg, seed_split)

            obj = hb.Objective(Xtr, ytr, cfg, cv_seed=seed_cv)
            hb.run_grid(obj, cfg, seed_opt)
            curve = obj.best_so_far_curve()

            for i, v in enumerate(curve, start=1):
                rec = {"dataset": name, "optimizer": "grid",
                       "eval": i, "best_so_far": v}
                rec[seedcol] = rep
                if seedcol == "rep" and "seed" in old.columns:
                    rec["seed"] = rep
                rows.append(rec)
        print(f"  {name}: {args.n_reps} reps done", flush=True)

    new_grid = pd.DataFrame(rows)

    # splice: keep every non-grid row, replace grid rows
    keep = old[old["optimizer"] != "grid"].copy()
    merged = pd.concat([keep, new_grid[keep.columns.intersection(new_grid.columns)]],
                       ignore_index=True)
    # ensure column order matches the original file
    merged = merged[[c for c in old.columns if c in merged.columns]]

    out_path = os.path.join(args.out, "anytime_curves.csv")
    merged.to_csv(out_path, index=False)

    print(f"\nWrote {out_path}")
    print(f"  rows: {len(merged)} (non-grid kept: {len(keep)}, new grid: {len(new_grid)})")
    print(f"  optimizers: {sorted(merged['optimizer'].unique())}")
    print("\nNext:")
    print(f"  python analyze_common_target.py --curves {out_path} "
          f"--out results/efficiency_gridrandom --targets 0.90,0.95,0.98")
    print( "  python analyze_efficiency_datasetlevel.py "
           "--by_run results/efficiency_gridrandom/common_target_by_run.csv "
           "--out results/efficiency_gridrandom --target 0.95")


if __name__ == "__main__":
    main()