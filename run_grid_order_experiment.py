"""
run_grid_order_experiment.py  --  Grid traversal-order sensitivity (Reviewer R2.6).

WHY
---
A grid has no intrinsic evaluation order: the same 100 lattice points can be visited
in any sequence. The submitted study used a fixed row-major sweep, which starts in the
low-C/low-gamma corner (a poor-accuracy region) and produces the staircase seen in the
anytime curve. Because "evaluations to reach a target" depends on WHEN good points are
visited, part of grid search's apparent inefficiency was an ordering artefact.

This script quantifies that. It runs grid search under four traversal policies on every
dataset and reports the distribution of efficiency, so the paper can (a) show how much
of the disadvantage was ordering, and (b) anchor its claim on an order-independent
number (the average over random orderings).

POLICIES
--------
  row_major : the submitted deterministic sweep (C ascending, gamma ascending)
  col_major : the same sweep with loops swapped
  spiral    : centre-outwards (a "smart" deterministic policy)
  random    : uniformly random permutation, a different one per repetition
              -> this is the order-independent reference

OUTPUT
------
  grid_order_by_run.csv  : one row per (dataset, policy, rep) with the efficiency
                           measures and final accuracy
  console                : per-policy median evals-to-target, and the spread across
                           random orderings

USAGE
-----
    python run_grid_order_experiment.py --out results/grid_order --n_reps 30
    # optionally restrict datasets for a quick pilot:
    #   --datasets banknote,car
"""

import argparse
import importlib.util
import os
import sys
import numpy as np
import pandas as pd


def load_bench(path="hpo_benchmark_v3.py"):
    spec = importlib.util.spec_from_file_location("hb", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hpo_benchmark_v3.py")
    ap.add_argument("--out", default="results/grid_order")
    ap.add_argument("--n_reps", type=int, default=30,
                    help="repetitions; for deterministic policies only rep 0 differs "
                         "by the train/test split, for 'random' the ordering also varies")
    ap.add_argument("--datasets", default="",
                    help="comma-separated dataset names; default = all in CONFIG")
    args = ap.parse_args()

    hb = load_bench(args.bench)
    os.makedirs(args.out, exist_ok=True)

    all_ds = hb.CONFIG["datasets"]
    if args.datasets:
        want = {d.strip() for d in args.datasets.split(",")}
        all_ds = [(oid, name) for oid, name in all_ds if name in want]
        if not all_ds:
            sys.exit(f"[ERROR] none of {want} found in CONFIG['datasets']")

    policies = ["row_major", "col_major", "spiral", "random"]
    rows = []

    from sklearn.model_selection import train_test_split

    for oid, name in all_ds:
        print(f"\n--- {name} ---", flush=True)
        X, y = hb.load_dataset(oid)
        for policy in policies:
            for rep in range(args.n_reps):
                cfg = dict(hb.CONFIG)
                cfg["cv_n_jobs"] = cfg.get("cv_n_jobs", 1)
                cfg["grid_order"] = policy
                # same seed structure as the baseline runs: split varies with rep
                seed_split, seed_cv, seed_opt = hb._resolve_seeds(rep, cfg)

                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=cfg["test_size"],
                    random_state=seed_split, stratify=y)
                Xtr, ytr = hb.maybe_subsample(Xtr, ytr, cfg, seed_split)

                obj = hb.Objective(Xtr, ytr, cfg, cv_seed=seed_cv)
                hb.run_grid(obj, cfg, seed_opt)

                bc = obj.best_config()
                if bc is None:
                    continue
                _, bC, bg, cv_best = bc
                test_acc = hb.evaluate_test(Xtr, ytr, Xte, yte, bC, bg, cfg)
                curve = obj.best_so_far_curve()

                # self-referential metric (for comparison with the submitted number)
                thr = 0.99 * cv_best
                reach99 = (int(np.argmax(curve >= thr)) + 1
                           if np.any(curve >= thr) else cfg["budget"] + 1)

                rows.append(dict(dataset=name, policy=policy, rep=rep,
                                 seed_split=seed_split, seed_cv=seed_cv,
                                 seed_opt=seed_opt,
                                 cv_best=cv_best, test_acc=test_acc,
                                 evals_to_99pct=reach99,
                                 final_curve_value=curve[-1]))
                # deterministic policies do not depend on seed_opt, but the split does
                # vary with rep, so we keep all reps for a fair comparison.
            print(f"  {policy:10} done", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(args.out, "grid_order_by_run.csv")
    df.to_csv(path, index=False)

    # ---------------- report ----------------
    print("\n" + "=" * 68)
    print("GRID TRAVERSAL-ORDER SENSITIVITY (R2.6)")
    print("=" * 68)

    print("\nMedian evals_to_99pct by policy (pooled over datasets & reps):")
    summ = df.groupby("policy")["evals_to_99pct"].agg(["median", "mean", "min", "max"])
    print(summ.round(2).to_string())

    print("\nPer-dataset median evals_to_99pct by policy:")
    piv = df.pivot_table(index="dataset", columns="policy",
                         values="evals_to_99pct", aggfunc="median")
    print(piv.round(1).to_string())

    print("\nSpread across RANDOM orderings (the order-independent reference):")
    r = df[df["policy"] == "random"]
    for ds, g in r.groupby("dataset"):
        v = g["evals_to_99pct"]
        print(f"  {ds:14} median={v.median():5.1f}  IQR=[{v.quantile(.25):.0f},"
              f"{v.quantile(.75):.0f}]  min={v.min()}  max={v.max()}")

    print("\nFinal accuracy is essentially unaffected by ordering (sanity check):")
    acc = df.groupby("policy")["test_acc"].mean()
    print(acc.round(4).to_string())
    print("\n-> Ordering changes WHEN good points are found, not WHICH points exist,")
    print("   so final accuracy should be stable while efficiency is not.")

    print("\nWrote:", path)


if __name__ == "__main__":
    main()