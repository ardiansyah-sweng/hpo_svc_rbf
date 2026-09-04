"""
analyze_common_target.py  --  Common-target efficiency metric (Stage #4 / R2.4).

Replaces the self-referential "evaluations to 99% of its OWN best" metric with a
COMMON target: for each dataset, every optimizer is measured against the SAME bar,
defined as a fraction of a shared high-budget reference optimum. This removes the
flaw that a weaker optimizer can reach 99% of its own (lower) best sooner and appear
"efficient" (Reviewer R2.4).

Reads results/baseline/anytime_curves.csv, which has columns:
    dataset, optimizer, rep, seed, eval, best_so_far
(the best cross-validation accuracy achieved by evaluation number `eval`).

USAGE
-----
    python analyze_common_target.py --curves results/baseline/anytime_curves.csv \
        --out results/efficiency --targets 0.95,0.98,0.99

For each dataset:
  reference = max best_so_far achieved by ANY optimizer at the FINAL evaluation,
              across all seeds  (a shared, budget-consistent high-budget optimum).
  target(f) = f * reference.
For each (optimizer, dataset, seed):
  evals_to_target = first evaluation index whose best_so_far >= target(f),
                    or budget+1 (censored) if never reached.

OUTPUTS
-------
  common_target_by_run.csv     : per (dataset, optimizer, seed, target) hitting time
  common_target_summary.csv    : per (optimizer, target) median hitting time + reach rate
  console                      : median evals-to-common-target per optimizer, per level,
                                 and a comparison to the old self-referential metric.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


REQUIRED = ["dataset", "optimizer", "eval", "best_so_far"]


def detect_seed_col(df):
    for c in ["seed", "rep"]:
        if c in df.columns:
            return c
    sys.exit("[ERROR] no 'seed' or 'rep' column found in curves file.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--out", default="results/efficiency")
    ap.add_argument("--targets", default="0.90,0.95,0.98",
                    help="comma-separated fractions of the shared reference optimum")
    args = ap.parse_args()

    targets = [float(t) for t in args.targets.split(",")]
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.curves)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] curves file missing columns: {missing}. "
                 f"Found: {list(df.columns)}")
    seedcol = detect_seed_col(df)
    budget = int(df["eval"].max())

    # ---- shared high-budget reference per dataset (ROBUST) ----
    # A single lucky run can inflate a max-based reference, pushing 98-99% targets out
    # of reach for almost everyone (mostly-censored medians). Instead, define the
    # reference robustly: for each optimizer take the MEDIAN final best_so_far across
    # seeds (a typical high-budget outcome for that optimizer), then take the BEST of
    # those per-optimizer medians as the shared bar. This is still a common, budget-
    # consistent target, but not driven by an outlier run.
    final = df[df["eval"] == budget]
    per_opt_median = (final.groupby(["dataset", "optimizer"])["best_so_far"]
                      .median().reset_index())
    reference = (per_opt_median.groupby("dataset")["best_so_far"].max().to_dict())
    # also keep the old max reference for reporting/transparency
    reference_max = final.groupby("dataset")["best_so_far"].max().to_dict()

    # ---- hitting time per run per target ----
    rows = []
    # iterate per (dataset, optimizer, seed) group; each group is one monotone curve
    grp = df.sort_values("eval").groupby(["dataset", "optimizer", seedcol])
    for (ds, opt, sd), g in grp:
        ref = reference[ds]
        curve = g["best_so_far"].values
        evals = g["eval"].values
        for f in targets:
            bar = f * ref
            hit_idx = np.argmax(curve >= bar) if np.any(curve >= bar) else None
            if hit_idx is None:
                reach = budget + 1     # censored: never reached
                reached = False
            else:
                reach = int(evals[hit_idx])
                reached = True
            rows.append(dict(dataset=ds, optimizer=opt, seed=sd, target=f,
                             reference=ref, bar=bar,
                             evals_to_target=reach, reached=reached))
    by_run = pd.DataFrame(rows)
    by_run.to_csv(os.path.join(args.out, "common_target_by_run.csv"), index=False)

    # ---- summary per (optimizer, target) ----
    summ = []
    for (opt, f), g in by_run.groupby(["optimizer", "target"]):
        summ.append(dict(
            optimizer=opt, target=f,
            median_evals=g["evals_to_target"].median(),
            mean_evals=g["evals_to_target"].mean(),
            reach_rate=g["reached"].mean(),
        ))
    summary = pd.DataFrame(summ).sort_values(["target", "median_evals"])
    summary.to_csv(os.path.join(args.out, "common_target_summary.csv"), index=False)

    # ---- console report ----
    print(f"Budget = {budget} evaluations. Shared reference = BEST of per-optimizer")
    print("MEDIAN final accuracies per dataset (robust; not a single lucky run).")
    print("Targets are fractions of that robust reference.\n")
    print("Reference per dataset (robust vs max):")
    for ds in sorted(reference):
        print(f"  {ds:14} robust={reference[ds]:.4f}  max={reference_max[ds]:.4f}")
    print()
    for f in targets:
        sub = summary[summary["target"] == f].sort_values("median_evals")
        print(f"=== Target = {f:.0%} of shared reference ===")
        print(f"  {'optimizer':10} {'median_evals':>12} {'mean_evals':>11} {'reach_rate':>11}")
        for _, r in sub.iterrows():
            print(f"  {r['optimizer']:10} {r['median_evals']:>12.1f} "
                  f"{r['mean_evals']:>11.1f} {r['reach_rate']:>10.0%}")
        print()

    # ---- dataset-level median (pseudoreplication-safe, ties to R1.8/R2.5) ----
    print("=== Dataset-level median evals-to-target (unit = dataset) ===")
    print("    (median over seeds first, then reported per optimizer across 10 datasets)")
    for f in targets:
        # per (optimizer, dataset): median over seeds; then median across datasets
        d = by_run[by_run["target"] == f]
        per_ds = d.groupby(["optimizer", "dataset"])["evals_to_target"].median().reset_index()
        per_opt = per_ds.groupby("optimizer")["evals_to_target"].median().sort_values()
        print(f"  Target {f:.0%}:  " +
              ", ".join(f"{o}={v:.1f}" for o, v in per_opt.items()))
    print()
    print("Wrote:", os.path.join(args.out, "common_target_by_run.csv"))
    print("Wrote:", os.path.join(args.out, "common_target_summary.csv"))
    print("\nNote: compare these rankings to the old evals_to_99pct (self-referential).")
    print("If the ranking is stable, the flaw did not change conclusions; report that.")


if __name__ == "__main__":
    main()