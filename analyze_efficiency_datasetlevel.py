"""
analyze_efficiency_datasetlevel.py  --  Pseudoreplication-safe efficiency test
                                        (Reviewers R1.8 / R2.5).

The submitted paper tested efficiency over 300 "dataset-seed blocks" as if they were
independent, which inflates significance (30 seeds from one dataset are clustered, not
30 independent problems). This script redoes the efficiency analysis with the DATASET
as the experimental unit:

  1. For each (optimizer, dataset), take the MEDIAN evals-to-common-target over the 30
     seeds  -> a 10 (datasets) x K (optimizers) matrix.
  2. Friedman test across the 10 datasets (matches the accuracy analysis exactly).
  3. If significant, Nemenyi post-hoc + critical-difference ranks.
  4. Report per-optimizer mean rank and which pairwise differences survive.

It also directly answers the SH question: does successive halving's apparent edge at
high targets survive when the dataset (not the seed) is the unit?

INPUT
-----
Reads the per-run common-target file produced by analyze_common_target.py:
    results/efficiency/common_target_by_run.csv
with columns: dataset, optimizer, seed, target, evals_to_target, reached

USAGE
-----
    python analyze_efficiency_datasetlevel.py \
        --by_run results/efficiency/common_target_by_run.csv \
        --out results/efficiency --target 0.98

Run once per target level of interest (0.90, 0.95, 0.98).

NOTE ON CENSORING
-----------------
evals_to_target = budget+1 for runs that never reached the target. Taking the median
over seeds is robust to a minority of censored runs. We also report the per-optimizer
reach_rate so a low median is never read without its attainment context.
"""

import argparse
import itertools
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

try:
    import scikit_posthocs as sp
    HAVE_POSTHOCS = True
except Exception:
    HAVE_POSTHOCS = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by_run", required=True)
    ap.add_argument("--out", default="results/efficiency")
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.by_run)
    for c in ["dataset", "optimizer", "seed", "target", "evals_to_target", "reached"]:
        if c not in df.columns:
            sys.exit(f"[ERROR] missing column: {c}")

    d = df[np.isclose(df["target"], args.target)].copy()
    if d.empty:
        sys.exit(f"[ERROR] no rows for target={args.target}. "
                 f"Available: {sorted(df['target'].unique())}")

    optimizers = sorted(d["optimizer"].unique())
    datasets = sorted(d["dataset"].unique())
    os.makedirs(args.out, exist_ok=True)

    # ---- Step 1: dataset-level summary (median over seeds) ----
    med = (d.groupby(["dataset", "optimizer"])["evals_to_target"]
             .median().unstack("optimizer"))
    med = med[optimizers]  # consistent column order
    reach = (d.groupby(["optimizer"])["reached"].mean())

    med.to_csv(os.path.join(args.out, f"efficiency_datasetlevel_{int(args.target*100)}.csv"))

    print(f"=== Dataset-level efficiency at target {args.target:.0%} "
          f"(unit = dataset, n={len(datasets)}) ===\n")
    print("Median evals-to-target per optimizer, per dataset:")
    print(med.round(1).to_string())
    print()

    # ---- Step 2: Friedman across datasets ----
    # each optimizer is a column; each dataset is a block (row)
    cols = [med[o].values for o in optimizers]
    stat, p = friedmanchisquare(*cols)
    print(f"Friedman across {len(datasets)} datasets: chi2 = {stat:.2f}, p = {p:.3g}")
    if p < args.alpha:
        print(f"  -> significant at alpha={args.alpha}: efficiency DOES differ.")
    else:
        print(f"  -> NOT significant at alpha={args.alpha}: no efficiency difference "
              f"survives dataset-level blocking.")
    print()

    # ---- per-optimizer mean rank (lower evals = better = lower rank) ----
    ranks = med.rank(axis=1, ascending=True, method="average")  # 1 = fastest
    mean_rank = ranks.mean(axis=0).sort_values()
    print("Mean rank (1 = most efficient), dataset-level:")
    for o, r in mean_rank.items():
        print(f"  {o:10} mean_rank={r:.2f}  reach_rate={reach[o]:.0%}")
    print()

    # ---- Step 3: Nemenyi post-hoc if significant ----
    if p < args.alpha and HAVE_POSTHOCS:
        nem = sp.posthoc_nemenyi_friedman(med.values)
        nem.index = optimizers
        nem.columns = optimizers
        nem.to_csv(os.path.join(args.out, f"nemenyi_{int(args.target*100)}.csv"))
        print("Nemenyi post-hoc p-values (pairs with p<0.05 differ significantly):")
        # report only the significant pairs to keep it readable
        sig_pairs = []
        for a, b in itertools.combinations(optimizers, 2):
            pv = nem.loc[a, b]
            if pv < args.alpha:
                sig_pairs.append((a, b, pv))
        if sig_pairs:
            for a, b, pv in sorted(sig_pairs, key=lambda x: x[2]):
                print(f"  {a:10} vs {b:10}  p={pv:.4f}")
        else:
            print("  (Friedman significant but no pair survives Nemenyi correction.)")
        print()

        # ---- Step 4: SH-specific verdict ----
        print("=== Successive-halving (sh) verdict at this target ===")
        if "sh" in optimizers:
            sh_rank = mean_rank.get("sh", np.nan)
            better_than = [o for o in optimizers if o != "sh"
                           and mean_rank["sh"] < mean_rank[o]]
            sh_sig = [(o, nem.loc["sh", o]) for o in optimizers if o != "sh"
                      and nem.loc["sh", o] < args.alpha]
            print(f"  sh mean rank = {sh_rank:.2f} (of {len(optimizers)}); "
                  f"nominally faster than {len(better_than)} optimizers.")
            sh_sig_names = [o for o, pv in sh_sig]
            sh_vs_adaptive = [o for o in sh_sig_names if o != "grid"]
            if sh_sig:
                print("  sh is SIGNIFICANTLY faster than: " +
                      ", ".join(f"{o} (p={pv:.3f})" for o, pv in sh_sig))
            if sh_vs_adaptive:
                print("  -> SH's edge SURVIVES over ADAPTIVE methods "
                      f"({', '.join(sh_vs_adaptive)}); a genuine high-target advantage.")
            elif sh_sig_names == ["grid"]:
                print("  -> SH is faster than GRID only, like every adaptive method. "
                      "Its apparent edge over the adaptive group does NOT survive "
                      "dataset-level blocking; report as within the adaptive cluster.")
            else:
                print("  -> SH is NOT significantly faster than any optimizer after "
                      "Nemenyi correction; report as non-significant.")
    elif p < args.alpha and not HAVE_POSTHOCS:
        print("[WARN] scikit-posthocs not installed; cannot run Nemenyi.")
        print("       pip install scikit-posthocs")
    else:
        print("Friedman not significant -> no post-hoc needed. No optimizer, including")
        print("sh, is significantly more efficient once the dataset is the unit.")

    print("\nWrote dataset-level matrix and (if significant) Nemenyi table to", args.out)


if __name__ == "__main__":
    main()