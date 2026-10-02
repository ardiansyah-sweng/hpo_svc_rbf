"""
table6_spread_noise.py -- regenerate the between-optimizer spread versus
run-to-run noise table, with the aggregation stated explicitly (Reviewer 1
comment 7).

What the reviewer found
-----------------------
The submitted Table 6 gave banknote a baseline SD of 0.11 percentage points and
a ratio of 3.05, while its footnote said seven of nine optimizers have exactly
zero run-to-run variance, and Tables 5 and 7 reported an aggregate SD of 0.0000.
Those statements cannot all describe the same quantity.

Two separate mistakes produced that:

  * the aggregation was never stated. The column is the MEAN of the nine
    per-optimizer standard deviations. Its MEDIAN is 0.00 on banknote, so the
    ratio would be undefined under a median.
  * the "seven of nine" count came from the isolation experiments (Table 7),
    not from the baseline configuration this table is computed on.

This script reports both aggregations side by side, counts the zero-variance
optimizers in the baseline configuration itself, and marks any dataset whose
noise denominator is degenerate, so the ratio is never read as meaningful when
it is not.

    python table6_spread_noise.py --summary results/baseline_fullfid/summary.csv --out results/table6_fullfid
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
ap.add_argument("--out", default=os.path.join("results", "table6"))
ap.add_argument("--alpha", type=float, default=0.05)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

s = pd.read_csv(a.summary)
seedcol = "seed" if "seed" in s.columns else "rep"

rows = []
for ds, d in s.groupby("dataset"):
    per_opt_mean = d.groupby("optimizer")["test_acc"].mean()
    per_opt_sd = d.groupby("optimizer")["test_acc"].std(ddof=1)

    spread = (per_opt_mean.max() - per_opt_mean.min()) * 100
    sd_mean = per_opt_sd.mean() * 100
    sd_median = per_opt_sd.median() * 100
    n_zero = int((per_opt_sd < 1e-12).sum())
    n_opt = len(per_opt_sd)

    # within-dataset Friedman: blocks = repetitions, treatments = optimizers
    piv = d.pivot_table(index=seedcol, columns="optimizer", values="test_acc").dropna()
    fr = stats.friedmanchisquare(*[piv[c] for c in piv.columns])

    # the denominator is degenerate once most optimizers contribute no variance
    degenerate = n_zero >= n_opt / 2 or sd_mean < 0.05
    rows.append(dict(dataset=ds, spread_pp=spread,
                     base_sd_mean_pp=sd_mean, base_sd_median_pp=sd_median,
                     n_zero_variance=n_zero, n_optimizers=n_opt,
                     ratio=np.nan if sd_mean == 0 else spread / sd_mean,
                     friedman_p=fr.pvalue, significant=fr.pvalue < a.alpha,
                     ratio_interpretable=not degenerate))

t = pd.DataFrame(rows).sort_values("spread_pp").reset_index(drop=True)
t.to_csv(os.path.join(a.out, "table6.csv"), index=False)

print("=== Table 6: between-optimizer spread vs baseline run-to-run noise ===")
print("Baseline configuration: train/test partition and optimizer seed vary, "
      "CV folds fixed.\nBoth SD columns are over the nine per-optimizer standard "
      "deviations; the table uses the MEAN.\n")
print(f"{'dataset':14}{'spread':>8}{'SD mean':>9}{'SD med':>8}{'ratio':>8}"
      f"{'zeroSD':>8}{'Friedman p':>12}")
for _, r in t.iterrows():
    star = "*" if r.significant else " "
    flag = "" if r.ratio_interpretable else "   <- degenerate denominator"
    ratio = "   n/a" if np.isnan(r.ratio) else f"{r.ratio:8.2f}"
    print(f"{r.dataset + star:14}{r.spread_pp:8.2f}{r.base_sd_mean_pp:9.2f}"
          f"{r.base_sd_median_pp:8.2f}{ratio}{r.n_zero_variance:>4}/{r.n_optimizers}"
          f"{r.friedman_p:12.4f}{flag}")

n_below = int((t.ratio < 1).sum())
print(f"\n  * = within-dataset Friedman significant at alpha={a.alpha} "
      f"({int(t.significant.sum())} of {len(t)} datasets).")
print(f"  {n_below} of {len(t)} datasets have a spread smaller than the baseline "
      f"noise (ratio < 1).")
deg = t[~t.ratio_interpretable]
if len(deg):
    print("  Ratio NOT interpretable on: " + ", ".join(
        f"{r.dataset} ({r.n_zero_variance}/{r.n_optimizers} optimizers with zero "
        f"variance, median SD {r.base_sd_median_pp:.2f} pp)" for _, r in deg.iterrows()))
print(f"\nWrote table6.csv to {a.out}")