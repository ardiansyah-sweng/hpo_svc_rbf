"""
accuracy_summary.py -- the dataset-level accuracy results in one place:
Table 4 means, the Friedman test with its effect size, and (Reviewer 1
comment 5) per-comparison versus simultaneous confidence intervals for the 36
pairwise equivalence tests.

Aggregation: the 30 repetitions of each optimizer on each dataset are averaged
(MEAN), consistent with Eq. (7). This is the convention that produced the
chi2 = 20.30 reported in the previous submission; a median-based matrix gives a
different statistic.

    python accuracy_summary.py --summary results/baseline_fullfid/summary.csv --out results/accuracy_fullfid
"""
import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
ap.add_argument("--out", default=os.path.join("results", "accuracy"))
ap.add_argument("--margin", type=float, default=0.01)
ap.add_argument("--alpha", type=float, default=0.05)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

s = pd.read_csv(a.summary)
m = s.groupby(["dataset", "optimizer"])["test_acc"].mean().unstack()
n, k = m.shape

# ---- Table 4 ------------------------------------------------------------
means = m.mean().sort_values(ascending=False)
print("=== Table 4: mean test accuracy (macro-average over datasets) ===")
for o, v in means.items():
    print(f"  {o:8} {v:.4f}")
print(f"  range {means.min():.4f} - {means.max():.4f}  "
      f"(spread {100*(means.max()-means.min()):.2f} pp)")

# ---- Friedman + Kendall's W -----------------------------------------------
fr = stats.friedmanchisquare(*[m[c] for c in m.columns])
W = fr.statistic / (n * (k - 1))
print(f"\n=== Friedman across {n} datasets (mean aggregation) ===")
print(f"  chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.4f}, Kendall's W = {W:.3f}")

# ---- per-comparison vs simultaneous CIs ------------------------------------
pairs = list(combinations(m.columns, 2))
t_pc = stats.t.ppf(1 - a.alpha, n - 1)                  # 90% two-sided = TOST level
t_sim = stats.t.ppf(1 - a.alpha / len(pairs), n - 1)    # Bonferroni over 36 pairs
rows = []
for o1, o2 in pairs:
    d = (m[o1] - m[o2]).values
    mu, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    rows.append(dict(opt_A=o1, opt_B=o2, mean_diff=mu,
                     ci90_low=mu - t_pc * se, ci90_high=mu + t_pc * se,
                     sim_low=mu - t_sim * se, sim_high=mu + t_sim * se))
ci = pd.DataFrame(rows)
d = a.margin
ci["inside_90"] = (ci.ci90_low > -d) & (ci.ci90_high < d)
ci["inside_sim"] = (ci.sim_low > -d) & (ci.sim_high < d)
ci.to_csv(os.path.join(a.out, "pairwise_ci.csv"), index=False)

pp = lambda x: f"{100*x:+.2f}"
print(f"\n=== Pairwise CIs, margin +/-{100*d:.1f} pp ({len(pairs)} pairs) ===")
print(f"  per-comparison 90%   : {ci.inside_90.sum()}/{len(ci)} inside; "
      f"outermost {pp(ci.ci90_low.min())} to {pp(ci.ci90_high.max())} pp")
print(f"  simultaneous (Bonf.) : {ci.inside_sim.sum()}/{len(ci)} inside; "
      f"outermost {pp(ci.sim_low.min())} to {pp(ci.sim_high.max())} pp")
bad = ci[~ci.inside_sim]
if len(bad):
    print("  pairs outside the margin under simultaneous bounds:")
    for _, r in bad.iterrows():
        print(f"    {r.opt_A:8} vs {r.opt_B:8} diff {pp(r.mean_diff)}  "
              f"[{pp(r.sim_low)}, {pp(r.sim_high)}]")

m.to_csv(os.path.join(a.out, "dataset_by_optimizer_mean.csv"))
print(f"\nWrote pairwise_ci.csv and dataset_by_optimizer_mean.csv to {a.out}")