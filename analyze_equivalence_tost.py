"""
analyze_equivalence_tost.py  --  Paired TOST equivalence testing (Stage #6 / R2.3).

Replaces the weak "no significant difference" claim with a positive claim of practical
equivalence: between-optimizer accuracy differences lie within a pre-specified margin.

Reads results/baseline/summary.csv (9 optimizers x 10 datasets x 30 seeds).

USAGE
-----
    python analyze_equivalence_tost.py --summary results/baseline/summary.csv \
        --out results/equivalence --margin 0.01 --alpha 0.05
    # margin in accuracy units (0.01 = 1 percentage point)
    # add --margin_sd to ALSO run a margin equal to each pair's run-to-run SD

TWO ANALYSES
------------
(b) Across-dataset (HEADLINE): for each optimizer pair, take the 10 per-dataset mean
    differences and TOST them against +/-margin. Unit = dataset (matches the paper's
    pseudoreplication-safe design).
(a) Per-dataset drill-down: within each dataset, paired TOST over 30 seeds per pair;
    used to show that even the Friedman-significant datasets stay within the margin.

Multiple comparisons across the 36 pairs are Holm-corrected.
"""

import argparse
import itertools
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats


def paired_tost(diffs, margin, alpha):
    """Paired TOST on a vector of differences d_i.
    Returns (mean_diff, ci_low, ci_high, equivalent_bool, p_tost).
    Equivalent if the (1-2*alpha) CI lies within [-margin, +margin]."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return np.nan, np.nan, np.nan, False, np.nan
    dbar = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        # zero variance: equivalent iff |dbar| < margin
        eq = abs(dbar) < margin
        return dbar, dbar, dbar, eq, (0.0 if eq else 1.0)
    dfree = n - 1
    # two one-sided tests
    t_lower = (dbar - (-margin)) / se      # H0: mu <= -margin
    p_lower = stats.t.sf(t_lower, dfree)   # reject if mu > -margin
    t_upper = ((margin) - dbar) / se       # H0: mu >= +margin
    p_upper = stats.t.sf(t_upper, dfree)   # reject if mu < +margin
    p_tost = max(p_lower, p_upper)
    # (1-2alpha) CI
    tcrit = stats.t.ppf(1 - alpha, dfree)
    ci_low, ci_high = dbar - tcrit * se, dbar + tcrit * se
    equivalent = (ci_low > -margin) and (ci_high < margin)
    return dbar, ci_low, ci_high, equivalent, p_tost


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def across_dataset(df, optimizers, datasets, margin, alpha):
    """Analysis (b): per-optimizer-pair TOST over the 10 per-dataset mean diffs."""
    # per-dataset mean accuracy per optimizer
    mean_by = (df.groupby(["dataset", "optimizer"])["test_acc"].mean()
                 .unstack("optimizer"))
    rows = []
    pairs = list(itertools.combinations(optimizers, 2))
    raw_p = []
    for a, b in pairs:
        diffs = (mean_by[a] - mean_by[b]).values  # 10 values, one per dataset
        dbar, lo, hi, eq, p = paired_tost(diffs, margin, alpha)
        rows.append(dict(opt_A=a, opt_B=b, mean_diff=dbar, ci_low=lo, ci_high=hi,
                         equivalent=eq, p_tost=p, max_abs_diff=np.nanmax(np.abs(diffs))))
        raw_p.append(p)
    res = pd.DataFrame(rows)
    res["p_tost_holm"] = holm(res["p_tost"].values)
    res["equivalent_holm"] = res["equivalent"] & (res["p_tost_holm"] < alpha)
    return res


def per_dataset(df, optimizers, datasets, margin, alpha):
    """Analysis (a): within each dataset, paired TOST over 30 seeds per pair.
    Report the fraction of pairs equivalent per dataset."""
    rows = []
    for ds in datasets:
        d = df[df["dataset"] == ds]
        piv = d.pivot_table(index="seed", columns="optimizer", values="test_acc")
        pairs = list(itertools.combinations(optimizers, 2))
        eqs = []
        for a, b in pairs:
            if a in piv and b in piv:
                diffs = (piv[a] - piv[b]).values
                _, _, _, eq, _ = paired_tost(diffs, margin, alpha)
                eqs.append(eq)
        frac = np.mean(eqs) if eqs else np.nan
        rows.append(dict(dataset=ds, n_pairs=len(eqs),
                         frac_equivalent=frac))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="results/equivalence")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--margin_sd", action="store_true",
                    help="also run a margin equal to the mean run-to-run SD")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    for c in ["dataset", "optimizer", "seed", "test_acc"]:
        if c not in df.columns:
            sys.exit(f"[ERROR] summary missing column: {c}")
    optimizers = sorted(df["optimizer"].unique())
    datasets = sorted(df["dataset"].unique())
    os.makedirs(args.out, exist_ok=True)

    def run_at(margin, tag):
        res_b = across_dataset(df, optimizers, datasets, margin, args.alpha)
        res_a = per_dataset(df, optimizers, datasets, margin, args.alpha)
        res_b.to_csv(os.path.join(args.out, f"tost_across_dataset_{tag}.csv"), index=False)
        res_a.to_csv(os.path.join(args.out, f"tost_per_dataset_{tag}.csv"), index=False)
        n_pairs = len(res_b)
        n_eq = int(res_b["equivalent_holm"].sum())
        print(f"\n=== Margin = {margin:.4f} ({margin*100:.1f} pp), tag={tag} ===")
        print(f"Across-dataset TOST (unit = dataset, Holm-corrected):")
        print(f"  {n_eq}/{n_pairs} optimizer pairs are EQUIVALENT within +/-{margin*100:.1f} pp.")
        print(f"  largest mean |diff| across all pairs: "
              f"{res_b['max_abs_diff'].max()*100:.2f} pp")
        print(f"Per-dataset drill-down (fraction of pairs equivalent):")
        for _, r in res_a.iterrows():
            print(f"  {r['dataset']:14} {r['frac_equivalent']*100:5.1f}% of {int(r['n_pairs'])} pairs equivalent")
        return res_b, res_a

    print("#"*64)
    print("PRIMARY MARGIN")
    run_at(args.margin, f"{int(round(args.margin*1000))}mpp")

    print("\n" + "#"*64)
    print("SENSITIVITY: margin = 2 pp")
    run_at(0.02, "20mpp")

    if args.margin_sd:
        # data-driven margin = mean run-to-run SD across optimizers/datasets
        sd = df.groupby(["dataset", "optimizer"])["test_acc"].std().mean()
        print("\n" + "#"*64)
        print(f"SENSITIVITY: margin = 1 run-to-run SD ({sd*100:.2f} pp)")
        run_at(float(sd), "1sd")

    print("\nWrote equivalence tables to", args.out)


if __name__ == "__main__":
    main()