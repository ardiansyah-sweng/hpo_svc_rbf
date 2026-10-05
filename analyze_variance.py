"""
analyze_variance.py  --  Variance decomposition analysis (Reviewer R1.11 / R2.2)

Reads the per-run summaries produced by hpo_benchmark_v3.py under the three
isolation modes and produces:
  - results/variance/seed_map.csv         (unique instances of each random source)
  - results/variance/variance_components.csv  (Var_S, Var_F, Var_O + shares)
  - console: median variance share of each source across (optimizer, dataset)

USAGE
-----
Run the benchmark three times, changing only CONFIG["decomp_mode"], and write each
mode's summary to a distinct file, e.g.:

    # in hpo_benchmark_v3.py CONFIG:  decomp_mode="isolate_F", results_dir="results/decomp_F"
    python hpo_benchmark_v3.py
    # then isolate_S -> results/decomp_S, isolate_O -> results/decomp_O

Then:

    python analyze_variance.py \
        --isolate_F results/decomp_F/summary.csv \
        --isolate_S results/decomp_S/summary.csv \
        --isolate_O results/decomp_O/summary.csv \
        --out results/variance

Each summary.csv must contain columns:
    dataset, optimizer, rep, seed_split, seed_cv, seed_opt, test_acc
(hpo_benchmark_v3.py writes all of these.)
"""

import argparse
import os
import numpy as np
import pandas as pd


def load(path, component):
    df = pd.read_csv(path)
    df["component"] = component
    return df


def seed_map(df_all):
    """How many unique instances of each random source appear across all runs.
    This is the table R1.11 asks for ('explicitly distinguishable')."""
    rows = []
    for comp, g in df_all.groupby("component"):
        rows.append({
            "isolation_mode": comp,
            "unique_train_test_partitions": g["seed_split"].nunique(),
            "unique_cv_partitions": g["seed_cv"].nunique(),
            "unique_optimizer_inits": g["seed_opt"].nunique(),
            "n_runs": len(g),
        })
    return pd.DataFrame(rows)


def decompose(df_all):
    """Per (optimizer, dataset): variance of test_acc within each isolation set,
    and each variance as a share of their sum."""
    recs = []
    for (opt, ds), g in df_all.groupby(["optimizer", "dataset"]):
        varF = g.loc[g.component == "isolate_F", "test_acc"].var(ddof=1)
        varS = g.loc[g.component == "isolate_S", "test_acc"].var(ddof=1)
        varO = g.loc[g.component == "isolate_O", "test_acc"].var(ddof=1)
        # grid has no optimizer RNG -> Var(O) may be exactly 0 or NaN; treat as 0
        varO = 0.0 if (np.isnan(varO) and opt == "grid") else varO
        total = np.nansum([varS, varF, varO])
        recs.append({
            "optimizer": opt, "dataset": ds,
            "var_split_S": varS, "var_cv_F": varF, "var_opt_O": varO,
            "share_split_S": varS / total if total > 0 else np.nan,
            "share_cv_F": varF / total if total > 0 else np.nan,
            "share_opt_O": varO / total if total > 0 else np.nan,
        })
    return pd.DataFrame(recs)


def additivity_check(dec, baseline_path):
    """Test how far the isolated variances depart from additivity.

    The one-factor-at-a-time design estimates the variance attributable to each
    random source with the other two held fixed. That is an attribution, not a full
    variance decomposition: interaction terms are not estimated, so the isolated
    contributions need not sum to the variance observed when several sources vary
    together.

    The baseline configuration provides a direct test, because it varies the
    train/test split and the optimizer seed together while holding the CV folds
    fixed. If the two contributions were additive we would expect

        Var(baseline)  ==  Var_split + Var_opt

    The ratio of the two is reported below. A ratio near 1 means the isolated
    contributions add up; a ratio below 1 means their sum overstates the variance
    actually observed when both sources vary, i.e. a negative interaction.
    """
    base = pd.read_csv(baseline_path)
    obs = (base.groupby(["dataset", "optimizer"])["test_acc"]
               .var(ddof=1).rename("var_baseline").reset_index())
    m = dec.merge(obs, on=["dataset", "optimizer"], how="inner")
    m["var_predicted"] = m["var_split_S"] + m["var_opt_O"]
    # drop degenerate pairs where both are numerically zero
    m = m[(m["var_baseline"] > 1e-12) & (m["var_predicted"] > 1e-12)].copy()
    m["ratio"] = m["var_baseline"] / m["var_predicted"]
    return m[["dataset", "optimizer", "var_baseline", "var_predicted", "ratio"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolate_F", required=True)
    ap.add_argument("--isolate_S", required=True)
    ap.add_argument("--isolate_O", required=True)
    ap.add_argument("--out", default="results/variance_fullfid")
    ap.add_argument("--baseline", default=None,
                    help="baseline summary.csv; enables the additivity check")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df_all = pd.concat([
        load(args.isolate_F, "isolate_F"),
        load(args.isolate_S, "isolate_S"),
        load(args.isolate_O, "isolate_O"),
    ], ignore_index=True)

    sm = seed_map(df_all)
    sm.to_csv(os.path.join(args.out, "seed_map.csv"), index=False)

    dec = decompose(df_all)
    dec.to_csv(os.path.join(args.out, "variance_components.csv"), index=False)

    print("=== Seed map ===")
    print(sm.to_string(index=False))
    print("\n=== Median variance share across (optimizer, dataset) ===")
    med = dec[["share_split_S", "share_cv_F", "share_opt_O"]].median()
    for k, v in med.items():
        print(f"  {k:16s}: {v:6.1%}")
    print("\nWrote:", os.path.join(args.out, "seed_map.csv"))
    print("Wrote:", os.path.join(args.out, "variance_components.csv"))

    if args.baseline:
        add = additivity_check(dec, args.baseline)
        add.to_csv(os.path.join(args.out, "additivity_check.csv"), index=False)
        r = add["ratio"]
        print("\n=== Additivity check: Var(baseline) / (Var_split + Var_opt) ===")
        print("The baseline varies split and optimizer together with folds fixed,")
        print("so a ratio near 1 indicates the isolated contributions are additive.")
        print(f"  pairs with non-zero variance : {len(add)}")
        print(f"  median ratio                 : {r.median():.2f}")
        print(f"  interquartile range          : {r.quantile(.25):.2f} - {r.quantile(.75):.2f}")
        print(f"  within 20% of additive       : {r.between(0.8, 1.2).mean():.0%}")
        print(f"  within 50% of additive       : {r.between(0.5, 1.5).mean():.0%}")
        print("\n  median ratio per dataset:")
        for ds, g in add.groupby("dataset"):
            print(f"    {ds:14} {g['ratio'].median():5.2f}  (n={len(g)})")
        print("\nWrote:", os.path.join(args.out, "additivity_check.csv"))

    # sanity check
    grid = dec[dec.optimizer == "grid"]
    if len(grid) and (grid["var_opt_O"].fillna(0) > 1e-9).any():
        print("\n[WARN] grid Var(O) is not ~0 -- check seed_opt wiring for grid.")
    else:
        print("\n[OK] grid Var(O) ~ 0 as expected (sanity check passed).")


if __name__ == "__main__":
    main()