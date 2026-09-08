"""
analyze_two_regime.py  --  Conventional vs noise-aware evaluation (Stage #2 / R2.1).

Extracts BOTH evaluation regimes from the existing baseline results and reports
whether apparent optimizer superiority survives, disappears, or reverses:

  Regime A (conventional): single run (one seed), rank optimizers by test accuracy.
  Regime B (noise-aware) : all seeds, rank by mean; Friedman test for any real diff.

No new experiments required. Reads results/baseline/summary.csv.

USAGE
-----
    python analyze_two_regime.py --summary results/baseline/summary.csv \
                                 --out results/two_regime
    # optional: --headline_seed 0   (which single seed is the Regime-A headline)
    #           --focus cuckoo,sh    (optimizers to spotlight in the verdict)

OUTPUTS
-------
  two_regime_winner_table.csv : per dataset, Regime-A winner (headline seed),
                                that winner's rank range across all seeds, and
                                whether Regime B finds any significant winner.
  rank_churn.csv              : per optimizer, distribution of its single-run rank
                                across seeds (how much a single-run rank is noise).
  console                     : winner-instability summary + Friedman p per dataset
                                + focus-optimizer verdict.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


def regime_a_winner(df_ds, seed):
    """Single-run winner on a given seed: optimizer with max test_acc."""
    sub = df_ds[df_ds["seed"] == seed]
    if sub.empty:
        return None
    r = sub.sort_values("test_acc", ascending=False)
    return r.iloc[0]["optimizer"]


def per_seed_ranks(df_ds):
    """For each seed, rank optimizers by test_acc (1 = best). Returns a
    seed x optimizer rank matrix (DataFrame)."""
    piv = df_ds.pivot_table(index="seed", columns="optimizer", values="test_acc")
    # rank within each seed row; highest acc -> rank 1
    ranks = piv.rank(axis=1, ascending=False, method="average")
    return piv, ranks


def tie_statistics(df):
    """How often is the single-run winner not uniquely defined?

    A "winner" is only meaningful if one optimizer strictly beats the rest. Where two
    or more optimizers attain exactly the same best accuracy the winner depends purely
    on tie-breaking, which is an implementation detail rather than a result. This
    function quantifies how often that happens.
    """
    rows = []
    n_tied_blocks = 0
    n_blocks = 0
    for ds, d in df.groupby("dataset"):
        tied, sizes = 0, []
        for _, g in d.groupby("seed"):
            k = int((g["test_acc"] == g["test_acc"].max()).sum())
            sizes.append(k)
            n_blocks += 1
            if k > 1:
                tied += 1
                n_tied_blocks += 1
        rows.append(dict(dataset=ds, n_reps=len(sizes), tied_reps=tied,
                         tied_pct=100.0 * tied / len(sizes),
                         mean_tied_at_top=float(np.mean(sizes))))
    tab = pd.DataFrame(rows).sort_values("tied_reps", ascending=False)
    overall = 100.0 * n_tied_blocks / n_blocks
    return tab, n_tied_blocks, n_blocks, overall


def analyze(df, headline_seed, focus):
    datasets = sorted(df["dataset"].unique())
    optimizers = sorted(df["optimizer"].unique())
    seeds = sorted(df["seed"].unique())

    winner_rows = []
    churn = {o: [] for o in optimizers}
    friedman_p = {}

    for ds in datasets:
        d = df[df["dataset"] == ds]
        piv, ranks = per_seed_ranks(d)

        # accumulate rank churn per optimizer
        for o in optimizers:
            if o in ranks.columns:
                churn[o].extend(ranks[o].dropna().tolist())

        # Regime A headline winner
        a_winner = regime_a_winner(d, headline_seed)

        # that winner's rank range across ALL seeds (instability of a single-run pick)
        if a_winner in ranks.columns:
            rr = ranks[a_winner]
            rank_lo, rank_hi = int(rr.min()), int(rr.max())
        else:
            rank_lo = rank_hi = None

        # Regime B: mean-accuracy nominal best + Friedman across seeds
        meanacc = piv.mean(axis=0)
        b_nominal = meanacc.idxmax()
        # Friedman needs each optimizer's per-seed vector (columns of piv)
        cols = [piv[o].values for o in piv.columns if piv[o].notna().all()]
        if len(cols) >= 3 and all(len(c) == len(cols[0]) for c in cols):
            try:
                _, p = friedmanchisquare(*cols)
            except Exception:
                p = np.nan
        else:
            p = np.nan
        friedman_p[ds] = p

        winner_rows.append({
            "dataset": ds,
            "regimeA_winner_seed{}".format(headline_seed): a_winner,
            "regimeA_winner_rank_min": rank_lo,
            "regimeA_winner_rank_max": rank_hi,
            "regimeB_nominal_best": b_nominal,
            "regimeB_friedman_p": round(p, 4) if not np.isnan(p) else np.nan,
            "regimeB_significant_winner": (not np.isnan(p)) and (p < 0.05),
        })

    winner_df = pd.DataFrame(winner_rows)

    # winner instability: for each dataset, how many DISTINCT single-run winners
    # appear as we vary the seed
    instab = []
    for ds in datasets:
        d = df[df["dataset"] == ds]
        winners = {regime_a_winner(d, s) for s in seeds}
        winners.discard(None)
        instab.append({"dataset": ds, "distinct_single_run_winners": len(winners),
                       "winners": ",".join(sorted(winners))})
    instab_df = pd.DataFrame(instab)

    # rank churn table
    churn_rows = []
    for o in optimizers:
        vals = np.array(churn[o], dtype=float)
        if len(vals):
            churn_rows.append({
                "optimizer": o,
                "rank_median": np.median(vals),
                "rank_min": vals.min(),
                "rank_max": vals.max(),
                "rank_iqr_lo": np.percentile(vals, 25),
                "rank_iqr_hi": np.percentile(vals, 75),
            })
    churn_df = pd.DataFrame(churn_rows).sort_values("rank_median")

    return winner_df, instab_df, churn_df, friedman_p, focus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="results/two_regime")
    ap.add_argument("--headline_seed", type=int, default=0)
    ap.add_argument("--focus", default="cuckoo,sh")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    for c in ["dataset", "optimizer", "seed", "test_acc"]:
        if c not in df.columns:
            sys.exit(f"[ERROR] summary missing column: {c}")

    focus = [f.strip() for f in args.focus.split(",") if f.strip()]
    os.makedirs(args.out, exist_ok=True)

    winner_df, instab_df, churn_df, fp, focus = analyze(df, args.headline_seed, focus)

    tie_tab, n_tied, n_blocks, tie_pct = tie_statistics(df)
    tie_tab.to_csv(os.path.join(args.out, "tie_statistics.csv"), index=False)

    winner_df.to_csv(os.path.join(args.out, "two_regime_winner_table.csv"), index=False)
    churn_df.to_csv(os.path.join(args.out, "rank_churn.csv"), index=False)
    instab_df.to_csv(os.path.join(args.out, "winner_instability.csv"), index=False)

    # ---- console report ----
    print("=== Ties at the top: is the single-run winner uniquely defined? ===")
    print(tie_tab.round(1).to_string(index=False))
    print(f"\n{n_tied} of {n_blocks} dataset-repetition blocks ({tie_pct:.0f}%) contain a tie at")
    print("the top, so in those blocks the single-run winner is decided by tie-breaking,")
    print("not by performance. Counts of distinct winners are therefore reported below")
    print("only as a lower bound on ranking instability.\n")

    print("=== Regime-A winner instability (distinct single-run winners per dataset) ===")
    print(instab_df.to_string(index=False))

    n_unstable = (instab_df["distinct_single_run_winners"] > 1).sum()
    print(f"\n{n_unstable}/{len(instab_df)} datasets have MORE THAN ONE distinct "
          f"single-run winner across seeds -> single-run 'best' is seed-dependent.")
    print("NOTE: the exact count of distinct winners per dataset depends on how ties are")
    print("broken and should not be quoted as a finding. The robust statements are the")
    print("tie rate above, the number of datasets with more than one winner, and the")
    print("rank range per optimizer below.")

    print("\n=== Regime B: any statistically significant winner? (Friedman) ===")
    sig = winner_df[winner_df["regimeB_significant_winner"]]
    if len(sig) == 0:
        print("None. No dataset has a significantly better optimizer in Regime B.")
    else:
        print(sig[["dataset", "regimeB_friedman_p"]].to_string(index=False))

    print("\n=== Rank churn (single-run rank distribution per optimizer) ===")
    print(churn_df.round(2).to_string(index=False))

    print("\n=== Verdict for focus optimizers ===")
    for o in focus:
        if o not in churn_df["optimizer"].values:
            print(f"  {o}: not found in data.")
            continue
        row = churn_df[churn_df["optimizer"] == o].iloc[0]
        # how often is o the single-run winner somewhere?
        won = instab_df["winners"].str.contains(rf"\b{o}\b").sum()
        print(f"  {o}: single-run rank ranges {int(row['rank_min'])}-{int(row['rank_max'])} "
              f"(median {row['rank_median']:.1f}); is single-run winner on {won} dataset(s). "
              f"-> its apparent standing is seed-dependent; Regime B shows no significant edge "
              f"unless flagged above.")

    print("\nWrote:", os.path.join(args.out, "two_regime_winner_table.csv"))
    print("Wrote:", os.path.join(args.out, "rank_churn.csv"))
    print("Wrote:", os.path.join(args.out, "winner_instability.csv"))


if __name__ == "__main__":
    main()