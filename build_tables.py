"""
build_tables.py -- regenerate Tables 3 to 9 from the result folders, so no number
in the manuscript is transcribed by hand.

Every value is read from a released CSV. The script prints each table as markdown
for reading and writes a .tsv beside it; in Word, paste the .tsv and use
Insert > Table > Convert Text to Table (tab-delimited) to get a real table.

Defaults point at the corrected, full-fidelity results:

    python build_tables.py

Override any folder if yours are named differently:

    python build_tables.py --baseline results/baseline_fullfid \
                           --variance results/variance_fullfid \
                           --equivalence results/equivalence_fullfid \
                           --gap results/efficiency_gap \
                           --table6 results/table6_fullfid \
                           --grid_order results/grid_order/grid_order_by_run.csv \
                           --out tables

The last section re-checks a few headline numbers against what the manuscript
says. If one of them disagrees, a folder argument is pointing at the wrong
results and the tables should not be pasted.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Paper-facing optimizer labels.
NAMES = {"grid": "GS", "random": "RS", "bo_tpe": "BO-TPE", "pso": "PSO",
         "de": "DE", "ga": "GA", "sa": "SA", "cuckoo": "CS", "sh": "SH"}


def label(o):
    return NAMES.get(o, o)


def need(path):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] not found: {path}\n"
                 f"        point the matching --flag at the folder that has it.")
    return path


def emit(df, number, title, note, out_dir, index_name=None, fmt=None):
    """Print as markdown and write as tab-separated.

    Formatting is applied per column before anything is printed. Letting pandas
    stringify a mixed-dtype row instead would upcast it, printing an integer 1 as
    1.0 and dropping the trailing zero from 0.8830.
    """
    def cell(v, f):
        if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
            return ""
        try:
            return format(v, f) if f else str(v)
        except (TypeError, ValueError):
            return str(v)

    fmt = fmt or {}
    disp = pd.DataFrame({c: [cell(v, fmt.get(c)) for v in df[c]] for c in df.columns},
                        index=df.index)
    print(f"\n### Table {number}. {title}\n")
    hdr = ([index_name] if index_name else []) + list(disp.columns)
    print("| " + " | ".join(str(h) for h in hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for idx, row in disp.iterrows():
        cells = ([str(idx)] if index_name else []) + list(row.values)
        print("| " + " | ".join(cells) + " |")
    if note:
        print(f"\n_{note}_")
    path = os.path.join(out_dir, f"table{number}.tsv")
    disp.to_csv(path, sep="\t", index=bool(index_name))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=os.path.join("results", "baseline_fullfid"))
    p.add_argument("--variance", default=os.path.join("results", "variance_fullfid"))
    p.add_argument("--equivalence", default=os.path.join("results", "equivalence_fullfid"))
    p.add_argument("--gap", default=os.path.join("results", "efficiency_gap"))
    p.add_argument("--table6", default=os.path.join("results", "table6_fullfid"))
    p.add_argument("--grid_order",
                   default=os.path.join("results", "grid_order", "grid_order_by_run.csv"))
    p.add_argument("--out", default="tables")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    written = []

    summary = pd.read_csv(need(os.path.join(a.baseline, "summary.csv")))
    meta = pd.read_csv(need(os.path.join(a.baseline, "dataset_meta.csv")))
    vc = pd.read_csv(need(os.path.join(a.variance, "variance_components.csv")))
    base = pd.read_csv(need(os.path.join(a.gap, "dataset_baseline.csv")))
    t6 = pd.read_csv(need(os.path.join(a.table6, "table6.csv")))

    # dataset_baseline.csv calls the no-skill floor "majority_rate" in files written
    # before the --floor option existed, and "floor" in later ones.
    floorcol = "floor" if "floor" in base.columns else "majority_rate"

    # Aggregate SD per dataset: median over optimizers of the total isolated SD.
    vc = vc.copy()
    vc["tot"] = vc[["var_split_S", "var_cv_F", "var_opt_O"]].sum(axis=1)
    agg_sd = vc.groupby("dataset")["tot"].median().pow(0.5)

    # ---------------- Table 3: datasets ----------------
    m = meta.set_index("dataset")
    b = base.set_index("dataset")
    t3 = pd.DataFrame({
        "OpenML id": m["openml_id"],
        "Samples": m["n_samples"],
        "Features": m["n_features"],
        "Classes": m["n_classes"],
        "Majority": b[floorcol].round(4),
        "Headroom (pp)": (b["headroom"] * 100).round(2),
    }).loc[sorted(m.index)]
    written.append(emit(
        t3, 3, "Benchmark datasets, with the no-skill floor and the headroom",
        "Feature counts are given before one-hot encoding; car and ilpd expand to 21 "
        "and 11 features respectively when categorical attributes are encoded. "
        "Majority is the majority-class rate b_d, the score of a constant predictor; "
        "headroom is r_d - b_d, the accuracy a tuner can add at all.",
        a.out, index_name="Dataset",
        fmt={"OpenML id": "d", "Samples": "d", "Features": "d", "Classes": "d",
             "Majority": ".4f", "Headroom (pp)": ".2f"}))

    # ---------------- Table 4: mean accuracy ----------------
    dm = summary.groupby(["dataset", "optimizer"])["test_acc"].mean().unstack()
    means = dm.mean().sort_values(ascending=False)
    t4 = pd.DataFrame({"Mean accuracy": means.round(4).values},
                      index=[label(o) for o in means.index])
    spread_pp = (means.max() - means.min()) * 100
    written.append(emit(
        t4, 4, "Mean test accuracy across datasets",
        f"Macro-average over the ten datasets; the nine optimizers span "
        f"{spread_pp:.2f} percentage points.",
        a.out, index_name="Optimizer", fmt={"Mean accuracy": ".4f"}))

    # ---------------- Table 5: equivalence rates ----------------
    sd_margin = summary.groupby(["dataset", "optimizer"])["test_acc"].std().mean()
    cols, tags = {}, [("10mpp", "1.00"), ("1sd", f"{sd_margin*100:.2f}"), ("20mpp", "2.00")]
    for tag, lab in tags:
        f = os.path.join(a.equivalence, f"tost_per_dataset_{tag}.csv")
        if not os.path.exists(f):
            print(f"[warn] missing {f}; column skipped "
                  f"(re-run analyze_equivalence_tost.py with --margin_sd)")
            continue
        d = pd.read_csv(f).set_index("dataset")
        cols[f"d={lab}"] = (d["frac_equivalent"] * 100).round(1)
    t5 = pd.DataFrame(cols)
    t5["Agg. SD"] = agg_sd.round(4)
    t5 = t5.loc[t5.iloc[:, 0].sort_values(ascending=False).index]
    written.append(emit(
        t5, 5, "Within-dataset pairwise equivalence rates",
        "A separate within-dataset analysis: the thirty repetitions of each dataset are "
        "the paired observations, not the ten dataset means, so these rates are not "
        "interchangeable with the cross-dataset conclusion. Agg. SD is the median "
        "aggregate run-to-run standard deviation in accuracy units.",
        a.out, index_name="Dataset",
        fmt={**{c: ".1f" for c in t5.columns if c.startswith("d=")},
             "Agg. SD": ".4f"}))

    # ---------------- Table 6: spread vs noise ----------------
    t6s = t6.sort_values("spread_pp").set_index("dataset")
    t6out = pd.DataFrame({
        "Spread": t6s["spread_pp"].round(2),
        "Base. SD": t6s["base_sd_mean_pp"].round(2),
        "Ratio": t6s["ratio"].round(2),
        "Friedman p": t6s["friedman_p"].round(4),
    })
    t6out.index = [f"{d}*" if s else d for d, s in zip(t6s.index, t6s["significant"])]
    n_zero = int(t6s.loc[t6s["ratio_interpretable"] == False, "n_zero_variance"].max()) \
        if (~t6s["ratio_interpretable"]).any() else 0
    written.append(emit(
        t6out, 6, "Between-optimizer accuracy spread and run-to-run standard deviation",
        f"Baseline configuration: the train/test partition and the optimizer seed vary "
        f"while the cross-validation folds are held fixed. Base. SD is the MEAN of the "
        f"nine per-optimizer standard deviations; its median agrees within 0.1 points on "
        f"every dataset except banknote, where {n_zero} of 9 optimizers have exactly zero "
        f"variance and the ratio is not interpretable. Datasets marked * yield a "
        f"significant within-dataset Friedman result.",
        a.out, index_name="Dataset",
        fmt={"Spread": ".2f", "Base. SD": ".2f", "Ratio": ".2f", "Friedman p": ".4f"}))

    # ---------------- Table 7: variance attribution ----------------
    sh = (vc.groupby("dataset")[["share_split_S", "share_cv_F", "share_opt_O"]]
          .median() * 100).sort_values("share_split_S", ascending=False)
    t7 = pd.DataFrame({"Split": sh["share_split_S"].round(1),
                       "CV": sh["share_cv_F"].round(1),
                       "Opt.": sh["share_opt_O"].round(1),
                       "Agg. SD": agg_sd.reindex(sh.index).round(4)})
    overall = vc[["share_split_S", "share_cv_F", "share_opt_O"]].median() * 100
    t7.loc["Median (all)"] = [round(overall.iloc[0], 1), round(overall.iloc[1], 1),
                              round(overall.iloc[2], 1), ""]
    rs = sh.sum(axis=1)
    written.append(emit(
        t7, 7, "Median relative attribution of isolated variance sources per dataset",
        f"Shares and standard deviations are medians across the nine optimizers, taken "
        f"independently for each component, so rows need not sum to exactly 100%: within "
        f"any optimizer the three shares sum to 100% by construction, but the optimizer "
        f"attaining the median differs by column. Observed row sums range from "
        f"{rs.min():.1f}% to {rs.max():.1f}%. Banknote is degenerate: almost every "
        f"optimizer returns identical accuracy on every repetition.",
        a.out, index_name="Dataset",
        fmt={"Split": ".1f", "CV": ".1f", "Opt.": ".1f", "Agg. SD": ".4f"}))

    # ---------------- Table 8: efficiency ----------------
    by_run = pd.read_csv(need(os.path.join(a.gap, "gap_target_by_run.csv")))
    eff, ranks, attain = {}, {}, {}
    for t in (90, 95, 98):
        f = os.path.join(a.gap, f"efficiency_datasetlevel_{t}.csv")
        if not os.path.exists(f):
            sys.exit(f"[ERROR] not found: {f}\n"
                     f"        run analyze_efficiency_datasetlevel.py at target 0.{t}.")
        mx = pd.read_csv(f, index_col=0)
        eff[t] = mx.median()
        ranks[t] = mx.rank(axis=1).mean()
        sub = by_run[np.isclose(by_run["target"], t / 100)]
        attain[t] = sub.groupby("optimizer")["reached"].mean() * 100
    t8 = pd.DataFrame({
        "90%": eff[90].round(1), "95%": eff[95].round(1), "98%": eff[98].round(1),
        "Rank 90%": ranks[90].round(2), "Rank 95%": ranks[95].round(2),
        "Rank 98%": ranks[98].round(2),
        "Att. 90%": attain[90].round(0).astype(int),
        "Att. 95%": attain[95].round(0).astype(int),
        "Att. 98%": attain[98].round(0).astype(int),
    }).sort_values("Rank 95%")
    t8.index = [label(o) for o in t8.index]
    written.append(emit(
        t8, 8, "Dataset-level search efficiency under randomized grid traversal",
        "Targets are fractions of the attainable gap, tau_d(alpha) = b_d + alpha(r_d - b_d), "
        "in full-fidelity-equivalent cost units; the unit of analysis is the dataset "
        "(n=10). Att. is the proportion of runs reaching the target within budget B. "
        "If the table is too wide, keep Rank 95% and drop the other two rank columns.",
        a.out, index_name="Optimizer",
        fmt={**{f"{t}%": ".1f" for t in (90, 95, 98)},
             **{f"Rank {t}%": ".2f" for t in (90, 95, 98)},
             **{f"Att. {t}%": "d" for t in (90, 95, 98)}}))

    # ---------------- Table 9: traversal policies ----------------
    go = pd.read_csv(need(a.grid_order))
    order = [("row_major", "Row-major (conventional)"), ("col_major", "Column-major"),
             ("random", "Random permutation"), ("spiral", "Centre-outwards spiral")]
    rows = []
    for key, lab in order:
        d = go[go["policy"] == key]
        if d.empty:
            continue
        rows.append(dict(Policy=lab,
                         Med=round(d["evals_to_99pct"].median(), 1),
                         Mean=round(d["evals_to_99pct"].mean(), 2),
                         Min=int(d["evals_to_99pct"].min()),
                         Max=int(d["evals_to_99pct"].max()),
                         Acc=round(d["test_acc"].mean(), 4)))
    t9 = pd.DataFrame(rows).set_index("Policy")
    acc_spread = t9["Acc"].max() - t9["Acc"].min()
    written.append(emit(
        t9, 9, "Grid search efficiency under four traversal policies",
        f"Pooled over ten datasets and thirty repetitions, against a within-run target of "
        f"99% of each run's own optimal objective; admissible here because all four "
        f"policies share an identical lattice and therefore an identical attainable "
        f"optimum. Mean test accuracy differs by at most {acc_spread:.4f}.",
        a.out, index_name="Traversal policy",
        fmt={"Med": ".1f", "Mean": ".2f", "Min": "d", "Max": "d", "Acc": ".4f"}))

    # ---------------- consistency check ----------------
    print("\n\n### Consistency check against the numbers used in the text\n")
    checks = [
        ("Table 4 spread (pp)", f"{spread_pp:.2f}", "0.11"),
        ("Table 7 median Split (%)", f"{overall.iloc[0]:.1f}", "80.2"),
        ("Table 7 median CV (%)", f"{overall.iloc[1]:.1f}", "11.8"),
        ("Table 7 median Opt. (%)", f"{overall.iloc[2]:.1f}", "6.9"),
        ("Table 6 ratios below 1", f"{int((t6s['ratio'] < 1).sum())}/{len(t6s)}", "10/10"),
        ("SD-based margin (pp)", f"{sd_margin*100:.2f}", "1.55"),
        ("Table 8 GS rank at 95%", f"{ranks[95].get('grid', float('nan')):.2f}", "4.75"),
        ("Table 9 accuracy spread", f"{acc_spread:.4f}", "0.0003"),
    ]
    if "d=1.00" in t5.columns and "ilpd" in t5.index:
        checks.append(("Table 5 ilpd at 1.0 pp (%)", f"{t5.loc['ilpd', 'd=1.00']:.1f}", "36.1"))
    bad = 0
    for name, got, expect in checks:
        ok = str(got) == str(expect)
        bad += not ok
        print(f"  [{'ok ' if ok else 'MISMATCH'}] {name:30} got {got:>8}   expected {expect}")
    if bad:
        print(f"\n  {bad} mismatch(es). Either a --flag points at the wrong folder, or a "
              f"number in the text is stale. Resolve before pasting these tables.")
    else:
        print("\n  All checks pass; the tables match the numbers used in the text.")

    print(f"\nWrote {len(written)} .tsv files to {a.out}/")


if __name__ == "__main__":
    main()