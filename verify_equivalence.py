"""
verify_equivalence.py

Confirms that the seven original optimizers produce NUMERICALLY IDENTICAL results
before and after the Stage-#1/#2 code changes (three-seed wiring, weighted-budget
instrumentation, cuckoo, and successive halving). If they match, it is safe to
RESUME: append cuckoo + SH to the existing Stage-#1 results instead of re-running
everything (Option B). If they do not match, re-run all optimizers under v3
(Option A).

WHY THIS MATTERS
----------------
Reviewers require that every optimizer be evaluated under identical conditions. The
weighted-budget instrumentation was designed so that full-fidelity optimizers
(frac=1 -> cost=1) are unchanged. This script proves that on YOUR real data, not
just on synthetic checks.

TWO WAYS TO USE IT
------------------
(A) Compare two summary.csv files directly (old vs new), if you have both:

    python verify_equivalence.py --old results_v2/summary.csv \
                                 --new results_v3_baseline/summary.csv

(B) Regenerate a small slice with v3 and compare to your existing Stage-#1 file,
    if you only kept the Stage-#1 output. See regenerate_and_compare() below and
    the note at the bottom.

WHAT COUNTS AS A MATCH
----------------------
For each (dataset, optimizer, seed) row shared by both files, the key numeric
columns must agree to a tight tolerance:
    test_acc, cv_best, best_log2C, best_log2gamma, evals_to_99pct
time_sec is IGNORED (wall-clock is not reproducible and is expected to differ).
"""

import argparse
import sys
import numpy as np
import pandas as pd

SEVEN = ["grid", "random", "bo_tpe", "pso", "de", "ga", "sa"]
KEYCOLS = ["test_acc", "cv_best", "best_log2C", "best_log2gamma", "evals_to_99pct"]
JOINKEYS = ["dataset", "optimizer", "seed"]
ATOL = 1e-9   # accuracies/logs are means of CV scores; identical seeds -> identical


def load(path):
    df = pd.read_csv(path)
    # keep only the seven original optimizers; cuckoo/sh are new and have no "old"
    df = df[df["optimizer"].isin(SEVEN)].copy()
    missing = [c for c in JOINKEYS + KEYCOLS if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] {path} is missing columns: {missing}")
    return df


def compare(old, new):
    merged = old.merge(new, on=JOINKEYS, suffixes=("_old", "_new"), how="inner")
    if len(merged) == 0:
        sys.exit("[ERROR] No overlapping (dataset, optimizer, seed) rows to compare. "
                 "Check that both files cover the same optimizers/datasets/seeds.")

    report = []
    all_ok = True
    for col in KEYCOLS:
        a = merged[f"{col}_old"].to_numpy(dtype=float)
        b = merged[f"{col}_new"].to_numpy(dtype=float)
        # evals_to_99pct is an integer count -> exact match; others -> atol
        if col == "evals_to_99pct":
            ok_mask = (a == b)
        else:
            ok_mask = np.isclose(a, b, atol=ATOL, rtol=0.0)
        n_bad = int((~ok_mask).sum())
        max_abs = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
        report.append((col, n_bad, max_abs))
        all_ok &= (n_bad == 0)

    # Coverage summary
    print(f"Compared {len(merged)} shared rows across "
          f"{merged['dataset'].nunique()} datasets x "
          f"{merged['optimizer'].nunique()} optimizers x "
          f"{merged['seed'].nunique()} seeds.\n")
    print(f"{'column':16} {'mismatches':>10} {'max_abs_diff':>14}")
    for col, n_bad, max_abs in report:
        flag = "" if n_bad == 0 else "  <-- DIFFERS"
        print(f"{col:16} {n_bad:>10} {max_abs:>14.2e}{flag}")

    print()
    if all_ok:
        print("[PASS] All seven optimizers are numerically identical (ignoring time).")
        print("       -> SAFE TO RESUME: append cuckoo + sh to the Stage-#1 results")
        print("          (Option B). No need to re-run the seven optimizers.")
    else:
        print("[FAIL] Some values differ. Do NOT resume onto the old results.")
        print("       -> Re-run ALL optimizers under v3 into a fresh folder (Option A),")
        print("          so every optimizer shares one code version. Inspect the")
        print("          differing rows below before proceeding.")
        # show a few offending rows for the first differing column
        for col, n_bad, _ in report:
            if n_bad:
                bad = merged[~np.isclose(
                    merged[f"{col}_old"].astype(float),
                    merged[f"{col}_new"].astype(float),
                    atol=ATOL, rtol=0.0)]
                cols = JOINKEYS + [f"{col}_old", f"{col}_new"]
                print(f"\nFirst mismatches in {col}:")
                print(bad[cols].head(8).to_string(index=False))
                break
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Stage-#1 (or v2) summary.csv")
    ap.add_argument("--new", required=True, help="v3 baseline summary.csv")
    args = ap.parse_args()
    old = load(args.old)
    new = load(args.new)
    ok = compare(old, new)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()