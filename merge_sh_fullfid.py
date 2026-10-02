"""
merge_sh_fullfid.py -- build a results folder in which Successive Halving's rows
come from the full-fidelity rerun while the other eight optimizers are copied
unchanged from an existing folder.

It never writes into --base or --sh; output always goes to --out.

Usage (run each line that applies)
----------------------------------
  # accuracy + TOST + Friedman (Tables 4-6):
  python merge_sh_fullfid.py --base results/baseline --sh results/sh_fullfid/baseline --out results/baseline_fullfid

  # efficiency with randomized grid traversal (Table 8):
  python merge_sh_fullfid.py --base results/baseline_gridrandom --sh results/sh_fullfid/baseline --out results/baseline_gridrandom_fullfid

  # variance attribution (Table 7), one line per isolation mode:
  python merge_sh_fullfid.py --base results/isolate_S --sh results/sh_fullfid/isolate_S --out results/isolate_S_fullfid
  python merge_sh_fullfid.py --base results/isolate_F --sh results/sh_fullfid/isolate_F --out results/isolate_F_fullfid
  python merge_sh_fullfid.py --base results/isolate_O --sh results/sh_fullfid/isolate_O --out results/isolate_O_fullfid

In the merged files, the column `best_so_far` / `cv_best` is the PRIMARY
(full-fidelity) value for every optimizer, so all existing analysis scripts run
on the merged folders unchanged. The columns `best_so_far_anyfid` /
`cv_best_anyfid` keep the original any-fidelity values for the sensitivity
analysis. For the eight full-fidelity optimizers both columns are identical.
"""
import argparse
import os
import shutil
import sys

import pandas as pd

SH = "sh"


def load(folder, name):
    path = os.path.join(folder, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="existing folder with all optimizers")
    ap.add_argument("--sh", required=True, help="folder from run_sh_fullfid.py")
    ap.add_argument("--out", required=True, help="new folder to create")
    args = ap.parse_args()

    for p in (args.base, args.sh):
        if not os.path.isdir(p):
            sys.exit(f"folder not found: {p}")
    if os.path.abspath(args.out) in (os.path.abspath(args.base), os.path.abspath(args.sh)):
        sys.exit("--out must differ from --base and --sh")
    os.makedirs(args.out, exist_ok=True)

    wrote = []

    # ---------------- summary.csv ----------------
    b, n = load(args.base, "summary.csv"), load(args.sh, "summary.csv")
    if b is not None and n is not None:
        n = n[n.optimizer == SH]
        old_sh = b[b.optimizer == SH]
        keep = b[b.optimizer != SH].copy()
        # the eight full-fidelity optimizers: both views coincide
        keep["cv_best_anyfid"] = keep["cv_best"]
        keep["n_fullfid_evals"] = keep["n_evals_used"]
        keep["full_fidelity_only"] = True
        # pairing check: SH must cover the same (dataset, rep, split seed) cells
        cols = [c for c in ("dataset", "rep", "seed_split", "seed_cv") if c in b.columns]
        a_keys = set(map(tuple, old_sh[cols].values))
        c_keys = set(map(tuple, n[cols].values))
        if a_keys != c_keys:
            sys.exit(f"SH cells differ between base and rerun on {cols}; "
                     f"{len(a_keys ^ c_keys)} mismatched. Check decomp_mode / n_seeds.")
        merged = pd.concat([keep, n], ignore_index=True)   # aligns columns by name
        merged.to_csv(os.path.join(args.out, "summary.csv"), index=False)
        wrote.append("summary.csv")

        print("\n=== SH before -> after (full-fidelity selection) ===")
        print(f"  runs               : {len(old_sh)} -> {len(n)}")
        print(f"  mean test_acc      : {old_sh.test_acc.mean():.4f} -> {n.test_acc.mean():.4f}")
        print(f"  median cv_best     : {old_sh.cv_best.median():.4f} -> {n.cv_best.median():.4f}")
        gap_old = (old_sh.cv_best - old_sh.test_acc).median() * 100
        gap_new = (n.cv_best - n.test_acc).median() * 100
        print(f"  median cv-test gap : {gap_old:.2f} pp -> {gap_new:.2f} pp")
        others = keep.groupby("optimizer").apply(
            lambda g: (g.cv_best - g.test_acc).median() * 100)
        print(f"  (other optimizers' cv-test gap: {others.min():.2f}-{others.max():.2f} pp)")

    # ---------------- anytime_curves.csv ----------------
    b, n = load(args.base, "anytime_curves.csv"), load(args.sh, "anytime_curves.csv")
    if b is not None and n is not None:
        n = n[n.optimizer == SH]
        keep = b[b.optimizer != SH].copy()
        keep["best_so_far_anyfid"] = keep["best_so_far"]
        merged = pd.concat([keep, n], ignore_index=True)   # aligns columns by name
        merged.to_csv(os.path.join(args.out, "anytime_curves.csv"), index=False)
        wrote.append("anytime_curves.csv")
        nan_share = n["best_so_far"].isna().mean() * 100
        print(f"\n  SH curve slots before its first full-fidelity evaluation: {nan_share:.1f}%")

    # ---------------- dataset_meta.csv ----------------
    for src in (args.base, args.sh):
        p = os.path.join(src, "dataset_meta.csv")
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(args.out, "dataset_meta.csv"))
            wrote.append("dataset_meta.csv")
            break

    if not wrote:
        sys.exit("nothing merged: neither summary.csv nor anytime_curves.csv found in both folders")
    print(f"\nWrote to {args.out}: {', '.join(wrote)}")


if __name__ == "__main__":
    main()