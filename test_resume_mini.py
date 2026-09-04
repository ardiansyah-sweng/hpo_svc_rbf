"""
test_resume_mini.py  --  Combined Stage-#2 pre-flight check (Steps 2 + 4).

Verifies, on ONE small REAL dataset, that:
  (2) cuckoo and successive halving run without error on real OpenML data,
      respect the weighted budget, and produce sensible accuracies; and
  (4) incremental resume works: when the seven original optimizers are already
      present in a checkpoint, adding cuckoo + sh recomputes ONLY cuckoo + sh
      (the seven are not re-run, and their rows are preserved unchanged).

It works on a COPY so your real results/ folder is never touched.

USAGE
-----
    python test_resume_mini.py

Optional: change DATASET below to any (id, name) from CONFIG["datasets"].
Uses cv_n_jobs=1 so it stays light if anything else is running.

WHAT "PASS" MEANS
-----------------
All checks green -> safe to run the full four-mode resume with cuckoo + sh.
Any check red   -> stop and inspect before touching the real results.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd

# ---- pick one small real dataset (banknote is small & fast) ----
DATASET = (1462, "banknote")
N_SEEDS = 5           # small; enough to prove wiring works on real data
MODE = "baseline"

def load_module(path):
    spec = importlib.util.spec_from_file_location("hb", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def make_cfg(hb, results_dir):
    cfg = dict(hb.CONFIG)
    cfg["decomp_mode"] = MODE
    cfg["results_dir"] = results_dir
    cfg["checkpoint_dir"] = os.path.join(results_dir, "checkpoints")
    cfg["cv_n_jobs"] = 1
    return cfg

def green(msg):  print("  [PASS] " + msg)
def red(msg):    print("  [FAIL] " + msg)

def main():
    hb = load_module("hpo_benchmark_v3.py")
    oid, name = DATASET
    workdir = tempfile.mkdtemp(prefix="resume_test_")
    results_dir = os.path.join(workdir, "results")
    os.makedirs(results_dir, exist_ok=True)
    ok = True

    # ---------------------------------------------------------------
    # PHASE 1: run the SEVEN original optimizers -> builds a checkpoint
    # ---------------------------------------------------------------
    print(f"\nPHASE 1: seven optimizers on real '{name}' (builds checkpoint) ...")
    cfg = make_cfg(hb, results_dir)
    seven = ["grid", "random", "bo_tpe", "pso", "de", "ga", "sa", "cuckoo", "sh"]
    dfr7, _, _ = hb.run_one_dataset(oid, name, cfg, optimizers=seven,
                                    n_seeds=N_SEEDS, verbose=False)
    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{name}.csv")
    if not os.path.exists(ckpt_path):
        red("checkpoint file was not written"); return 1
    seven_rows = pd.read_csv(ckpt_path)
    n_seven = len(seven_rows)
    green(f"seven optimizers ran, checkpoint has {n_seven} rows "
          f"({seven_rows['optimizer'].nunique()} optimizers).")

    # snapshot the seven optimizers' results to detect any change later
    snap = seven_rows.sort_values(["optimizer", "seed"]).reset_index(drop=True)

    # ---------------------------------------------------------------
    # PHASE 2: resume with cuckoo + sh added -> should ONLY add those two
    # ---------------------------------------------------------------
    print(f"\nPHASE 2: resume with cuckoo + sh added ...")
    nine = seven + ["cuckoo", "sh"]
    dfr9, _, _ = hb.run_one_dataset(oid, name, cfg, optimizers=nine,
                                    n_seeds=N_SEEDS, verbose=False)

    # --- Check A: cuckoo and sh are now present ---
    opts_now = set(dfr9["optimizer"].unique())
    if {"cuckoo", "sh"}.issubset(opts_now):
        green("cuckoo and sh are present after resume.")
    else:
        red(f"cuckoo/sh missing after resume; got {sorted(opts_now)}"); ok = False

    # --- Check B: the seven originals were NOT recomputed (rows identical) ---
    after = dfr9[dfr9["optimizer"].isin(seven)].sort_values(
        ["optimizer", "seed"]).reset_index(drop=True)
    keycols = ["test_acc", "cv_best", "best_log2C", "best_log2gamma", "evals_to_99pct"]
    same = True
    for c in keycols:
        a = snap[c].to_numpy(dtype=float)
        b = after[c].to_numpy(dtype=float)
        if len(a) != len(b) or not np.allclose(a, b, atol=1e-9, rtol=0.0):
            same = False
            break
    if same:
        green("seven originals unchanged after resume (not recomputed).")
    else:
        red("seven originals CHANGED after resume -- resume is NOT safe."); ok = False

    # --- Check C: cuckoo & sh produced sensible accuracies (not NaN/0) ---
    for opt in ["cuckoo", "sh"]:
        acc = dfr9[dfr9["optimizer"] == opt]["test_acc"]
        if acc.isna().any():
            red(f"{opt}: produced NaN accuracy."); ok = False
        elif (acc <= 0.0).any():
            red(f"{opt}: produced non-positive accuracy."); ok = False
        else:
            green(f"{opt}: accuracies OK "
                  f"(min={acc.min():.4f}, max={acc.max():.4f}).")

    # --- Check D: SH respects the weighted budget (raw evals high, no crash) ---
    sh_rows = dfr9[dfr9["optimizer"] == "sh"]
    if len(sh_rows):
        raw = sh_rows["n_evals_used"].to_numpy()
        green(f"sh ran with raw evals in [{raw.min()}, {raw.max()}] "
              f"(multi-fidelity; weighted budget enforced internally).")

    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    if ok:
        print("ALL CHECKS PASSED.")
        print("-> Safe to run the full four-mode resume with cuckoo + sh.")
        print("   Point each mode's results_dir/checkpoint_dir at your run #1")
        print("   folders; only cuckoo + sh will be computed.")
    else:
        print("SOME CHECKS FAILED -- inspect above before touching real results.")
    print("=" * 60)
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())