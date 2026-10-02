"""
apply_sh_fullfid_patch.py -- patch YOUR local hpo_benchmark_v3.py so that
multi-fidelity evaluations (Successive Halving) are verified at full fidelity.

Why (Reviewer 1 comment 2, Reviewer 2 comment 1)
------------------------------------------------
The current harness records every evaluation in one history, whatever its data
fraction rho. Two things then read that history without looking at rho:

  * best_so_far_curve()  -> SH could "reach" a target on a 1/27 subsample score
  * best_config()        -> SH's FINAL selected configuration could be one that
                            only looked good on a tiny subsample; that config is
                            then refitted and scored on the test set

Standard SH returns the survivor of the last (full-fidelity) rung. This patch
makes both functions able to ignore evaluations with rho < 1, and makes the
main loop record BOTH views so the old behaviour remains available as a
sensitivity analysis.

For the eight full-fidelity optimizers every evaluation has rho = 1, so their
results are numerically identical with or without the flag. Only SH changes.

Default behaviour is UNCHANGED: the new CONFIG flag "full_fidelity_only" is
False, so any existing run reproduces exactly. The companion runner
run_sh_fullfid.py switches it on.

Usage
-----
    python apply_sh_fullfid_patch.py --check      # report only, writes nothing
    python apply_sh_fullfid_patch.py              # patch hpo_benchmark_v3.py
    python apply_sh_fullfid_patch.py --file other_name.py

A backup is written to <file>.bak before any change. If any expected code
fragment is missing, NOTHING is written and the missing fragment is reported.
"""
import argparse
import ast
import shutil
import sys

MARKER = "full_fidelity_only"

# (label, old, new). Each `old` must occur exactly once in the target file.
EDITS = [
    (
        "1. record the data fraction rho in every history entry",
        "        self.history.append((self.n_eval, log2C, log2g, score, self.spent))",
        "        # (raw_index, log2C, log2g, score, weighted_spent_after, rho)\n"
        "        self.history.append((self.n_eval, log2C, log2g, score, self.spent,\n"
        "                             float(frac)))",
    ),
    (
        "2a. best_so_far_curve: accept a full_fidelity_only flag",
        "    def best_so_far_curve(self):",
        "    def best_so_far_curve(self, full_fidelity_only=False):",
    ),
    (
        "2b. best_so_far_curve: skip rho < 1 when the flag is set",
        "        for h in self.history:\n"
        "            s = h[3]\n",
        "        for h in self.history:\n"
        "            # With full_fidelity_only, an evaluation at rho < 1 still consumes\n"
        "            # budget (the slot index below advances with `spent`) but cannot\n"
        "            # raise the best-so-far: a target counts as reached only once a\n"
        "            # full-fidelity evaluation attains it.\n"
        "            if full_fidelity_only and len(h) > 5 and h[5] < 1.0 - 1e-9:\n"
        "                continue\n"
        "            s = h[3]\n",
    ),
    (
        "2c. best_so_far_curve: slots before the first full-fidelity eval -> NaN",
        "        last = -np.inf\n"
        "        for i in range(len(curve)):\n"
        "            if np.isnan(curve[i]):\n"
        "                curve[i] = last\n"
        "            else:\n"
        "                last = curve[i]\n"
        "        return curve",
        "        last = -np.inf\n"
        "        for i in range(len(curve)):\n"
        "            if np.isnan(curve[i]):\n"
        "                curve[i] = last\n"
        "            else:\n"
        "                last = curve[i]\n"
        "        if full_fidelity_only:\n"
        "            # no full-fidelity result yet -> undefined, not -inf\n"
        "            curve[np.isneginf(curve)] = np.nan\n"
        "        return curve",
    ),
    (
        "3. best_config: select among full-fidelity evaluations when flagged",
        "    def best_config(self):\n"
        "        if not self.history:\n"
        "            return None\n",
        "    def best_config(self, full_fidelity_only=False):\n"
        "        if not self.history:\n"
        "            return None\n"
        "        if full_fidelity_only:\n"
        "            # Standard successive halving returns the survivor of its last,\n"
        "            # full-fidelity rung. Selecting over subsample scores instead can\n"
        "            # pick a configuration that merely looked good on a tiny fraction.\n"
        "            full = [h for h in self.history if len(h) <= 5 or h[5] >= 1.0 - 1e-9]\n"
        "            if full:\n"
        "                b = max(full, key=lambda h: h[3])\n"
        "                return (b[0], b[1], b[2], b[3])\n",
    ),
    (
        "4a. main loop: primary selection uses the flag",
        "            bc = obj.best_config()\n",
        "            ffo = cfg.get(\"full_fidelity_only\", False)\n"
        "            bc = obj.best_config(full_fidelity_only=ffo)\n"
        "            bc_any = obj.best_config(full_fidelity_only=False)\n",
    ),
    (
        "4b. main loop: primary curve uses the flag, any-fidelity kept as well",
        "            curve = obj.best_so_far_curve()\n",
        "            curve = obj.best_so_far_curve(full_fidelity_only=ffo)\n"
        "            curve_any = obj.best_so_far_curve(full_fidelity_only=False)\n",
    ),
    (
        "4c. main loop: summary rows get the any-fidelity best and fidelity counts",
        "                         \"n_evals_used\": obj.n_eval})",
        "                         \"n_evals_used\": obj.n_eval,\n"
        "                         \"cv_best_anyfid\": bc_any[3],\n"
        "                         \"n_fullfid_evals\": sum(1 for h in obj.history\n"
        "                                                 if len(h) <= 5 or h[5] >= 1.0 - 1e-9),\n"
        "                         \"full_fidelity_only\": bool(ffo)})",
    ),
    (
        "4d. main loop: curves get the any-fidelity column as well",
        "                               \"eval\": i + 1, \"best_so_far\": v})",
        "                               \"eval\": i + 1, \"best_so_far\": v,\n"
        "                               \"best_so_far_anyfid\": curve_any[i]})",
    ),
    (
        "5. CONFIG: add the flag, default off so existing runs reproduce exactly",
        "    \"decomp_mode\": \"baseline\",\n",
        "    \"decomp_mode\": \"baseline\",\n"
        "    # Reviewer 1 comment 2 / Reviewer 2 comment 1: count only rho = 1\n"
        "    # evaluations toward targets and toward the final selection. False keeps\n"
        "    # the original behaviour; run_sh_fullfid.py sets it to True. It has no\n"
        "    # effect on full-fidelity optimizers.\n"
        "    \"full_fidelity_only\": False,\n",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="hpo_benchmark_v3.py")
    ap.add_argument("--check", action="store_true",
                    help="report whether every fragment is found; write nothing")
    args = ap.parse_args()

    src = open(args.file, encoding="utf-8").read()
    if MARKER in src and not args.check:
        print(f"[skip] {args.file} already contains '{MARKER}'; not patching twice.")
        return

    problems, new = [], src
    for label, old, rep in EDITS:
        n = new.count(old)
        status = "found" if n == 1 else ("MISSING" if n == 0 else f"AMBIGUOUS x{n}")
        print(f"  [{status:>11}] {label}")
        if n != 1:
            problems.append(label)
        else:
            new = new.replace(old, rep, 1)

    if problems:
        print(f"\n{len(problems)} fragment(s) not matched exactly once. Nothing written.")
        print("Your local file differs from the version this patch was built for;")
        print("send the reported sections and the patch will be adapted.")
        sys.exit(1)

    try:
        ast.parse(new)
    except SyntaxError as e:
        print(f"\nPatched source does not parse ({e}). Nothing written.")
        sys.exit(1)

    if args.check:
        print("\nAll fragments found; the patched file would parse. (--check: nothing written)")
        return

    shutil.copyfile(args.file, args.file + ".bak")
    open(args.file, "w", encoding="utf-8").write(new)
    print(f"\nPatched {args.file}  (backup: {args.file}.bak)")


if __name__ == "__main__":
    main()