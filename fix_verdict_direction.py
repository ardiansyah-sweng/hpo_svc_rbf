"""
fix_verdict_direction.py -- patch analyze_efficiency_datasetlevel.py so the
successive-halving verdict reads the DIRECTION of a significant difference.

The bug: `sh_sig` collects every pair whose Nemenyi p-value is below alpha and
then announces "sh is SIGNIFICANTLY faster than ...". A small p-value only says
the pair differs; which one is faster is decided by the mean ranks, where a
LOWER rank is faster. After full-fidelity verification SH has mean rank 9.00 of
9 -- the slowest -- yet the old code reported it as "a genuine high-target
advantage", the exact opposite of the data.

    python fix_verdict_direction.py --check     # report only, writes nothing
    python fix_verdict_direction.py             # patch, backup to .bak
"""
import argparse
import ast
import shutil
import sys

OLD = '''            sh_sig = [(o, nem.loc["sh", o]) for o in optimizers if o != "sh"
                      and nem.loc["sh", o] < args.alpha]
            print(f"  sh mean rank = {sh_rank:.2f} (of {len(optimizers)}); "
                  f"nominally faster than {len(better_than)} optimizers.")
            sh_sig_names = [o for o, pv in sh_sig]
            sh_vs_adaptive = [o for o in sh_sig_names if o != "grid"]
            if sh_sig:
                print("  sh is SIGNIFICANTLY faster than: " +
                      ", ".join(f"{o} (p={pv:.3f})" for o, pv in sh_sig))
            if sh_vs_adaptive:
                print("  -> SH's edge SURVIVES over ADAPTIVE methods "
                      f"({', '.join(sh_vs_adaptive)}); a genuine high-target advantage.")
            elif sh_sig_names == ["grid"]:
                print("  -> SH is faster than GRID only, like every adaptive method. "
                      "Its apparent edge over the adaptive group does NOT survive "
                      "dataset-level blocking; report as within the adaptive cluster.")
            else:
                print("  -> SH is NOT significantly faster than any optimizer after "
                      "Nemenyi correction; report as non-significant.")'''

NEW = '''            sh_sig = [(o, nem.loc["sh", o]) for o in optimizers if o != "sh"
                      and nem.loc["sh", o] < args.alpha]
            print(f"  sh mean rank = {sh_rank:.2f} (of {len(optimizers)}); "
                  f"nominally faster than {len(better_than)} optimizers.")
            # A small Nemenyi p-value says only that the pair differs. The
            # direction comes from the mean ranks: a LOWER mean rank is faster.
            faster = [(o, pv) for o, pv in sh_sig if mean_rank["sh"] < mean_rank[o]]
            slower = [(o, pv) for o, pv in sh_sig if mean_rank["sh"] > mean_rank[o]]
            fmt = lambda xs: ", ".join(f"{o} (p={pv:.3f})" for o, pv in xs)
            if faster:
                print("  sh is SIGNIFICANTLY FASTER than: " + fmt(faster))
            if slower:
                print("  sh is SIGNIFICANTLY SLOWER than: " + fmt(slower))
            faster_adaptive = [o for o, _ in faster if o != "grid"]
            if faster and slower:
                print("  -> Mixed: SH is faster than some optimizers and slower than "
                      "others. Report the two lists separately, not as an overall edge.")
            elif slower:
                print("  -> SH is significantly SLOWER than the optimizers listed above. "
                      "Any apparent efficiency advantage does not survive; report SH as "
                      "the slowest member of the group at this target.")
            elif faster_adaptive:
                print("  -> SH's edge SURVIVES over ADAPTIVE methods "
                      f"({', '.join(faster_adaptive)}); a genuine high-target advantage.")
            elif [o for o, _ in faster] == ["grid"]:
                print("  -> SH is faster than GRID only, like every adaptive method. "
                      "Its apparent edge over the adaptive group does NOT survive "
                      "dataset-level blocking; report as within the adaptive cluster.")
            else:
                print("  -> SH is NOT significantly faster or slower than any optimizer "
                      "after Nemenyi correction; report as non-significant.")'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="analyze_efficiency_datasetlevel.py")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    src = open(args.file, encoding="utf-8").read()
    if "SIGNIFICANTLY SLOWER" in src and not args.check:
        print(f"[skip] {args.file} already patched.")
        return
    n = src.count(OLD)
    if n != 1:
        print(f"[{'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}] verdict block not matched "
              f"exactly once in {args.file}. Nothing written.")
        sys.exit(1)
    new = src.replace(OLD, NEW, 1)
    try:
        ast.parse(new)
    except SyntaxError as e:
        print(f"Patched source does not parse ({e}). Nothing written.")
        sys.exit(1)
    if args.check:
        print("Verdict block found; patched file would parse. (--check: nothing written)")
        return
    shutil.copyfile(args.file, args.file + ".bak")
    open(args.file, "w", encoding="utf-8").write(new)
    print(f"Patched {args.file}  (backup: {args.file}.bak)")
    print("Re-run the three dataset-level commands to regenerate the verdict lines.")


if __name__ == "__main__":
    main()