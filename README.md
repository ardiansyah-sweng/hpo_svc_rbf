# Budget-Controlled HPO Benchmark for RBF-SVM

Code and results for *A Budget-Controlled Protocol for Separating Optimizer
Quality from Efficiency in SVM Hyperparameter Tuning*.

The study compares **nine optimizers** — grid search (GS), random search (RS),
TPE-based Bayesian optimization (BO-TPE), particle swarm optimization (PSO),
differential evolution (DE), genetic algorithm (GA), simulated annealing (SA),
cuckoo search (CS) and successive halving (SH) — over the two-dimensional
(C, γ) space of an RBF-SVM, on **ten OpenML datasets**, with **30 repetitions**
under **four seed configurations**: **10,800 runs** in total.

Every optimizer receives the same search box, the same budget of **B = 100
full-fidelity-equivalent evaluations**, and the same cross-validation protocol.
Three controls distinguish this benchmark from a conventional comparison:

1. **Full-fidelity verification.** A configuration is returned, and a target
   counts as reached, only on the strength of a full-fidelity evaluation.
   Low-fidelity evaluations still consume budget but never satisfy a target.
2. **Variance attribution.** Run-to-run variability is attributed to the
   train/test partition, the cross-validation folds and optimizer
   initialization by varying one seed at a time.
3. **Order randomization.** The grid lattice has no intrinsic traversal order,
   so the efficiency analysis uses a random permutation and the deterministic
   policies are analysed separately.

## Install

```
pip install -r requirements.txt
```

Python 3.11+. Datasets are fetched from OpenML on first run.

## Where the published numbers live

Every figure quoted in the paper comes from the files below. Directories
carrying the `_fullfid` suffix hold the corrected results; see **Provenance**.

| Paper | File |
| --- | --- |
| Table 3 — datasets, no-skill floor, headroom | `tables/table3.tsv`, `results/efficiency_gap/dataset_baseline.csv` |
| Table 4 — mean test accuracy | `tables/table4.tsv`, `results/accuracy_fullfid/dataset_by_optimizer_mean.csv` |
| §4.1 — 36/36 pairs equivalent, CI −0.39 to +0.37 pp | `results/equivalence_fullfid/tost_across_dataset_10mpp.csv` |
| Table 5 — within-dataset equivalence rates | `tables/table5.tsv`, `results/equivalence_fullfid/tost_per_dataset_10mpp.csv` |
| Table 6 — between-optimizer spread vs run-to-run SD | `tables/table6.tsv`, `results/table6_fullfid/table6.csv` |
| §4.1 — ties: 189 of 300 blocks (63%), banknote 8.9 of 9 | `results/two_regime_fullfid/tie_statistics.csv`, `rank_churn.csv` |
| Table 7 — variance attribution 80.2 / 11.8 / 6.9 % | `tables/table7.tsv`, `results/variance_fullfid/variance_components.csv` |
| §4.2 — additivity, median 0.92 (IQR 0.75–1.00) | `results/variance_fullfid/additivity_check.csv` |
| Table 8 — dataset-level efficiency, ranks, attainment | `tables/table8.tsv`, `results/efficiency_gap/efficiency_datasetlevel_{90,95,98}.csv`, `nemenyi_{90,95,98}.csv` |
| §4.3 — eight-dataset headroom subset, attainment 98–99 % | `results/efficiency_gap_headroom/` |
| §4.3 — GS bootstrap interval, 20.5 to 56.5 at the 98 % target | `results/traversal_uncertainty_gap/grid_bootstrap_ci.csv` |
| §4.3 — permutation accounts for a median 48 % of GS efficiency variance | `results/traversal_uncertainty_gap/permutation_share.csv` |
| Table 9 — GS under four traversal policies | `tables/table9.tsv`, `results/grid_order/grid_order_by_run.csv` |
| §4.3 — common-target cross-check | `results/efficiency_gridrandom_fullfid/` |
| §4.5 — balanced accuracy robustness check | `results/accuracy_balanced/`, `results/equivalence_balanced/`, `results/table6_balanced/`, `results/efficiency_balanced_gap/` |
| Figure 1 — composition of run-to-run variance | `figures/fig1_variance.png` |
| Figure 2 — effect of grid traversal order | `figures/fig2_traversal.png` (panels: `fig2a_efficiency.png`, `fig2b_quality.png`) |

## Provenance: `_fullfid` and what came before

`results/baseline/`, `results/isolate_S|F|O/` and `results/baseline_gridrandom/`
are the **raw runs**, recorded before successive halving was restricted to
full-fidelity verification. In those runs SH selects configurations on
subsample scores, which makes it look both more accurate and far faster than it
is. The corrected results are produced by rerunning SH alone under
`full_fidelity_only` and merging its rows back in; those merged folders carry
the `_fullfid` suffix and are the ones the paper reports.

The raw folders are kept because the merge is built from them and because
`check_env_equivalence.py` uses `results/baseline/summary.csv` as its
determinism reference. **Analyses run against the raw folders will not
reproduce the published numbers.**

## Reproducing from scratch

The full run takes several hours on a laptop CPU. Every stage checkpoints per
dataset, so an interrupted run resumes where it stopped.

### 1. Apply the patches

Both are idempotent; pass `--check` to verify without writing.

```
python apply_sh_fullfid_patch.py      # adds full_fidelity_only + fidelity bookkeeping
python apply_metric_patch.py          # adds the metric switch (accuracy / balanced_accuracy)
```

### 2. Main runs

`hpo_benchmark_v3.py` is driven by the `CONFIG` dict at the top of the file.
Set `decomp_mode` to each of `baseline`, `isolate_S`, `isolate_F`, `isolate_O`
in turn, adjusting `results_dir` to match, and run:

```
python hpo_benchmark_v3.py
```

This produces `results/baseline/`, `results/isolate_S/`, `results/isolate_F/`
and `results/isolate_O/` — 2,700 runs each.

### 3. Grid traversal experiment

```
python run_grid_order_experiment.py
python rebuild_curve_grid_random.py --curves results/baseline/anytime_curves.csv
```

The first sweeps row-major, column-major, spiral and random permutation into
`results/grid_order/`. The second rebuilds the anytime curves under randomized
traversal into `results/baseline_gridrandom/`, which the efficiency analysis
needs.

### 4. Successive halving, full fidelity

```
python run_sh_fullfid.py --modes all
```

Reruns SH alone — 1,200 runs — into `results/sh_fullfid/<mode>/`.

### 5. Merge

```
python merge_sh_fullfid.py --base results/baseline            --sh results/sh_fullfid/baseline  --out results/baseline_fullfid
python merge_sh_fullfid.py --base results/baseline_gridrandom --sh results/sh_fullfid/baseline  --out results/baseline_gridrandom_fullfid
python merge_sh_fullfid.py --base results/isolate_S           --sh results/sh_fullfid/isolate_S --out results/isolate_S_fullfid
python merge_sh_fullfid.py --base results/isolate_F           --sh results/sh_fullfid/isolate_F --out results/isolate_F_fullfid
python merge_sh_fullfid.py --base results/isolate_O           --sh results/sh_fullfid/isolate_O --out results/isolate_O_fullfid
```

The merge never writes into `--base` or `--sh`, and refuses to run unless the
eight non-SH optimizers pair cell for cell.

### 6. Balanced accuracy

```
python run_balanced_accuracy.py --out results/balanced_accuracy
```

Repeats the baseline configuration under balanced accuracy, with randomized
traversal and full-fidelity verification.

### 7. Analyses

```
python accuracy_summary.py          --summary results/baseline_fullfid/summary.csv --out results/accuracy_fullfid
python analyze_equivalence_tost.py  --summary results/baseline_fullfid/summary.csv --out results/equivalence_fullfid --margin_sd
python table6_spread_noise.py       --summary results/baseline_fullfid/summary.csv --out results/table6_fullfid
python analyze_two_regime.py        --summary results/baseline_fullfid/summary.csv --out results/two_regime_fullfid

python analyze_variance.py --baseline  results/baseline_fullfid \
                           --isolate_S results/isolate_S_fullfid \
                           --isolate_F results/isolate_F_fullfid \
                           --isolate_O results/isolate_O_fullfid \
                           --out       results/variance_fullfid

python gap_target_analysis.py   --curves results/baseline_gridrandom_fullfid/anytime_curves.csv --out results/efficiency_gap
python analyze_common_target.py --curves results/baseline_gridrandom_fullfid/anytime_curves.csv --out results/efficiency_gridrandom_fullfid

python traversal_uncertainty.py --grid_order results/grid_order/grid_order_by_run.csv \
                                --by_run     results/efficiency_gap/gap_target_by_run.csv \
                                --out        results/traversal_uncertainty_gap
```

`--margin_sd` adds the noise-floor margin alongside the 1.00 and 2.00
percentage-point margins. `gap_target_analysis.py` takes `--floor majority` for
accuracy and `--floor uniform` for balanced accuracy.

### 8. Tables and figures

```
python build_tables.py
python plot_figures.py --variance   results/variance_fullfid/variance_components.csv \
                       --grid_order results/grid_order/grid_order_by_run.csv \
                       --out        figures
```

`build_tables.py` writes `tables/table3.tsv` through `table9.tsv` and runs a
nine-item consistency check across them; it exits non-zero if any table
disagrees with another.

## Verification helpers

```
python check_env_equivalence.py                  # is this machine's grid identical to the stored one?
python verify_equivalence.py --old A --new B     # compare two result folders cell by cell
python apply_sh_fullfid_patch.py --check         # confirm the patch is applied
python fix_verdict_direction.py --check          # confirm the rank-direction fix is applied
```

`check_env_equivalence.py` reruns grid search on one dataset and compares
against `results/baseline/summary.csv`. Grid search is deterministic given a
fixed lattice, traversal order and software environment, so any difference
indicates a changed environment rather than a changed method.

## Layout

```
hpo_benchmark_v3.py              benchmark driver (CONFIG dict at the top)
apply_*.py                       idempotent patches
run_*.py, rebuild_*.py           experiment runners
merge_sh_fullfid.py              merges the SH rerun into a stored run
analyze_*.py, accuracy_summary.py, gap_target_analysis.py,
  table6_spread_noise.py, traversal_uncertainty.py
                                 analyses
build_tables.py, plot_figures.py tables and figures
check_env_equivalence.py, verify_equivalence.py, fix_verdict_direction.py
                                 verification
figures/                         figures as published
tables/                          tables as published
results/                         raw runs, merged runs, analysis outputs
```
## License

Released under CC BY-SA 4.0, matching the licence of the article.