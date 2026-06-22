"""
=============================================================================
HPO Benchmark for SVC: Grid / Random / Bayesian (TPE) / PSO / DE / GA
=============================================================================
A reproducible benchmark comparing hyperparameter optimization algorithms
for Support Vector Classification (RBF kernel) over the (C, gamma) space.

Angle: cost-aware / anytime performance + dataset-characteristic-based
       algorithm selection, with rigorous statistics (Friedman + Nemenyi).

Author: (your name)
Designed to run on a normal laptop (CPU only).
=============================================================================
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION  -- edit here only
# =============================================================================
CONFIG = {
    # Search space in log2 units (fairness: all optimizers share this exact box)
    "log2C_bounds":     (-5.0, 15.0),     # C    in [2^-5 , 2^15]
    "log2gamma_bounds": (-15.0, 3.0),     # gamma in [2^-15, 2^3]

    # Identical evaluation budget per optimizer (fairness #2)
    "budget": 100,                        # number of objective evaluations

    # Cross-validation for the objective
    "cv_folds": 5,
    "cv_seed": 42,                        # fixed during optimization (noise control)

    # Repetitions for statistics
    "n_seeds": 2,                        # optimization runs per (optimizer, dataset)
    "test_size": 0.20,                    # hold-out test fraction

    # Output
    "results_dir": "results",

    # Datasets: OpenML dataset IDs, stratified by size x dimensionality.
    # (name is for reporting only)
    "datasets": [
        # (openml_id, friendly_name)
        (37,   "diabetes"),        # small n, low dim
        (15,   "breast-w"),        # small n, low-mid dim
        (54,   "vehicle"),         # mid n, mid dim
        (1510, "wdbc"),            # small n, high dim
        (40975,"car"),             # mid n, low dim
        (1480, "ilpd"),            # small n, low dim
        (1494, "qsar-biodeg"),     # mid n, high dim
        (1462, "banknote"),        # mid n, low dim
        (40982,"steel-faults"),    # mid n, mid dim
        (1487, "ozone-8hr"),       # mid n, high dim
    ],
}

OPTIMIZERS = ["grid", "random", "bo_tpe", "pso", "de", "ga"]


# =============================================================================
# OBJECTIVE
# =============================================================================
class Objective:
    """Wraps SVC CV-accuracy as a function of (log2C, log2gamma).
    Records every evaluation so we can reconstruct anytime curves."""

    def __init__(self, X, y, cfg):
        self.X = X
        self.y = y
        self.cfg = cfg
        self.cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True,
                                  random_state=cfg["cv_seed"])
        self.history = []   # list of (eval_idx, log2C, log2g, score)
        self.n_eval = 0

    def evaluate(self, log2C, log2g):
        if self.n_eval >= self.cfg["budget"]:
            # Budget exhausted: return current best without spending more.
            best = max((h[3] for h in self.history), default=0.0)
            return best
        C = 2.0 ** log2C
        gamma = 2.0 ** log2g
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler()),
            ("svc", SVC(C=C, gamma=gamma, kernel="rbf")),
        ])
        score = cross_val_score(pipe, self.X, self.y, cv=self.cv,
                                scoring="accuracy", n_jobs=1).mean()
        self.n_eval += 1
        self.history.append((self.n_eval, log2C, log2g, score))
        return score

    def best_so_far_curve(self):
        """Return anytime best-so-far array of length == budget."""
        curve = np.full(self.cfg["budget"], np.nan)
        best = -np.inf
        for (idx, _, _, s) in self.history:
            best = max(best, s)
            curve[idx - 1] = best
        # forward-fill (if optimizer used < budget evals)
        last = -np.inf
        for i in range(len(curve)):
            if np.isnan(curve[i]):
                curve[i] = last
            else:
                last = curve[i]
        return curve

    def best_config(self):
        if not self.history:
            return None
        return max(self.history, key=lambda h: h[3])


# =============================================================================
# OPTIMIZERS  (each must respect the shared budget exactly)
# =============================================================================
def run_grid(obj, cfg, seed):
    """Deterministic grid 10x10 = 100 (seed unused, kept for API symmetry)."""
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    side = int(round(cfg["budget"] ** 0.5))
    cs = np.linspace(lc[0], lc[1], side)
    gs = np.linspace(lg[0], lg[1], side)
    for c in cs:
        for g in gs:
            if obj.n_eval >= cfg["budget"]:
                return
            obj.evaluate(c, g)


def run_random(obj, cfg, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def fn(trial):
        c = trial.suggest_float("log2C", lc[0], lc[1])
        g = trial.suggest_float("log2g", lg[0], lg[1])
        return obj.evaluate(c, g)
    study.optimize(fn, n_trials=cfg["budget"], show_progress_bar=False)


def run_bo_tpe(obj, cfg, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def fn(trial):
        c = trial.suggest_float("log2C", lc[0], lc[1])
        g = trial.suggest_float("log2g", lg[0], lg[1])
        return obj.evaluate(c, g)
    study.optimize(fn, n_trials=cfg["budget"], show_progress_bar=False)


def _pymoo_problem(obj, cfg):
    from pymoo.core.problem import ElementwiseProblem
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]

    class SVCProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=1, n_constr=0,
                             xl=np.array([lc[0], lg[0]]),
                             xu=np.array([lc[1], lg[1]]))

        def _evaluate(self, x, out, *args, **kwargs):
            # minimize negative accuracy
            out["F"] = -obj.evaluate(x[0], x[1])
    return SVCProblem()


def _run_pymoo(obj, cfg, seed, algorithm):
    from pymoo.optimize import minimize
    problem = _pymoo_problem(obj, cfg)
    # n_gen chosen so pop_size * n_gen ~= budget; termination also capped below
    minimize(problem, algorithm,
             termination=("n_evals", cfg["budget"]),
             seed=seed, verbose=False)


def run_pso(obj, cfg, seed):
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    algo = PSO(pop_size=20)
    _run_pymoo(obj, cfg, seed, algo)


def run_de(obj, cfg, seed):
    from pymoo.algorithms.soo.nonconvex.de import DE
    algo = DE(pop_size=20)
    _run_pymoo(obj, cfg, seed, algo)


def run_ga(obj, cfg, seed):
    from pymoo.algorithms.soo.nonconvex.ga import GA
    algo = GA(pop_size=20)
    _run_pymoo(obj, cfg, seed, algo)


RUNNERS = {
    "grid": run_grid, "random": run_random, "bo_tpe": run_bo_tpe,
    "pso": run_pso, "de": run_de, "ga": run_ga,
}


# =============================================================================
# DATA
# =============================================================================
def load_dataset(openml_id):
    import openml
    ds = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute,
                             dataset_format="dataframe")
    # keep numeric features only (simple, robust for a 1-month study)
    X = X.select_dtypes(include=[np.number])
    X = X.values.astype(float)
    # encode target
    y = pd.factorize(y)[0]
    return X, y


def dataset_meta(X, y):
    n, d = X.shape
    n_classes = len(np.unique(y))
    return {"n_samples": int(n), "n_features": int(d),
            "n_classes": int(n_classes), "ratio_nd": float(n / d)}


# =============================================================================
# MAIN LOOP
# =============================================================================
def evaluate_test(X_tr, y_tr, X_te, y_te, log2C, log2g):
    from sklearn.metrics import accuracy_score
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc", StandardScaler()),
        ("svc", SVC(C=2.0**log2C, gamma=2.0**log2g, kernel="rbf")),
    ])
    pipe.fit(X_tr, y_tr)
    return accuracy_score(y_te, pipe.predict(X_te))


def run_all(cfg, datasets=None, optimizers=None, n_seeds=None, verbose=True):
    os.makedirs(cfg["results_dir"], exist_ok=True)
    datasets = datasets if datasets is not None else cfg["datasets"]
    optimizers = optimizers if optimizers is not None else OPTIMIZERS
    n_seeds = n_seeds if n_seeds is not None else cfg["n_seeds"]

    rows = []          # summary rows
    curves = []        # anytime curves (long format)
    meta_rows = []

    for (oid, name) in datasets:
        if verbose:
            print(f"\n=== Dataset: {name} (OpenML {oid}) ===", flush=True)
        X, y = load_dataset(oid)
        meta = dataset_meta(X, y)
        meta_rows.append({"dataset": name, "openml_id": oid, **meta})
        if verbose:
            print(f"    n={meta['n_samples']} d={meta['n_features']} "
                  f"classes={meta['n_classes']}", flush=True)

        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=cfg["test_size"], random_state=seed, stratify=y)

            for opt in optimizers:
                obj = Objective(X_tr, y_tr, cfg)
                t0 = time.time()
                RUNNERS[opt](obj, cfg, seed)
                elapsed = time.time() - t0

                bc = obj.best_config()
                if bc is None:
                    continue
                _, bC, bg, cv_best = bc
                test_acc = evaluate_test(X_tr, y_tr, X_te, y_te, bC, bg)
                curve = obj.best_so_far_curve()

                # evals to reach 99% of own final best
                thr = 0.99 * cv_best
                reach = int(np.argmax(curve >= thr) + 1) if np.any(curve >= thr) else cfg["budget"]

                rows.append({
                    "dataset": name, "optimizer": opt, "seed": seed,
                    "cv_best": cv_best, "test_acc": test_acc,
                    "best_log2C": bC, "best_log2gamma": bg,
                    "time_sec": elapsed, "evals_to_99pct": reach,
                    "n_evals_used": obj.n_eval,
                })
                for i, v in enumerate(curve):
                    curves.append({"dataset": name, "optimizer": opt,
                                   "seed": seed, "eval": i + 1, "best_so_far": v})
            if verbose and (seed + 1) % max(1, n_seeds // 5) == 0:
                print(f"    seed {seed+1}/{n_seeds} done", flush=True)

    df = pd.DataFrame(rows)
    dfc = pd.DataFrame(curves)
    dfm = pd.DataFrame(meta_rows)
    df.to_csv(os.path.join(cfg["results_dir"], "summary.csv"), index=False)
    dfc.to_csv(os.path.join(cfg["results_dir"], "anytime_curves.csv"), index=False)
    dfm.to_csv(os.path.join(cfg["results_dir"], "dataset_meta.csv"), index=False)
    if verbose:
        print("\nSaved: summary.csv, anytime_curves.csv, dataset_meta.csv")
    return df, dfc, dfm


if __name__ == "__main__":
    run_all(CONFIG)
