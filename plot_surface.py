"""
=============================================================================
Plot the CV-accuracy SURFACE over the (log2 C, log2 gamma) search space for
selected datasets. This is an objective-landscape visualization (NOT ELA):
it shows directly why the 2-D RBF-SVC tuning space is benign, supporting the
finding that all optimizers perform equivalently.

Run on a machine with OpenML access. Produces one heatmap per dataset, plus
an optional combined figure.

Usage:
    python plot_surface.py
Edit DATASETS and GRID_RES below as needed.
=============================================================================
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# --- config (must mirror the benchmark) ---
LOG2C = (-5.0, 15.0)
LOG2G = (-15.0, 3.0)
CV_FOLDS = 5
CV_SEED = 42
GRID_RES = 40          # 40x40 = 1600 fits per dataset (heavier than the 100-eval budget; for visualization only)
DATASETS = [(1462, "banknote"), (1480, "ilpd")]   # (openml_id, name): easy vs challenging

def load_dataset(openml_id):
    import openml
    ds = openml.datasets.get_dataset(openml_id)
    Xdf, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    num = Xdf.select_dtypes(include=[np.number])
    cat = Xdf.select_dtypes(exclude=[np.number])
    parts = []
    if num.shape[1]: parts.append(num.values.astype(float))
    if cat.shape[1]:
        cat = cat.astype("object").where(cat.notna(), other="__m__")
        parts.append(OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(cat))
    X = np.hstack(parts).astype(float)
    return X, pd.factorize(y)[0]

def surface(X, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    cs = np.linspace(*LOG2C, GRID_RES)
    gs = np.linspace(*LOG2G, GRID_RES)
    Z = np.zeros((GRID_RES, GRID_RES))
    for i, lc in enumerate(cs):
        for j, lg in enumerate(gs):
            pipe = Pipeline([("imp", SimpleImputer(strategy="mean")),
                             ("sc", StandardScaler()),
                             ("svc", SVC(C=2.0**lc, gamma=2.0**lg, kernel="rbf"))])
            Z[j, i] = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    return cs, gs, Z

def plot_one(name, cs, gs, Z, ax):
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[cs[0], cs[-1], gs[0], gs[-1]], cmap="viridis")
    # mark the best cell
    jmax, imax = np.unravel_index(np.argmax(Z), Z.shape)
    ax.plot(cs[imax], gs[jmax], "r*", markersize=14, markeredgecolor="white")
    ax.set_xlabel(r"$\log_2 C$"); ax.set_ylabel(r"$\log_2 \gamma$")
    ax.set_title(f"{name}  (best CV acc = {Z.max():.3f})")
    return im

if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6*len(DATASETS), 4.6))
    if len(DATASETS) == 1: axes = [axes]
    for ax, (oid, name) in zip(axes, DATASETS):
        print(f"computing surface for {name} ...", flush=True)
        X, y = load_dataset(oid)
        cs, gs, Z = surface(X, y)
        im = plot_one(name, cs, gs, Z, ax)
        fig.colorbar(im, ax=ax, label="CV accuracy")
    plt.tight_layout()
    plt.savefig("surface_heatmaps.png", dpi=200)
    print("saved surface_heatmaps.png")
