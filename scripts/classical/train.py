"""
scripts/classical/train.py

Training script for Implementation 1 — Classical RF Baseline.

Pipeline:
    1. Extract features + targets from data/train/ using features.py
    2. Grid search with 5-fold CV to find best RF hyperparameters
    3. Retrain final RF on full training set with best params
    4. Evaluate on data/val/ → MAE and RMSE for each of the 3 targets
    5. Save model and results

Three separate Random Forest Regressors are trained — one per target:
    RF_0: predicts elevation std  (primary target, most informative)
    RF_1: predicts elevation mean (secondary, nearly always ~0)
    RF_2: predicts dominant slope direction

Usage:
    python scripts/classical/train.py
    python scripts/classical/train.py --max_patches 2000   # quick smoke test
    python scripts/classical/train.py --skip_cv            # skip grid search, use defaults
"""

import os
import sys
import time
import argparse
import pickle
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Make project root importable regardless of where the script is called from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.classical.features import build_feature_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TARGET_NAMES = ["elev_std", "elev_mean", "slope_dir"]

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [10, 20, None],
}

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth":    None,
}

CV_FOLDS   = 5
RANDOM_SEED = 42

OUTPUT_DIR = "outputs/classical"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RF baseline for MarsDEMNet")
    parser.add_argument("--train_dir",    default="data/train/", help="Path to training patches")
    parser.add_argument("--val_dir",      default="data/val/",   help="Path to validation patches")
    parser.add_argument("--max_patches",  type=int, default=None,
                        help="Limit patches loaded (None = all). Use e.g. 2000 for smoke test.")
    parser.add_argument("--skip_cv",      action="store_true",
                        help="Skip grid search CV and use DEFAULT_PARAMS directly.")
    parser.add_argument("--output_dir",   default=OUTPUT_DIR,   help="Where to save model + results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Feature extraction ───────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Extracting training features")
    print("=" * 60)
    t0 = time.time()
    X_train, y_train, ids_train = build_feature_matrix(
        args.train_dir, max_patches=args.max_patches, verbose=True
    )
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}  [{time.time()-t0:.1f}s]")

    print()
    print("Step 1b: Extracting validation features")
    t0 = time.time()
    X_val, y_val, ids_val = build_feature_matrix(
        args.val_dir, max_patches=args.max_patches, verbose=True
    )
    print(f"  X_val:   {X_val.shape}   y_val:   {y_val.shape}  [{time.time()-t0:.1f}s]")

    # ── Step 2: Train one RF per target ──────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 2: Training Random Forest regressors")
    print("=" * 60)

    trained_models = {}
    best_params_all = {}
    results = {}

    for i, target_name in enumerate(TARGET_NAMES):
        print(f"\n── Target {i}: {target_name} ──")
        y_tr = y_train[:, i]
        y_vl = y_val[:,   i]

        # ── Grid search CV ───────────────────────────────────────────────────
        if args.skip_cv:
            best_params = DEFAULT_PARAMS.copy()
            print(f"  Skipping CV. Using defaults: {best_params}")
        else:
            print(f"  Running {CV_FOLDS}-fold CV over {PARAM_GRID} ...")
            t0 = time.time()
            rf_base = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
            search  = GridSearchCV(
                rf_base,
                PARAM_GRID,
                cv=CV_FOLDS,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                verbose=0,
                refit=False,   # we'll refit manually below
            )
            search.fit(X_train, y_tr)
            best_params = search.best_params_
            print(f"  Best params: {best_params}  (CV took {time.time()-t0:.1f}s)")

        best_params_all[target_name] = best_params

        # ── Retrain on full training set with best params ─────────────────────
        print("  Retraining on full training set ...")
        t0 = time.time()
        rf = RandomForestRegressor(
            **best_params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X_train, y_tr)
        print(f"  Training done [{time.time()-t0:.1f}s]")

        # ── Evaluate on validation set ────────────────────────────────────────
        y_pred_train = rf.predict(X_train)
        y_pred_val   = rf.predict(X_val)

        train_mae  = mean_absolute_error(y_tr, y_pred_train)
        train_rmse = root_mean_squared_error(y_tr, y_pred_train)
        val_mae    = mean_absolute_error(y_vl, y_pred_val)
        val_rmse   = root_mean_squared_error(y_vl, y_pred_val)

        results[target_name] = {
            "best_params":  best_params,
            "train_mae":    round(float(train_mae),  4),
            "train_rmse":   round(float(train_rmse), 4),
            "val_mae":      round(float(val_mae),    4),
            "val_rmse":     round(float(val_rmse),   4),
        }

        print(f"  Train  MAE={train_mae:.4f}  RMSE={train_rmse:.4f}")
        print(f"  Val    MAE={val_mae:.4f}    RMSE={val_rmse:.4f}")

        trained_models[target_name] = rf

    # ── Step 3: Save models ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 3: Saving models and results")
    print("=" * 60)

    model_path   = os.path.join(args.output_dir, "rf_models.pkl")
    results_path = os.path.join(args.output_dir, "rf_results.json")

    with open(model_path, "wb") as f:
        pickle.dump(trained_models, f)
    print(f"  Models saved → {model_path}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {results_path}")

    # ── Step 4: Summary table ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    _print_table(results)

    # ── Step 5: Feature importance (elev_std only — the primary target) ───────
    print()
    print("Feature importance — elev_std (top 10):")
    _print_feature_importance(trained_models["elev_std"], top_n=10)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(results: dict):
    """Print a compact results table to stdout."""
    header = f"{'Target':<20} {'Train MAE':>12} {'Train RMSE':>12} {'Val MAE':>10} {'Val RMSE':>10}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        unit = "(m)" if name in ("elev_std", "elev_mean") else "(°)"
        print(
            f"{name:<20} {r['train_mae']:>12.4f} {r['train_rmse']:>12.4f} "
            f"{r['val_mae']:>10.4f} {r['val_rmse']:>10.4f}  {unit}"
        )


def _print_feature_importance(rf: RandomForestRegressor, top_n: int = 10):
    """Print top N feature importances with human-readable names."""
    feature_names = _build_feature_names()
    importances   = rf.feature_importances_
    ranked        = np.argsort(importances)[::-1]

    for rank, idx in enumerate(ranked[:top_n]):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"  {rank+1:2d}. {name:<30} {importances[idx]:.4f}")


def _build_feature_names() -> list:
    """Returns human-readable names for all 63 features, in extraction order."""
    names = []

    # LBP: 26 bins
    for b in range(26):
        names.append(f"lbp_bin_{b:02d}")

    # Gabor means: 12 filters
    from scripts.classical.features import GABOR_WAVELENGTHS, GABOR_ORIENTATIONS
    for wl in GABOR_WAVELENGTHS:
        for theta in GABOR_ORIENTATIONS:
            names.append(f"gabor_mean_wl{wl}_theta{theta}")

    # Gabor variances: 12 filters
    for wl in GABOR_WAVELENGTHS:
        for theta in GABOR_ORIENTATIONS:
            names.append(f"gabor_var_wl{wl}_theta{theta}")

    # HOG: 9 bins
    for b in range(9):
        names.append(f"hog_bin_{b:02d}")

    # Intensity moments: 4
    for m in ["mean", "std", "skewness", "kurtosis"]:
        names.append(f"intensity_{m}")

    return names  # 26 + 12 + 12 + 9 + 4 = 63


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
