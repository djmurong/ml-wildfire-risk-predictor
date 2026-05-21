"""Shared helpers for hazard-score evaluation and log1p burned-area transforms."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/processed"


def clip_log1p_burned_area(
    log_pred: np.ndarray,
    train_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Clip log1p(burned_area) predictions to the training range (linear models extrapolate)."""
    log_pred = np.asarray(log_pred, dtype=float)
    if train_df is None:
        train_path = PROCESSED_DIR / "train.csv"
        if not train_path.exists():
            return np.maximum(log_pred, 0.0)
        train_df = pd.read_csv(train_path)

    train_log_pos = train_df.loc[train_df["burned_area"] > 0, "log_burned_area"].dropna()
    if len(train_log_pos) == 0:
        return np.maximum(log_pred, 0.0)

    log_upper_cap = float(train_log_pos.quantile(0.999))
    return np.clip(log_pred, 0.0, log_upper_cap)


def hazard_from_heads(
    p_ignition: np.ndarray,
    log_burned_area: np.ndarray,
    train_df: Optional[pd.DataFrame] = None,
    clip_log: bool = True,
) -> np.ndarray:
    """Hazard score = P(ignition) × expm1(predicted log1p burned area)."""
    p_ignition = np.asarray(p_ignition, dtype=float)
    log_burned_area = np.asarray(log_burned_area, dtype=float)
    if clip_log:
        log_burned_area = clip_log1p_burned_area(log_burned_area, train_df)
    burned_area = np.maximum(np.expm1(log_burned_area), 0.0)
    return p_ignition * burned_area


def actual_hazard(ignition: np.ndarray, burned_area: np.ndarray) -> np.ndarray:
    """Ground-truth hazard = ignition × burned_area (pixel-count units)."""
    return np.asarray(ignition, dtype=float) * np.asarray(burned_area, dtype=float)


def evaluate_hazard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Hazard metrics on original scale plus log1p(hazard) where actual hazard > 0.

    log1p metrics use a fixed mask (y_true > 0) so all models are scored on the
    same fire days; zero or small predictions still count as errors.

    Returns keys: rmse, mae, r2, spearman, spearman_positive,
    log1p_hazard_rmse, log1p_hazard_mae, log1p_hazard_n.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    spearman_all, _ = spearmanr(y_true, y_pred)
    pos_mask = y_true > 0
    if pos_mask.sum() > 0:
        spearman_pos, _ = spearmanr(y_true[pos_mask], y_pred[pos_mask])
    else:
        spearman_pos = np.nan

    log_mask = y_true > 0
    if log_mask.sum() > 0:
        y_true_log = np.log1p(y_true[log_mask])
        y_pred_log = np.log1p(y_pred[log_mask])
        log1p_hazard_rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
        log1p_hazard_mae = float(mean_absolute_error(y_true_log, y_pred_log))
        log1p_hazard_n = int(log_mask.sum())
    else:
        log1p_hazard_rmse = None
        log1p_hazard_mae = None
        log1p_hazard_n = 0

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearman_all),
        "spearman_positive": float(spearman_pos),
        "log1p_hazard_rmse": log1p_hazard_rmse,
        "log1p_hazard_mae": log1p_hazard_mae,
        "log1p_hazard_n": log1p_hazard_n,
    }


def print_hazard_metrics(metrics: Dict[str, Any], title: str = "Combined Hazard") -> None:
    """Print hazard metrics with explicit labels."""
    if title:
        print(f"\n{title}:")
    print(f"  RMSE:                 {metrics['rmse']:.2f} hectares")
    print(f"  MAE:                  {metrics['mae']:.2f} hectares")
    print(f"  R²:                   {metrics['r2']:.4f}")
    print(f"  Spearman ρ (all):     {metrics['spearman']:.4f}")
    if not np.isnan(metrics["spearman_positive"]):
        print(f"  Spearman ρ (hazard>0): {metrics['spearman_positive']:.4f}")
    if metrics["log1p_hazard_rmse"] is not None:
        print(
            f"  log1p(hazard) RMSE:   {metrics['log1p_hazard_rmse']:.4f} "
            f"(n={metrics['log1p_hazard_n']}, actual hazard > 0)"
        )
        print(f"  log1p(hazard) MAE:    {metrics['log1p_hazard_mae']:.4f}")
    else:
        print("  log1p(hazard) RMSE/MAE: N/A (no samples with actual hazard > 0)")
