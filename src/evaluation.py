"""Regression metrics and evaluation plots for fitted model pipelines."""

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use('Agg')  # render to files only, so the pipeline runs without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def compute_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, log_target: bool = True) -> Dict[str, float]:
    """
    Computes regression metrics for a fitted model.

    `rmse`, `mae` and `r2` are computed on the scale the model was trained on
    (log1p(shares) when `log_target` is true). When the target is logged, the
    errors are also reported in raw shares (`mae_shares`, `median_ae_shares`)
    after inverting with expm1, which is easier to interpret.

    Args:
        model (Pipeline): A fitted regressor pipeline.
        X (pd.DataFrame): Features.
        y (pd.Series): True target values, on the training scale.
        log_target (bool): Whether `y` is log1p(shares).
    Returns:
        Dict[str, float]: rmse, mae, r2, and (if `log_target`) mae_shares and median_ae_shares.
    """
    y_pred = model.predict(X)
    metrics = {
        'rmse': float(root_mean_squared_error(y, y_pred)),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2': float(r2_score(y, y_pred)),
    }
    if log_target:
        y_raw, y_pred_raw = np.expm1(y), np.expm1(y_pred)
        metrics['mae_shares'] = float(mean_absolute_error(y_raw, y_pred_raw))
        metrics['median_ae_shares'] = float(median_absolute_error(y_raw, y_pred_raw))
    return metrics


def plot_model_comparison(metrics: Dict[str, Dict[str, Any]], path: Path) -> None:
    """
    Plots validation RMSE and R² of every model side by side and saves the figure.

    Args:
        metrics (Dict[str, Dict[str, Any]]): Model name -> metrics from `compute_metrics`.
        path (Path): Output image path.
    """
    df = pd.DataFrame(metrics).T[['rmse', 'r2']].astype(float).sort_values('rmse')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    axes[0].barh(df.index, df['rmse'], color='tab:blue')
    axes[0].set_title('Validation RMSE (lower is better)')
    axes[1].barh(df.index, df['r2'], color='tab:green')
    axes[1].set_title('Validation R² (higher is better)')
    axes[1].axvline(0, color='black', linewidth=0.8)
    for ax, col in zip(axes, ['rmse', 'r2']):
        for i, value in enumerate(df[col]):
            ax.annotate(f'{value:.3f}', (value, i), xytext=(3, 0), textcoords='offset points',
                        va='center', fontsize=9)
    axes[0].invert_yaxis()
    _save_figure(fig, path)


def plot_predicted_vs_actual(model: Pipeline, X: pd.DataFrame, y: pd.Series, path: Path, title: str) -> None:
    """
    Plots predicted against actual target values, with the y = x line, and saves the figure.

    Args:
        model (Pipeline): A fitted regressor pipeline.
        X (pd.DataFrame): Features to evaluate on.
        y (pd.Series): True target values, on the training scale.
        path (Path): Output image path.
        title (str): Figure title.
    """
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, y_pred, s=6, alpha=0.3)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, color='black', linestyle='--', linewidth=1, label='perfect prediction')
    ax.set_xlabel('actual')
    ax.set_ylabel('predicted')
    ax.set_title(title)
    ax.legend(loc='upper left')
    _save_figure(fig, path)


def plot_residuals(model: Pipeline, X: pd.DataFrame, y: pd.Series, path: Path, title: str) -> None:
    """
    Plots residuals (actual - predicted) against predictions, plus their
    histogram, and saves the figure.

    Args:
        model (Pipeline): A fitted regressor pipeline.
        X (pd.DataFrame): Features to evaluate on.
        y (pd.Series): True target values, on the training scale.
        path (Path): Output image path.
        title (str): Figure title.
    """
    y_pred = model.predict(X)
    residuals = np.asarray(y) - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, s=6, alpha=0.3)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('predicted')
    axes[0].set_ylabel('residual (actual - predicted)')
    axes[0].set_title('Residuals vs predicted')
    axes[1].hist(residuals, bins=60)
    axes[1].set_xlabel('residual')
    axes[1].set_title('Residual distribution')
    fig.suptitle(title)
    _save_figure(fig, path)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure to %s", path)
