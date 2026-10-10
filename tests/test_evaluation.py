"""Tests for evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from src.evaluation import compute_metrics, plot_model_comparison, plot_predicted_vs_actual, plot_residuals


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 10, 100)})
    y = pd.Series(2 + 0.5 * X["x"])
    return X, y


def test_perfect_predictor(linear_data):
    X, y = linear_data
    metrics = compute_metrics(LinearRegression().fit(X, y), X, y)
    assert metrics["rmse"] == pytest.approx(0, abs=1e-9)
    assert metrics["mae"] == pytest.approx(0, abs=1e-9)
    assert metrics["r2"] == pytest.approx(1)
    assert metrics["mae_shares"] == pytest.approx(0, abs=1e-6)


def test_mean_predictor_has_zero_r2(linear_data):
    X, y = linear_data
    metrics = compute_metrics(DummyRegressor().fit(X, y), X, y)
    assert metrics["r2"] == pytest.approx(0, abs=1e-9)
    assert metrics["rmse"] == pytest.approx(y.std(ddof=0))


def test_raw_scale_metrics_only_with_log_target(linear_data):
    X, y = linear_data
    metrics = compute_metrics(LinearRegression().fit(X, y), X, y, log_target=False)
    assert set(metrics) == {"rmse", "mae", "r2"}


def test_plots_are_written(tmp_path, linear_data):
    X, y = linear_data
    model = LinearRegression().fit(X, y)
    metrics = {"a": compute_metrics(model, X, y), "b": {"rmse": 1.0, "r2": 0.1, "best_params": {}}}

    plot_model_comparison(metrics, tmp_path / "comparison.png")
    plot_predicted_vs_actual(model, X, y, tmp_path / "pred.png", title="t")
    plot_residuals(model, X, y, tmp_path / "res.png", title="t")

    for name in ("comparison.png", "pred.png", "res.png"):
        assert (tmp_path / name).stat().st_size > 0
