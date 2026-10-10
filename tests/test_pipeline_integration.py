"""Tests for pipeline_integration."""

import numpy as np
import pytest

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

METRIC_KEYS = ("rmse", "mae", "r2", "mae_shares", "median_ae_shares")
ALL_MODELS = {
    "dummy_mean", "linear_regression",
    "ridge", "lasso", "decision_tree", "random_forest", "gradient_boosting",
}


@pytest.fixture
def splits(base_config, synthetic_raw_df):
    data_preparation = DataPreparation(base_config)
    df_cleaned = data_preparation.clean_data(synthetic_raw_df)
    model_training = ModelTraining(base_config, data_preparation.create_preprocessor)
    return model_training, df_cleaned, model_training.split_data(df_cleaned)


def test_split_sizes_and_log_target(splits, synthetic_raw_df):
    _, df_cleaned, (X_train, X_val, X_test, y_train, y_val, y_test) = splits

    assert len(X_train) + len(X_val) + len(X_test) == len(df_cleaned)
    assert "shares" not in X_train.columns
    np.testing.assert_allclose(np.expm1(y_train), df_cleaned.loc[y_train.index, "shares"])


def test_end_to_end_pipeline_runs(base_config, splits):
    model_training, _, (X_train, X_val, X_test, y_train, y_val, y_test) = splits

    baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(
        X_train, y_train, X_val, y_val
    )
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    assert set(all_models) == ALL_MODELS
    for metrics in all_metrics.values():
        assert set(METRIC_KEYS) <= set(metrics)
        assert metrics["rmse"] >= 0
    assert all("best_params" in tuned_metrics[name] for name in tuned_models)
    # The synthetic target depends on kw_avg_avg, so a linear model must beat the mean.
    assert all_metrics["linear_regression"]["rmse"] < all_metrics["dummy_mean"]["rmse"]

    best_name = min(all_metrics, key=lambda name: all_metrics[name]["rmse"])
    refit_model = model_training.refit_on_train_val(all_models[best_name], X_train, X_val, y_train, y_val)
    final_metrics = model_training.evaluate_final_model(refit_model, X_test, y_test, best_name)
    assert set(METRIC_KEYS) <= set(final_metrics)

    importance = model_training.explain_model(refit_model, X_test, y_test, best_name)
    assert set(importance) == {*base_config["numeric_features"], *base_config["nominal_features"]}


def test_missing_param_grid_raises(base_config, splits):
    model_training, _, (X_train, X_val, _, y_train, y_val, _) = splits
    del base_config["param_grids"]["lasso"]

    with pytest.raises(ValueError, match="lasso"):
        model_training.train_and_evaluate_tuned_models(X_train, y_train, X_val, y_val)
