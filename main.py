"""Entry point: run the end-to-end ML pipeline."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.config_validation import validate_config
from src.data_preparation import DataPreparation
from src.evaluation import plot_model_comparison, plot_predicted_vs_actual, plot_residuals
from src.model_training import ModelTraining

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path('./src/config.yaml')
MODELS_DIR = Path('models')
REPORTS_DIR = Path('reports')
FIGURES_DIR = REPORTS_DIR / 'figures'

# Created during cleaning, so they are not expected in the raw CSV.
ENGINEERED_FEATURES = {'is_no_text', 'is_weekend', 'pub_year', 'pub_month'}


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Loads and validates the pipeline YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at '{config_path}'.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def _load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Loads the raw dataset referenced by the config and checks required columns exist."""
    file_path = Path(config['file_path'])
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at '{file_path}'.")

    logger.info("Loading data from CSV file.")
    df = pd.read_csv(file_path)
    logger.info("Data loading completed: %d rows, %d columns.", *df.shape)

    required_columns = {
        config['target_column'],
        *DataPreparation.REQUIRED_RAW_COLUMNS,
        *(set(config['numeric_features']) - ENGINEERED_FEATURES),
        *config['nominal_features'],
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Data file is missing configured column(s): {sorted(missing_columns)}.")

    return df


def _select_best_model(
    all_models: Dict[str, Pipeline],
    all_metrics: Dict[str, Dict[str, Any]],
) -> Tuple[str, Pipeline]:
    """Selects the model with the lowest validation RMSE."""
    best_model_name = min(all_metrics, key=lambda name: all_metrics[name]['rmse'])
    best_model = all_models[best_model_name]
    logger.info(
        "Best model: %s with validation RMSE %.4f (R² %.4f)",
        best_model_name, all_metrics[best_model_name]['rmse'], all_metrics[best_model_name]['r2'],
    )
    return best_model_name, best_model


def _persist_artifacts(
    refit_model: Pipeline,
    final_metrics: Dict[str, Any],
    best_model_name: str,
    best_model_metrics: Dict[str, Any],
    feature_importance: Dict[str, Dict[str, float]],
) -> None:
    """Saves the final model and its accompanying reports to disk."""
    joblib.dump(refit_model, MODELS_DIR / 'best_pipeline.joblib')
    _save_json(final_metrics, REPORTS_DIR / 'final_metrics.json')
    _save_json(
        {
            'model_name': best_model_name,
            'hyperparameters': best_model_metrics.get('best_params', {}),
        },
        REPORTS_DIR / 'selected_hyperparameters.json',
    )
    _save_json(feature_importance, REPORTS_DIR / 'feature_importance.json')
    logger.info("Model saved to %s, reports saved to %s", MODELS_DIR.resolve(), REPORTS_DIR.resolve())


def main():
    """
    Runs the end-to-end regression pipeline:

    1. Load and validate the YAML config.
    2. Load the raw dataset and verify required columns are present.
    3. Generate a data validation report, then clean the data and engineer features.
    4. Split into train/validation/test sets (target log1p-transformed).
    5. Train and evaluate the baseline models (dummy mean, Linear Regression),
       then tune Ridge, Lasso, Decision Tree, Random Forest and Gradient
       Boosting with GridSearchCV, and plot every model's validation scores.
    6. Select the best model by validation RMSE.
    7. Refit the best model on train+validation data and evaluate it on the
       test set (metrics, predicted-vs-actual and residual plots).
    8. Compute permutation feature importance for the final model.
    9. Persist the final model and all reports to disk.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config = _load_config(CONFIG_PATH)
    df = _load_data(config)

    # Data preparation
    data_preparation = DataPreparation(config)

    start = time.perf_counter()
    validation_report = data_preparation.generate_validation_report(df)
    _save_json(validation_report, REPORTS_DIR / 'data_validation_report.json')
    logger.info("Data validation completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    df_cleaned = data_preparation.clean_data(df)
    logger.info("Data cleaning completed in %.2fs.", time.perf_counter() - start)

    # Model training and evaluation
    model_training = ModelTraining(config, data_preparation.create_preprocessor)

    start = time.perf_counter()
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(df_cleaned)
    logger.info("Data splitting completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(
        X_train, y_train, X_val, y_val
    )
    logger.info("Baseline model training completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )
    logger.info("Tuned model training completed in %.2fs.", time.perf_counter() - start)

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}
    _save_json(all_metrics, REPORTS_DIR / 'validation_metrics.json')
    plot_model_comparison(all_metrics, FIGURES_DIR / 'model_comparison_validation.png')

    best_model_name, best_model = _select_best_model(all_models, all_metrics)

    start = time.perf_counter()
    refit_model = model_training.refit_on_train_val(best_model, X_train, X_val, y_train, y_val)
    final_metrics = model_training.evaluate_final_model(refit_model, X_test, y_test, best_model_name)
    logger.info("Final refit and evaluation completed in %.2fs.", time.perf_counter() - start)

    final_metrics = {
        'model_name': best_model_name,
        'target_scale': 'log1p(shares)' if config['log_target'] else 'shares',
        **final_metrics,
        'dummy_mean_test': model_training.evaluate_final_model(
            baseline_models['dummy_mean'], X_test, y_test, 'dummy_mean',
        ),
    }
    plot_predicted_vs_actual(
        refit_model, X_test, y_test, FIGURES_DIR / 'predicted_vs_actual_test.png',
        title=f'Predicted vs actual (test set): {best_model_name}',
    )
    plot_residuals(
        refit_model, X_test, y_test, FIGURES_DIR / 'residuals_test.png',
        title=f'Residuals (test set): {best_model_name}',
    )

    start = time.perf_counter()
    feature_importance = model_training.explain_model(refit_model, X_test, y_test, best_model_name)
    logger.info("Permutation importance completed in %.2fs.", time.perf_counter() - start)

    _persist_artifacts(
        refit_model, final_metrics, best_model_name, all_metrics[best_model_name], feature_importance,
    )


if __name__ == "__main__":
    main()
