"""Build, tune and train regression model pipelines."""

import logging
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)


class ModelTraining:
    """
    Class for regression model training and evaluation.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for model training.
        preprocessor_factory (Callable[[], ColumnTransformer]): Builds a fresh,
            unfitted preprocessor. Called once per model pipeline so every
            pipeline owns its own independent preprocessor instance.
    """

    TUNED_MODEL_NAMES = ('ridge', 'lasso', 'decision_tree', 'random_forest', 'gradient_boosting')

    def __init__(self, config: Dict[str, Any], preprocessor_factory: Callable[[], ColumnTransformer]):
        """
        Initializes the ModelTraining class with the given configuration and preprocessor factory.

        Args:
            config (Dict[str, Any]): Configuration dictionary for model training.
            preprocessor_factory (Callable[[], ColumnTransformer]): Callable that
                returns a new, unfitted ColumnTransformer each time it's called.
        """
        self.config = config
        self.preprocessor_factory = preprocessor_factory

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Separates the target (log1p-transformed if `log_target` is set) and
        splits the data into training, validation, and testing sets.

        Args:
            df (pd.DataFrame): Input DataFrame containing cleaned data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: Training, validation, and testing sets for features and target variable.
        """
        logger.info("Starting data splitting process.")
        target = self.config['target_column']
        X = df.drop(columns=[target])
        y = df[target].astype(float)
        if self.config['log_target']:
            y = np.log1p(y)
            logger.info("Target transformed with log1p.")

        random_state = self.config['random_state']
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config['val_test_size'], random_state=random_state,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config['validation_size'], random_state=random_state,
        )

        for split_name, y_split in (('train', y_train), ('validation', y_val), ('test', y_test)):
            logger.info("%s set: %d rows, target mean %.3f", split_name, len(y_split), y_split.mean())
        logger.info("Data splitting process completed.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        """
        Trains and evaluates the untuned models: a mean-predicting dummy
        regressor, which sets the floor every model must beat, and ordinary
        Linear Regression, which has no hyperparameters to tune.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]: Trained baseline models and their evaluation metrics on the validation set.
        """
        logger.info("Starting baseline model training and evaluation.")
        models = {
            'dummy_mean': DummyRegressor(strategy='mean'),
            'linear_regression': LinearRegression(),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            logger.info("Training %s.", model_name)
            pipeline = self._build_pipeline(model)
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline

            logger.info("Evaluating %s on validation set.", model_name)
            metrics[model_name] = self._evaluate_model(pipeline, X_val, y_val, model_name)

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        """
        Tunes Ridge, Lasso, Decision Tree, Random Forest and Gradient Boosting
        regressors with GridSearchCV over the per-model grids in the config,
        then evaluates each best estimator on the validation set.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]: Trained tuned models and their evaluation metrics on the validation set.
        Raises:
            ValueError: If a model has no entry in the config's `param_grids`.
        """
        logger.info("Starting tuned model training and evaluation.")
        param_grids = self.config['param_grids']
        missing_grids = [name for name in self.TUNED_MODEL_NAMES if name not in param_grids]
        if missing_grids:
            raise ValueError(f"Config 'param_grids' is missing an entry for model(s): {missing_grids}.")

        random_state = self.config['random_state']
        models = {
            'ridge': Ridge(),
            'lasso': Lasso(max_iter=10000),
            'decision_tree': DecisionTreeRegressor(random_state=random_state),
            'random_forest': RandomForestRegressor(random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=random_state),
        }
        tuned_models = {}
        tuned_metrics = {}

        for model_name, model in models.items():
            logger.info("Training %s with hyperparameter tuning.", model_name)
            grid_search = GridSearchCV(
                estimator=self._build_pipeline(model),
                param_grid=param_grids[model_name],
                cv=self.config['cv'],
                scoring=self.config['scoring'],
                n_jobs=self.config['n_jobs'],
            )
            grid_search.fit(X_train, y_train)
            logger.info(
                "%s best CV %s: %.4f with %s",
                model_name, self.config['scoring'], grid_search.best_score_, grid_search.best_params_,
            )

            tuned_models[model_name] = grid_search.best_estimator_
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name + " (tuned)",
            )
            tuned_metrics[model_name]['best_params'] = grid_search.best_params_
            tuned_metrics[model_name]['best_cv_score'] = float(grid_search.best_score_)

        logger.info("Tuned model training and evaluation completed.")
        return tuned_models, tuned_metrics

    def refit_on_train_val(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> Pipeline:
        """
        Refits a clone of the selected model on the combined training and
        validation data, so the final test evaluation reflects a model trained
        on as much data as possible once model/hyperparameter selection is finished.

        Args:
            model (Pipeline): The selected model pipeline (fitted on X_train only).
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            y_train (pd.Series): Training target variable.
            y_val (pd.Series): Validation target variable.
        Returns:
            Pipeline: A new pipeline, refit on the combined train+val data.
        """
        logger.info("Refitting selected model on combined train+val data.")
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        refit_model = clone(model)
        refit_model.fit(X_train_val, y_train_val)
        return refit_model

    def evaluate_final_model(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluates the final selected model on the testing set.

        Args:
            model (Pipeline): The final selected model pipeline.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target variable.
            model_name (str): Name of the model being evaluated.
        Returns:
            Dict[str, Any]: Evaluation metrics for the final model on the testing set.
        """
        logger.info("Evaluating final model on testing set.")
        return self._evaluate_model(model, X_test, y_test, model_name)

    def explain_model(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimates feature importance by permutation: each configured input
        feature is shuffled in turn and the increase in RMSE is measured. Works
        for every model type and reports importance per original feature
        rather than per one-hot column, so it can be compared directly with
        the EDA's feature ranking.

        Args:
            model (Pipeline): The final fitted model pipeline.
            X (pd.DataFrame): Held-out features to measure importance on.
            y (pd.Series): Held-out target variable.
            model_name (str): Name of the model being explained.
        Returns:
            Dict[str, Dict[str, float]]: Feature name -> {'mean', 'std'} of the
                RMSE increase, sorted by descending mean.
        """
        features = [*self.config['numeric_features'], *self.config['nominal_features']]
        result = permutation_importance(
            model, X[features], y,
            scoring='neg_root_mean_squared_error',
            n_repeats=5,
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs'],
        )
        importance = {
            feature: {'mean': float(mean), 'std': float(std)}
            for feature, mean, std in zip(features, result.importances_mean, result.importances_std)
        }
        importance = dict(sorted(importance.items(), key=lambda item: item[1]['mean'], reverse=True))
        logger.info(
            "Permutation importance for %s: %s",
            model_name, {feature: round(values['mean'], 4) for feature, values in importance.items()},
        )
        return importance

    def _build_pipeline(self, model: Any) -> Pipeline:
        """Wraps a regressor with its own freshly built preprocessor."""
        return Pipeline(steps=[('preprocessor', self.preprocessor_factory()), ('regressor', model)])

    def _evaluate_model(self, model: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """
        Evaluates a given model and calculates regression metrics.

        Args:
            model (Pipeline): The model pipeline to be evaluated.
            X (pd.DataFrame): Evaluation features.
            y (pd.Series): Evaluation target variable.
            model_name (str): Name of the model being evaluated.
        Returns:
            Dict[str, Any]: RMSE, MAE and R² (plus raw-share errors when the target is logged).
        """
        metrics = compute_metrics(model, X, y, log_target=self.config['log_target'])
        logger.info(
            "Evaluation metrics for %s: %s", model_name, {k: round(v, 4) for k, v in metrics.items()},
        )
        if metrics['r2'] <= 0 and not model_name.startswith('dummy'):
            logger.warning(
                "%s has R² %.4f, i.e. it does no better than predicting the mean.", model_name, metrics['r2'],
            )
        return metrics
