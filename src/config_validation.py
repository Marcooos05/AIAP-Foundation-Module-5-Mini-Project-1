"""Validation for the pipeline's YAML configuration."""

from typing import Any, Dict

REQUIRED_KEYS = {
    "file_path": str,
    "target_column": str,
    "log_target": bool,
    "val_test_size": float,
    "validation_size": float,
    "random_state": int,
    "param_grids": dict,
    "cv": int,
    "scoring": str,
    "n_jobs": int,
    "numeric_features": list,
    "log_features": list,
    "nominal_features": list,
}

FEATURE_ROLE_KEYS = ("numeric_features", "nominal_features")

# `scoring` is passed to GridSearchCV. Model selection always uses validation RMSE.
SUPPORTED_SCORING = (
    "neg_root_mean_squared_error",
    "neg_mean_squared_error",
    "neg_mean_absolute_error",
    "r2",
)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the pipeline configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
    Raises:
        ValueError: If a required key is missing, has the wrong type, or an
            out-of-range value, or if the feature roles are inconsistent.
    """
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required key(s): {missing}")

    for key, expected_type in REQUIRED_KEYS.items():
        value = config[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Config key '{key}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )

    for key in ("val_test_size", "validation_size"):
        value = config[key]
        if not 0.0 < value < 1.0:
            raise ValueError(f"Config key '{key}' must be in (0, 1), got {value}.")

    if config["cv"] < 2:
        raise ValueError(f"Config key 'cv' must be >= 2, got {config['cv']}.")

    if config["scoring"] not in SUPPORTED_SCORING:
        raise ValueError(
            f"Config key 'scoring' must be one of {list(SUPPORTED_SCORING)}, got '{config['scoring']}'."
        )

    if config["n_jobs"] == 0 or config["n_jobs"] < -1:
        raise ValueError(f"Config key 'n_jobs' must be -1 (all cores) or >= 1, got {config['n_jobs']}.")

    for key in (*FEATURE_ROLE_KEYS, "log_features"):
        if not all(isinstance(item, str) for item in config[key]):
            raise ValueError(f"Config key '{key}' must be a list of strings.")

    if not config["numeric_features"] and not config["nominal_features"]:
        raise ValueError("Config must list at least one feature in 'numeric_features' or 'nominal_features'.")

    seen_roles: Dict[str, str] = {}
    for key in FEATURE_ROLE_KEYS:
        for feature in config[key]:
            if feature in seen_roles:
                raise ValueError(
                    f"Feature '{feature}' is listed in both '{seen_roles[feature]}' and '{key}'."
                )
            seen_roles[feature] = key

    not_numeric = [f for f in config["log_features"] if f not in config["numeric_features"]]
    if not_numeric:
        raise ValueError(f"'log_features' must be a subset of 'numeric_features'; not numeric: {not_numeric}.")

    if config["target_column"] in seen_roles:
        raise ValueError(f"Target column '{config['target_column']}' cannot also be a feature.")

    for model_name, grid in config["param_grids"].items():
        if not isinstance(grid, dict):
            raise ValueError(
                f"Config key 'param_grids.{model_name}' must be a dict of hyperparameter lists."
            )
