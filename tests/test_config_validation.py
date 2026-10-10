"""Tests for config_validation."""

import pytest

from src.config_validation import validate_config


def test_valid_config_passes(base_config):
    validate_config(base_config)


def test_missing_key_raises(base_config):
    del base_config["log_target"]
    with pytest.raises(ValueError, match="log_target"):
        validate_config(base_config)


def test_wrong_type_raises(base_config):
    base_config["cv"] = "5"
    with pytest.raises(ValueError, match="'cv' must be of type int"):
        validate_config(base_config)


@pytest.mark.parametrize("key", ["val_test_size", "validation_size"])
@pytest.mark.parametrize("value", [0.0, 1.0, 1.5])
def test_split_size_out_of_range_raises(base_config, key, value):
    base_config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(base_config)


def test_cv_below_two_raises(base_config):
    base_config["cv"] = 1
    with pytest.raises(ValueError, match="'cv' must be >= 2"):
        validate_config(base_config)


def test_classification_scoring_rejected(base_config):
    base_config["scoring"] = "roc_auc"
    with pytest.raises(ValueError, match="scoring"):
        validate_config(base_config)


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_invalid_n_jobs_raises(base_config, n_jobs):
    base_config["n_jobs"] = n_jobs
    with pytest.raises(ValueError, match="n_jobs"):
        validate_config(base_config)


def test_feature_in_two_roles_raises(base_config):
    base_config["nominal_features"].append("num_imgs")
    with pytest.raises(ValueError, match="num_imgs"):
        validate_config(base_config)


def test_log_feature_must_be_numeric(base_config):
    base_config["log_features"].append("data_channel")
    with pytest.raises(ValueError, match="subset of 'numeric_features'"):
        validate_config(base_config)


def test_target_as_feature_raises(base_config):
    base_config["numeric_features"].append("shares")
    with pytest.raises(ValueError, match="Target column"):
        validate_config(base_config)


def test_param_grid_must_be_dict(base_config):
    base_config["param_grids"]["ridge"] = [1.0]
    with pytest.raises(ValueError, match="param_grids.ridge"):
        validate_config(base_config)
