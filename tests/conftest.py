"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CHANNELS = ["world", "technology", "entertainment", "business", "social_media", "lifestyle"]


@pytest.fixture
def base_config():
    """A valid pipeline config with small grids, shared across test modules."""
    return {
        "file_path": "./data/mini_project_1_data.csv",
        "target_column": "shares",
        "log_target": True,
        "val_test_size": 0.3,
        "validation_size": 0.5,
        "random_state": 42,
        "cv": 2,
        "scoring": "neg_root_mean_squared_error",
        "n_jobs": 1,
        "numeric_features": [
            "kw_avg_min", "kw_avg_max", "kw_avg_avg", "self_reference_avg_shares",
            "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos", "num_keywords",
            "n_tokens_content", "n_tokens_title", "n_non_stop_unique_tokens",
            "average_token_length", "timedelta", "pub_month", "is_weekend", "is_no_text",
        ],
        "log_features": [
            "kw_avg_min", "kw_avg_max", "kw_avg_avg", "self_reference_avg_shares",
            "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos", "n_tokens_content",
        ],
        "nominal_features": ["data_channel", "weekday"],
        "param_grids": {
            "ridge": {"regressor__alpha": [1.0]},
            "lasso": {"regressor__alpha": [0.01]},
            "decision_tree": {"regressor__max_depth": [3]},
            "random_forest": {"regressor__n_estimators": [10], "regressor__max_depth": [3]},
            "gradient_boosting": {"regressor__n_estimators": [10], "regressor__max_depth": [2]},
        },
    }


@pytest.fixture
def synthetic_raw_df():
    """
    ~200 raw rows shaped like mini_project_1_data.csv, including the known data
    issues (missing values, -1 keyword placeholders, articles with no text),
    with a target that depends on `kw_avg_avg` so models can learn something.
    """
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.Timestamp("2013-01-07") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    n_tokens_content = rng.integers(50, 2000, n).astype(float)
    n_tokens_content[:10] = 0  # articles without text
    kw_avg_avg = rng.uniform(1000, 6000, n)
    self_ref = rng.uniform(0, 10000, n)

    df = pd.DataFrame({
        "ID": np.arange(n),
        "URL": [f"http://mashable.com/{d:%Y/%m/%d}/article-{i}/" for i, d in enumerate(dates)],
        "timedelta": (pd.Timestamp("2015-01-04") - dates).days.astype(float),
        "weekday": rng.choice(WEEKDAYS, n),
        "n_tokens_title": rng.integers(5, 15, n).astype(float),
        "n_tokens_content": n_tokens_content,
        "n_unique_tokens": np.where(n_tokens_content == 0, 0, rng.uniform(0.4, 0.7, n)),
        "n_non_stop_words": np.where(n_tokens_content == 0, 0, 1.0),
        "n_non_stop_unique_tokens": np.where(n_tokens_content == 0, 0, rng.uniform(0.5, 0.8, n)),
        "num_hrefs": rng.integers(0, 30, n).astype(float),
        "num_self_hrefs": rng.integers(0, 10, n).astype(float),
        "num_imgs": rng.integers(0, 20, n).astype(float),
        "num_videos": rng.integers(0, 3, n).astype(float),
        "n_comments": rng.integers(0, 200, n).astype(float),
        "average_token_length": np.where(n_tokens_content == 0, 0, rng.uniform(4, 5.5, n)),
        "data_channel": rng.choice(CHANNELS, n).astype(object),
        "self_reference_min_shares": self_ref * 0.5,
        "self_reference_max_shares": self_ref * 2,
        "self_reference_avg_shares": self_ref,
        "num_keywords": rng.integers(3, 11, n).astype(float),
        "kw_min_min": rng.choice([-1.0, 4.0, 217.0], n),
        "kw_max_min": rng.uniform(0, 3000, n),
        "kw_avg_min": np.where(rng.random(n) < 0.05, -1.0, rng.uniform(0, 1000, n)),
        "kw_min_max": rng.uniform(0, 10000, n),
        "kw_max_max": np.full(n, 843300.0),
        "kw_avg_max": rng.uniform(1e5, 8e5, n),
        "kw_min_avg": np.where(rng.random(n) < 0.05, -1.0, rng.uniform(0, 3000, n)),
        "kw_max_avg": kw_avg_avg * 2,
        "kw_avg_avg": kw_avg_avg,
    })
    df["shares"] = np.round(np.exp(5 + kw_avg_avg / 2000 + rng.normal(0, 0.5, n))).astype(int)

    # Missing values in the same pattern as the real data.
    df.loc[rng.random(n) < 0.4, "num_videos"] = np.nan
    df.loc[rng.random(n) < 0.05, "num_imgs"] = np.nan
    df.loc[rng.random(n) < 0.15, "data_channel"] = np.nan
    link_gap = rng.random(n) < 0.05
    df.loc[link_gap, ["num_hrefs", "num_self_hrefs", "self_reference_min_shares",
                      "self_reference_max_shares", "self_reference_avg_shares"]] = np.nan
    return df
