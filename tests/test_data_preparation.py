"""Tests for data_preparation."""

import numpy as np
import pandas as pd
import pytest

from src.data_preparation import DataPreparation


@pytest.fixture
def cleaned(base_config, synthetic_raw_df):
    return DataPreparation(base_config).clean_data(synthetic_raw_df)


def test_validation_report_counts_issues(base_config, synthetic_raw_df):
    report = DataPreparation(base_config).generate_validation_report(synthetic_raw_df)

    assert report["n_rows"] == len(synthetic_raw_df)
    assert report["missing_values_per_column"]["num_videos"] == synthetic_raw_df["num_videos"].isna().sum()
    assert report["n_negative_keyword_values"]["kw_min_min"] == (synthetic_raw_df["kw_min_min"] < 0).sum()
    assert report["n_zero_content_articles"] == 10
    assert report["target_summary"]["min"] <= report["target_summary"]["median"] <= report["target_summary"]["max"]


def test_no_missing_values_after_cleaning(cleaned):
    assert cleaned.isna().sum().sum() == 0


def test_missing_channel_becomes_other(base_config, synthetic_raw_df, cleaned):
    n_missing = synthetic_raw_df["data_channel"].isna().sum()
    assert (cleaned["data_channel"] == "other").sum() == n_missing


def test_missing_counts_filled_with_zero(synthetic_raw_df, cleaned):
    for col in DataPreparation.ZERO_FILL_COLUMNS:
        was_missing = synthetic_raw_df[col].isna()
        assert (cleaned.loc[was_missing, col] == 0).all(), col


def test_keyword_placeholders_clipped(cleaned):
    for col in DataPreparation.KEYWORD_SENTINEL_COLUMNS:
        assert (cleaned[col] >= 0).all(), col


def test_no_text_articles_kept_and_flagged(synthetic_raw_df, cleaned):
    assert len(cleaned) == len(synthetic_raw_df)
    assert cleaned["is_no_text"].sum() == 10
    assert (cleaned.loc[cleaned["is_no_text"] == 1, "n_tokens_content"] == 0).all()


def test_publish_date_extracted_from_url(cleaned):
    assert cleaned["pub_year"].between(2013, 2014).all()
    assert cleaned["pub_month"].between(1, 12).all()


def test_is_weekend_flag(cleaned):
    expected = cleaned["weekday"].isin(["saturday", "sunday"]).astype(int)
    pd.testing.assert_series_equal(cleaned["is_weekend"], expected, check_names=False)


def test_identifiers_dropped(cleaned):
    assert "ID" not in cleaned.columns
    assert "URL" not in cleaned.columns


def test_duplicates_dropped(base_config, synthetic_raw_df):
    df = pd.concat([synthetic_raw_df, synthetic_raw_df.iloc[:5]])
    assert len(DataPreparation(base_config).clean_data(df)) == len(synthetic_raw_df)


def test_input_not_mutated(base_config, synthetic_raw_df):
    original = synthetic_raw_df.copy()
    DataPreparation(base_config).clean_data(synthetic_raw_df)
    pd.testing.assert_frame_equal(synthetic_raw_df, original)


def test_empty_dataframe_raises(base_config):
    with pytest.raises(ValueError, match="empty"):
        DataPreparation(base_config).clean_data(pd.DataFrame())


def test_missing_required_column_raises(base_config, synthetic_raw_df):
    with pytest.raises(ValueError, match="URL"):
        DataPreparation(base_config).clean_data(synthetic_raw_df.drop(columns="URL"))


def test_preprocessor_output_shape_and_scaling(base_config, cleaned):
    preprocessor = DataPreparation(base_config).create_preprocessor()
    X = preprocessor.fit_transform(cleaned)

    n_numeric = len(base_config["numeric_features"])
    n_onehot = cleaned["data_channel"].nunique() + cleaned["weekday"].nunique()
    assert X.shape == (len(cleaned), n_numeric + n_onehot)
    # Scaled numeric columns have mean ~0 and std ~1.
    numeric_block = np.asarray(X[:, :n_numeric], dtype=float)
    np.testing.assert_allclose(numeric_block.mean(axis=0), 0, atol=1e-8)


def test_preprocessor_drops_unconfigured_columns(base_config, cleaned):
    preprocessor = DataPreparation(base_config).create_preprocessor().fit(cleaned)
    names = preprocessor.get_feature_names_out()
    assert not any("n_comments" in name for name in names)
    assert not any("pub_year" in name for name in names)


def test_preprocessor_ignores_unseen_category(base_config, cleaned):
    preprocessor = DataPreparation(base_config).create_preprocessor().fit(cleaned)
    unseen = cleaned.iloc[:1].copy()
    unseen["data_channel"] = "brand_new_channel"
    assert preprocessor.transform(unseen).shape[0] == 1
