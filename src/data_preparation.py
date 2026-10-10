"""Data validation, cleaning, feature engineering and preprocessing for online news shares."""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Class for data validation, cleaning, and feature engineering of the online
    news dataset, following the decisions made in eda.ipynb.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for data preparation.
    """

    REQUIRED_RAW_COLUMNS = ('URL', 'weekday', 'data_channel', 'n_tokens_content')

    # EDA §3.2: these gaps mean "none present" (no videos / images / links), so fill with 0.
    ZERO_FILL_COLUMNS = (
        'num_videos', 'num_imgs', 'num_hrefs', 'num_self_hrefs',
        'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_shares',
    )

    # EDA §3.3: share counts cannot be negative; -1 is a placeholder for "unknown".
    KEYWORD_SENTINEL_COLUMNS = ('kw_min_min', 'kw_avg_min', 'kw_min_avg')

    # Identifiers: unique per article, cannot generalise (EDA §3.6).
    COLUMNS_TO_DROP = ('ID', 'URL')

    WEEKEND_DAYS = ('saturday', 'sunday')

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for data preparation.
        """
        self.config = config

    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes data quality statistics on the raw DataFrame before cleaning,
        so every run is auditable.

        Args:
            df (pd.DataFrame): Raw input DataFrame.
        Returns:
            Dict[str, Any]: Row/column counts, missing-value counts, duplicate
                count, counts of each known data-quality issue, and a summary of
                the target distribution.
        """
        report = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "missing_values_per_column": {
                col: int(count) for col, count in df.isna().sum().items() if count > 0
            },
            "n_duplicate_rows": int(df.duplicated().sum()),
        }
        report["n_negative_keyword_values"] = {
            col: int((df[col] < 0).sum()) for col in self.KEYWORD_SENTINEL_COLUMNS if col in df.columns
        }
        if "n_tokens_content" in df.columns:
            report["n_zero_content_articles"] = int((df["n_tokens_content"] == 0).sum())
        target = self.config.get("target_column")
        if target in df.columns:
            report["target_summary"] = {
                "min": float(df[target].min()),
                "median": float(df[target].median()),
                "mean": float(df[target].mean()),
                "max": float(df[target].max()),
                "skew": float(df[target].skew()),
            }

        logger.info("Data validation report: %s", report)
        return report

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw news articles and engineers model features.

        Steps (see eda.ipynb for the evidence behind each one):
            1. Drop exact duplicate rows and rows without a target.
            2. Fill missing `data_channel` with a new level, 'other' (blog pages).
            3. Fill missing video / image / link / self-reference counts with 0.
            4. Clip the -1 placeholder in the keyword columns at 0.
            5. Flag articles without text (`is_no_text`).
            6. Extract `pub_year` / `pub_month` from the date in the URL.
            7. Flag weekend publication (`is_weekend`).
            8. Drop identifier columns.

        Args:
            df (pd.DataFrame): Input DataFrame containing raw data.
        Returns:
            pd.DataFrame: The cleaned DataFrame (a new object; the input is not modified).
        Raises:
            ValueError: If the DataFrame is empty or is missing a required raw column.
        """
        if df.empty:
            raise ValueError("Cannot clean an empty DataFrame.")

        missing_columns = [col for col in self.REQUIRED_RAW_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Data is missing required raw column(s): {missing_columns}.")

        logger.info("Starting data cleaning process. Rows before cleaning: %d", len(df))

        # EDA §3.1: no duplicates in the current data, but guard against them.
        n_duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
        logger.info("Removed %d duplicate row(s).", n_duplicates)

        target = self.config['target_column']
        if target in df.columns:
            n_missing_target = int(df[target].isna().sum())
            if n_missing_target:
                df = df.dropna(subset=[target])
                logger.warning("Dropped %d row(s) with a missing '%s'.", n_missing_target, target)

        # EDA §3.2: uncategorised articles are blog pages and share more than average.
        n_missing_channel = int(df['data_channel'].isna().sum())
        df['data_channel'] = df['data_channel'].fillna('other')
        logger.info("Filled %d missing 'data_channel' value(s) with 'other'.", n_missing_channel)

        # EDA §3.2: 721 rows miss all link columns together; no links means no self-reference shares.
        zero_fill = [col for col in self.ZERO_FILL_COLUMNS if col in df.columns]
        n_filled = {col: int(df[col].isna().sum()) for col in zero_fill}
        df[zero_fill] = df[zero_fill].fillna(0)
        logger.info("Filled missing values with 0: %s", n_filled)

        # EDA §3.3: -1 marks unknown keyword statistics in more recent articles.
        sentinel_cols = [col for col in self.KEYWORD_SENTINEL_COLUMNS if col in df.columns]
        n_negative = {col: int((df[col] < 0).sum()) for col in sentinel_cols}
        df[sentinel_cols] = df[sentinel_cols].clip(lower=0)
        logger.info("Clipped negative keyword placeholder value(s) at 0: %s", n_negative)

        # EDA §3.4: articles with no text are kept (image/video posts); their text ratios
        # are 0 by construction, so a flag lets models separate them.
        df['is_no_text'] = (df['n_tokens_content'] == 0).astype(int)

        # EDA §3.5: every URL holds the publish date as /YYYY/MM/DD/.
        pub_date = pd.to_datetime(
            df['URL'].str.extract(r'/(\d{4}/\d{2}/\d{2})/')[0], format='%Y/%m/%d', errors='coerce',
        )
        n_unparsed = int(pub_date.isna().sum())
        if n_unparsed:
            logger.warning("%d URL(s) had no parseable publish date; pub_year/pub_month left missing.", n_unparsed)
        df['pub_year'] = pub_date.dt.year
        df['pub_month'] = pub_date.dt.month

        df['is_weekend'] = df['weekday'].isin(self.WEEKEND_DAYS).astype(int)

        df = df.drop(columns=[col for col in self.COLUMNS_TO_DROP if col in df.columns])
        logger.info("Data cleaning process completed. Rows after cleaning: %d", len(df))
        return df

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Builds a new preprocessor for numeric and nominal features. Called once
        per model pipeline so each pipeline owns an independent, unfitted
        preprocessor that is only ever fitted on training data. Columns not
        listed in the config are dropped, so features excluded in the EDA (and
        `n_comments`, excluded as leakage) never reach the model.

        - `log_features`: log1p, then standardise (heavy right tails, EDA §4.2).
        - other numeric features: standardise.
        - nominal features: one-hot encode; levels unseen in training are ignored.

        Returns:
            ColumnTransformer: A preprocessor for data transformation.
        """
        log_features = list(self.config['log_features'])
        scaled_features = [f for f in self.config['numeric_features'] if f not in log_features]
        nominal_features = list(self.config['nominal_features'])

        log_transformer = Pipeline(steps=[
            ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
            ('scaler', StandardScaler()),
        ])
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
        ])
        nominal_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])

        transformers = []
        if log_features:
            transformers.append(('log', log_transformer, log_features))
        if scaled_features:
            transformers.append(('num', numeric_transformer, scaled_features))
        if nominal_features:
            transformers.append(('cat', nominal_transformer, nominal_features))

        return ColumnTransformer(transformers=transformers, remainder='drop')
