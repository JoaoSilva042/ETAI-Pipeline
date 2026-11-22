"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from category_encoders import CountEncoder



def clean_dataset(
    df: pd.DataFrame,
    diagnostics_config: dict,
) -> pd.DataFrame:

    out = df.copy()

    placeholder_tokens = set(
        diagnostics_config.get("placeholder_tokens", [])
    )

    normalized_placeholders = {
        str(token).strip().lower()
        for token in placeholder_tokens
    }

    
    for col in out.columns:
        normalized = (
            out[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        placeholder_mask = normalized.isin(
            normalized_placeholders
        )

        out.loc[placeholder_mask, col] = np.nan

    
    for col in diagnostics_config.get(
        "numeric_text_columns",
        [],
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    
    validity_rules = diagnostics_config.get(
        "validity_rules",
        {},
    )

    for col, rule in validity_rules.items():
        if col not in out.columns:
            continue

        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

        if "min" in rule:
            out.loc[
                out[col] < rule["min"],
                col,
            ] = np.nan

        if "max" in rule:
            out.loc[
                out[col] > rule["max"],
                col,
            ] = np.nan

    # Canonicalize categorical values.
    category_mappings = diagnostics_config.get(
        "canonical_categories",
        diagnostics_config.get(
            "category_mappings",
            {},
        ),
    )

    for col, mapping in category_mappings.items():
        if col not in out.columns:
            continue

        normalized = (
            out[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        mapped = normalized.map(mapping)

        out[col] = mapped.fillna(out[col])

    
    columns_to_drop = diagnostics_config.get(
        "redundant_columns",
        diagnostics_config.get(
            "columns_to_drop",
            [],
        ),
    )

    existing_columns_to_drop = [
        col
        for col in columns_to_drop
        if col in out.columns
    ]

    out = out.drop(
        columns=existing_columns_to_drop
    )

    return out



def drop_duplicate_rows(
    df: pd.DataFrame,
    id_column: str | None = None,
) -> pd.DataFrame:

    out = df.drop_duplicates()

    if id_column and id_column in out.columns:
        out = out.drop_duplicates(
            subset=id_column,
            keep="first",
        )

    return out



def split_features_target(
    df: pd.DataFrame,
    data_config: dict,
    mnar_indicator_sources: list,
):
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(
        df,
        mnar_indicator_sources
    )

    y = df[target] if target in df.columns else None

    extras_cols = [
        c
        for c in [sensitive_attr, "score_text"]
        if c in df.columns
    ]

    extras = (
        df[extras_cols].copy()
        if extras_cols
        else None
    )

    always_drop = set(drop_columns) | {
        target,
        sensitive_attr,
        "score_text",
    }

    feature_cols = [
        c
        for c in df.columns
        if c not in always_drop
    ]

    X = df[feature_cols]

    return X, y, extras

def add_missingness_indicators(
    df: pd.DataFrame,
    indicator_sources: list,
) -> pd.DataFrame:

    out = df.copy()

    for col in indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)

    return out

def split_dev_test(
    X: pd.DataFrame,
    y: pd.Series,
    extras: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X,
        y,
        extras,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
    )

_SCALERS = {
    "none": "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    ),

    "ordinal": lambda seed: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ),

    "count": lambda seed: CountEncoder(
        handle_unknown=0,
        handle_missing=0,
    ),

    "target": lambda seed: TargetEncoder(
        target_type="binary",
        cv=5,
        shuffle=True,
        random_state=seed,
    ),

}


_NUMERIC_IMPUTERS = {
    "median": lambda cfg, seed: SimpleImputer(strategy="median"),
    "mean": lambda cfg, seed: SimpleImputer(strategy="mean"),
    "knn": lambda cfg, seed: KNNImputer(n_neighbors=cfg.get("n_neighbors", 5)),
    "iterative": lambda cfg, seed: IterativeImputer(max_iter=cfg.get("max_iter", 10), random_state=seed),
    "random_forest": lambda cfg, seed: IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=cfg.get("n_estimators", 50), min_samples_leaf=5,
                                        random_state=seed, n_jobs=-1),
        max_iter=cfg.get("max_iter", 10), random_state=seed),
}

def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds the unfitted ColumnTransformer described by `preprocessing_config`: scaling and
    imputation for numeric columns (scaled first, so distance-based imputers compare columns
    on one scale; scalers ignore NaN when fitting), imputation and encoding for categorical
    ones, and the missingness flags passed through. Every encoder tolerates unseen categories.
    """
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))
    numeric_strategy = imputation.get("numeric_strategy", "median")
    if numeric_strategy not in _NUMERIC_IMPUTERS:
        raise ValueError(f"Unknown numeric_strategy: {numeric_strategy}. Options: {list(_NUMERIC_IMPUTERS)}")

    numeric_pipeline = Pipeline([
        ("scale", scaler),
        ("impute", _NUMERIC_IMPUTERS[numeric_strategy](imputation, preprocessing_config.get("random_state"))),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_cols),
    ])



