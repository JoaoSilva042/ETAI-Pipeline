import numpy as np
import pandas as pd 
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V effect size for a chi-square test of association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1)))


def test_missingness_mechanism(
    df: pd.DataFrame,
    column: str,
    comparison_columns: list,
    placeholder_tokens: list,
) -> pd.DataFrame:

    normalized_tokens = {
        str(token).strip().lower()
        for token in placeholder_tokens
    }

    normalized = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    missing_indicator = (
        df[column].isna()
        | normalized.isin(normalized_tokens)
    )

    results = []

    for comparison_col in comparison_columns:

        if comparison_col not in df.columns:
            continue

        contingency_table = pd.crosstab(
            missing_indicator,
            df[comparison_col]
        )

        chi2, p_value, _, _ = chi2_contingency(
            contingency_table
        )

        association = cramers_v(
            contingency_table
        )

        results.append({
            "missing_column": column,
            "comparison_column": comparison_col,
            "p_value": p_value,
            "cramers_v": association,
        })

    return pd.DataFrame(results)

def domain_rule_report(
    df: pd.DataFrame,
    validity_rules: dict,
) -> pd.DataFrame:

    results = []

    for col, rule in validity_rules.items():

        if col not in df.columns:
            continue

        numeric_values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        invalid_mask = pd.Series(
            False,
            index=df.index,
        )

        if "min" in rule:
            invalid_mask |= numeric_values < rule["min"]

        if "max" in rule:
            invalid_mask |= numeric_values > rule["max"]

        results.append({
            "column": col,
            "invalid_count": invalid_mask.sum(),
            "invalid_percentage": invalid_mask.mean() * 100,
        })

    return pd.DataFrame(results)




def duplicate_report(
    df: pd.DataFrame,
    id_column: str | None = None,
) -> dict:

    duplicate_rows = df.duplicated().sum()

    duplicated_ids = None

    if id_column and id_column in df.columns:
        duplicated_ids = df[id_column].duplicated().sum()

    return {
        "duplicate_rows": duplicate_rows,
        "duplicated_ids": duplicated_ids,
    }



def correlation_report(
    df: pd.DataFrame,
    columns: list,
    threshold: float = 0.8,
) -> pd.DataFrame:

    existing_columns = [
        col
        for col in columns
        if col in df.columns
    ]

    numeric_df = df[existing_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation_matrix = numeric_df.corr()

    results = []

    for i, col1 in enumerate(existing_columns):
        for col2 in existing_columns[i + 1:]:

            correlation = correlation_matrix.loc[
                col1,
                col2
            ]

            if abs(correlation) >= threshold:
                results.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": correlation,
                })

    return pd.DataFrame(results)



def vif_report(
    df: pd.DataFrame,
    columns: list,
) -> pd.DataFrame:

    existing_columns = [
        col
        for col in columns
        if col in df.columns
    ]

    vif_data = df[existing_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    vif_data = vif_data.dropna()

    results = []

    for i, col in enumerate(existing_columns):

        vif = variance_inflation_factor(
            vif_data.values,
            i,
        )

        results.append({
            "feature": col,
            "vif": vif,
        })

    return pd.DataFrame(results).sort_values(
        by="vif",
        ascending=False,
    )