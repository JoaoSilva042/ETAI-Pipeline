"""Multi-metric cross-validation, OOF predictions and fairness checks."""

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    get_scorer,
    make_scorer,
)

from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    scorers = {}

    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_", 1)[1])

            if beta <= 0:
                raise ValueError("F-beta requires beta > 0")

            scorers[name] = make_scorer(
                fbeta_score,
                beta=beta,
                zero_division=0,
            )
        else:
            scorers[name] = get_scorer(name)

    return scorers


def cross_validate_pipeline(
    pipeline, X, y, cv, scorers: dict,
    selection_metric: str, n_jobs: int = 1,
):
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorers,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs,
    )

    fold_scores = pd.DataFrame({
        "fold": range(
            1, len(scores[f"test_{selection_metric}"]) + 1
        ),
        "train": scores[f"train_{selection_metric}"],
        "validation": scores[f"test_{selection_metric}"],
    })

    fold_scores["gap"] = (
        fold_scores["train"] - fold_scores["validation"]
    )

    for metric in scorers:
        fold_scores[metric] = scores[f"test_{metric}"]

    oof_proba = np.full(len(X), np.nan, dtype=float)

    for estimator, indices in zip(
        scores["estimator"], scores["indices"]["test"]
    ):
        classes = list(estimator.classes_)

        if 1 not in classes:
            raise ValueError("Expected binary classes 0 and 1")

        oof_proba[indices] = estimator.predict_proba(
            X.iloc[indices]
        )[:, classes.index(1)]

    if np.isnan(oof_proba).any():
        raise ValueError("Missing out-of-fold probabilities")

    return fold_scores, oof_proba


def cv_report(
    fold_scores: pd.DataFrame,
    metrics: list,
    selection_metric: str,
    title: str,
) -> str:
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, "
        f"selection metric: {selection_metric})",
        "",
        fold_scores.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
        ),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(
            float_format=lambda v: f"{v:.3f}"
        ),
        (
            f"\n{selection_metric}: train-validation gap mean = "
            f"{fold_scores['gap'].mean():+.3f}"
        ),
    ]

    return "\n".join(lines)


def corrected_paired_test(
    scores_a, scores_b, test_train_ratio: float,
) -> dict:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if (
        a.ndim != 1
        or b.ndim != 1
        or len(a) != len(b)
        or len(a) < 2
    ):
        raise ValueError("Provide paired scores of equal length")

    diff = a - b
    k = len(diff)

    se = np.sqrt(
        (1 / k + test_train_ratio)
        * diff.var(ddof=1)
    )

    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = (
        float(2 * stats.t.sf(abs(t), df=k - 1))
        if se > 0 else 1.0
    )

    return {
        "mean_diff": float(diff.mean()),
        "wins": f"{int((diff > 0).sum())}/{k}",
        "t": t,
        "p_value": p_value,
    }


def oof_classification_report(
    y_true, oof_proba, threshold: float = 0.5,
) -> str:
    y_pred = (
        np.asarray(oof_proba) >= threshold
    ).astype(int)

    cm = pd.DataFrame(
        confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ),
        index=["true 0", "true 1"],
        columns=["predicted 0", "predicted 1"],
    )

    return (
        f"Out-of-fold predictions, development set "
        f"(threshold {threshold}):\n\n"
        f"{cm.to_string()}\n\n"
        f"{classification_report(y_true, y_pred, zero_division=0)}"
    )


def fairness_report(
    y_true,
    oof_proba,
    extras: pd.DataFrame,
    sensitive_attr: str = "race",
    threshold: float = 0.5,
) -> str:

    df = extras.reset_index(drop=True).copy()

    df[sensitive_attr] = (
        df[sensitive_attr]
        .astype(object)
        .fillna("unknown")
    )

    df["y_true"] = np.asarray(y_true)

    df["y_pred_model"] = (
        np.asarray(oof_proba) >= threshold
    ).astype(int)

    df["y_pred_compas"] = (
        df["score_text"] != "Low"
    ).astype(int)

    lines = [
        f"False positive rate by {sensitive_attr}",
        "(share of people who did NOT reoffend, "
        "but were predicted to)",
        "",
    ]

    for label, col in [
        ("Our model", "y_pred_model"),
        ("COMPAS's own score", "y_pred_compas"),
    ]:
        lines.append(f"  {label}:")

        for group, subset in df.groupby(sensitive_attr):
            negatives = subset[subset["y_true"] == 0]

            if negatives.empty:
                continue

            fpr = (negatives[col] == 1).mean()

            lines.append(
                f"    {group:<20s} "
                f"FPR = {fpr:.2f}  (n={len(negatives)})"
            )

        lines.append("")

    return "\n".join(lines)