"""Run the predictive pipeline with multi-metric evaluation."""

import hashlib
import json

from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedKFold

from src.data import load_data

from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test,
)

from src.model import build_pipeline

from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)

from src.tuning import (
    tune_pipeline,
    nested_cross_validate,
    tuning_report,
)

from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def comparison_key(config: dict) -> str:
    """Identify experiments with comparable configurations."""

    data_path = Path(config["data"]["path"])

    data_digest = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest()

    comparable = {
        "data_hash": data_digest,
        "data": config["data"],
        "diagnostics": config["diagnostics"],
        "preprocessing": config["preprocessing"],
        "test_set": config["test_set"],
        "cv_folds": {
            k: config["cv"].get(k)
            for k in (
                "n_splits",
                "shuffle",
                "random_state",
            )
        },
    }

    blob = json.dumps(
        comparable,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(blob).hexdigest()[:16]


def main():
    config = load_config()

    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    # Load and clean data
    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])

    df = drop_duplicate_rows(
        df,
        config["diagnostics"].get("id_column"),
    )

    X, y, extras = split_features_target(
        df,
        config["data"],
        config["preprocessing"].get(
            "mnar_indicator_sources", []
        ),
    )

    # Keep the locked test set untouched
    (
        X_dev,
        X_test,
        y_dev,
        _y_test,
        extras_dev,
        _extras_test,
    ) = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # Cross-validation
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=(
            cv_config.get("random_state")
            if shuffle else None
        ),
    )

    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]

    if selection_metric not in metrics:
        raise ValueError(
            "cv.selection_metric must appear in cv.metrics"
        )

    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)

    print(
        f"Model: {model_type} | "
        f"selection metric: {selection_metric} | "
        f"{cv_config['n_splits']}-fold CV"
    )

    print(
        f"Data: {len(X_dev)} development rows | "
        f"{len(X_test)} test rows locked away"
    )

    # Build pipelines
    pipeline = build_pipeline(
        config["preprocessing"],
        config["model"],
    )

    floor_pipeline = build_pipeline(
        config["preprocessing"],
        {"type": "dummy"},
    )

    # Dummy baseline
    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline,
        X_dev,
        y_dev,
        cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric,
        n_jobs,
    )

    # Default model evaluation
    fold_scores, oof_proba = cross_validate_pipeline(
        pipeline,
        X_dev,
        y_dev,
        cv,
        scorers,
        selection_metric,
        n_jobs,
    )

    floor = float(
        floor_scores["validation"].mean()
    )

    untuned = float(
        fold_scores["validation"].mean()
    )

    print(
        f"Default configuration: "
        f"{selection_metric}={untuned:.3f} "
        f"(no-model floor={floor:.3f})"
    )

    report = (
        f"Hyperparameters from config.yaml: "
        f"{config['model'].get('params')}\n\n"
        + cv_report(
            fold_scores,
            metrics,
            selection_metric,
            "Cross-validation",
        )
        + f"\n\nNo-model floor (dummy, same folds): "
        f"{selection_metric} = {floor:.3f}"
    )

    final_scores = fold_scores

    # Hyperparameter tuning
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get(
        "enabled", False
    )

    study = None

    if tuning_enabled:
        search_spaces = (
            tuning_config.get("search_spaces") or {}
        )

        if model_type not in search_spaces:
            raise ValueError(
                f"No tuning search space for '{model_type}'"
            )

        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        optuna.logging.set_verbosity(
            optuna.logging.WARNING
        )

        print(
            f"Tuning: {n_trials} trials x "
            f"{cv.get_n_splits()} outer x "
            f"{inner_cv.get_n_splits()} inner folds",
            flush=True,
        )

        # Nested CV
        final_scores, oof_proba = nested_cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            scorers,
            selection_metric,
            search_spaces[model_type],
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )

        # Final tuning on development set
        pipeline, study = tune_pipeline(
            pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scorers[selection_metric],
            search_spaces[model_type],
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )

        report += (
            "\n\n"
            + cv_report(
                final_scores,
                metrics,
                selection_metric,
                "Nested cross-validation",
            )
            + "\n\n"
            + tuning_report(
                study,
                final_scores,
                selection_metric,
            )
        )

        print(
            f"Tuned {selection_metric} (nested CV): "
            f"{final_scores['validation'].mean():.3f} "
            f"+/- {final_scores['validation'].std(ddof=1):.3f}"
        )

        print(
            f"Selected hyperparameters: "
            f"{study.best_params}"
        )

    # All evaluation metrics
    print("\nAll metrics (mean +/- std across folds)")

    for name in metrics:
        print(
            f"  {name:<20} "
            f"{final_scores[name].mean():.3f} +/- "
            f"{final_scores[name].std(ddof=1):.3f}"
        )

    # Out-of-fold evaluation
    report += "\n\n" + oof_classification_report(
        y_dev,
        oof_proba,
    )

    report += "\n\n" + fairness_report(
        y_dev,
        oof_proba,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    # Final refit
    pipeline.fit(X_dev, y_dev)

    refit = (
        f"Final model: {model_type} refit on "
        f"all {len(X_dev)} development rows."
    )

    locked = (
        f"Locked test set: {len(X_test)} rows "
        f"set aside, not evaluated."
    )

    print(f"\n{refit}\n{locked}")

    report += f"\n\n{refit}\n{locked}\n"

    # Save results to scoreboard
    row = {
        "run": run,
        "data": config["data"]["path"],
        "comparison_key": comparison_key(config),
        "model": model_type,
        "tuned": tuning_enabled,
        "n_trials": (
            int(tuning_config["n_trials"])
            if tuning_enabled else 0
        ),
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "params": str(config["model"].get("params")),
        "tuned_params": (
            str(study.best_params) if study else ""
        ),
        "metric": selection_metric,
        "mean": float(
            final_scores["validation"].mean()
        ),
        "std": float(
            final_scores["validation"].std(ddof=1)
        ),
        "untuned": untuned,
        "floor": floor,
        **{
            f"fold_{i}": float(value)
            for i, value in enumerate(
                final_scores["validation"],
                start=1,
            )
        },
        **{
            f"mean_{name}": float(
                final_scores[name].mean()
            )
            for name in metrics
        },
    }

    output = config.get("output", {})

    scoreboard_path = output.get(
        "scoreboard", "results/runs.csv"
    )

    runs = save_scores(scoreboard_path, row)

    comparable_runs = runs[
        (runs["comparison_key"] == row["comparison_key"])
        & (runs["metric"] == selection_metric)
    ].copy()

    print(
        "\nComparable runs "
        "(same dataset, preprocessing and CV folds):"
    )

    print(
        comparable_runs[
            ["run", "model", "tuned", "metric", "mean", "std"]
        ]
        .sort_values("mean", ascending=False)
        .to_string(index=False)
    )

    # Save complete text report
    path = save_run(
        output.get("results_dir", "results"),
        config,
        report,
        run_id=run,
    )

    print(f"\nFull report saved to {path}")
    print(f"Scoreboard saved to {scoreboard_path}")


if __name__ == "__main__":
    main()