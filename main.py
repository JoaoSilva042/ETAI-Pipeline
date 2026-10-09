"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data

from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test,
    build_preprocessor,
)

from src.model import build_model

from src.evaluate import (
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

from src.results import save_run




def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])

    df_clean = clean_dataset(
        df,
        config["diagnostics"],
    )

    df_clean = drop_duplicate_rows(
    df_clean,
    id_column=config["diagnostics"].get(
        "id_column"
    ),
)

    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        config["preprocessing"]["mnar_indicator_sources"],
    )

    (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
    ) = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    preprocessor = build_preprocessor(
        config["preprocessing"]
    )

    model = build_model(
        config["model"]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    

    cv_config = config["cv"]

    shuffle = cv_config.get("shuffle", True) 

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=(
            cv_config.get("random_state")
            if shuffle
            else None
        ),
    )

    scoring = cv_config.get("scoring", "accuracy", )
    n_jobs = cv_config.get("n_jobs", 1)

    fold_scores, y_oof = cross_validate_pipeline(pipeline,X_dev,y_dev,cv, scoring, n_jobs=n_jobs)

    report = cv_report( fold_scores,scoring,)

    # Week 5: hyperparameter tuning
    tuning_config = config.get("tuning", {})

    if tuning_config.get("enabled", False):
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces", {})

        if model_type not in search_spaces:
            raise ValueError(
                f"No tuning search space configured for model '{model_type}'."
            )

        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        print("\nNested cross-validation (tuned model)\n")

        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )

        nested_report = cv_report(nested_scores, scoring)
        report += "\n\nNested cross-validation (tuned model)\n" + nested_report

        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )

        tuning_text = tuning_report(study, nested_scores, scoring)
        report += "\n\n" + tuning_text
    
    


    report += "\n\n" + oof_classification_report(y_dev, y_oof,)

    report += "\n" + fairness_report(y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"],)

   

    final_model = pipeline.fit(X_dev, y_dev,)

    refit = (f"Final model: {config['model']['type']} "f"refit on all {len(X_dev)} development rows.")

    print(refit)
    report += "\n" + refit + "\n"

    locked = (
        f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
        f"Development set: {len(X_dev)} rows."
    )

    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
