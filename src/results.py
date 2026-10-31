"""Save experiment reports and CSV scoreboard."""

import os
from datetime import datetime

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    if os.path.exists(path):
        runs = pd.read_csv(
            path, dtype={"run": str}
        )
    else:
        runs = pd.DataFrame()

    runs = pd.concat(
        [runs, pd.DataFrame([row])],
        ignore_index=True,
    )

    runs.to_csv(path, index=False)

    return runs


def save_run(
    results_dir: str,
    config: dict,
    report_text: str,
    run_id: str = None,
) -> str:

    os.makedirs(results_dir, exist_ok=True)

    run_id = run_id or datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = os.path.join(
        results_dir,
        f"run_{run_id}.txt",
    )

    cv = config["cv"]
    tuning = config.get("tuning", {})

    if tuning.get("enabled"):
        tuning_line = (
            f"Tuning: enabled=True "
            f"trials={tuning['n_trials']} "
            f"inner_folds={tuning['n_splits']} "
            f"random_state={tuning['random_state']}"
        )
    else:
        tuning_line = "Tuning: enabled=False"

    header = (
        f"Run: {run_id}\n"
        f"Model: {config['model']['type']} "
        f"params={config['model'].get('params')}\n"
        f"Preprocessing: "
        f"encoder={config['preprocessing']['encoder']} "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: {cv['n_splits']} stratified folds "
        f"shuffle={cv.get('shuffle', True)} "
        f"random_state={cv.get('random_state')} "
        f"metrics={cv['metrics']} "
        f"selection_metric={cv['selection_metric']}\n"
        f"{tuning_line}\n"
        f"Locked test set: "
        f"size={config['test_set']['size']} "
        f"random_state={config['test_set']['random_state']} "
        f"(not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as out:
        out.write(header + report_text)

    return path