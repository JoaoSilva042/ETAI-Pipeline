"""Predict a CSV with the model saved by main.py (no training)."""

import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Returns the id, the probability of the positive class and the prediction (at 0.5) for every row."""
    clean = clean_dataset(df, config["diagnostics"])          # same cleaning as training, but NO drop_duplicate_rows()
    X, _, _ = split_features_target(clean, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    probability = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({"id": df[config["diagnostics"]["id_column"]],
                         "probability": probability,
                         "prediction": (probability >= 0.5).astype(int)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="config used to train the model")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config["output"]["model_path"])
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")