import pandas as pd
from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target


def predict_new_data(pipeline, config: dict, threshold: float = 0.5) -> pd.DataFrame:
    data_cfg = config["data"]
    id_col = config["diagnostics"]["id_column"]

    
    new = load_data(data_cfg["new_data_path"])
    new = clean_dataset(new, config["diagnostics"])

    
    X_new, _, _ = split_features_target(
        new,
        data_cfg,
        config["preprocessing"].get("mnar_indicator_sources", []),
    )

    
    proba = pipeline.predict_proba(X_new)[:, 1]

    out = pd.DataFrame({
        id_col: new[id_col].astype(int).values,      
        "probability": proba,
        "prediction": (proba >= threshold).astype(int),
    })

    
    assert len(out) == len(new), "Perderam-se linhas na previsão!"

    out.to_csv(data_cfg["predictions_path"], index=False)
    return out