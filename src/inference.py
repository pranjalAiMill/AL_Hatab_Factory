# src/inference.py
import argparse
import os
import json
from pathlib import Path

import pandas as pd
import joblib
from train_model import prepare_features

def align_features(X_raw, feature_columns):
    for col in feature_columns:
        if col not in X_raw.columns:
            X_raw[col] = 0
    return X_raw[feature_columns]

def main(args):
    df = pd.read_csv(args.data)
    print("Loaded", len(df), "rows")

    df = prepare_features(df, is_training=False)

    feature_columns = json.load(open("models/feature_columns.json"))
    model = joblib.load("models/factory_model.joblib")

    cat_cols = [c for c in ["production_shift", "category", "factory_city", "line_id", "sku_id"] if c in df.columns]
    X_cats = pd.get_dummies(df[cat_cols], dummy_na=False) if cat_cols else pd.DataFrame(index=df.index)
    numeric_cols = [c for c in feature_columns if c in df.columns and c not in cat_cols]
    X_num = df[numeric_cols] if numeric_cols else pd.DataFrame(index=df.index)

    X_pre = pd.concat([X_num.reset_index(drop=True), X_cats.reset_index(drop=True)], axis=1).fillna(0)
    X_aligned = align_features(X_pre, feature_columns)

    df["y_pred"] = model.predict(X_aligned)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/predictions.csv", index=False)
    print("Saved predictions to outputs/predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/factory_hourly_synthetic.csv")
    args = parser.parse_args()
    main(args)
