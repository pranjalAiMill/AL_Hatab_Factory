# src/train_model.py
import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# -----------------------
# Feature engineering
# -----------------------
def prepare_features(df, is_training=True, target_col="prod_plan_qty"):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Create timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns and "hour" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    else:
        raise ValueError("timestamp or (date + hour) required in dataset")

    # Sort for grouped operations
    df = df.sort_values(["factory_id", "sku_id", "timestamp"]).reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)

    # Shift
    df["production_shift"] = df["hour"].apply(lambda h: "night" if h < 7 else ("afternoon" if h < 15 else "evening"))

    # Group lags and rolling features
    grp = df.groupby(["factory_id", "sku_id"], sort=False)

    for lag in [1, 24, 168]:  # 1h, 24h, 7d
        df[f"lag_{lag}h"] = grp["prod_actual_qty"].shift(lag)

    df["rolling_6h_mean"] = grp["prod_actual_qty"].rolling(window=6, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df["rolling_24h_mean"] = grp["prod_actual_qty"].rolling(window=24, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df["rolling_7d_mean"] = grp["prod_actual_qty"].rolling(window=24*7, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # Derived rolling rates
    df["rolling_defect_rate"] = (
        grp["defect_qty"].rolling(window=24, min_periods=1).sum() /
        (grp["prod_actual_qty"].rolling(window=24, min_periods=1).sum() + 1e-9)
    ).reset_index(level=[0,1], drop=True)

    df["rolling_energy_usage_per_unit"] = (
        grp["energy_kwh"].rolling(window=24, min_periods=1).sum() /
        (grp["prod_actual_qty"].rolling(window=24, min_periods=1).sum() + 1e-9)
    ).reset_index(level=[0,1], drop=True)

    df["rolling_labor_hours_per_unit"] = (
        grp["labor_hours"].rolling(window=24, min_periods=1).sum() /
        (grp["prod_actual_qty"].rolling(window=24, min_periods=1).sum() + 1e-9)
    ).reset_index(level=[0,1], drop=True)

    # Demand signals
    if "released_to_dc_qty" in df.columns:
        df["dc_demand_24h"] = grp["released_to_dc_qty"].rolling(window=24, min_periods=1).sum().reset_index(level=[0,1], drop=True)
        df["dc_demand_48h"] = grp["released_to_dc_qty"].rolling(window=48, min_periods=1).sum().reset_index(level=[0,1], drop=True)

    df.fillna(0, inplace=True)
    return df

# -----------------------
# Training & evaluation
# -----------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, X_all):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]))*100 if mask.sum() > 0 else np.nan
    bias = np.mean(y_pred - y_test)

    print("\n✅ Model performance:")
    print(f"MAE  : {mae:.3f}")
    print(f"MAPE : {mape:.3f}%")
    print(f"BIAS : {bias:.3f}")

    feat_imp = pd.DataFrame({"feature": X_all.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    print("\nTop 30 features:")
    print(feat_imp.head(30).to_string(index=False))

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(model, "models/factory_model.joblib")
    with open("models/feature_columns.json", "w") as f:
        json.dump(list(X_all.columns), f)

    df_out = X_test.copy()
    df_out["y_true"] = y_test
    df_out["y_pred"] = y_pred
    df_out.to_csv("outputs/test_predictions_sample.csv", index=False)

    return model, feat_imp

# -----------------------
# Main
# -----------------------
def main(args):
    df = pd.read_csv(args.data)
    print("Loaded", len(df), "rows")

    df = prepare_features(df)

    if "prod_actual_qty" in df.columns:
        TARGET = "prod_actual_qty"
    else:
    # Fallback if actual isn't available
        TARGET = "prod_plan_qty"
        print("Warning: prod_actual_qty not found. Defaulting to prod_plan_qty.") 
        
    print("Using target:", TARGET)
    

    features = [
        "hour", "day_of_week", "is_weekend", "week_of_year",
        "is_month_start", "is_month_end", "production_shift",
        "category", "shelf_life_days", "line_id",
        "batch_size_units", "factory_city", "sku_id",
        "lag_1h", "lag_24h", "lag_168h",
        "rolling_6h_mean", "rolling_24h_mean", "rolling_7d_mean",
        "rolling_defect_rate", "rolling_energy_usage_per_unit", "rolling_labor_hours_per_unit",
        "dc_demand_24h", "dc_demand_48h",
        "base_demand_index", "energy_kwh", "labor_hours"
    ]
    features = [f for f in features if f in df.columns]
    X = df[features].copy()
    X = X.loc[:, ~X.columns.duplicated()]
    y = df[TARGET]

    cat_cols = [c for c in ["production_shift", "category", "factory_city", "line_id", "sku_id"] if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

   
    X.columns = ["_".join(c.split()) for c in X.columns]
    

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].nunique() < 10:
        train_size = int(0.8*len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    else:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=7)
        train_mask = df["timestamp"] <= cutoff
        test_mask = df["timestamp"] > cutoff
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]

        if X_train.shape[0]==0 or X_test.shape[0]==0:
            train_size = int(0.8*len(X))
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    train_and_evaluate(X_train, y_train, X_test, y_test, X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/factory_hourly_synthetic.csv", help="CSV path")
    args = parser.parse_args()
    main(args)
