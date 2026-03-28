from typing import List, Optional

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MIN_DATA_POINTS = 15


def forecast_sku(df: pd.DataFrame, sku: str) -> Optional[dict]:
    df_sku = df[df["sku"] == sku][["date", "total_sales"]].copy()
    df_sku = df_sku.sort_values("date").reset_index(drop=True)

    if len(df_sku) < MIN_DATA_POINTS:
        return None

    # Create lag features
    for lag in [1, 2, 3, 6, 12]:
        df_sku[f"lag_{lag}"] = df_sku["total_sales"].shift(lag)
    df_sku["month"] = df_sku["date"].dt.month
    df_sku = df_sku.dropna()

    if len(df_sku) < 5:
        return None

    feature_cols = [f"lag_{l}" for l in [1, 2, 3, 6, 12]] + ["month"]
    X = df_sku[feature_cols]
    y = df_sku["total_sales"]

    split = max(1, len(X) - 3)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, verbosity=0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    return {
        "dates_train": df_sku["date"].iloc[:split].tolist(),
        "dates_test": df_sku["date"].iloc[split:].tolist(),
        "actual_train": y_train.tolist(),
        "actual_test": y_test.tolist(),
        "predicted": pred.tolist(),
        "mae": mae,
        "rmse": rmse,
        "feature_importance": dict(zip(feature_cols, model.feature_importances_.tolist())),
    }


def get_forecastable_skus(df: pd.DataFrame) -> List[str]:
    counts = df.groupby("sku").size()
    return counts[counts >= MIN_DATA_POINTS].index.tolist()
