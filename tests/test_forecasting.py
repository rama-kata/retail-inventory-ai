import pandas as pd
import numpy as np
from pipeline.forecasting import forecast_sku, get_forecastable_skus


def make_sku_series():
    dates = pd.date_range("2018-01-01", periods=36, freq="MS")
    np.random.seed(42)
    sales = np.random.poisson(50, size=36).astype(float)
    return pd.DataFrame({
        "date": dates, "supplier": "Acme", "sku": "SKU1",
        "description": "Widget", "item_type": "BEER",
        "retail_sales": sales * 0.6, "wh_sales": sales * 0.4,
        "retail_transfers": np.zeros(36), "total_sales": sales,
    })


def test_forecast_returns_predictions():
    df = make_sku_series()
    result = forecast_sku(df, "SKU1")
    assert "actual_train" in result
    assert "actual_test" in result
    assert "predicted" in result
    assert "mae" in result
    assert "rmse" in result
    assert len(result["predicted"]) > 0


def test_forecast_metrics_are_reasonable():
    df = make_sku_series()
    result = forecast_sku(df, "SKU1")
    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["rmse"] >= result["mae"]


def test_forecast_rejects_insufficient_data():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")],
        "sku": ["SKU1"], "total_sales": [10.0],
        "supplier": ["X"], "description": ["X"], "item_type": ["X"],
        "retail_sales": [5.0], "wh_sales": [5.0], "retail_transfers": [0.0],
    })
    result = forecast_sku(df, "SKU1")
    assert result is None


def test_get_forecastable_skus_filters_by_min_data_points():
    df = make_sku_series()  # SKU1 with 36 rows
    short_df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")],
        "sku": ["SKU_SHORT"], "total_sales": [10.0],
        "supplier": ["X"], "description": ["X"], "item_type": ["X"],
        "retail_sales": [5.0], "wh_sales": [5.0], "retail_transfers": [0.0],
    })
    combined = pd.concat([df, short_df], ignore_index=True)
    skus = get_forecastable_skus(combined)
    assert "SKU1" in skus
    assert "SKU_SHORT" not in skus
