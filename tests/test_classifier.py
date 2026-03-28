import pandas as pd
from pipeline.classifier import classify_slow_movers


def make_kpis():
    return pd.DataFrame({
        "sku": ["SKU1", "SKU2", "SKU3", "SKU4", "SKU5"],
        "turnover_rate": [50.0, 40.0, 10.0, 5.0, 2.0],
        "days_on_hand": [5.0, 10.0, 30.0, 60.0, 150.0],
        "abc_class": ["A", "A", "B", "C", "C"],
    })


def test_classify_returns_slow_mover_column():
    kpis = make_kpis()
    result = classify_slow_movers(kpis, percentile=30)
    assert "slow_mover" in result.columns


def test_classify_labels_low_turnover_as_slow():
    kpis = make_kpis()
    result = classify_slow_movers(kpis, percentile=30)
    # SKU5 has the lowest turnover, should be slow
    sku5 = result[result["sku"] == "SKU5"].iloc[0]
    assert sku5["slow_mover"] == 1


def test_classify_labels_high_turnover_as_fast():
    kpis = make_kpis()
    result = classify_slow_movers(kpis, percentile=30)
    sku1 = result[result["sku"] == "SKU1"].iloc[0]
    assert sku1["slow_mover"] == 0


def test_classify_custom_percentile():
    kpis = make_kpis()
    result_low = classify_slow_movers(kpis, percentile=10)
    result_high = classify_slow_movers(kpis, percentile=80)
    slow_low = result_low["slow_mover"].sum()
    slow_high = result_high["slow_mover"].sum()
    assert slow_high >= slow_low
