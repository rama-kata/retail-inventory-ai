import pandas as pd
from pipeline.kpis import compute_kpis


def make_test_data():
    rows = []
    for month in range(1, 13):
        rows.append({
            "date": pd.Timestamp(f"2020-{month:02d}-01"),
            "supplier": "Acme", "sku": "SKU1", "description": "Widget",
            "item_type": "BEER", "retail_sales": 100.0, "wh_sales": 50.0,
            "retail_transfers": 10.0, "total_sales": 150.0,
        })
        rows.append({
            "date": pd.Timestamp(f"2020-{month:02d}-01"),
            "supplier": "Acme", "sku": "SKU2", "description": "Gadget",
            "item_type": "WINE", "retail_sales": 5.0, "wh_sales": 2.0,
            "retail_transfers": 0.0, "total_sales": 7.0,
        })
    return pd.DataFrame(rows)


def test_compute_kpis_returns_expected_columns():
    df = make_test_data()
    kpis = compute_kpis(df)
    assert "sku" in kpis.columns
    assert "turnover_rate" in kpis.columns
    assert "days_on_hand" in kpis.columns
    assert "abc_class" in kpis.columns


def test_compute_kpis_abc_classification():
    df = make_test_data()
    kpis = compute_kpis(df)
    # SKU1 has much higher sales, should be class A
    sku1 = kpis[kpis["sku"] == "SKU1"].iloc[0]
    assert sku1["abc_class"] == "A"


def test_compute_kpis_row_count():
    df = make_test_data()
    kpis = compute_kpis(df)
    assert len(kpis) == 2  # SKU1 and SKU2


def test_compute_kpis_turnover_is_meaningful():
    """turnover_rate should differ between SKUs with different sales/inventory ratios."""
    df = make_test_data()
    kpis = compute_kpis(df)
    sku1 = kpis[kpis["sku"] == "SKU1"].iloc[0]
    sku2 = kpis[kpis["sku"] == "SKU2"].iloc[0]
    # Both have constant wh_sales, so turnover = total_sales_sum / avg_wh_sales.
    # SKU1: 1800 / 50 = 36.0, SKU2: 84 / 2 = 42.0
    assert sku1["turnover_rate"] == 36.0
    assert sku2["turnover_rate"] == 42.0
    # Crucially, turnover is NOT just the count of rows (12 for both)
    assert sku1["turnover_rate"] != 12.0
