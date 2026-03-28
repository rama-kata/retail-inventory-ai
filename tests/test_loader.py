import pytest
import pandas as pd
from pipeline.loader import load_csv, validate_schema, clean_data

REQUIRED_COLUMNS = [
    "YEAR", "MONTH", "SUPPLIER", "ITEM CODE",
    "ITEM DESCRIPTION", "ITEM TYPE", "RETAIL SALES",
    "RETAIL TRANSFERS", "WAREHOUSE SALES",
]


def test_validate_schema_accepts_valid_df():
    df = pd.DataFrame({col: [1] for col in REQUIRED_COLUMNS})
    assert validate_schema(df) is True


def test_validate_schema_rejects_missing_columns():
    df = pd.DataFrame({"YEAR": [1], "MONTH": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_clean_data_creates_expected_columns():
    df = pd.DataFrame({
        "YEAR": [2020, 2020],
        "MONTH": [1, 2],
        "SUPPLIER": ["Acme", "Acme"],
        "ITEM CODE": ["SKU1", "SKU1"],
        "ITEM DESCRIPTION": ["Widget", "Widget"],
        "ITEM TYPE": ["BEER", "BEER"],
        "RETAIL SALES": [10.0, 20.0],
        "RETAIL TRANSFERS": [1.0, 2.0],
        "WAREHOUSE SALES": [5.0, 8.0],
    })
    result = clean_data(df)
    assert "date" in result.columns
    assert "total_sales" in result.columns
    assert "sku" in result.columns
    assert result["total_sales"].iloc[0] == 15.0  # 10 + 5


def test_load_csv_with_default_data(tmp_path):
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "YEAR": [2020], "MONTH": [1], "SUPPLIER": ["Acme"],
        "ITEM CODE": ["SKU1"], "ITEM DESCRIPTION": ["Widget"],
        "ITEM TYPE": ["BEER"], "RETAIL SALES": [10.0],
        "RETAIL TRANSFERS": [1.0], "WAREHOUSE SALES": [5.0],
    })
    df.to_csv(csv_path, index=False)
    result = load_csv(str(csv_path))
    assert len(result) == 1
    assert "total_sales" in result.columns
