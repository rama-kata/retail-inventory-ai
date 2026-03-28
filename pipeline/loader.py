from functools import lru_cache

import pandas as pd

REQUIRED_COLUMNS = [
    "YEAR", "MONTH", "SUPPLIER", "ITEM CODE",
    "ITEM DESCRIPTION", "ITEM TYPE", "RETAIL SALES",
    "RETAIL TRANSFERS", "WAREHOUSE SALES",
]

DEFAULT_CSV = "data/Warehouse_and_Retail_Sales.csv"


def validate_schema(df: pd.DataFrame) -> bool:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01"
    )
    df = df.rename(columns={
        "RETAIL SALES": "retail_sales",
        "WAREHOUSE SALES": "wh_sales",
        "RETAIL TRANSFERS": "retail_transfers",
        "ITEM CODE": "sku",
        "ITEM DESCRIPTION": "description",
        "ITEM TYPE": "item_type",
        "SUPPLIER": "supplier",
    })
    df = df[["date", "supplier", "sku", "description", "item_type",
             "retail_sales", "wh_sales", "retail_transfers"]]
    df["total_sales"] = df["retail_sales"].fillna(0) + df["wh_sales"].fillna(0)
    return df


@lru_cache(maxsize=4)
def _load_csv_cached(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_schema(df)
    return clean_data(df)


def load_csv(path: str = DEFAULT_CSV) -> pd.DataFrame:
    return _load_csv_cached(path).copy()
