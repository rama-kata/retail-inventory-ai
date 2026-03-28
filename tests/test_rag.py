import pandas as pd
from pipeline.rag import build_index, retrieve, format_context


def make_test_df():
    return pd.DataFrame({
        "sku": ["SKU1", "SKU2", "SKU3"],
        "description": [
            "Patron Tequila Silver 750ml",
            "Budweiser Beer 12pk",
            "Cabernet Sauvignon Reserve 750ml",
        ],
        "total_sales": [150.0, 500.0, 75.0],
        "item_type": ["LIQUOR", "BEER", "WINE"],
        "supplier": ["Patron", "AB InBev", "Napa Valley"],
    })


def test_build_index_returns_searchable_index():
    df = make_test_df()
    index, docs = build_index(df)
    assert index is not None
    assert len(docs) == 3


def test_retrieve_finds_relevant_items():
    df = make_test_df()
    index, docs = build_index(df)
    results = retrieve("tequila", index, docs, top_k=1)
    assert len(results) == 1
    assert "Patron" in results[0] or "Tequila" in results[0]


def test_format_context_produces_string():
    docs = ["SKU1: Patron Tequila - Sales: 150", "SKU2: Budweiser - Sales: 500"]
    context = format_context(docs)
    assert isinstance(context, str)
    assert "Patron" in context
