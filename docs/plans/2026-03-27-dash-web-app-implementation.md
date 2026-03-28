# Retail Inventory AI — Dash Web App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Plotly Dash web app that serves an interactive retail inventory dashboard with KPI visualization, XGBoost demand forecasting, slow-mover classification, and RAG-powered Q&A using Ollama.

**Architecture:** Dash multi-page app with 3 pages (Dashboard, Forecasting, Classification) plus a collapsible RAG chat sidebar. Backend pipeline modules process CSV data on demand. FAISS indexes item descriptions for retrieval, Ollama generates answers locally.

**Tech Stack:** Plotly Dash, Dash Bootstrap Components, pandas, XGBoost, scikit-learn, Sentence-Transformers, faiss-cpu, Ollama (local LLM), pytest

---

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore` (update existing)
- Create: `app.py` (skeleton)

**Step 1: Create requirements.txt**

```txt
dash==2.18.2
dash-bootstrap-components==1.6.0
pandas==2.2.3
numpy==1.26.4
plotly==5.24.1
xgboost==2.1.3
scikit-learn==1.5.2
sentence-transformers==3.3.1
faiss-cpu==1.9.0
requests==2.32.3
pytest==8.3.4
```

**Step 2: Install dependencies**

Run: `cd /Users/riinygiir/RAG_Project && pip install -r requirements.txt`
Expected: All packages install successfully.

**Step 3: Update .gitignore**

Add to existing `.gitignore`:
```
data/*.csv
__pycache__/
*.egg-info/
.env
venv/
```

**Step 4: Move CSV into data/ directory**

Run:
```bash
mkdir -p data
mv Warehouse_and_Retail_Sales.csv data/
```

**Step 5: Create app.py skeleton**

```python
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="/")),
            dbc.NavItem(dbc.NavLink("Forecasting", href="/forecasting")),
            dbc.NavItem(dbc.NavLink("Classification", href="/classification")),
        ],
        brand="Retail Inventory AI",
        brand_href="/",
        color="primary",
        dark=True,
    ),
    dash.page_container,
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
```

**Step 6: Verify the skeleton runs**

Run: `cd /Users/riinygiir/RAG_Project && python app.py`
Expected: Dash app starts at http://127.0.0.1:8050, shows navbar with 3 links and empty page container.

**Step 7: Commit**

```bash
git add requirements.txt .gitignore app.py data/
git commit -m "feat: project setup with Dash skeleton and dependencies"
```

---

### Task 2: Data Loader Pipeline

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/loader.py`
- Create: `tests/__init__.py`
- Create: `tests/test_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_loader.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline'`

**Step 3: Write the implementation**

```python
# pipeline/__init__.py
# (empty)
```

```python
# tests/__init__.py
# (empty)
```

```python
# pipeline/loader.py
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


def load_csv(path: str = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_schema(df)
    return clean_data(df)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_loader.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add pipeline/ tests/
git commit -m "feat: add data loader with validation and cleaning"
```

---

### Task 3: KPI Pipeline

**Files:**
- Create: `pipeline/kpis.py`
- Create: `tests/test_kpis.py`

**Step 1: Write the failing test**

```python
# tests/test_kpis.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_kpis.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

```python
# pipeline/kpis.py
import pandas as pd


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    sales_sum = df.groupby("sku")["total_sales"].sum()
    sales_mean = df.groupby("sku")["total_sales"].mean()

    turnover = (sales_sum / sales_mean).rename("turnover_rate")

    proxy_stock = df.groupby("sku").last()["total_sales"]
    days_on_hand = (proxy_stock / (sales_mean / 30)).rename("days_on_hand")

    annual_sales = sales_sum.sort_values(ascending=False)
    cum_pct = annual_sales.cumsum() / annual_sales.sum()
    abc_class = cum_pct.map(
        lambda x: "A" if x <= 0.70 else ("B" if x <= 0.90 else "C")
    ).rename("abc_class")

    kpis = pd.concat([turnover, days_on_hand, abc_class], axis=1).reset_index()
    return kpis
```

**Step 4: Run tests**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_kpis.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add pipeline/kpis.py tests/test_kpis.py
git commit -m "feat: add KPI computation pipeline"
```

---

### Task 4: Forecasting Pipeline

**Files:**
- Create: `pipeline/forecasting.py`
- Create: `tests/test_forecasting.py`

**Step 1: Write the failing test**

```python
# tests/test_forecasting.py
import pandas as pd
import numpy as np
from pipeline.forecasting import forecast_sku


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
    assert "actual" in result
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_forecasting.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

```python
# pipeline/forecasting.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MIN_DATA_POINTS = 15


def forecast_sku(df: pd.DataFrame, sku: str) -> dict | None:
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


def get_forecastable_skus(df: pd.DataFrame) -> list[str]:
    counts = df.groupby("sku").size()
    return counts[counts >= MIN_DATA_POINTS].index.tolist()
```

**Step 4: Run tests**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_forecasting.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add pipeline/forecasting.py tests/test_forecasting.py
git commit -m "feat: add XGBoost forecasting pipeline"
```

---

### Task 5: Classification Pipeline

**Files:**
- Create: `pipeline/classifier.py`
- Create: `tests/test_classifier.py`

**Step 1: Write the failing test**

```python
# tests/test_classifier.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_classifier.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

```python
# pipeline/classifier.py
import pandas as pd


def classify_slow_movers(kpis: pd.DataFrame, percentile: int = 30) -> pd.DataFrame:
    kpis = kpis.copy()
    cutoff = kpis["turnover_rate"].quantile(percentile / 100)
    kpis["slow_mover"] = (kpis["turnover_rate"] < cutoff).astype(int)
    return kpis
```

**Step 4: Run tests**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_classifier.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add pipeline/classifier.py tests/test_classifier.py
git commit -m "feat: add slow-mover classification pipeline"
```

---

### Task 6: RAG Pipeline

**Files:**
- Create: `pipeline/rag.py`
- Create: `tests/test_rag.py`

**Step 1: Write the failing test**

```python
# tests/test_rag.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_rag.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

```python
# pipeline/rag.py
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import requests

_model = None


def _get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_index(df: pd.DataFrame) -> tuple:
    docs = []
    for _, row in df.iterrows():
        doc = (
            f"SKU: {row['sku']} | {row['description']} | "
            f"Type: {row.get('item_type', 'N/A')} | "
            f"Supplier: {row.get('supplier', 'N/A')} | "
            f"Total Sales: {row.get('total_sales', 0):.1f}"
        )
        docs.append(doc)

    embedder = _get_embedder()
    embeddings = embedder.encode(docs, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, docs


def retrieve(query: str, index, docs: list[str], top_k: int = 5) -> list[str]:
    embedder = _get_embedder()
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [docs[i] for i in indices[0] if i < len(docs)]


def format_context(docs: list[str]) -> str:
    return "\n".join(f"- {doc}" for doc in docs)


def query_ollama(question: str, context: str, model: str = "llama3") -> str:
    prompt = (
        f"You are a retail inventory analyst. Use the following data to answer the question.\n\n"
        f"DATA:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer concisely based only on the data provided."
    )
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Start it with `ollama serve`."
    except Exception as e:
        return f"Error querying Ollama: {e}"
```

**Step 4: Run tests**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/test_rag.py -v`
Expected: 3 passed (note: first run downloads the embedding model, may take a minute)

**Step 5: Commit**

```bash
git add pipeline/rag.py tests/test_rag.py
git commit -m "feat: add RAG pipeline with FAISS and Ollama"
```

---

### Task 7: Dashboard Page

**Files:**
- Create: `pages/__init__.py`
- Create: `pages/dashboard.py`

**Step 1: Create the page**

```python
# pages/__init__.py
# (empty)
```

```python
# pages/dashboard.py
import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import io

from pipeline.loader import load_csv, validate_schema, clean_data
from pipeline.kpis import compute_kpis

dash.register_page(__name__, path="/", name="Dashboard")

layout = dbc.Container([
    # Upload section
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id="upload-data",
                children=dbc.Button("Upload CSV", color="secondary", className="me-2"),
                multiple=False,
            ),
            html.Small("Or using pre-loaded dataset", className="text-muted"),
        ], width=12, className="my-3"),
    ]),

    # Status
    html.Div(id="upload-status"),

    # Store for processed data
    dcc.Store(id="store-data"),
    dcc.Store(id="store-kpis"),

    # Filters
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="filter-supplier", placeholder="Filter by Supplier", multi=True), width=4),
        dbc.Col(dcc.Dropdown(id="filter-item-type", placeholder="Filter by Item Type", multi=True), width=4),
        dbc.Col(dcc.Dropdown(id="filter-sku", placeholder="Filter by SKU", multi=True), width=4),
    ], className="mb-3"),

    # KPI Cards
    dbc.Row(id="kpi-cards", className="mb-3"),

    # Charts
    dbc.Row([
        dbc.Col(dcc.Graph(id="turnover-hist"), width=6),
        dbc.Col(dcc.Graph(id="doh-scatter"), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="sales-trend"), width=12),
    ]),

    # Data table
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id="kpi-table",
            page_size=20,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#303030", "color": "white"},
            style_cell={"backgroundColor": "#222", "color": "white"},
        ), width=12),
    ], className="mt-3"),
], fluid=True)


@callback(
    Output("store-data", "data"),
    Output("store-kpis", "data"),
    Output("upload-status", "children"),
    Output("filter-supplier", "options"),
    Output("filter-item-type", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def load_data(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            raw = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            validate_schema(raw)
            df = clean_data(raw)
        except Exception as e:
            return None, None, dbc.Alert(str(e), color="danger"), [], []
        status = dbc.Alert(f"Loaded {filename}: {len(df)} rows", color="success")
    else:
        df = load_csv()
        status = dbc.Alert(f"Using default dataset: {len(df)} rows", color="info")

    kpis = compute_kpis(df)
    supplier_opts = [{"label": s, "value": s} for s in sorted(df["supplier"].dropna().unique())]
    item_type_opts = [{"label": t, "value": t} for t in sorted(df["item_type"].dropna().unique())]

    return df.to_json(date_format="iso"), kpis.to_json(), status, supplier_opts, item_type_opts


@callback(
    Output("filter-sku", "options"),
    Input("store-data", "data"),
    Input("filter-supplier", "value"),
    Input("filter-item-type", "value"),
)
def update_sku_options(data_json, suppliers, item_types):
    if data_json is None:
        return []
    df = pd.read_json(io.StringIO(data_json))
    if suppliers:
        df = df[df["supplier"].isin(suppliers)]
    if item_types:
        df = df[df["item_type"].isin(item_types)]
    skus = sorted(df["sku"].dropna().unique())
    return [{"label": s, "value": s} for s in skus[:500]]


@callback(
    Output("kpi-cards", "children"),
    Output("turnover-hist", "figure"),
    Output("doh-scatter", "figure"),
    Output("sales-trend", "figure"),
    Output("kpi-table", "data"),
    Output("kpi-table", "columns"),
    Input("store-data", "data"),
    Input("store-kpis", "data"),
    Input("filter-supplier", "value"),
    Input("filter-item-type", "value"),
    Input("filter-sku", "value"),
)
def update_dashboard(data_json, kpis_json, suppliers, item_types, skus):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark")
    if data_json is None or kpis_json is None:
        return [], empty_fig, empty_fig, empty_fig, [], []

    df = pd.read_json(io.StringIO(data_json))
    kpis = pd.read_json(io.StringIO(kpis_json))

    # Apply filters to main data
    if suppliers:
        df = df[df["supplier"].isin(suppliers)]
    if item_types:
        df = df[df["item_type"].isin(item_types)]
    if skus:
        df = df[df["sku"].isin(skus)]
        kpis = kpis[kpis["sku"].isin(skus)]

    # KPI cards
    cards = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total SKUs"), html.H3(f"{kpis['sku'].nunique():,}")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Avg Turnover"), html.H3(f"{kpis['turnover_rate'].mean():.1f}")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total Sales"), html.H3(f"{df['total_sales'].sum():,.0f}")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Class A SKUs"), html.H3(f"{(kpis['abc_class'] == 'A').sum():,}")
        ])), width=3),
    ]

    # Turnover histogram
    hist_fig = px.histogram(kpis, x="turnover_rate", nbins=30, title="Turnover Rate Distribution")
    hist_fig.update_layout(template="plotly_dark")

    # Days-on-hand scatter
    scatter_fig = px.scatter(
        kpis, x="days_on_hand", y="turnover_rate", color="abc_class",
        title="Days-on-Hand vs Turnover by ABC Class",
        color_discrete_map={"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"},
    )
    scatter_fig.update_layout(template="plotly_dark")

    # Sales trend
    monthly = df.groupby("date")["total_sales"].sum().reset_index()
    trend_fig = px.line(monthly, x="date", y="total_sales", title="Total Sales Over Time")
    trend_fig.update_layout(template="plotly_dark")

    # Table
    kpi_display = kpis.round(2)
    columns = [{"name": c, "id": c} for c in kpi_display.columns]

    return cards, hist_fig, scatter_fig, trend_fig, kpi_display.to_dict("records"), columns
```

**Step 2: Verify the page loads**

Run: `cd /Users/riinygiir/RAG_Project && python app.py`
Expected: Dashboard loads at http://127.0.0.1:8050 with charts, KPI cards, filters, and data table populated from the default CSV.

**Step 3: Commit**

```bash
git add pages/
git commit -m "feat: add Dashboard page with KPIs, charts, filters, and upload"
```

---

### Task 8: Forecasting Page

**Files:**
- Create: `pages/forecasting.py`

**Step 1: Create the page**

```python
# pages/forecasting.py
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import io

from pipeline.loader import load_csv
from pipeline.forecasting import forecast_sku, get_forecastable_skus

dash.register_page(__name__, path="/forecasting", name="Forecasting")

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("Demand Forecasting (XGBoost)"),
            html.P("Select a SKU to see actual vs predicted sales.", className="text-muted"),
        ]),
    ], className="my-3"),

    dbc.Row([
        dbc.Col(dcc.Dropdown(id="forecast-sku", placeholder="Select SKU..."), width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="forecast-chart")), width=9),
        dbc.Col([
            dbc.Card(dbc.CardBody([html.H6("MAE"), html.H3(id="forecast-mae")]), className="mb-2"),
            dbc.Card(dbc.CardBody([html.H6("RMSE"), html.H3(id="forecast-rmse")])),
        ], width=3),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id="feature-importance"), width=12),
    ], className="mt-3"),
], fluid=True)


@callback(
    Output("forecast-sku", "options"),
    Input("forecast-sku", "id"),  # triggers on page load
)
def populate_skus(_):
    df = load_csv()
    skus = get_forecastable_skus(df)
    return [{"label": s, "value": s} for s in sorted(skus)]


@callback(
    Output("forecast-chart", "figure"),
    Output("forecast-mae", "children"),
    Output("forecast-rmse", "children"),
    Output("feature-importance", "figure"),
    Input("forecast-sku", "value"),
)
def run_forecast(sku):
    empty = go.Figure()
    empty.update_layout(template="plotly_dark")
    if not sku:
        return empty, "-", "-", empty

    df = load_csv()
    result = forecast_sku(df, sku)
    if result is None:
        return empty, "N/A", "N/A", empty

    # Forecast chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["dates_train"], y=result["actual_train"],
        mode="lines", name="Training (Actual)", line=dict(color="#3498db"),
    ))
    fig.add_trace(go.Scatter(
        x=result["dates_test"], y=result["actual_test"],
        mode="lines+markers", name="Test (Actual)", line=dict(color="#2ecc71"),
    ))
    fig.add_trace(go.Scatter(
        x=result["dates_test"], y=result["predicted"],
        mode="lines+markers", name="Predicted", line=dict(color="#e74c3c", dash="dash"),
    ))
    fig.update_layout(
        title=f"XGBoost Forecast — SKU {sku}",
        xaxis_title="Date", yaxis_title="Total Sales",
        template="plotly_dark",
    )

    # Feature importance
    fi = result["feature_importance"]
    fi_fig = go.Figure(go.Bar(
        x=list(fi.values()), y=list(fi.keys()), orientation="h",
        marker_color="#3498db",
    ))
    fi_fig.update_layout(
        title="Feature Importance", template="plotly_dark",
        yaxis=dict(autorange="reversed"),
    )

    return fig, f"{result['mae']:.2f}", f"{result['rmse']:.2f}", fi_fig
```

**Step 2: Verify the page loads**

Run: `cd /Users/riinygiir/RAG_Project && python app.py`
Navigate to http://127.0.0.1:8050/forecasting
Expected: SKU dropdown populates, selecting one shows forecast chart + metrics + feature importance.

**Step 3: Commit**

```bash
git add pages/forecasting.py
git commit -m "feat: add Forecasting page with XGBoost predictions"
```

---

### Task 9: Classification Page

**Files:**
- Create: `pages/classification.py`

**Step 1: Create the page**

```python
# pages/classification.py
import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from pipeline.loader import load_csv
from pipeline.kpis import compute_kpis
from pipeline.classifier import classify_slow_movers

dash.register_page(__name__, path="/classification", name="Classification")

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("Slow-Mover Classification"),
            html.P("Adjust the percentile threshold to identify slow-moving inventory.", className="text-muted"),
        ]),
    ], className="my-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Slow-Mover Threshold (percentile)"),
            dcc.Slider(id="threshold-slider", min=5, max=80, step=5, value=30,
                       marks={i: f"{i}%" for i in range(5, 81, 10)}),
        ], width=8),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="class-pie"), width=4),
        dbc.Col([
            dbc.Card(dbc.CardBody([html.H6("Slow Movers"), html.H3(id="slow-count")]), className="mb-2"),
            dbc.Card(dbc.CardBody([html.H6("Fast Movers"), html.H3(id="fast-count")])),
        ], width=2),
        dbc.Col(dcc.Graph(id="class-scatter"), width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Flagged Slow Movers", className="mt-3"),
            dbc.Button("Export CSV", id="export-btn", color="secondary", size="sm", className="mb-2"),
            dcc.Download(id="export-csv"),
            dash_table.DataTable(
                id="slow-table",
                page_size=20,
                sort_action="native",
                style_header={"backgroundColor": "#303030", "color": "white"},
                style_cell={"backgroundColor": "#222", "color": "white"},
            ),
        ], width=12),
    ]),
], fluid=True)


@callback(
    Output("class-pie", "figure"),
    Output("class-scatter", "figure"),
    Output("slow-count", "children"),
    Output("fast-count", "children"),
    Output("slow-table", "data"),
    Output("slow-table", "columns"),
    Input("threshold-slider", "value"),
)
def update_classification(percentile):
    df = load_csv()
    kpis = compute_kpis(df)
    classified = classify_slow_movers(kpis, percentile=percentile)

    slow = classified["slow_mover"].sum()
    fast = len(classified) - slow

    # Pie chart
    pie_fig = px.pie(
        values=[fast, slow], names=["Fast Movers", "Slow Movers"],
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
    )
    pie_fig.update_layout(template="plotly_dark")

    # Scatter
    scatter_fig = px.scatter(
        classified, x="days_on_hand", y="turnover_rate",
        color=classified["slow_mover"].map({0: "Fast", 1: "Slow"}),
        color_discrete_map={"Fast": "#2ecc71", "Slow": "#e74c3c"},
        title="Inventory Classification",
    )
    scatter_fig.update_layout(template="plotly_dark")

    # Table of slow movers
    slow_df = classified[classified["slow_mover"] == 1].round(2)
    columns = [{"name": c, "id": c} for c in slow_df.columns]

    return pie_fig, scatter_fig, f"{slow:,}", f"{fast:,}", slow_df.to_dict("records"), columns


@callback(
    Output("export-csv", "data"),
    Input("export-btn", "n_clicks"),
    Input("threshold-slider", "value"),
    prevent_initial_call=True,
)
def export_slow_movers(n_clicks, percentile):
    if not n_clicks:
        return None
    df = load_csv()
    kpis = compute_kpis(df)
    classified = classify_slow_movers(kpis, percentile=percentile)
    slow_df = classified[classified["slow_mover"] == 1]
    return dcc.send_data_frame(slow_df.to_csv, "slow_movers.csv", index=False)
```

**Step 2: Verify the page loads**

Run: `cd /Users/riinygiir/RAG_Project && python app.py`
Navigate to http://127.0.0.1:8050/classification
Expected: Slider adjusts classification, pie chart + scatter + table update.

**Step 3: Commit**

```bash
git add pages/classification.py
git commit -m "feat: add Classification page with threshold slider and export"
```

---

### Task 10: RAG Chat Sidebar

**Files:**
- Modify: `app.py` — add sidebar layout and chat callbacks

**Step 1: Update app.py with chat sidebar**

Replace `app.py` with:

```python
# app.py
import dash
from dash import html, dcc, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd

from pipeline.loader import load_csv
from pipeline.rag import build_index, retrieve, format_context, query_ollama

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

# Pre-build the FAISS index on startup
_df = load_csv()
_index, _docs = build_index(_df)

chat_sidebar = dbc.Offcanvas(
    [
        html.Div(id="chat-history", style={
            "height": "60vh", "overflowY": "auto", "marginBottom": "10px",
        }),
        dbc.InputGroup([
            dbc.Input(id="chat-input", placeholder="Ask about your inventory...", type="text"),
            dbc.Button("Send", id="chat-send", color="primary"),
        ]),
        html.Div([
            html.Small("Try: ", className="text-muted"),
            html.A("Which SKUs should I reorder?", id="suggest-1", href="#", className="me-2 small"),
            html.A("What are my slowest movers?", id="suggest-2", href="#", className="small"),
        ], className="mt-2"),
    ],
    id="chat-offcanvas",
    title="Inventory AI Chat",
    placement="end",
    is_open=False,
    style={"width": "400px"},
)

app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="/")),
            dbc.NavItem(dbc.NavLink("Forecasting", href="/forecasting")),
            dbc.NavItem(dbc.NavLink("Classification", href="/classification")),
            dbc.NavItem(dbc.Button("Chat", id="chat-toggle", color="light", size="sm", className="ms-2")),
        ],
        brand="Retail Inventory AI",
        brand_href="/",
        color="primary",
        dark=True,
    ),
    dash.page_container,
    chat_sidebar,
    dcc.Store(id="chat-store", data=[]),
], fluid=True)


@callback(
    Output("chat-offcanvas", "is_open"),
    Input("chat-toggle", "n_clicks"),
    State("chat-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_chat(n_clicks, is_open):
    return not is_open


@callback(
    Output("chat-history", "children"),
    Output("chat-store", "data"),
    Output("chat-input", "value"),
    Input("chat-send", "n_clicks"),
    Input("suggest-1", "n_clicks"),
    Input("suggest-2", "n_clicks"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True,
)
def handle_chat(send_clicks, s1_clicks, s2_clicks, user_input, history):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "suggest-1":
        question = "Which SKUs should I reorder?"
    elif trigger == "suggest-2":
        question = "What are my slowest movers?"
    else:
        question = user_input

    if not question:
        return dash.no_update, dash.no_update, ""

    # RAG: retrieve + query
    relevant = retrieve(question, _index, _docs, top_k=10)
    context = format_context(relevant)
    answer = query_ollama(question, context)

    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    # Render chat bubbles
    bubbles = []
    for msg in history:
        if msg["role"] == "user":
            bubbles.append(dbc.Card(
                dbc.CardBody(html.P(msg["content"], className="mb-0")),
                className="mb-2 ms-5", color="primary", inverse=True,
            ))
        else:
            bubbles.append(dbc.Card(
                dbc.CardBody(html.P(msg["content"], className="mb-0", style={"whiteSpace": "pre-wrap"})),
                className="mb-2 me-5",
            ))

    return bubbles, history, ""


if __name__ == "__main__":
    app.run(debug=True, port=8050)
```

**Step 2: Verify chat works**

Prerequisites:
```bash
ollama pull llama3
ollama serve  # in a separate terminal
```

Run: `cd /Users/riinygiir/RAG_Project && python app.py`
Click "Chat" button in navbar, ask a question.
Expected: Chat sidebar opens, question is sent, Ollama returns an answer based on retrieved inventory data.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add RAG chat sidebar with Ollama integration"
```

---

### Task 11: Final Cleanup & README

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Replace README.md with documentation covering:
- Project description (keep existing)
- Screenshots section (placeholder)
- Setup instructions: pip install, ollama setup, data download
- How to run: `python app.py`
- Project structure
- Tech stack

**Step 2: Run full test suite**

Run: `cd /Users/riinygiir/RAG_Project && python -m pytest tests/ -v`
Expected: All tests pass.

**Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: update README with setup and usage instructions"
```

**Step 4: Push**

```bash
git push origin main
```
