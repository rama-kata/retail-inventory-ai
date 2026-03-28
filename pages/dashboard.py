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
