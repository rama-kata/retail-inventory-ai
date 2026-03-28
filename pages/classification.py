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
