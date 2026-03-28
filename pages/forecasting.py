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
