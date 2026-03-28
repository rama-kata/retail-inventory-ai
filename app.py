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
