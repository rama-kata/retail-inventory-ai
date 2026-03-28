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

# Lazy FAISS index — built on first chat, not on startup
_index = None
_docs = None


def _get_index():
    global _index, _docs
    if _index is None:
        df = load_csv()
        df_unique = df.groupby("sku").agg({
            "description": "first", "item_type": "first",
            "supplier": "first", "total_sales": "sum",
        }).reset_index()
        _index, _docs = build_index(df_unique)
    return _index, _docs

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
    index, docs = _get_index()
    relevant = retrieve(question, index, docs, top_k=10)
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
