# Retail Inventory AI — Dash Web App Design

## Overview

A Plotly Dash web application that turns the existing Jupyter notebook pipeline into an interactive dashboard with KPI visualization, demand forecasting, slow-mover classification, and RAG-powered Q&A using a local Ollama model.

## Architecture

```
retail-inventory-ai/
├── app.py                  # Main Dash app, page routing, sidebar chat
├── pages/
│   ├── dashboard.py        # KPI overview + CSV upload
│   ├── forecasting.py      # XGBoost demand forecasting
│   └── classification.py   # Slow-mover detection
├── pipeline/
│   ├── loader.py           # CSV loading, validation, cleaning
│   ├── kpis.py             # Turnover, days-on-hand, ABC classification
│   ├── forecasting.py      # XGBoost regression model
│   ├── classifier.py       # Slow-mover classification
│   └── rag.py              # FAISS indexing + Ollama query
├── assets/                 # Dash CSS/static files
├── data/
│   └── (Warehouse_and_Retail_Sales.csv — gitignored, downloaded separately)
├── requirements.txt
└── README.md
```

## Data Flow

CSV → `loader.py` validates schema & cleans → pipeline modules compute on demand → Dash pages render Plotly charts + tables.

## RAG Flow

CSV rows + item descriptions + computed stats → Sentence-Transformers embeddings → FAISS index → user question → retrieve top-k relevant rows → send context + question to Ollama → display streamed answer.

## Pages

### 1. Dashboard (with Upload)

- Drag-and-drop CSV upload at top, pre-loaded dataset as default
- Schema validation with preview of first 10 rows before processing
- Dropdown filters: supplier, item type, SKU
- KPI cards: total SKUs, avg turnover, top movers
- Plotly charts: turnover distribution histogram, days-on-hand vs turnover scatter (colored by ABC class), sales trend over time
- Sortable data table with KPIs per SKU

### 2. Forecasting

- SKU selector dropdown (filtered to SKUs with sufficient data points)
- XGBoost model only (Prophet evaluated and outperformed in notebook analysis)
- Line chart: actual vs predicted sales
- Metrics display: MAE, RMSE
- Confidence intervals on predictions

### 3. Classification

- Slow-mover threshold slider (default 30th percentile, adjustable)
- Pie chart: fast vs slow mover split
- Table of flagged slow-movers with turnover rate, days-on-hand
- Export flagged items as CSV download

### 4. RAG Chat (Sidebar — available on all pages)

- Collapsible sidebar with chat interface
- FAISS index built on data load from item descriptions + sales stats
- User types question, top-k rows retrieved, sent to Ollama with context
- Streamed response in chat bubbles
- Suggested starter questions

## Tech Stack

- **Frontend:** Plotly Dash, Dash Bootstrap Components
- **ML Pipeline:** pandas, XGBoost, scikit-learn
- **RAG:** Sentence-Transformers, FAISS (faiss-cpu)
- **LLM:** Ollama (local, running Llama 3 or Mistral)
- **Data:** CSV-based, no database needed

## Key Decisions

- **No Prophet:** XGBoost outperformed it significantly (MAE 3.68 vs 71.02). Simplifies the forecasting page.
- **Upload on Dashboard, not separate page:** Reduces navigation, users see data immediately.
- **Chat as sidebar, not page:** Accessible from any view, enables contextual questions.
- **CSV not in repo:** 27MB file gitignored, downloaded separately or uploaded by user.
- **Ollama for LLM:** Local inference on dual NVIDIA 6090s, no API costs.
