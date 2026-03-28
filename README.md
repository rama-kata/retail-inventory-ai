# retail-inventory-ai

A step-by-step portfolio project demonstrating how to optimize retail inventory management by combining traditional analytics, machine learning forecasting, transformer fine-tuning, and a Retrieval-Augmented Generation (RAG) Q&A system.

## Project Description

SmartInventoryAI is an end-to-end proof-of-concept showing how small retailers can leverage data-driven techniques to:

1. **Understand current operations**
   - Compute key performance indicators (turnover, days-on-hand, ABC classification)
   - Visualize SKU-level patterns and identify outliers

2. **Forecast demand**
   - Use XGBoost with lag features and monthly seasonality
   - Evaluate accuracy with MAE and RMSE on a held-out test set

3. **Classify slow-moving inventory**
   - Label SKUs as "fast movers" vs. "slow movers" based on turnover percentiles
   - Adjust threshold interactively via a slider

4. **RAG-powered Q&A**
   - Index item descriptions with Sentence-Transformers + FAISS
   - Answer inventory questions via Ollama (local LLM) in a chat sidebar

## Dash Web App

An interactive multi-page dashboard built with Plotly Dash:

- **Dashboard** (`/`) — KPI cards, turnover histogram, days-on-hand scatter, sales trend, filterable data table, CSV upload
- **Forecasting** (`/forecasting`) — SKU-level XGBoost demand forecasting with actual vs predicted chart and feature importance
- **Classification** (`/classification`) — Slow-mover classification with adjustable percentile threshold, pie chart, scatter plot, CSV export
- **Chat Sidebar** — RAG-powered Q&A using FAISS retrieval + Ollama, accessible from any page

## Tech Stack

- **Frontend:** Plotly Dash, Dash Bootstrap Components (DARKLY theme)
- **Data:** pandas, NumPy
- **ML:** XGBoost, scikit-learn
- **RAG:** Sentence-Transformers (all-MiniLM-L6-v2), faiss-cpu, Ollama
- **Testing:** pytest

## Project Structure

```
├── app.py                  # Dash app entrypoint with chat sidebar
├── pages/
│   ├── dashboard.py        # KPI dashboard page
│   ├── forecasting.py      # XGBoost forecasting page
│   └── classification.py   # Slow-mover classification page
├── pipeline/
│   ├── loader.py           # CSV loading, validation, cleaning (cached)
│   ├── kpis.py             # KPI computation (turnover, DoH, ABC)
│   ├── forecasting.py      # XGBoost demand forecasting
│   ├── classifier.py       # Slow-mover classification
│   └── rag.py              # FAISS indexing, retrieval, Ollama query
├── tests/                  # pytest test suite
├── data/                   # CSV data (gitignored)
├── docs/                   # Design docs and plans
├── requirements.txt        # Python dependencies
└── rag_project.ipynb       # Original exploration notebook
```

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) (for RAG chat)

### Setup

```bash
git clone https://github.com/rama-kata/retail-inventory-ai.git
cd retail-inventory-ai

pip install -r requirements.txt
```

### Data

Place `Warehouse_and_Retail_Sales.csv` in the `data/` directory. The app loads it automatically on startup.

### Ollama (for chat)

```bash
ollama pull llama3
ollama serve   # keep running in a separate terminal
```

### Run

```bash
python app.py
```

Open http://127.0.0.1:8050 in your browser.

### Tests

```bash
python -m pytest tests/ -v
```
