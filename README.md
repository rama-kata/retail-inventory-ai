# retail-inventory-ai

A step-by-step portfolio project demonstrating how to optimize retail inventory management by combining traditional analytics, machine learning forecasting, transformer fine-tuning, and a Retrieval-Augmented Generation (RAG) Q&A system.

## Project Description

SmartInventoryAI is an end-to-end proof-of-concept showing how small retailers can leverage data-driven techniques to:

1. **Understand current operations**
   - Compute key performance indicators (turnover, days-on-hand, ABC classification)
   - Visualize SKU-level patterns and identify outliers

2. **Forecast demand**
   - Use both statistical (ARIMA/Prophet) and machine-learning (XGBoost/LightGBM) approaches
   - Compare model accuracy (MAE, RMSE, MAPE) and select a winner

3. **Fine-tune a transformer for classification**
   - Label SKUs as "fast movers" vs. "slow movers" based on turnover percentiles
   - Fine-tune DistilBERT/T5 on product descriptions to automate reorder urgency

4. **Build a Retail-specific RAG system**
   - Index your CSV data and process documentation with Sentence-Transformers + FAISS
   - Serve up answers to questions like "When should I reorder SKU 123?" via a simple CLI or notebook

## Tech Stack

- Python, Jupyter Notebook
- Sentence-Transformers, FAISS
- XGBoost, LightGBM, ARIMA, Prophet
- DistilBERT/T5 (Hugging Face)

## Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/riiny-riiny/retail-inventory-ai.git
   cd retail-inventory-ai
   ```

2. Open the notebook
   ```bash
   jupyter notebook rag_project.ipynb
   ```
