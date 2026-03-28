from typing import List, Tuple

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


def build_index(df: pd.DataFrame) -> Tuple:
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


def retrieve(query: str, index, docs: List[str], top_k: int = 5) -> List[str]:
    embedder = _get_embedder()
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [docs[i] for i in indices[0] if i < len(docs)]


def format_context(docs: List[str]) -> str:
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
