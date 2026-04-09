#!/usr/bin/env python3
"""
Generate FinBERT embeddings for articles_max.

Uses ProsusAI/finbert (financial domain BERT, ~440MB) as specified in PDF.
Falls back to bert-base-uncased if FinBERT cannot load.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

# Absolute path from script location
script_file = Path(__file__).resolve()
service_root = script_file.parent.parent.parent.parent
data_path = service_root / "training_data"

articles = pd.read_csv(data_path / "articles_max.csv")

print(f"[EMBED] {len(articles)} articles")

# Use FinBERT (financial domain) as specified in PDF Section 3
MODEL_NAME = "ProsusAI/finbert"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    print(f"[MODEL] Using FinBERT ({MODEL_NAME}) — financial domain embeddings")
except Exception as e:
    print(f"[WARN] FinBERT load failed ({e}), falling back to bert-base-uncased")
    MODEL_NAME = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
print(f"[MODEL] Device: {device}\n")

embeddings_list = []
for i in range(0, len(articles), 32):  # Smaller batch (FinBERT uses more RAM)
    batch = articles.iloc[i:i+32]
    texts = batch['title'].astype(str) + " " + batch['content'].astype(str)
    texts = [str(t)[:512] for t in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        # CLS token embedding (position 0)
        embeddings_list.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    pct = min(i+32, len(articles)) / len(articles) * 100
    print(f"  {pct:.0f}%", end="\r", flush=True)

embeddings = np.vstack(embeddings_list)
np.save(data_path / "btcusdt_article_embeddings_max.npy", embeddings)
print(f"\n[OK] Saved: btcusdt_article_embeddings_max.npy {embeddings.shape}")
print(f"     Model: {MODEL_NAME}")
