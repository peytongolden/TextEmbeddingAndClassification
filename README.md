# Text Embedding Spam Classifier

Compares commercial and local text embedding models for spam classification, evaluating two classifier approaches on a spam/ham dataset.

## Overview

This project benchmarks:
- **Cohere API** (`embed-v4.0`, 1536-dim) vs. **Sentence Transformers** (`all-MiniLM-L6-v2`, 384-dim)
- **Logistic Regression** vs. **Cosine Similarity** classifiers

Both embedding sources and both classifiers are tested against 100 training / 100 test labeled samples.

## Results

| Embeddings | Classifier | Test Accuracy |
|---|---|---|
| Cohere API | Logistic Regression | 99% |
| Cohere API | Cosine Similarity | 99% |
| Local MiniLM | Logistic Regression | 98% |
| Local MiniLM | Cosine Similarity | 98% |

Angular separation between spam and ham centroids: **67°** (Cohere) / **81°** (Local).

## Setup

**Install dependencies:**
```bash
pip install cohere sentence-transformers scikit-learn numpy
```

**Add API keys** (gitignored):
- `cohere.txt` — Cohere API key
- `huggingface.txt` — HuggingFace token

## Usage

Open [`classifier.ipynb`](classifier.ipynb) in Jupyter.

- Set `compute_embeddings = False` to use cached embeddings in `spam/` (no API calls)
- Set `compute_embeddings = True` to recompute (hits the Cohere API)

## Files

```
classifier.ipynb          # Main notebook with experiments and write-up
train.pkl / test.pkl      # 100-sample labeled datasets (0=ham, 1=spam)
spam/                     # Cached embeddings (Cohere + local, train + test)
```
