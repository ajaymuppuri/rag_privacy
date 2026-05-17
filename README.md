# Privacy RAG Analysis

Analyzes a RAG pipeline built over privacy documentation in `/Users/ajaymuppuri/rag_documents` — corpus stats, chunking, retrieval accuracy, and privacy-relevant retrieval behavior (cross-document leakage, low-confidence matches).

## Setup

```bash
cd /Users/ajaymuppuri/code/rag_privacy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run analysis

```bash
python analyze.py
```

Writes `output/rag_analysis.json` and prints a summary to the terminal.

## What the analysis covers

1. **Corpus** — document sizes, character counts, previews
2. **Index** — chunk counts per source, size distribution, overlap settings
3. **Retrieval** — probe queries with expected source documents; Hit@1 and Hit@k accuracy
4. **Privacy** — cross-document chunks in top-k results, low-confidence retrievals, exposure counts

## Configuration

See `.env.example` for `DOCS_PATH`, chunk size, `TOP_K`, and embedding model.
