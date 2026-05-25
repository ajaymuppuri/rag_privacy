# Privacy RAG Analysis

Analyzes a local RAG pipeline built over privacy documentation — corpus stats, chunking,
retrieval accuracy, and privacy-relevant retrieval behavior (cross-document leakage,
low-confidence matches). Also benchmarks 4 embedding models against the same corpus
and visualises each model's vector space in 3D.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

This project uses the standard `.env` / `.env.example` pattern for configuration.

| File | Committed? | Purpose |
|---|---|---|
| `.env.example` | ✅ yes | Template showing what variables the app expects. Safe to share — no secrets. |
| `.env`         | ❌ no (gitignored) | Your local values. Overrides defaults from `.env.example`. |

To configure your own values:

```bash
cp .env.example .env
# then edit .env with your paths / settings
```

`config.py` calls `load_dotenv()` on import, so any value in `.env` is automatically
picked up. If `.env` is missing or a variable is unset, the defaults in `config.py`
apply (e.g. `DOCS_PATH` defaults to `~/rag_documents`).

Why split the two files: `.env.example` documents what config the app needs without
exposing personal paths or (in projects that have them) API keys. New contributors
clone the repo, copy `.env.example` → `.env`, and fill in their own values without
needing to read source code to figure out what env vars exist.

### Available variables

| Variable | Default | What it does |
|---|---|---|
| `DOCS_PATH` | `~/rag_documents` | Folder of `.txt` source documents to index |
| `CHROMA_PATH` | `data/chroma` | Where the default Chroma store is persisted |
| `COLLECTION_NAME` | `privacy_docs` | Default collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model used for embeddings |
| `TOP_K` | `4` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between consecutive chunks |

## Scripts

### `analyze.py` — single-model corpus + retrieval analysis

```bash
python analyze.py
```

Indexes `DOCS_PATH` with the configured `EMBEDDING_MODEL`, runs probe queries from
`rag/analyze.py:PROBE_QUERIES`, and writes `output/rag_analysis.json`. Covers:

1. **Corpus** — document sizes, character counts, previews
2. **Index** — chunk counts per source, size distribution, overlap settings
3. **Retrieval** — Hit@1 and Hit@K accuracy on probe queries
4. **Privacy** — cross-document chunks in top-K, low-confidence retrievals, exposure counts

### `benchmark_models.py` — compare 4 embedding models

```bash
python benchmark_models.py                # full run (ingests into 4 separate stores)
python benchmark_models.py --skip-ingest  # reuse stores; auto-invalidates if corpus changed
```

Ingests the same corpus into 4 per-model Chroma stores under `data/chroma_<model>/`,
then measures Hit@1, Hit@K, average top-1 cosine score, ingest time, and median + p95
query latency for each. Outputs:

- Terminal comparison table + per-query heatmap (✓ / ~ / ✗)
- `output/runs/<timestamp>/benchmark_results.json` — slim summary
- `output/runs/<timestamp>/benchmark_per_query.csv` — per-(model, query) detail
- `output/benchmark_latest.json` — pointer to the most-recent run

Models benchmarked:

| Model | Dims | Size |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ~80 MB |
| `all-MiniLM-L12-v2` | 384 | ~120 MB |
| `all-mpnet-base-v2` | 768 | ~420 MB |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3 GB |

### `visualize.py` — 3D vector-space viewer (one model at a time)

```bash
python visualize.py                                          # uses EMBEDDING_MODEL
python visualize.py --model all-mpnet-base-v2 "your query"
python visualize.py --all --no-show "your query"             # one HTML per model
```

Pulls all stored vectors from a model's Chroma store, embeds the query with the same
model, reduces to 3D with UMAP, and renders an interactive Plotly chart with the query
(red diamond), top-K matches (bright green), and the rest tinted by source file.

Output: `output/viz_<model_safe_name>.html`. Open in a browser; hover any dot for
source, chunk index, cosine score, and a text preview.

**Note:** these are intentionally separate HTML files, not a 2x2 grid. UMAP projects
each model's embedding space independently, so absolute positions across models are
not directly comparable — use `benchmark_models.py` for accuracy comparison, and
`visualize.py` for understanding a single model's topology.

## Project structure

```
rag_privacy/
├── .env.example          template; copy to .env to override defaults
├── config.py             reads .env via python-dotenv
├── requirements.txt
├── analyze.py            entrypoint for single-model analysis
├── benchmark_models.py   4-model comparison
├── visualize.py          per-model 3D viewer
├── rag/
│   ├── loader.py         loads .txt files from DOCS_PATH
│   ├── chunker.py        splits documents into chunks
│   ├── indexer.py        SentenceTransformer + Chroma persistent store
│   ├── retriever.py      cosine top-K retrieval
│   ├── pipeline.py       RAGPipeline wires loader+chunker+indexer+retriever
│   └── analyze.py        probe queries + shared make_probe helper
├── data/                 gitignored — Chroma stores live here
│   ├── chroma/                  default store (analyze.py)
│   └── chroma_<model>/          per-model stores (benchmark_models.py)
└── output/               gitignored — JSON / CSV / HTML reports
    ├── rag_analysis.json
    ├── benchmark_latest.json
    ├── runs/<timestamp>/
    └── viz_<model>.html
```
