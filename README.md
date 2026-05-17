# Privacy RAG Pipeline

Retrieval-augmented generation over privacy documentation in `/Users/ajaymuppuri/rag_documents`.

## Setup

```bash
cd /Users/ajaymuppuri/code/rag_privacy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: set OPENAI_API_KEY
```

## Usage

**1. Ingest documents** (load, chunk, embed, store in ChromaDB):

```bash
python main.py ingest
```

Use `--reset` to rebuild the index from scratch.

**2. Ask a question:**

```bash
python main.py ask "What is data minimization?"
python main.py ask "What rights does GDPR give users?"
```

**3. Interactive chat:**

```bash
python main.py chat
```

**4. Check index status:**

```bash
python main.py status
```

## How it works

1. **Load** — Reads `.txt` / `.md` / `.html` from `DOCS_PATH`. HTML (e.g. GDPR scrape) is stripped to plain text.
2. **Chunk** — Splits text with overlap at sentence/paragraph boundaries.
3. **Embed** — `sentence-transformers` (`all-MiniLM-L6-v2`) produces vectors stored in ChromaDB.
4. **Retrieve** — Cosine similarity search returns top-k chunks for each query.
5. **Generate** — Uses OpenAI if `OPENAI_API_KEY` is set, else Ollama if running locally, else returns top passages.

## Documents indexed

- `encryption_guide.txt`
- `gdpr_overview.txt`
- `hipaa_privacy_rule.txt`
- `privacy_attacks.txt`
- `privacy_by_design.txt`
- `privacy_principles.txt`
- `user_privacy_rights.txt`

## Configuration

See `.env.example` for `DOCS_PATH`, chunk size, `TOP_K`, and LLM settings.
