"""
4-model RAG benchmark — pure comparison output (no 3D).

Ingests the same corpus into 4 separate Chroma stores, then measures:
  • Hit@1, Hit@K accuracy on PROBE_QUERIES from rag.analyze
  • Average top-1 cosine similarity (model confidence)
  • Ingest time (one-off cost)
  • Median + p95 query latency (per-query cost)

Outputs (each run timestamped, never overwrites history):
  • output/runs/<ts>/benchmark_results.json   slim summary
  • output/runs/<ts>/benchmark_per_query.csv  per-(model, query) rows
  • output/benchmark_latest.json              pointer to most-recent run

For 3D topology visualisation of a single model, see visualize.py.

Usage:
    python benchmark_models.py
    python benchmark_models.py --skip-ingest   # reuse existing stores (auto-invalidated if corpus changed)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Optional

import config
from rag.analyze import PROBE_QUERIES, make_probe
from rag.chunker import chunk_documents
from rag.loader import load_documents
from rag.pipeline import RAGPipeline

# ── Model registry ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    name: str        # huggingface identifier (what SentenceTransformer takes)
    label: str       # short name for tables
    dims: int        # output dimension
    size_mb: int     # approximate download size


MODELS: list[ModelSpec] = [
    ModelSpec("all-MiniLM-L6-v2",        "MiniLM-L6",   384,   80),
    ModelSpec("all-MiniLM-L12-v2",       "MiniLM-L12",  384,  120),
    ModelSpec("all-mpnet-base-v2",       "MPNet",       768,  420),
    ModelSpec("BAAI/bge-large-en-v1.5",  "BGE-large",  1024, 1300),
]

# ── Named constants (no more magic in the body) ──────────────────────────────
COLLECTION_NAME      = "benchmark"
WARMUP_QUERY         = "warmup query for first-call model overhead"
FINGERPRINT_FILENAME = ".corpus_fingerprint"
P95                  = 0.95
QUERY_TRUNCATION     = 47   # chars shown in heatmap before ellipsis

# ── TTY-aware ANSI colors ────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()
def _c(code: str) -> str: return code if _USE_COLOR else ""
GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    _c("\033[92m"), _c("\033[91m"), _c("\033[93m"),
    _c("\033[2m"),  _c("\033[1m"),  _c("\033[0m"),
)


# ── Path helpers (shared with visualize.py) ──────────────────────────────────
def chroma_path_for(model_name: str) -> Path:
    """Where each model's Chroma store lives."""
    safe = model_name.replace("/", "_").replace("-", "_")
    return config.PROJECT_ROOT / "data" / f"chroma_{safe}"


def _fingerprint_path_for(model_name: str) -> Path:
    return chroma_path_for(model_name) / FINGERPRINT_FILENAME


def corpus_fingerprint(docs_path: Path) -> str:
    """Stable hash of all docs — lets --skip-ingest detect a changed corpus."""
    h = hashlib.sha256()
    for p in sorted(Path(docs_path).glob("*.txt")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest_all(skip: bool) -> dict[str, tuple[RAGPipeline, float]]:
    """Returns {model_name: (pipeline, ingest_seconds)}.  ingest_seconds=0 if skipped."""
    documents = load_documents(config.DOCS_PATH)
    chunks    = chunk_documents(documents, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    fp        = corpus_fingerprint(config.DOCS_PATH)
    print(f"  corpus fingerprint: {fp}  ({len(chunks)} chunks from {len(documents)} docs)")

    out: dict[str, tuple[RAGPipeline, float]] = {}

    for m in MODELS:
        path    = chroma_path_for(m.name)
        fp_file = _fingerprint_path_for(m.name)

        try:
            pipeline = RAGPipeline(
                chroma_path=path,
                collection_name=COLLECTION_NAME,
                embedding_model=m.name,
            )
        except Exception as e:
            print(f"  [{m.label:<11}] {RED}FAILED to load model{RESET}: {e}")
            continue

        existing_fp = fp_file.read_text().strip() if fp_file.exists() else None
        needs_reindex = (
            not skip
            or pipeline.chunk_count == 0
            or existing_fp != fp
        )

        if needs_reindex:
            reason = (
                "forced"          if not skip
                else "empty"      if pipeline.chunk_count == 0
                else "corpus changed"
            )
            print(f"  [{m.label:<11}] reindexing ({reason}) ... ", end="", flush=True)
            t0 = time.time()
            try:
                # Re-ingest using the existing pipeline (which handles chunking + indexing)
                pipeline.ingest(reset=True)
                elapsed = time.time() - t0
                fp_file.write_text(fp)
                print(f"done in {elapsed:.1f}s")
                out[m.name] = (pipeline, elapsed)
            except Exception as e:
                print(f"{RED}FAILED{RESET}: {e}")
        else:
            print(f"  [{m.label:<11}] reusing existing index ({pipeline.chunk_count} chunks)")
            out[m.name] = (pipeline, 0.0)
    return out


# ── Evaluate ──────────────────────────────────────────────────────────────────
@dataclass
class ModelResult:
    model: str
    label: str
    dims: int
    size_mb: int
    n_probes: int
    hit_at_1: int
    hit_at_k: int
    hit_at_1_rate: float
    hit_at_k_rate: float
    avg_top_score: float
    ingest_seconds: float
    latency_median_ms: float
    latency_p95_ms: float
    probes: list[dict]   # serialised RetrievalProbe per query


def evaluate(pipelines: dict[str, tuple[RAGPipeline, float]]) -> list[ModelResult]:
    """Run probes against each model, measuring accuracy AND latency."""
    results: list[ModelResult] = []

    for m in MODELS:
        if m.name not in pipelines:
            continue
        pipeline, ingest_seconds = pipelines[m.name]

        # Warm-up — first call after model load pays JIT / cache cost we don't want to time
        pipeline.retrieve(WARMUP_QUERY)

        # Single pass: time each call and reuse its result via make_probe
        latencies_ms: list[float] = []
        probes = []
        for query, expected in PROBE_QUERIES:
            t0 = time.perf_counter()
            chunks = pipeline.retrieve(query)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            probes.append(make_probe(query, expected, chunks))

        n          = len(probes)
        hits1      = sum(p.hit_at_1 for p in probes)
        hits_k     = sum(p.hit_at_k for p in probes)
        top_scores = [p.top_score for p in probes]

        results.append(ModelResult(
            model             = m.name,
            label             = m.label,
            dims              = m.dims,
            size_mb           = m.size_mb,
            n_probes          = n,
            hit_at_1          = hits1,
            hit_at_k          = hits_k,
            hit_at_1_rate     = hits1 / n,
            hit_at_k_rate     = hits_k / n,
            avg_top_score     = sum(top_scores) / len(top_scores) if top_scores else 0.0,
            ingest_seconds    = round(ingest_seconds, 2),
            latency_median_ms = round(median(latencies_ms), 1),
            latency_p95_ms    = round(_percentile(latencies_ms, P95), 1),
            probes            = [asdict(p) for p in probes],
        ))
    return results


# ── Output: comparison table ─────────────────────────────────────────────────
def print_comparison_table(results: list[ModelResult]) -> None:
    n = results[0].n_probes
    print(f"\n{BOLD}{'═'*92}{RESET}")
    print(f"{BOLD}MODEL COMPARISON{RESET}  ({n} probes, top_k={config.TOP_K})")
    print("═" * 92)
    print(
        f"  {'Model':<12} {'Dims':>5} {'Size':>6}  "
        f"{'Hit@1':>10} {'Hit@K':>10}  "
        f"{'Avg score':>10}  {'Ingest':>8}  {'p50':>8}  {'p95':>8}"
    )
    print("  " + "-" * 90)

    best_h1  = max(r.hit_at_1_rate     for r in results)
    best_hk  = max(r.hit_at_k_rate     for r in results)
    best_lat = min(r.latency_median_ms for r in results)

    for r in results:
        h1  = f"{r.hit_at_1:>2}/{n} {r.hit_at_1_rate:>4.0%}"
        hk  = f"{r.hit_at_k:>2}/{n} {r.hit_at_k_rate:>4.0%}"
        lat = f"{r.latency_median_ms:>5.1f}ms"
        if r.hit_at_1_rate     == best_h1:  h1  = f"{GREEN}{h1}{RESET}"
        if r.hit_at_k_rate     == best_hk:  hk  = f"{GREEN}{hk}{RESET}"
        if r.latency_median_ms == best_lat: lat = f"{GREEN}{lat}{RESET}"

        print(
            f"  {r.label:<12} {r.dims:>5} {r.size_mb:>4}MB  "
            f"{h1:>19} {hk:>19}  "
            f"{r.avg_top_score:>10.4f}  "
            f"{r.ingest_seconds:>6.1f}s  "
            f"{lat:>17}  {r.latency_p95_ms:>5.1f}ms"
        )

    print()
    print(f"  {DIM}Hit@1     correct doc was the TOP result{RESET}")
    print(f"  {DIM}Hit@K     correct doc was anywhere in top {config.TOP_K} results{RESET}")
    print(f"  {DIM}Avg score mean cosine similarity of top-1 across all probes (confidence){RESET}")
    print(f"  {DIM}p50 / p95 end-to-end query latency (embed + search), after warm-up{RESET}")


# ── Output: per-query heatmap ────────────────────────────────────────────────
def print_per_query_heatmap(results: list[ModelResult]) -> None:
    print(f"\n{BOLD}PER-QUERY RESULTS{RESET}  (✓ top-1   ~ in top-K   ✗ miss)")
    print("═" * 92)
    labels        = [r.label for r in results]
    header_models = "  ".join(f"{l:>10}" for l in labels)
    sep           = "-" * len(header_models)
    print(f"  {'Query':<{QUERY_TRUNCATION + 1}}  {header_models}")
    print(f"  {'-' * (QUERY_TRUNCATION + 1)}  {sep}")

    for i, (query, _) in enumerate(PROBE_QUERIES):
        q_short = query if len(query) <= QUERY_TRUNCATION else query[: QUERY_TRUNCATION - 3] + "..."
        cells = []
        for r in results:
            p = r.probes[i]
            if   p["hit_at_1"]: cells.append(f"{GREEN}{'✓':>10}{RESET}")
            elif p["hit_at_k"]: cells.append(f"{YELLOW}{'~':>10}{RESET}")
            else:               cells.append(f"{RED}{'✗':>10}{RESET}")
        print(f"  {q_short:<{QUERY_TRUNCATION + 1}}  {'  '.join(cells)}")


# ── Output: save reports (timestamped — never clobbers history) ──────────────
def save_reports(results: list[ModelResult]) -> Path:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.PROJECT_ROOT / "output" / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Slim JSON (no per-probe details)
    summary = [{k: v for k, v in asdict(r).items() if k != "probes"} for r in results]
    (run_dir / "benchmark_results.json").write_text(json.dumps(summary, indent=2))

    # Convenience pointer
    latest = config.PROJECT_ROOT / "output" / "benchmark_latest.json"
    latest.write_text(json.dumps({"run": ts, "path": str(run_dir)}, indent=2))

    # Full CSV
    csv_path = run_dir / "benchmark_per_query.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "label", "dims", "query", "expected", "got",
            "top_score", "hit_at_1", "hit_at_k",
            "retrieved_sources_json", "retrieved_scores_json",
        ])
        for r in results:
            for p in r.probes:
                w.writerow([
                    r.model, r.label, r.dims,
                    p["query"], p["expected_source"], p["top_source"],
                    p["top_score"], int(p["hit_at_1"]), int(p["hit_at_k"]),
                    json.dumps(p["retrieved_sources"]),
                    json.dumps(p["scores"]),
                ])

    print(f"\n  Saved → {run_dir / 'benchmark_results.json'}")
    print(f"  Saved → {csv_path}")
    print(f"  Latest pointer → {latest}")
    print(f"\n  For 3D topology of each model:")
    print(f"    python visualize.py --all --no-show")
    print(f"    python visualize.py --model BAAI/bge-large-en-v1.5 \"your query\"")
    return run_dir


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="4-model RAG accuracy benchmark")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Reuse existing stores (auto-invalidated if corpus changed)")
    args = parser.parse_args()

    print("\n[1/3] Ingesting corpus into 4 model stores...")
    pipelines = ingest_all(skip=args.skip_ingest)
    if not pipelines:
        print(f"\n{RED}No models could be loaded.{RESET}  Aborting.")
        return 1

    print("\n[2/3] Running probe queries...")
    results = evaluate(pipelines)

    print("\n[3/3] Results:")
    print_comparison_table(results)
    print_per_query_heatmap(results)
    save_reports(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
