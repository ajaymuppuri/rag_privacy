"""
3D vector-space viewer for a SINGLE model's Chroma store.

Use this AFTER running benchmark_models.py — the benchmark creates the per-model
stores at data/chroma_<model>/, and this script visualises any one of them.

Why one-at-a-time and not a 2x2 grid:
  UMAP projects each model's embedding space independently, so absolute
  positions across panels are not comparable.  Use the benchmark TABLE to
  compare *accuracy*; use this viewer to understand a single model's *topology*.

Usage:
    python visualize.py                                   # uses config.EMBEDDING_MODEL
    python visualize.py --model all-mpnet-base-v2
    python visualize.py --model BAAI/bge-large-en-v1.5 "How is patient data protected?"
    python visualize.py --all --no-show                   # all 4 models → 4 HTML files

Output:
    output/viz_<model_safe_name>.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import config
from benchmark_models import COLLECTION_NAME, MODELS, chroma_path_for
from rag.pipeline import RAGPipeline

# ── Named constants ──────────────────────────────────────────────────────────
EPS                   = 1e-9   # cosine-norm denominator guard
HOVER_PREVIEW_CHARS   = 160
DEFAULT_QUERY         = "What is data minimization?"

# Visual styling — top-K ranking
RANK_COLORS = ["#00e676", "#69f0ae", "#b9f6ca", "#e8f5e9"]
RANK_SIZE_BASE, RANK_SIZE_STEP, RANK_SIZE_FLOOR = 18, 3, 8

# Source-cluster tinting (non-top-K dots, by source file)
SOURCE_PALETTE = [
    "#5c6bc0", "#26a69a", "#ab47bc", "#ef5350",
    "#ffa726", "#8d6e63", "#789262",
]
MUTED_ALPHA_HEX = "66"   # ~40% opacity suffix

# Plot chrome
PLOT_BG, PAPER_BG = "rgb(8,8,18)", "rgb(12,12,22)"
GRID_COLOR        = "rgb(40,40,60)"


# ── Dimensionality reduction ──────────────────────────────────────────────────
def reduce_to_3d(matrix: np.ndarray) -> np.ndarray:
    """UMAP to 3D using cosine metric, with deterministic seed."""
    import umap
    return umap.UMAP(
        n_components=3,
        metric="cosine",
        random_state=42,
    ).fit_transform(matrix)


# ── Resolve where this model's vectors live ──────────────────────────────────
def resolve_pipeline(model_name: str) -> RAGPipeline:
    """Use the per-model benchmark store if the model is registered;
    otherwise fall back to the default config.CHROMA_PATH / COLLECTION_NAME."""
    registry_match = next((m for m in MODELS if m.name == model_name), None)
    if registry_match:
        path, coll = chroma_path_for(model_name), COLLECTION_NAME
    else:
        path, coll = config.CHROMA_PATH, config.COLLECTION_NAME

    if not path.exists():
        print(f"  No store at {path}.")
        print(f"  Run `python benchmark_models.py` first (or `python analyze.py` for the default store).")
        sys.exit(1)

    return RAGPipeline(
        chroma_path=path,
        collection_name=coll,
        embedding_model=model_name,
    )


# ── Build one 3D figure for one model ────────────────────────────────────────
def build_figure(model_name: str, query: str, top_k: int) -> go.Figure:
    pipeline = resolve_pipeline(model_name)
    indexer  = pipeline.indexer
    total    = indexer.collection.count()
    if total == 0:
        print(f"  Store for {model_name} is empty.  Run benchmark_models.py first.")
        sys.exit(1)

    # Pull everything Chroma has stored
    stored    = indexer.collection.get(include=["embeddings", "documents", "metadatas"])
    all_vecs  = np.asarray(stored["embeddings"], dtype=np.float32)
    all_texts = stored["documents"]
    all_metas = stored["metadatas"]

    # Embed query with the same model (single Python call → one model forward pass)
    q_vec = np.asarray(indexer.embed([query])[0], dtype=np.float32)

    # Vectorised cosine sim against every stored chunk (no Python loop)
    norms  = np.linalg.norm(all_vecs, axis=1)
    q_norm = np.linalg.norm(q_vec)
    scores = (all_vecs @ q_vec) / (norms * q_norm + EPS)

    # Ranks
    order      = np.argsort(scores)[::-1]
    top_idx    = order[:top_k]
    bottom_idx = order[-2:]

    # Single UMAP fit on (all docs + query) so the query lands in the same space
    combined = np.vstack([all_vecs, q_vec])
    reduced  = reduce_to_3d(combined)
    docs_3d  = reduced[:-1]
    query_3d = reduced[-1]

    # Style: start by tinting every chunk by its source file (so clusters are visible)
    sources       = sorted({m["source"] for m in all_metas})
    source_colour = {s: SOURCE_PALETTE[i % len(SOURCE_PALETTE)] for i, s in enumerate(sources)}
    colors        = [source_colour[m["source"]] + MUTED_ALPHA_HEX for m in all_metas]
    sizes         = [7] * len(all_texts)

    # Then override the top-K with bright green markers (overlay wins visually)
    for rank, idx in enumerate(top_idx):
        colors[idx] = RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]
        sizes[idx]  = max(RANK_SIZE_BASE - rank * RANK_SIZE_STEP, RANK_SIZE_FLOOR)

    fig = go.Figure()

    # Document chunks
    fig.add_trace(go.Scatter3d(
        x=docs_3d[:, 0], y=docs_3d[:, 1], z=docs_3d[:, 2],
        mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.9,
                    line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
        hovertext=[
            f"<b>{m['source']}</b>  chunk #{m['chunk_index']}<br>"
            f"cosine sim: {scores[i]:.4f}<br><br>"
            f"{t[:HOVER_PREVIEW_CHARS].replace(chr(10), ' ')}..."
            for i, (t, m) in enumerate(zip(all_texts, all_metas))
        ],
        hoverinfo="text",
        name="Chunks",
    ))

    # Query point
    fig.add_trace(go.Scatter3d(
        x=[query_3d[0]], y=[query_3d[1]], z=[query_3d[2]],
        mode="markers+text",
        marker=dict(size=18, color="#ff1744", symbol="diamond",
                    line=dict(color="#b71c1c", width=2)),
        text=["QUERY"],
        textposition="middle right",
        textfont=dict(color="white", size=11),
        hovertext=f"<b>QUERY</b><br>{query}",
        hoverinfo="text",
        name="Query",
    ))

    # Lines: query → top-K matches (width and color encode rank)
    for rank, idx in enumerate(top_idx):
        fig.add_trace(go.Scatter3d(
            x=[query_3d[0], docs_3d[idx, 0]],
            y=[query_3d[1], docs_3d[idx, 1]],
            z=[query_3d[2], docs_3d[idx, 2]],
            mode="lines",
            line=dict(
                color=RANK_COLORS[min(rank, len(RANK_COLORS) - 1)],
                width=max(4 - rank, 1),
            ),
            hoverinfo="skip",
            name=f"#{rank+1}  {all_metas[idx]['source']}  sim={scores[idx]:.3f}",
        ))

    subtitle = "  |  ".join(
        f"#{r+1} {all_metas[i]['source']} ({scores[i]:.3f})"
        for r, i in enumerate(top_idx)
    )
    fig.update_layout(
        title=dict(
            text=(
                f'<b>{model_name}</b>  ({all_vecs.shape[1]}-dim, {total} chunks)<br>'
                f'<sup>Query: "{query}"</sup><br>'
                f'<sup style="color:#aaa">{subtitle}</sup>'
            ),
            font=dict(size=13, color="white"),
        ),
        scene=dict(
            xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3",
            bgcolor=PLOT_BG,
            xaxis=dict(gridcolor=GRID_COLOR, color="grey"),
            yaxis=dict(gridcolor=GRID_COLOR, color="grey"),
            zaxis=dict(gridcolor=GRID_COLOR, color="grey"),
        ),
        paper_bgcolor=PAPER_BG,
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(30,30,50,0.8)", bordercolor="grey", borderwidth=1),
        margin=dict(l=0, r=0, t=110, b=0),
    )

    # Terminal summary (matches in-figure ranks)
    print(f"\n  Top {top_k} matches for {model_name}:")
    for rank, idx in enumerate(top_idx):
        print(f"    [{rank+1}]  {scores[idx]:.4f}  {all_metas[idx]['source']}  #{all_metas[idx]['chunk_index']}")
    print(f"  Worst 2 (for context):")
    for idx in bottom_idx:
        print(f"     ✗  {scores[idx]:.4f}  {all_metas[idx]['source']}  #{all_metas[idx]['chunk_index']}")

    return fig


def save_figure(fig: go.Figure, model_name: str) -> Path:
    """Filename includes model name so concurrent runs don't collide."""
    safe = model_name.replace("/", "_").replace("-", "_")
    out_dir = config.PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"viz_{safe}.html"
    fig.write_html(str(out))
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────
def visualize_one(model_name: str, query: str, top_k: int, show: bool) -> None:
    print(f"\n[{model_name}]")
    t0  = time.time()
    fig = build_figure(model_name, query, top_k)
    out = save_figure(fig, model_name)
    print(f"  Saved → {out}  ({time.time() - t0:.1f}s)")
    if show:
        fig.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="3D vector-space viewer (per model)")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY,
                        help="Query to embed and visualise")
    parser.add_argument("--model", default=config.EMBEDDING_MODEL,
                        help="Model name (default: config.EMBEDDING_MODEL)")
    parser.add_argument("--all", action="store_true",
                        help="Run for all 4 benchmarked models (4 HTML files)")
    parser.add_argument("--top-k", type=int, default=config.TOP_K,
                        help=f"Matches to highlight (default: {config.TOP_K})")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not auto-open in browser (useful with --all)")
    args = parser.parse_args()

    if args.all:
        for m in MODELS:
            visualize_one(m.name, args.query, args.top_k, show=not args.no_show)
        print(f"\nDone.  Open the HTML files in {config.PROJECT_ROOT / 'output'} to compare topologies.")
    else:
        visualize_one(args.model, args.query, args.top_k, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
