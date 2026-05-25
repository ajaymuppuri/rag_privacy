"""RAG analysis: corpus stats, retrieval behavior, and privacy-relevant observations."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import config
from rag.chunker import Chunk, chunk_documents
from rag.loader import load_documents
from rag.pipeline import RAGPipeline
# Probe queries mapped to expected source documents (for retrieval accuracy checks)
PROBE_QUERIES = [
    # Clear, unambiguous — any model should get these right
    ("What is data minimization?", "privacy_principles.txt"),
    ("How does encryption protect data at rest?", "encryption_guide.txt"),
    ("What is a membership inference attack?", "privacy_attacks.txt"),
    ("What rights do users have over their data?", "user_privacy_rights.txt"),
    ("What does HIPAA require for PHI?", "hipaa_privacy_rule.txt"),
    ("What is privacy by design?", "privacy_by_design.txt"),
    ("What is the right to erasure under GDPR?", "user_privacy_rights.txt"),
    # Overlapping topics — harder, better models should win here
    ("How long should patient health records be retained?", "hipaa_privacy_rule.txt"),
    ("What is TLS and why is it used for data in transit?", "encryption_guide.txt"),
    ("How can anonymized data be re-identified?", "privacy_attacks.txt"),
    ("Should privacy settings default to most or least restrictive?", "privacy_by_design.txt"),
    ("What data can be collected under the purpose limitation principle?", "privacy_principles.txt"),
    ("Can a patient request a copy of their medical records?", "hipaa_privacy_rule.txt"),
    ("What is end-to-end encryption and when should it be used?", "encryption_guide.txt"),
    ("What 18 identifiers must be removed to de-identify health data?", "hipaa_privacy_rule.txt"),
    ("What are side channel attacks?", "privacy_attacks.txt"),
    ("How should encryption keys be managed and rotated?", "encryption_guide.txt"),
    ("Can users export their data in machine-readable format?", "user_privacy_rights.txt"),
    ("What does proactive privacy mean in system design?", "privacy_by_design.txt"),
    ("How does differential privacy protect individual records?", "privacy_attacks.txt"),
]


@dataclass
class CorpusStats:
    document_count: int
    total_chars: int
    documents: list[dict[str, Any]]


@dataclass
class IndexStats:
    chunk_count: int
    chunks_per_source: dict[str, int]
    avg_chunk_chars: float
    min_chunk_chars: int
    max_chunk_chars: int
    overlap_chars: int


@dataclass
class RetrievalProbe:
    query: str
    expected_source: str
    top_source: str
    top_score: float
    hit_at_1: bool
    hit_at_k: bool
    retrieved_sources: list[str]
    scores: list[float]


def analyze_corpus(docs_path: Path) -> CorpusStats:
    documents = load_documents(docs_path)
    doc_rows = [
        {
            "source": d.source,
            "chars": len(d.text),
            "lines": d.text.count("\n") + 1,
            "preview": d.text[:120].replace("\n", " "),
        }
        for d in documents
    ]
    return CorpusStats(
        document_count=len(documents),
        total_chars=sum(r["chars"] for r in doc_rows),
        documents=doc_rows,
    )


def analyze_index(chunks: list[Chunk]) -> IndexStats:
    by_source = Counter(c.source for c in chunks)
    lengths = [len(c.text) for c in chunks]
    return IndexStats(
        chunk_count=len(chunks),
        chunks_per_source=dict(by_source),
        avg_chunk_chars=sum(lengths) / len(lengths) if lengths else 0,
        min_chunk_chars=min(lengths) if lengths else 0,
        max_chunk_chars=max(lengths) if lengths else 0,
        overlap_chars=config.CHUNK_OVERLAP,
    )


def make_probe(query: str, expected: str, chunks) -> RetrievalProbe:
    """Build a RetrievalProbe from already-retrieved chunks.

    Shared by analyze_retrieval() and benchmark_models.evaluate() so the
    hit@1 / hit@k logic lives in exactly one place.
    """
    sources = [c.source for c in chunks]
    scores = [round(c.score, 4) for c in chunks]
    return RetrievalProbe(
        query=query,
        expected_source=expected,
        top_source=sources[0] if sources else "",
        top_score=scores[0] if scores else 0.0,
        hit_at_1=expected in sources[:1],
        hit_at_k=expected in sources,
        retrieved_sources=sources,
        scores=scores,
    )


def analyze_retrieval(
    pipeline: RAGPipeline, probes: list[tuple[str, str]]
) -> list[RetrievalProbe]:
    return [make_probe(q, e, pipeline.retrieve(q)) for q, e in probes]


def analyze_privacy_exposure(
    pipeline: RAGPipeline, probes: list[tuple[str, str]]
) -> dict[str, Any]:
    """Privacy-relevant RAG behaviors: cross-doc leakage and low-confidence retrieval."""
    cross_doc_queries: list[dict[str, Any]] = []
    low_confidence: list[dict[str, Any]] = []
    source_exposure = Counter()

    for query, expected in probes:
        chunks = pipeline.retrieve(query)
        unique_sources = {c.source for c in chunks}
        for c in chunks:
            source_exposure[c.source] += 1

        unexpected = [c for c in chunks if c.source != expected]
        if unexpected:
            cross_doc_queries.append(
                {
                    "query": query,
                    "expected": expected,
                    "unexpected_chunks": [
                        {
                            "source": c.source,
                            "score": round(c.score, 4),
                            "excerpt": c.text[:200],
                        }
                        for c in unexpected
                    ],
                }
            )

        if chunks and chunks[0].score < 0.5:
            low_confidence.append(
                {
                    "query": query,
                    "top_score": round(chunks[0].score, 4),
                    "note": "Weak semantic match — risk of irrelevant context in LLM prompt",
                }
            )

    return {
        "cross_document_retrieval": cross_doc_queries,
        "low_confidence_queries": low_confidence,
        "source_exposure_counts": dict(source_exposure),
        "observations": [
            "RAG sends retrieved chunks to the LLM context — any off-topic chunk is a form of context leakage.",
            "Overlapping chunks can duplicate the same facts in one prompt (redundant exposure).",
            "Low similarity scores mean the retriever may inject unrelated privacy content into answers.",
            "Probe queries test whether retrieval stays within the expected document boundary.",
        ],
    }


def run_analysis(
    docs_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    *,
    rebuild_index: bool = True,
) -> dict[str, Any]:
    docs_path = docs_path or config.DOCS_PATH
    output_dir = output_dir or config.PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(docs_path)
    chunks = chunk_documents(
        documents,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )

    corpus = analyze_corpus(docs_path)
    index = analyze_index(chunks)

    pipeline = RAGPipeline(docs_path=docs_path)
    if rebuild_index or pipeline.chunk_count == 0:
        pipeline.ingest(reset=True)

    probes = analyze_retrieval(pipeline, PROBE_QUERIES)
    privacy = analyze_privacy_exposure(pipeline, PROBE_QUERIES)

    hit_at_1 = sum(1 for p in probes if p.hit_at_1)
    hit_at_k = sum(1 for p in probes if p.hit_at_k)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "docs_path": str(docs_path),
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        "corpus": asdict(corpus),
        "index": asdict(index),
        "retrieval": {
            "probe_count": len(probes),
            "hit_at_1": hit_at_1,
            "hit_at_k": hit_at_k,
            "hit_at_1_rate": hit_at_1 / len(probes) if probes else 0,
            "hit_at_k_rate": hit_at_k / len(probes) if probes else 0,
            "probes": [asdict(p) for p in probes],
        },
        "privacy_analysis": privacy,
    }

    out_path = output_dir / "rag_analysis.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output_path"] = str(out_path)
    return report


def print_summary(report: dict[str, Any]) -> None:
    corpus = report["corpus"]
    index = report["index"]
    retrieval = report["retrieval"]

    print("=" * 60)
    print("RAG ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nCorpus: {corpus['document_count']} documents, {corpus['total_chars']:,} chars")
    for doc in corpus["documents"]:
        print(f"  - {doc['source']}: {doc['chars']:,} chars")

    print(f"\nIndex: {index['chunk_count']} chunks (overlap={index['overlap_chars']} chars)")
    for src, n in sorted(index["chunks_per_source"].items()):
        print(f"  - {src}: {n} chunks")

    print(
        f"\nRetrieval accuracy ({retrieval['probe_count']} probes, top_k={report['config']['top_k']}):"
    )
    print(f"  Hit@1: {retrieval['hit_at_1']}/{retrieval['probe_count']} ({retrieval['hit_at_1_rate']:.0%})")
    print(f"  Hit@k: {retrieval['hit_at_k']}/{retrieval['probe_count']} ({retrieval['hit_at_k_rate']:.0%})")

    print("\nProbe results:")
    for p in retrieval["probes"]:
        mark = "OK" if p["hit_at_1"] else ("partial" if p["hit_at_k"] else "MISS")
        print(f"  [{mark}] {p['query'][:50]}")
        print(f"       expected: {p['expected_source']}, got: {p['top_source']} (score {p['top_score']})")

    privacy = report["privacy_analysis"]
    print(f"\nPrivacy observations:")
    print(f"  Cross-doc chunks in top-k: {len(privacy['cross_document_retrieval'])} queries")
    print(f"  Low-confidence retrievals: {len(privacy['low_confidence_queries'])} queries")
    for obs in privacy["observations"]:
        print(f"  - {obs}")

    print(f"\nFull report: {report['output_path']}")
    print("=" * 60)
