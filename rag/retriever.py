from dataclasses import dataclass

from rag.indexer import Indexer


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float
    chunk_index: int


class Retriever:
    def __init__(self, indexer: Indexer, top_k: int):
        self._indexer = indexer
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_embedding = self._indexer.embed([query])[0]
        results = self._indexer.collection.query(
            query_embeddings=[query_embedding],
            n_results=self._top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        if not results["documents"] or not results["documents"][0]:
            return chunks

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma cosine distance: lower is more similar
            score = 1.0 - dist
            chunks.append(
                RetrievedChunk(
                    source=meta["source"],
                    text=doc,
                    score=score,
                    chunk_index=meta["chunk_index"],
                )
            )
        return chunks
