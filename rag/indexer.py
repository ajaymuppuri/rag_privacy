from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag.chunker import Chunk


class Indexer:
    def __init__(
        self,
        chroma_path: Path,
        collection_name: str,
        embedding_model: str,
    ):
        self._model = SentenceTransformer(embedding_model)
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=True).tolist()

    def index(self, chunks: list[Chunk], *, reset: bool = False) -> int:
        if reset and self._collection.count() > 0:
            self._client.delete_collection(self._collection.name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection.name,
                metadata={"hnsw:space": "cosine"},
            )

        if not chunks:
            return 0

        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]
        embeddings = self.embed(documents)

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
            )

        return len(chunks)

    @property
    def collection(self):
        return self._collection
