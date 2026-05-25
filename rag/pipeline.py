from pathlib import Path
from typing import List, Optional

import config
from rag.chunker import chunk_documents
from rag.indexer import Indexer
from rag.loader import load_documents
from rag.retriever import Retriever, RetrievedChunk


class RAGPipeline:
    def __init__(
        self,
        docs_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.docs_path = docs_path or config.DOCS_PATH
        self.chroma_path = chroma_path or config.CHROMA_PATH
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self._indexer = Indexer(
            chroma_path=self.chroma_path,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        self._retriever = Retriever(self._indexer, top_k=top_k or config.TOP_K)

    @property
    def indexer(self) -> Indexer:
        """Expose the underlying Indexer for callers that need raw vector access."""
        return self._indexer

    def ingest(self, *, reset: bool = False) -> dict:
        documents = load_documents(self.docs_path)
        chunks = chunk_documents(
            documents,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        count = self._indexer.index(chunks, reset=reset)
        return {
            "documents": len(documents),
            "chunks": count,
            "sources": [d.source for d in documents],
        }

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        return self._retriever.retrieve(question)

    @property
    def chunk_count(self) -> int:
        return self._indexer.collection.count()
