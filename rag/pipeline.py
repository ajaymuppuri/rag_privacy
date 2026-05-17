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
    ):
        self.docs_path = docs_path or config.DOCS_PATH
        self.chroma_path = chroma_path or config.CHROMA_PATH
        self._indexer = Indexer(
            chroma_path=self.chroma_path,
            collection_name=config.COLLECTION_NAME,
            embedding_model=config.EMBEDDING_MODEL,
        )
        self._retriever = Retriever(self._indexer, top_k=config.TOP_K)

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
