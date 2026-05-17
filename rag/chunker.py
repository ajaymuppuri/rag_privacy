from dataclasses import dataclass

from rag.loader import Document


@dataclass
class Chunk:
    source: str
    text: str
    chunk_index: int


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            break_at = max(
                chunk.rfind("\n\n"),
                chunk.rfind("\n"),
                chunk.rfind(". "),
                chunk.rfind(" "),
            )
            if break_at > chunk_size // 3:
                chunk = chunk[: break_at + 1]
                end = start + len(chunk)

        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return [c for c in chunks if c]


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        pieces = _split_text(doc.text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(source=doc.source, text=piece, chunk_index=i)
            )
    return all_chunks
