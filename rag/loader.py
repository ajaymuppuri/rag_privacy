from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup


@dataclass
class Document:
    source: str
    text: str


def _strip_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _read_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"} or raw.lstrip().startswith("<!DOCTYPE"):
        return _strip_html(raw)
    return raw.strip()


def load_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    documents: list[Document] = []
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".html", ".htm"}:
            continue
        text = _read_file(path)
        if text:
            documents.append(Document(source=path.name, text=text))
    return documents
