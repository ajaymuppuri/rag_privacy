import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_PATH = Path(os.getenv("DOCS_PATH", str(Path.home() / "rag_documents")))
CHROMA_PATH = PROJECT_ROOT / os.getenv("CHROMA_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "privacy_docs")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
