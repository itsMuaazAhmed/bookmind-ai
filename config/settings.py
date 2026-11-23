import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (local development)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)

# Core configuration – values come from environment or defaults
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.0-pro")
DEMO_QUERY_LIMIT = int(os.getenv("DEMO_QUERY_LIMIT", "5"))
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval settings
TOP_K = int(os.getenv("TOP_K", "5"))

# FAISS index filename (in‑memory; not persisted to disk)
FAISS_INDEX_FILENAME = "faiss_index.bin"
