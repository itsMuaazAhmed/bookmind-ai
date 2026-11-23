import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import CHUNK_SIZE

# The model is loaded once and cached globally
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model lazily and cache it.
    The model ``all-MiniLM-L6-v2`` is downloaded the first time this function is called.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _model


def embed_chunks(texts: List[str]) -> np.ndarray:
    """Return a 2‑D numpy array of normalized embeddings for the supplied texts.
    Normalization (L2) is performed because FAISS inner‑product search expects that.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype='float32')
