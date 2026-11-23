import os
import faiss
import numpy as np
from typing import List, Dict

class VectorStore:
    """FAISS in‑memory store for a single document.
    Stores embeddings and the corresponding chunk metadata.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # inner product (cosine after normalization)
        self.metadata: List[Dict] = []  # parallel list; same order as vectors

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add a batch of embeddings and their metadata.
        ``embeddings`` must be a 2‑D float32 array of shape (n, dimension).
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def clear(self):
        self.index.reset()
        self.metadata.clear()

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Return the top‑k most similar chunk metadata entries.
        ``query_vec`` should be a 1‑D normalized vector (float32).
        """
        if self.index.ntotal == 0:
            return []
        query_vec = query_vec.reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(dist)
            results.append(meta)
        return results
