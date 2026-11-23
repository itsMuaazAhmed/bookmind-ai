import re
from typing import List, Dict
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def _split_into_sentences(text: str) -> List[str]:
    """Very simple sentence splitter – split on punctuation followed by whitespace.
    This is sufficient for most English PDFs and keeps the implementation lightweight.
    """
    # Keep the delimiter attached to the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(page: Dict) -> List[Dict]:
    """Create overlapping chunks for a single page.
    Returns a list of dicts with keys: chunk_id, text, page_number, document_name.
    """
    text = page["text"]
    if not text:
        return []
    sentences = _split_into_sentences(text)
    # Join sentences back while respecting chunk size
    chunks = []
    start_idx = 0
    while start_idx < len(sentences):
        cur_len = 0
        end_idx = start_idx
        while end_idx < len(sentences) and cur_len + len(sentences[end_idx]) + 1 <= CHUNK_SIZE:
            cur_len += len(sentences[end_idx]) + 1  # +1 for space
            end_idx += 1
        chunk_text = " ".join(sentences[start_idx:end_idx])
        chunk_id = f"{page['document_name']}_p{page['page_number']}_c{len(chunks)+1}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "page_number": page["page_number"],
            "document_name": page["document_name"]
        })
        # Overlap: move back by overlap sentences (approx). Use character overlap for simplicity.
        if CHUNK_OVERLAP > 0:
            # Approximate overlap by characters
            overlap_chars = CHUNK_OVERLAP
            # Find new start_idx so that we keep overlap characters from the end of current chunk
            if end_idx < len(sentences):
                # Count characters backwards until overlap reached
                overlap_start = end_idx - 1
                chars = len(sentences[overlap_start])
                while overlap_start > start_idx and chars < overlap_chars:
                    overlap_start -= 1
                    chars += len(sentences[overlap_start]) + 1
                start_idx = overlap_start
            else:
                break
        else:
            start_idx = end_idx
    return chunks


def chunk_document(pages: List[Dict]) -> List[Dict]:
    """Chunk an entire document (list of page dicts)."""
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    return all_chunks
