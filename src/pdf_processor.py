import fitz  # PyMuPDF
import os
from typing import List, Dict

from config.settings import MAX_PDF_SIZE_MB

class PDFProcessor:
    """Validate and extract text from a PDF on a per‑page basis."""

    @staticmethod
    def validate_file(file_path: str) -> None:
        """Raise ValueError if the file is not a suitable PDF.
        Checks extension, size and that the file can be opened.
        """
        if not file_path.lower().endswith('.pdf'):
            raise ValueError('File must have a .pdf extension')
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > MAX_PDF_SIZE_MB:
            raise ValueError(f'PDF exceeds maximum size of {MAX_PDF_SIZE_MB} MB')
        try:
            doc = fitz.open(file_path)
            # quick sanity check – at least one page
            if doc.page_count == 0:
                raise ValueError('PDF contains no pages')
        except Exception as e:
            raise ValueError(f'Unable to open PDF: {e}')

    @staticmethod
    def extract_pages(file_path: str) -> List[Dict]:
        """Return a list of dictionaries, one per page, preserving page number.
        Empty or whitespace‑only pages are skipped but their original numbers are
        retained in the metadata for correct citation offsets.
        """
        doc = fitz.open(file_path)
        pages = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text").strip()
            if not text:
                # keep placeholder with empty text to preserve numbering
                pages.append({
                    "page_number": i + 1,
                    "text": "",
                    "document_name": os.path.basename(file_path)
                })
                continue
            pages.append({
                "page_number": i + 1,
                "text": text,
                "document_name": os.path.basename(file_path)
            })
        return pages
