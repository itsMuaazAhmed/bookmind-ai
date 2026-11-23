# BookMind AI

**Chat with your books. Understand every answer. Find the original page.**

BookMind AI is a Streamlit‑based Retrieval‑Augmented Generation (RAG) application that lets users upload a PDF (text‑based) and ask natural‑language questions. Answers are grounded in the document and include page‑level citations.

## Features
- PDF upload with per‑page text extraction (PyMuPDF)
- Page‑aware chunking (~800 characters) with overlap
- Local embeddings (`sentence‑transformers/all‑MiniLM‑L6‑v2`)
- FAISS‑CPU vector store for semantic search
- Google Gemini LLM (model configurable via env)
- Demo limit: 5 free queries per anonymous session
- Clean, responsive UI built with Streamlit
- Full test suite (`pytest`)

## Quick start (local)
```bash
git clone https://github.com/yourname/bookmind-ai.git
cd bookmind-ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your Gemini API key
streamlit run app.py
```

## Project structure
```
bookmind-ai/
│   app.py               # Streamlit entry point
│   README.md
│   requirements.txt
│   .gitignore
│   .env.example
│   LICENSE
│
├── config/
│   └── settings.py      # Central configuration values
│
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py # PDF validation & extraction
│   ├── chunker.py       # Page‑aware chunking logic
│   ├── embeddings.py    # Embedding model handling
│   ├── vector_store.py  # FAISS index wrapper
│   ├── retriever.py     # Retrieval pipeline
│   ├── llm_service.py   # Gemini API wrapper
│   ├── prompts.py       # System & user prompts
│   ├── chat_service.py  # Orchestrates RAG workflow
│   ├── usage_limiter.py # Demo query counter
│   └── utils.py         # Helper utilities
│
├── ui/
│   ├── __init__.py
│   ├── components.py   # Re‑usable Streamlit components
│   ├── styles.py        # CSS & theming
│   └── chat_interface.py# Chat UI layout
│
├── tests/
│   ├── test_pdf_processor.py
│   ├── test_chunker.py
│   ├── test_usage_limiter.py
│   ├── test_retriever.py
│   └── test_utils.py
│
└── assets/
    └── logo.png        # Optional logo
```

## Detailed installation & testing
See the **Hugging Face Spaces deployment guide** (section 9 of the README) for step‑by‑step instructions on deploying the app publicly.

## License
MIT – see the `LICENSE` file.
