import os
import streamlit as st
from src.pdf_processor import PDFProcessor
from src.chunker import chunk_document
from src.embeddings import embed_chunks
from src.vector_store import VectorStore
from src.llm_service import LLMService
from src.prompts import SYSTEM_PROMPT
from config.settings import DEMO_QUERY_LIMIT, TOP_K, GEMINI_MODEL, GEMINI_API_KEY

# ---------------------------------------------------------------
# Helper: display header / branding
# ---------------------------------------------------------------
def render_header():
    st.markdown("""
    <h1 style='text-align: center; color:#1e3a8a;'>BookMind AI</h1>
    <p style='text-align: center; font-size:1.1rem; color:#4b5563;'>Chat with your books. Understand every answer. Find the original page.</p>
    """, unsafe_allow_html=True)
    st.caption("AI PDF Assistant – free demo (5 queries per session)")

# ---------------------------------------------------------------
# Session‑state initialisation
# ---------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)
if "queries_used" not in st.session_state:
    st.session_state.queries_used = 0

# ---------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------
render_header()

st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save to a temporary location
    tmp_path = os.path.join("/tmp", uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Validate & extract
    try:
        PDFProcessor.validate_file(tmp_path)
        pages = PDFProcessor.extract_pages(tmp_path)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    # Chunking
    chunks = chunk_document(pages)
    if not chunks:
        st.error("⚠️ No extractable text found in the PDF.")
        st.stop()

    # Embeddings & indexing
    texts = [c["text"] for c in chunks]
    embeddings = embed_chunks(texts)
    store = VectorStore(dimension=embeddings.shape[1])
    store.add_embeddings(embeddings, chunks)

    # Store in session state
    st.session_state.vector_store = store
    st.session_state.doc_name = uploaded_file.name
    st.session_state.num_pages = len(pages)
    st.session_state.num_chunks = len(chunks)
    st.success(f"✅ Processed **{uploaded_file.name}** – {st.session_state.num_pages} pages → {st.session_state.num_chunks} chunks.")

# ---------------------------------------------------------------
# Show document card if a document is loaded
# ---------------------------------------------------------------
if st.session_state.vector_store:
    with st.expander("Document information", expanded=True):
        st.write(f"**File:** {st.session_state.doc_name}")
        st.write(f"**Pages:** {st.session_state.num_pages}")
        st.write(f"**Chunks:** {st.session_state.num_chunks}")
        remaining = DEMO_QUERY_LIMIT - st.session_state.queries_used
        st.info(f"Demo questions remaining: **{remaining}/{DEMO_QUERY_LIMIT}**")

# ---------------------------------------------------------------
# Chat interface (only when a document is loaded)
# ---------------------------------------------------------------
if st.session_state.vector_store:
    st.subheader("Chat with the document")

    # Display previous messages
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    # Input box
    if st.session_state.queries_used >= DEMO_QUERY_LIMIT:
        st.warning("🚫 You have reached the demo limit of 5 questions.")
    else:
        user_input = st.chat_input("Ask a question about the document")
        if user_input:
            # Append user message
            st.session_state.chat_history.append(("user", user_input))
            st.chat_message("user").write(user_input)
            st.session_state.queries_used += 1

            # -----------------------------------------------------------------
            # Retrieval
            # -----------------------------------------------------------------
            llm = LLMService()
            query_vec = embed_chunks([user_input])[0]
            top_chunks = st.session_state.vector_store.search(query_vec, top_k=TOP_K)
            if not top_chunks:
                answer = "I couldn't find a reliable answer to this question in the uploaded document."
                st.chat_message("assistant").write(answer)
                st.session_state.chat_history.append(("assistant", answer))
                st.stop()

            # Build prompt
            context_parts = [f"Page {c['page_number']}: {c['text']}" for c in top_chunks]
            context = "\n\n".join(context_parts)
            prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {user_input}\n\nAnswer (cite page numbers):"

            # -----------------------------------------------------------------
            # LLM generation
            # -----------------------------------------------------------------
            try:
                response_text = llm.generate_text(prompt)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.session_state.chat_history.append(("assistant", "Sorry, an internal error occurred."))
                st.stop()

            # Show answer
            st.chat_message("assistant").write(response_text)
            st.session_state.chat_history.append(("assistant", response_text))

            # -----------------------------------------------------------------
            # Sources section (simple display of retrieved pages)
            # -----------------------------------------------------------------
            with st.expander("Sources"):
                for c in top_chunks:
                    st.caption(f"Page {c['page_number']} – excerpt:")
                    st.write(c['text'][:300] + ("…" if len(c['text']) > 300 else ""))

else:
    st.info("Upload a PDF on the left sidebar to begin.")
