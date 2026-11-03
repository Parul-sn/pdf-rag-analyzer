import streamlit as st

from src.extract import extract_text_from_pdf
from src.chunk import chunk_text
from src.embed import build_store, DB_PATH
from src.retrieve import Retriever
from src.llm import answer_question

st.set_page_config(page_title="PDF RAG", page_icon="📄", layout="wide")

st.title("📄 PDF RAG")
st.caption("Upload one or more PDFs, then ask questions about them.")


@st.cache_resource
def get_retriever(version: int) -> Retriever:
    """Cache a Retriever keyed on `version` so it refreshes after each index."""
    return Retriever()


# --- Sidebar: upload ---
st.sidebar.header("1. Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Drop PDFs here or click to browse:",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.sidebar.write("**Selected:**")
    for f in uploaded_files:
        st.sidebar.write(f"- {f.name} ({f.size // 1024} KB)")

# --- Sidebar: index ---
st.sidebar.header("2. Build index")
index_clicked = st.sidebar.button(
    "Index selected PDFs",
    disabled=not uploaded_files,
    use_container_width=True,
)

# --- Sidebar: clear chat ---
st.sidebar.header("3. Reset")
if st.sidebar.button("Clear chat history", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()


# --- Indexing ---
if index_clicked:
    with st.status(f"Indexing {len(uploaded_files)} PDF(s)...", expanded=True) as status:
        all_chunks = []
        for f in uploaded_files:
            st.write(f"Extracting **{f.name}**...")
            try:
                pages = extract_text_from_pdf(f)
            except Exception as e:
                st.write(f"  ❌ Could not read: {e}")
                continue
            chunks = chunk_text(pages, chunk_size=1000, overlap=100)
            st.write(f"  ✓ {len(pages)} page(s) → {len(chunks)} chunk(s)")
            all_chunks.extend((c, f.name) for c in chunks)

        if not all_chunks:
            status.update(label="No text extracted", state="error")
            st.error("Try different PDFs — these may be scanned/image-only.")
        else:
            st.write(f"Embedding **{len(all_chunks)}** chunk(s) (may take ~20s)...")
            try:
                n = build_store(all_chunks)
                st.write(f"Stored **{n}** chunk(s) in `{DB_PATH}`")
                status.update(
                    label=f"Indexed {n} chunks from {len(uploaded_files)} PDF(s)",
                    state="complete",
                )
                st.session_state["indexed_pdfs"] = [f.name for f in uploaded_files]
                st.session_state["index_version"] = st.session_state.get("index_version", 0) + 1
                st.session_state["messages"] = []   # fresh chat on new index
            except Exception as e:
                status.update(label="Indexing failed", state="error")
                st.error(f"Embedding failed: {e}")


# --- Main: Q&A ---
if "indexed_pdfs" not in st.session_state:
    st.info("👈 Upload one or more PDFs in the sidebar and click **Index selected PDFs** to begin.")
else:
    st.success(f"Currently indexed: {', '.join(st.session_state['indexed_pdfs'])}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render existing chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption(f"[sources: {', '.join(msg['sources'])}]")

    # Chat input at the bottom
    question = st.chat_input("Ask a question about the indexed PDFs")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                retriever = get_retriever(st.session_state.get("index_version", 0))
                results = retriever.retrieve_hybrid(question, k=4)
                if not results:
                    answer = "No relevant chunks found for that question."
                    sources = []
                else:
                    chunks = [t for t, _, _ in results]
                    sources = sorted({s for _, s, _ in results if s})
                    try:
                        answer = answer_question(question, chunks)
                    except Exception as e:
                        answer = f"LLM error: {e}"
                        sources = []

                st.write(answer)
                if sources:
                    st.caption(f"[sources: {', '.join(sources)}]")

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )