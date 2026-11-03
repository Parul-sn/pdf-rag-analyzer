import os
from pathlib import Path

from src.extract import extract_text_from_pdf
from src.chunk import chunk_text
from src.embed import build_store, DB_PATH
from src.retrieve import Retriever
from src.llm import summarize_chunks, answer_question


DATA_DIR = "data"
PDF_GLOB = "*.pdf"


def ensure_store() -> None:
    """Build chunks.db from all PDFs in data/ only if it doesn't exist yet."""
    if os.path.exists(DB_PATH):
        print(f"[setup] Reusing existing {DB_PATH}")
        return

    pdf_paths = sorted(Path(DATA_DIR).glob(PDF_GLOB))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {DATA_DIR}/. Drop at least one .pdf file there.")

    print(f"[setup] {DB_PATH} not found — building from {len(pdf_paths)} PDF(s)...")
    all_chunks: list[tuple[str, str]] = []
    for pdf_path in pdf_paths:
        pages = extract_text_from_pdf(str(pdf_path))
        chunks = chunk_text(pages, chunk_size=1000, overlap=100)
        print(f"  {pdf_path.name}: {len(pages)} page(s) -> {len(chunks)} chunk(s)")
        all_chunks.extend((c, pdf_path.name) for c in chunks)

    if not all_chunks:
        raise SystemExit("No text extracted from any PDF. Aborting.")

    n = build_store(all_chunks)
    print(f"[setup] Stored {n} chunk(s) in {DB_PATH}\n")


def distinct_sources(retriever: Retriever, k: int = 20) -> list[str]:
    """Ask for a wide slice of chunks and collect unique source filenames."""
    rows = retriever.retrieve("document overview main topic", k=k)
    seen: list[str] = []
    for _, source, _ in rows:
        if source and source not in seen:
            seen.append(source)
    return seen


def summarize_all(retriever: Retriever) -> None:
    print("=== PER-DOCUMENT SUMMARY ===")
    sources = distinct_sources(retriever)
    if not sources:
        print("(no sources found in store)")
        return

    for source in sources:
        # Retrieve a handful of chunks, then keep only the ones from this PDF
        candidates = retriever.retrieve(f"what is {source} about", k=20)
        source_chunks = [text for text, src, _ in candidates if src == source][:5]
        if not source_chunks:
            print(f"\n--- {source} ---\n(no chunks retrieved)")
            continue

        print(f"\n--- {source} ---")
        print(summarize_chunks(source_chunks))
    print()


def qa_loop(retriever: Retriever) -> None:
    print("=== ASK QUESTIONS ===")
    print("Type a question about any PDF, or 'exit' to quit.\n")
    while True:
        try:
            question = input("Q: ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        results = retriever.retrieve_hybrid(question, k=4)
        if not results:
            print("A: No relevant chunks found.\n")
            continue

        # Deduplicate sources for display
        used_sources = sorted({src for _, src, _ in results if src})
        chunks = [text for text, _, _ in results]

        answer = answer_question(question, chunks)
        print(f"A: {answer}")
        print(f"   [sources: {', '.join(used_sources) or 'unknown'}]\n")


def main() -> None:
    ensure_store()

    retriever = Retriever()
    try:
        summarize_all(retriever)
        qa_loop(retriever)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()