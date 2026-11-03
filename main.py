import os

from src.extract import extract_text_from_pdf
from src.chunk import chunk_text
from src.embed import build_store, DB_PATH
from src.retrieve import Retriever
from src.llm import summarize_chunks, answer_question


PDF_PATH = "data/sample.pdf"


def ensure_store() -> None:
    """Build chunks.db only if it doesn't exist yet."""
    if os.path.exists(DB_PATH):
        print(f"[setup] Reusing existing {DB_PATH}")
        return

    print(f"[setup] {DB_PATH} not found — building it now...")
    pages = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(pages, chunk_size=1000, overlap=100)
    print(f"[setup] Extracted {len(pages)} page(s) -> {len(chunks)} chunk(s)")
    n = build_store(chunks)
    print(f"[setup] Stored {n} chunk(s) in {DB_PATH}\n")


def main() -> None:
    ensure_store()

    retriever = Retriever()
    try:
        # --- 1. Summary ---
        print("=== SUMMARY ===")
        summary_chunks = [t for t, _ in retriever.retrieve(
            "what is this document about", k=5
        )]
        print(summarize_chunks(summary_chunks))
        print()

        # --- 2. Q&A loop ---
        print("=== ASK QUESTIONS ===")
        print("Type a question about the PDF, or 'exit' to quit.\n")
        while True:
            try:
                question = input("Q: ").strip()
            except EOFError:
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                break

            qa_chunks = [t for t, _ in retriever.retrieve(question, k=3)]
            answer = answer_question(question, qa_chunks)
            print(f"A: {answer}\n")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()