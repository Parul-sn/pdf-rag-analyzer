import os
import sqlite3
from typing import List

import sqlite_vec
from fastembed import TextEmbedding


DB_PATH = "chunks.db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-dim, small (~130 MB), CPU-friendly


def _open_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def build_store(chunks: List[tuple], db_path: str = DB_PATH) -> int:
    """
    Embed (chunk_text, source) pairs and store them in a fresh sqlite-vec table.
    Returns the number of rows stored.
    """
    if os.path.exists(db_path):
        os.remove(db_path)

    # Accept either plain strings (backward-compat) or (text, source) tuples.
    normalized = [
        (c, "") if isinstance(c, str) else (c[0], c[1])
        for c in chunks
    ]

    print(f"Loading embedding model: {MODEL_NAME} (first run downloads ~130 MB)...")
    model = TextEmbedding(model_name=MODEL_NAME)

    texts = [text for text, _ in normalized]
    print(f"Embedding {len(texts)} chunk(s)...")
    embeddings = [vec.tolist() for vec in model.embed(texts)]

    db = _open_db(db_path)
    try:
        dim = len(embeddings[0])
        print(f"Embedding dimension: {dim}")

        db.execute(
            f"""
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                embedding float[{dim}],
                chunk_text text,
                source text
            )
            """
        )
                # --- NEW: FTS5 full-text index (empty for now) ---
        db.execute(
            """
            CREATE VIRTUAL TABLE fts_chunks USING fts5(
                chunk_text,
                tokenize='unicode61'
            )
            """
        )

        for i, ((text, source), vec) in enumerate(zip(normalized, embeddings)):
            rowid = i + 1
            db.execute(
                "INSERT INTO vec_chunks(rowid, embedding, chunk_text, source) VALUES (?, ?, ?, ?)",
                (rowid, sqlite_vec.serialize_float32(vec), text, source),
            )
            db.execute(
                "INSERT INTO fts_chunks(rowid, chunk_text) VALUES (?, ?)",
                (rowid, text),
            )
        db.commit()

        count = db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        return count
    finally:
        db.close()


if __name__ == "__main__":
    from pathlib import Path
    from extract import extract_text_from_pdf
    from chunk import chunk_text

    pdf_paths = sorted(Path("data").glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit("No PDFs found in data/. Drop at least one .pdf file there.")

    all_chunks: list[tuple[str, str]] = []
    for pdf_path in pdf_paths:
        pages = extract_text_from_pdf(str(pdf_path))
        chunks = chunk_text(pages, chunk_size=1000, overlap=100)
        print(f"  {pdf_path.name}: {len(pages)} page(s) -> {len(chunks)} chunk(s)")
        all_chunks.extend((c, pdf_path.name) for c in chunks)

    print(f"\nTotal: {len(all_chunks)} chunk(s) from {len(pdf_paths)} PDF(s)\n")
    n = build_store(all_chunks)
    print(f"Stored {n} chunk(s) in {DB_PATH}")