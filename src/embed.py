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


def build_store(chunks: List[str], db_path: str = DB_PATH) -> int:
    """Embed chunks and store them in a fresh sqlite-vec table. Returns row count."""
    # Start clean every run so we don't duplicate rows while learning.
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"Loading embedding model: {MODEL_NAME} (first run downloads ~130 MB)...")
    model = TextEmbedding(model_name=MODEL_NAME)

    print(f"Embedding {len(chunks)} chunk(s)...")
    # fastembed yields numpy arrays (float32) — convert to plain lists
    embeddings = [vec.tolist() for vec in model.embed(chunks)]

    db = _open_db(db_path)
    try:
        # Determine dimension from the first vector
        dim = len(embeddings[0])
        print(f"Embedding dimension: {dim}")

        db.execute(
            f"""
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                embedding float[{dim}],
                chunk_text text
            )
            """
        )

        # Insert chunk-by-chunk using sqlite_vec's float32 serializer
        for chunk, vec in zip(chunks, embeddings):
            db.execute(
                "INSERT INTO vec_chunks(embedding, chunk_text) VALUES (?, ?)",
                (sqlite_vec.serialize_float32(vec), chunk),
            )
        db.commit()

        count = db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        return count
    finally:
        db.close()


if __name__ == "__main__":
    from extract import extract_text_from_pdf
    from chunk import chunk_text

    pages = extract_text_from_pdf("data/sample.pdf")
    chunks = chunk_text(pages, chunk_size=1000, overlap=100)
    n = build_store(chunks)
    print(f"Stored {n} chunk(s) in {DB_PATH}")