import sqlite3
from typing import List, Tuple

import sqlite_vec
from fastembed import TextEmbedding


DB_PATH = "chunks.db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _open_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


class Retriever:
    def __init__(self, db_path: str = DB_PATH, model_name: str = MODEL_NAME):
        self.db = _open_db(db_path)
        self.model = TextEmbedding(model_name=model_name)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return up to k (chunk_text, distance) tuples, closest first."""
        # Embed the query (fastembed returns a generator of np arrays)
        query_vec = next(self.model.embed([query])).tolist()

        rows = self.db.execute(
            """
            SELECT chunk_text, distance
            FROM vec_chunks
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (sqlite_vec.serialize_float32(query_vec), k),
        ).fetchall()

        return [(text, float(dist)) for text, dist in rows]

    def close(self):
        self.db.close()


if __name__ == "__main__":
    r = Retriever()
    query = "What is this document about?"
    results = r.retrieve(query, k=3)

    print(f"Query: {query}")
    print(f"Got {len(results)} result(s).\n")
    for i, (text, dist) in enumerate(results, 1):
        print(f"--- Result {i} (distance={dist:.4f}) ---")
        print(text[:400])
        print()
    r.close()