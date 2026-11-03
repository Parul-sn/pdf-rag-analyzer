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

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Return up to k (chunk_text, source, distance) tuples, closest first."""
        query_vec = next(self.model.embed([query])).tolist()

        rows = self.db.execute(
            """
            SELECT chunk_text, source, distance
            FROM vec_chunks
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (sqlite_vec.serialize_float32(query_vec), k),
        ).fetchall()

        return [(text, src or "", float(dist)) for text, src, dist in rows]

    def _retrieve_fts_rowids(self, query: str, k: int = 20) -> List[int]:
        """
        FTS5 keyword search. Returns rowids ordered by rank (best first).
        Returns an empty list if the query has invalid FTS5 syntax
        (e.g. special characters like ?, :, *, quotes).
        """
        try:
            rows = self.db.execute(
                """
                SELECT rowid
                FROM fts_chunks
                WHERE chunk_text MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, k),
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.OperationalError:
            return []

    def _retrieve_vector_rowids(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """
        Vector search. Returns (rowid, distance) tuples ordered by distance
        (best first). rowid is our join key into fts_chunks.
        """
        query_vec = next(self.model.embed([query])).tolist()
        rows = self.db.execute(
            """
            SELECT rowid, distance
            FROM vec_chunks
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (sqlite_vec.serialize_float32(query_vec), k),
        ).fetchall()
        return [(int(rowid), float(dist)) for rowid, dist in rows]

    def retrieve_hybrid(self, query: str, k: int = 4, rrf_k: int = 60) -> List[Tuple[str, str, float]]:
        """
        Hybrid retrieval: vector + FTS5, fused with Reciprocal Rank Fusion.

        Returns (chunk_text, source, rrf_score) tuples, best first.
        Note: rrf_score is HIGHER = better (opposite of the `distance`
        returned by the pure-vector retrieve()).
        """
        # 1. Retrieve both ranked lists (rowids only)
        vec_rows = self._retrieve_vector_rowids(query, k=rrf_k)
        fts_rowids = self._retrieve_fts_rowids(query, k=rrf_k)

        # 2. RRF fusion: score(rowid) = sum over lists of 1/(rrf_k + rank)
        #    rrf_k here is the "constant" (default 60) — same symbol, different
        #    meaning than the k argument; that's standard RRF notation.
        scores: dict[int, float] = {}
        for rank, (rowid, _dist) in enumerate(vec_rows):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (60 + rank + 1)
        for rank, rowid in enumerate(fts_rowids):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (60 + rank + 1)

        if not scores:
            return []

        # 3. Take the top-k rowids by fused score
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        top_rowids = [rowid for rowid, _ in top]

        # 4. Fetch text + source for those rowids in one query
        placeholders = ",".join("?" * len(top_rowids))
        rows = self.db.execute(
            f"SELECT rowid, chunk_text, source FROM vec_chunks WHERE rowid IN ({placeholders})",
            top_rowids,
        ).fetchall()
        by_rowid = {rid: (text, src or "") for rid, text, src in rows}

        # 5. Reorder by fused score and return
        return [
            (by_rowid[rowid][0], by_rowid[rowid][1], score)
            for rowid, score in top
            if rowid in by_rowid
        ]

    def close(self):
        self.db.close()


if __name__ == "__main__":
    r = Retriever()
    query = "What is this document about?"
    results = r.retrieve(query, k=3)

    print(f"Query: {query}")
    print(f"Got {len(results)} result(s).\n")
    for i, (text, source, dist) in enumerate(results, 1):
        print(f"--- Result {i} [{source}] (distance={dist:.4f}) ---")
        print(text[:400])
        print()

    print("\n--- FTS smoke test ---")
    for term in ["Pillar", "CPLD", "Deloitte"]:
        hits = r._retrieve_fts_rowids(term, k=5)
        print(f"'{term}' → {len(hits)} FTS hit(s): rowids={hits}")

    print("\n--- Vector rowids smoke test ---")
    vec_hits = r._retrieve_vector_rowids("What is CPLD?", k=5)
    print(f"'What is CPLD?' → {len(vec_hits)} vector hit(s):")
    for rowid, dist in vec_hits:
        print(f"  rowid={rowid} distance={dist:.4f}")

    print("\n--- HYBRID vs VECTOR comparison ---")
    test_query = "What does the tax report say about Pillar Two?"

    print(f"\n[vector-only] {test_query}")
    for i, (text, src, dist) in enumerate(r.retrieve(test_query, k=3), 1):
        print(f"  {i}. [{src}] dist={dist:.4f} | {text[:60].strip()}...")

    print(f"\n[hybrid] {test_query}")
    for i, (text, src, rrf) in enumerate(r.retrieve_hybrid(test_query, k=3), 1):
        print(f"  {i}. [{src}] rrf={rrf:.4f} | {text[:60].strip()}...")

    r.close()