from typing import List


def chunk_text(
    pages: List[str],
    chunk_size: int = 1000,
    overlap: int = 100,
) -> List[str]:
    """
    Join pages into one long string, then split into overlapping chunks.

    chunk_size: number of characters per chunk
    overlap:    number of characters shared between consecutive chunks
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    full_text = "\n\n".join(pages)

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


if __name__ == "__main__":
    from extract import extract_text_from_pdf

    pages = extract_text_from_pdf("data/sample.pdf")
    chunks = chunk_text(pages, chunk_size=1000, overlap=100)

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    if chunks:
        print("--- Preview of first chunk (first 300 chars) ---")
        print(chunks[0][:300])