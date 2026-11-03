from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """Return a list of text strings, one per page (pages with no text are skipped)."""
    reader = PdfReader(pdf_path)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    return pages


if __name__ == "__main__":
    pages = extract_text_from_pdf("data/sample.pdf")
    print(f"Extracted {len(pages)} page(s) with text.")
    if not pages:
        print("WARNING: no text extracted — PDF may be scanned/image-only.")
    else:
        print("--- Preview of first page (first 500 chars) ---")
        print(pages[0][:500])