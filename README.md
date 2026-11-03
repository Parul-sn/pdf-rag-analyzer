# pdf-rag-analyzer

A minimal **Retrieval-Augmented Generation (RAG)** pipeline over **one or many PDFs** — built from scratch with **no heavy frameworks**, designed to run comfortably on **low-memory devices**.

Two ways to use it:
- **CLI** — drop PDFs in `data/`, run `python main.py`
- **Web UI** — upload PDFs in the browser, chat with them

Answers cite their source PDFs. Multi-PDF supported. Hybrid retrieval (vector + keyword, RRF-fused). Powered by local embeddings and a hosted LLM.

---

## Why this project exists

Most RAG tutorials use `sentence-transformers` (pulls in PyTorch, ~2 GB) + ChromaDB (~800 MB RAM). That's overkill for a small device.

This project swaps those for lightweight equivalents **without losing any RAG concept**:

| Component | Standard choice | This project | Why |
|---|---|---|---|
| Embeddings | `sentence-transformers` + PyTorch | **fastembed** (ONNX Runtime) | No PyTorch, ~130 MB model, CPU-optimized |
| Vector store | ChromaDB | **sqlite-vec** | SQLite-native, <150 MB RAM |
| Keyword store | Elasticsearch / Whoosh | **SQLite FTS5** | Built into SQLite, zero extra dependency |
| LLM | OpenAI / local 7B model | **Groq** (`openai/gpt-oss-20b`) | Fast, free tier, no local GPU/RAM needed |
| UI | React / Flask app | **Streamlit** | ~40 MB install, zero server setup |

Same pipeline. Same concepts. Runs on a laptop with 2 GB free.

---

## Pipeline

```
PDFs (data/*.pdf or uploads via UI)
      │
      ▼  src/extract.py     (pypdf)                    → pages of text per PDF
      ▼  src/chunk.py       (pure Python)              → overlapping chunks
      ▼  src/embed.py       (fastembed + sqlite-vec)   → vectors + source tag in chunks.db
      ▼                                                        + FTS5 keyword index
      ▼  src/retrieve.py    (vector + FTS5 + RRF)      → top-k (text, source, score)
      ▼  src/llm.py         (Groq)                     → per-document summary + answers
      ▼
  main.py (CLI)   /   app.py (Streamlit UI)
```

**Concepts demonstrated:**

- Overlapping chunking (preserves context at boundaries)
- Local embedding with a 384-dim model (`BAAI/bge-small-en-v1.5`)
- Vector similarity search via SQLite extension
- **Multi-document indexing** — each chunk is tagged with its source PDF
- **Source-grounded answers** — every answer shows which PDF(s) it drew from
- **Hybrid retrieval** — vector (semantic) + FTS5 (keyword) fused with Reciprocal Rank Fusion
- Grounded LLM answers (context-only, refuses to hallucinate)
- Build-once caching (`chunks.db` reused across runs)
- Upload-from-browser via file-like objects (`BytesIO`-compatible extraction)

---

## Quickstart

### 1. Clone and enter

```bash
git clone https://github.com/Parul-sn/pdf-rag-analyzer.git
cd pdf-rag-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the data folder

```bash
mkdir -p data
```

(Only needed for the CLI. The UI lets you upload from anywhere.)

### 5. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Create an API key (starts with `gsk_`)

### 6. Configure the key

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
```

> **Note:** No quotes around the value, no spaces around `=`.

### 7. Run it — pick a mode

**CLI (uses `data/*.pdf`):**

```bash
python main.py
```

**Web UI (uploads from browser):**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## CLI usage

The CLI reads **every `*.pdf` in `data/`** — any filenames, any number.

Place your PDFs in `data/`, then:

```bash
python main.py
```

On the **first run**, the pipeline will:

- Discover every PDF in `data/`
- Extract text from each
- Chunk it into overlapping segments
- Embed each chunk (downloads the ~130 MB model once, cached)
- Store vectors **and the source filename** in `chunks.db`
- Build the FTS5 keyword index over the same chunks
- Print **one summary per PDF**
- Enter an interactive Q&A loop that cites sources

Type `exit` to quit.

**Subsequent runs** reuse `chunks.db` and skip re-embedding — starting instantly.

To force a rebuild after adding/removing PDFs or changing chunk settings:

```bash
rm chunks.db
python main.py
```

---

## Web UI (Streamlit)

Prefer clicking to typing commands? There's a browser interface.

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

**Flow:**

1. **Upload** one or more PDFs in the sidebar (any files from your computer — no need to put them in `data/`).
2. Click **Index selected PDFs** → the app extracts, chunks, embeds, and stores them.
3. **Ask questions** in the chat box at the bottom. Answers cite their source PDFs, e.g. `[sources: report.pdf]`.
4. **Clear chat history** in the sidebar resets the conversation (re-indexing also resets it).

**Notes:**

- Both the CLI and the UI write to the same `chunks.db`. Running one rebuilds what the other indexed. Re-click **Index** in the UI when you switch back.
- Indexing a fresh set of PDFs replaces the previous index — scope is what's currently selected in the sidebar.
- Uploaded files live only in browser memory + `chunks.db`; they are not saved back to `data/`.

---

## Multiple PDFs

Drop as many `.pdf` files as you like into `data/` (CLI) or upload multiple in the UI — any filenames, no config needed. On first index, all of them are chunked, embedded, and tagged with their source filename.

- **Summary** prints one summary per PDF.
- **Q&A** shows which PDF(s) each answer draws from, e.g.:

  ```
  Q: What does the tax report say about Pillar Two?
  A: <grounded answer here>
     [sources: dttl-tax-technology-report-2023.pdf]
  ```

- To rebuild the CLI's index after adding/removing PDFs:

  ```bash
  rm chunks.db && python main.py
  ```

- In the UI, just click **Index selected PDFs** again with the new set.

---

## Hybrid search

Retrieval uses **two indexes** and fuses their results:

- **Vector** (sqlite-vec) — finds semantic matches ("programmable logic" ↔ "CPLD")
- **FTS5** (SQLite full-text) — finds exact keyword matches ("XC9500", "Pillar Two")

The two ranked lists are fused with **Reciprocal Rank Fusion (RRF)**:
each document contributes `1/(60 + rank)` from each list, and documents
appearing in both rise to the top. RRF needs no score normalization —
it only compares rank positions.

**Why this matters:** vector search alone misses rare tokens the embedding
model never learned; keyword search alone misses paraphrases. Together they
cover each other. RRF is ~10 lines of Python, no tuning required.

**In the code:**

- `Retriever.retrieve()` — vector-only (used by `distinct_sources()` in `main.py` for source enumeration)
- `Retriever.retrieve_hybrid()` — vector + FTS, RRF-fused (**used by both CLI Q&A and UI Q&A**)

Both return `(text, source, score)` tuples. In `retrieve()` the score is
`distance` (lower is better); in `retrieve_hybrid()` it's `rrf_score`
(higher is better). The two methods are kept separate because the score
semantics differ.

---

## Project structure

```
pdf-rag-analyzer/
├── data/                   # drop any number of *.pdf files here (CLI, not committed)
├── src/
│   ├── extract.py          # PDF path OR file object → list of page texts
│   ├── chunk.py            # pages → overlapping chunks
│   ├── embed.py            # (chunk, source) tuples → vec_chunks + fts_chunks → chunks.db
│   ├── retrieve.py         # vector search, FTS5 keyword search, RRF fusion
│   └── llm.py              # Groq wrapper: summarize + answer
├── main.py                 # CLI orchestrator (multi-PDF discovery, summary, Q&A)
├── app.py                  # Streamlit web UI (upload, index, chat)
├── requirements.txt
├── .env.example            # copy to .env and add your key
├── .gitignore
├── LICENSE
└── README.md
```

---

## Configuration knobs

All in one place — edit these to experiment:

| File | Setting | Default | What it does |
|---|---|---|---|
| `src/chunk.py` | `chunk_size` | `1000` | Characters per chunk. Smaller = more precise, less context. |
| `src/chunk.py` | `overlap` | `100` | Characters shared between chunks. Preserves boundary context. |
| `src/embed.py` | `MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Swap for other [fastembed-supported](https://qdrant.github.io/fastembed/examples/Supported_Models/) models. |
| `src/llm.py` | `MODEL` | `openai/gpt-oss-20b` | Any Groq-hosted model. |
| `src/retrieve.py` | `rrf_k` (in `retrieve_hybrid`) | `60` | RRF damping constant. 60 is the standard from the original paper; rarely worth changing. |
| `main.py` / `app.py` | `k` in `retrieve_hybrid()` | `4` | Number of fused chunks sent to the LLM per query. |
| `main.py` | `k` in `distinct_sources()` | `20` | Wide vector-only slice to enumerate source filenames. |

---

## Tuning tips

- **Getting vague summaries?** Increase `k` in the summary call (try `8–10`) or use longer PDFs.
- **Getting hallucinations?** Lower `temperature` in `src/llm.py` (try `0.1`) and reduce `k`.
- **Chunks feel too short?** Increase `chunk_size` to `1500–2000`.
- **Retrieval misses context?** Increase `overlap` to `200`.
- **Mixing very different documents?** Consider a `rrf_score` floor (drop results with score `< 0.02`) — mixed corpora can confuse generic queries.
- **Want the best signal?** Ask specific questions. `"What is CPLD?"` gives stronger matches than `"What is this document about?"`.
- **Want to see what each half contributes?** Call `Retriever._retrieve_vector_rowids()` and `_retrieve_fts_rowids()` directly — both return ranked rowids, useful for diagnosing why a particular answer surfaced.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Extracted 0 page(s)` | PDF is scanned/image-only | Use a text-based PDF, or add OCR (`pytesseract`) |
| `no such module: vec0` | `sqlite-vec` extension didn't load | Confirm `sqlite_vec.load(db)` runs before queries |
| `no such module: fts5` | Python's SQLite built without FTS5 | Check `python -c "import sqlite3; print(sqlite3.sqlite_version)"` — most builds include it |
| `groq.AuthenticationError` | Invalid/missing API key | Check `.env` is in project root, no quotes/spaces around key |
| `model_not_found` | Groq model unavailable on your tier | Run `client.models.list()` and pick a free-tier model |
| `ModuleNotFoundError: No module named 'src'` | Ran `main.py` from a subfolder | Run from project root |
| `chunks.db` missing after clone | Not committed (by design) | Just run `python main.py` or click **Index** in the UI |
| `ValueError: too many values to unpack` | Old `main.py` / new `retrieve.py` mismatch | Update `main.py` to unpack `(text, source, score)` |
| Flood of `fontTools is required...` warnings | pypdf can't parse some font encodings | `fonttools` is in `requirements.txt` — reinstall if missing |
| `command not found: streamlit` | `.venv` not activated | `source .venv/bin/activate` |
| Streamlit port 8501 already in use | Another Streamlit is running | `streamlit run app.py --server.port 8502` |
| `chunks.db` resets when I switch CLI ↔ UI | Shared DB path (by design) | Use one entry point at a time, or re-index when switching |
| `IndentationError` in `retrieve.py` | Method added outside `class Retriever` or at wrong indent | Method bodies must be indented 8 spaces from file margin |
| `AttributeError: 'Retriever' object has no attribute 'retrieve_hybrid'` | `retrieve.py` not saved / method outside class | Verify `retrieve_hybrid()` is inside `class Retriever` |

---

## Requirements

- Python 3.10+
- ~2 GB free RAM (peak during embedding)
- ~200 MB disk (model + vector DB)
- Groq API key (free tier)
- Modern browser (for the UI)

---

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, learn from it.