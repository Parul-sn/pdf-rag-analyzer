# pdf-rag-analyzer

A minimal **Retrieval-Augmented Generation (RAG)** pipeline over **one or many PDFs** — built from scratch with **no heavy frameworks**, designed to run comfortably on **low-memory devices**.

Drop any number of PDFs into `data/`, ask questions, and get grounded answers with source citations — plus a per-document summary. Powered by local embeddings and a hosted LLM.

---

## Why this project exists

Most RAG tutorials use `sentence-transformers` (pulls in PyTorch, ~2 GB) + ChromaDB (~800 MB RAM). That's overkill for a small device.

This project swaps those for lightweight equivalents **without losing any RAG concept**:

| Component | Standard choice | This project | Why |
|---|---|---|---|
| Embeddings | `sentence-transformers` + PyTorch | **fastembed** (ONNX Runtime) | No PyTorch, ~130 MB model, CPU-optimized |
| Vector store | ChromaDB | **sqlite-vec** | SQLite-native, <150 MB RAM |
| LLM | OpenAI / local 7B model | **Groq** (`openai/gpt-oss-20b`) | Fast, free tier, no local GPU/RAM needed |

Same pipeline. Same concepts. Runs on a laptop with 2 GB free.

---

## Pipeline

```
data/*.pdf   (any number of PDFs, any filenames)
      │
      ▼  src/extract.py     (pypdf)                    → pages of text per PDF
      ▼  src/chunk.py       (pure Python)              → overlapping chunks
      ▼  src/embed.py       (fastembed + sqlite-vec)   → vectors + source tag in chunks.db
      ▼  src/retrieve.py    (cosine search in SQLite)  → top-k (text, source, distance)
      ▼  src/llm.py         (Groq)                     → per-document summary + answers
      ▼  main.py            (wires it all together)
```

**Concepts demonstrated:**

- Overlapping chunking (preserves context at boundaries)
- Local embedding with a 384-dim model (`BAAI/bge-small-en-v1.5`)
- Vector similarity search via SQLite extension
- **Multi-document indexing** — each chunk is tagged with its source PDF
- **Source-grounded answers** — every answer shows which PDF(s) it drew from
- Grounded LLM answers (context-only, refuses to hallucinate)
- Build-once caching (`chunks.db` reused across runs)

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

### 4. Create the data folder and add PDFs

```bash
mkdir -p data
```

Place **one or more** PDFs into `data/`. Any filenames, any number. The pipeline picks up every `*.pdf` in that folder automatically.

> **Note:** Scanned/image-only PDFs won't extract text with `pypdf`. Use text-based PDFs (e.g. research papers, documentation, reports).

### 5. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Create an API key (starts with `gsk_`)

### 6. Configure the key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

### 7. Run it

```bash
python main.py
```

On the **first run**, the pipeline will:

- Discover every PDF in `data/`
- Extract text from each
- Chunk it into overlapping segments
- Embed each chunk (downloads the ~130 MB model once, cached)
- Store vectors **and the source filename** in `chunks.db`
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

## Multiple PDFs

Drop as many `.pdf` files as you like into `data/` — any filenames, no config needed.
On first run, all of them are chunked, embedded, and tagged with their source filename.

- **Summary** prints one summary per PDF.
- **Q&A** shows which PDF(s) each answer draws from, e.g.:

  ```
  Q: What does the tax report say about Pillar Two?
  A: <grounded answer here>
     [sources: dttl-tax-technology-report-2023.pdf]
  ```

- To rebuild after adding/removing PDFs:

  ```bash
  rm chunks.db && python main.py
  ```

---

## Project structure

```
pdf-rag-analyzer/
├── data/                   # drop any number of *.pdf files here (not committed)
├── src/
│   ├── extract.py          # PDF → list of page texts
│   ├── chunk.py            # pages → overlapping chunks
│   ├── embed.py            # (chunk, source) tuples → vectors → chunks.db
│   ├── retrieve.py         # query → top-k (text, source, distance)
│   └── llm.py              # Groq wrapper: summarize + answer
├── main.py                 # orchestration (multi-PDF discovery, summary, Q&A)
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
| `main.py` | `k` in `retriever.retrieve()` | `4` (Q&A), `20` (source discovery) | Number of chunks sent to the LLM per query. |

---

## Tuning tips

- **Getting vague summaries?** Increase `k` in the summary call (try `8–10`) or use longer PDFs.
- **Getting hallucinations?** Lower `temperature` in `src/llm.py` (try `0.1`) and reduce `k`.
- **Chunks feel too short?** Increase `chunk_size` to `1500–2000`.
- **Retrieval misses context?** Increase `overlap` to `200`.
- **Mixing very different documents?** Consider a score threshold (drop results with `distance > 0.85`) — mixed corpora can confuse generic queries.
- **Want the best signal?** Ask specific questions. `"What is CPLD?"` gives stronger matches than `"What is this document about?"`.

---

## Extending this project

Ideas for going further (all buildable in a few lines):

- **Score threshold** — in `main.py`, drop results with `distance > 0.85` before calling the LLM.
- **`SELECT DISTINCT source`** — replace the wide-retrieve hack in `distinct_sources()` with a direct SQL query.
- **Web UI** — `pip install streamlit` and wrap `main.py` in a chat interface (~30 lines).
- **Hybrid search** — combine vector search with SQLite FTS5 keyword search, then re-rank.
- **Filter by source** — extend `retrieve()` with an optional `source_filter` parameter so you can ask questions about *one* PDF only.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Extracted 0 page(s)` | PDF is scanned/image-only | Use a text-based PDF, or add OCR (`pytesseract`) |
| `no such module: vec0` | `sqlite-vec` extension didn't load | Confirm `sqlite_vec.load(db)` runs before queries |
| `groq.AuthenticationError` | Invalid/missing API key | Check `.env` is in project root, no quotes/spaces around key |
| `model_not_found` | Groq model unavailable on your tier | Run `client.models.list()` and pick a free-tier model |
| `ModuleNotFoundError: No module named 'src'` | Ran `main.py` from a subfolder | Run from project root |
| `chunks.db` missing after clone | Not committed (by design) | Just run `python main.py` — it rebuilds automatically |
| `ValueError: too many values to unpack` | Old `main.py` / new `retrieve.py` mismatch | Update `main.py` to unpack `(text, source, distance)` |
| Flood of `fontTools is required...` warnings | pypdf can't parse some font encodings | Already fixed — `fonttools` is in `requirements.txt` |

---

## Requirements

- Python 3.10+
- ~2 GB free RAM (peak during embedding)
- ~200 MB disk (model + vector DB)
- Groq API key (free tier)

---

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, learn from it.