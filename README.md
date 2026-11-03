# pdf-rag-analyzer

A minimal **Retrieval-Augmented Generation (RAG)** pipeline over a single PDF — built from scratch with **no heavy frameworks**, designed to run comfortably on **low-memory devices**.

Ask questions about any PDF and get grounded answers, plus a document summary — all powered by local embeddings and a hosted LLM.

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
data/sample.pdf
      │
      ▼  src/extract.py     (pypdf)                    → pages of text
      ▼  src/chunk.py       (pure Python)              → overlapping chunks
      ▼  src/embed.py       (fastembed + sqlite-vec)   → vectors in chunks.db
      ▼  src/retrieve.py    (cosine search in SQLite)  → top-k chunks
      ▼  src/llm.py         (Groq)                     → summary + answers
      ▼  main.py            (wires it all together)
```

**Concepts demonstrated:**
- Overlapping chunking (preserves context at boundaries)
- Local embedding with a 384-dim model (`BAAI/bge-small-en-v1.5`)
- Vector similarity search via SQLite extension
- Grounded LLM answers (context-only, refuses to hallucinate)
- Build-once caching (`chunks.db` reused across runs)

---

## Quickstart

### 1. Clone and enter

```bash
git clone https://github.com/<your-username>/pdf-rag-analyzer.git
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

### 4. Add a PDF

Place the PDF you want to query at `data/sample.pdf`.

> **Note:** Scanned/image-only PDFs won't extract text with `pypdf`. Use a text-based PDF (e.g. research papers, documentation, reports).

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

On the first run, the pipeline will:
- Extract text from the PDF
- Chunk it into overlapping segments
- Embed each chunk (downloads the ~130 MB model once, cached)
- Store vectors in `chunks.db`
- Print a summary
- Enter an interactive Q&A loop

Type `exit` to quit.

**Subsequent runs** reuse `chunks.db` and skip re-embedding — starting instantly.

To force a rebuild after changing the PDF or chunk settings:

```bash
rm chunks.db
python main.py
```

---

## Project structure

```
pdf-rag-analyzer/
├── data/
│   └── sample.pdf          # your PDF (not committed)
├── src/
│   ├── extract.py          # PDF → list of page texts
│   ├── chunk.py            # pages → overlapping chunks
│   ├── embed.py            # chunks → vectors → chunks.db
│   ├── retrieve.py         # query → top-k relevant chunks
│   └── llm.py              # Groq wrapper: summarize + answer
├── main.py                 # orchestration
├── requirements.txt
├── .env                    # GROQ_API_KEY (not committed)
├── .gitignore
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
| `main.py` | `k` in `retrieve()` | `5` (summary), `3` (QA) | Number of chunks sent to the LLM. |

---

## Tuning tips

- **Getting vague summaries?** Increase `k` in the summary retrieval (try `8–10`) or use a longer PDF.
- **Getting hallucinations?** Lower `temperature` in `src/llm.py` (try `0.1`) and reduce `k`.
- **Chunks feel too short?** Increase `chunk_size` to `1500–2000`.
- **Retrieval misses context?** Increase `overlap` to `200`.
- **Want to test on a real PDF?** A 20-page text PDF produces far better results than a 1-page one — retrieval needs material to work with.

---

## Extending this project

Ideas for going further (all buildable in a few lines):

- **Score threshold** — in `retrieve.py`, drop results with `distance > 0.7`. In `answer_question`, skip the LLM entirely if no chunks pass.
- **Multi-PDF support** — add a `source` column to `vec_chunks` and print it alongside each result.
- **Web UI** — `pip install streamlit` and wrap `main.py` in a chat interface (~30 lines).
- **Hybrid search** — combine vector search with SQLite FTS5 keyword search, then re-rank.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Extracted 0 page(s)` | PDF is scanned/image-only | Use a text-based PDF, or add OCR (`pytesseract`) |
| `no such module: vec0` | `sqlite-vec` extension didn't load | Confirm `sqlite_vec.load(db)` runs before queries |
| `groq.AuthenticationError` | Invalid/missing API key | Check `.env` is in project root, no quotes/spaces around key |
| `model_not_found` | Groq model unavailable on your tier | Run the model-list snippet in README → pick a free-tier model |
| `ModuleNotFoundError: No module named 'src'` | Ran `main.py` from a subfolder | Run from project root |
| `chunks.db` missing after clone | Not committed (by design) | Just run `python main.py` — it rebuilds automatically |

---

## Requirements

- Python 3.10+
- ~2 GB free RAM (peak during embedding)
- ~200 MB disk (model + vector DB)
- Groq API key (free tier)

---

## License

MIT — use it, fork it, learn from it.