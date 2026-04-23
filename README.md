# CompliGuard-FR

A RAG (Retrieval Augmented Generation) assistant for the **AMF Règlement Général**, the core regulatory text of the *Autorité des marchés financiers* (French financial markets authority).

Ask questions in **French or English** and get answers grounded strictly in the regulatory text, with cited article references.



## Architecture

```
INGESTION
──────────────────────────────────────────────────────
  PDF  →  Extract  →  Segment by Article  →  Chunk
          (PyMuPDF)   (LIVRE/TITRE/        (1200 chars,
                       CHAPITRE/Section)    paragraph-aware)
                                  ↓
                             Embed  →  FAISS Index
                          (multilingual)

QUERY
──────────────────────────────────────────────────────
  Question  →  Embed  →  FAISS Search  →  Top-K Chunks
                                               ↓
                                    LLM (Claude / Ollama)
                                               ↓
                                    Answer + Citations

INTERFACES
──────────────────────────────────────────────────────
  Gradio UI                    REST API (FastAPI)
```



## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure your API key**

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

**3. Add regulatory PDFs**

Place AMF PDF files in the `data/` directory.

**4. Run the ingestion pipeline**

```bash
python ingest.py
```

This extracts text, segments by article, embeds with a multilingual model, and builds a FAISS index in `vector_store/`.

---

## Usage

### Gradio UI

```bash
python ui.py
```

![CompliGuard-FR UI](image.png)

Full-featured interface with conversation history sidebar, source excerpts panel, and LIVRE filter.

### REST API

```bash
uvicorn api:app --reload
```

`POST /ask`

```json
{
  "question": "Quelles sont les obligations de transparence pour un émetteur ?",
  "model": "claude",
  "top_k": 8,
  "livre": "LIVRE II"
}
```

### CLI

```bash
# Interactive chat
python llm.py

# Single retrieval query
python query.py "Article 111-1" --top-k 5
```

---

## Models

| Backend | How to use | Notes |
|---|---|---|
| **Claude** (default) | Set `ANTHROPIC_API_KEY` in `.env` | Best quality |
| **Ollama** | Run `ollama pull mistral` locally | Offline fallback |

---

## Demo questions

| Question | What it shows |
|---|---|
| `Quelles sont les obligations de transparence pour un émetteur ?` | French, rich citations |
| `What are the sanctions for market abuse?` | English support |
| *(follow-up)* `Can you give me a specific example?` | Conversation memory |
| `What is the AMF's position on crypto assets?` | Out-of-corpus guardrail |

---

## Project structure

```
CompliGuard-FR/
├── data/                  # AMF PDFs (git-ignored)
├── vector_store/          # FAISS index + metadata (git-ignored)
├── extract/
│   └── pdf_reader.py      # PDF text extraction + noise cleanup
├── transform/
│   ├── segmenter.py       # Article-level segmentation
│   └── chunker.py         # Paragraph-aware chunker
├── vector/
│   ├── embeddings.py      # fastembed wrapper (multilingual)
│   └── vector_store.py    # FAISS build / load / search
├── storage/
│   └── models.py          # Chunk pydantic model
├── ingest.py              # ETL pipeline entry point
├── generate_metadata.py   # Metadata-only pass (no embedding)
├── query.py               # Retrieval layer
├── llm.py                 # Generation layer (Claude + Ollama)
├── ui.py                  # Gradio UI
├── api.py                 # FastAPI REST layer
└── test_embeddings.py     # Embedding smoke test
```
