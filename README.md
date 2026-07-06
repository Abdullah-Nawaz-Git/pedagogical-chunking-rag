# ragkit

`ragkit` is a small toolkit for reproducing and comparing three RAG ingestion
pipelines over Arabic high-school mathematics PDFs:

- `proposed`: Gemini structured extraction + pedagogical chunking
- `b1`: Tesseract OCR + fixed-window chunking
- `b2`: Gemini structured extraction + fixed-window chunking

The code keeps rendering, embedding, and Pinecone upsert logic shared so the
experiments differ only in the variables under study.

## Layout

- `ragkit/` contains the reusable pipeline components.
- `experiments/` contains one runnable entry point per configuration.
- `requirements.txt` lists the Python dependencies.

## Prerequisites

- Python 3.10+.
- Tesseract OCR with Arabic language data for the OCR baselines.
- Google Cloud / Vertex AI credentials for Gemini extraction and embeddings.
- Pinecone API access.
- Cloudinary credentials for diagram uploads used by `proposed` and `b2`.

On macOS:

```bash
brew install tesseract tesseract-lang
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values for your project.

- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION` (defaults to `us-central1` if omitted)
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` for the proposed pipeline
- `PINECONE_INDEX_NAME_B1`
- `PINECONE_INDEX_NAME_B2`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`

Cloudinary is only required when the Gemini-based pipelines run the diagram
crop/upload stage. You can skip that stage with `--skip-bbox`.

## Run

Each experiment module accepts the same core arguments:

- `--pdf` path to the input PDF
- `--semester` `1` or `2`
- `--cache-dir` output cache directory
- `--start-page`, `--end-page`, and `--max-pages` for page ranges

Example runs:

```bash
python -m experiments.proposed --pdf book.pdf --semester 1
python -m experiments.b1 --pdf book.pdf --semester 1
python -m experiments.b2 --pdf book.pdf --semester 1
```

The proposed and B2 pipelines also support:

- `--skip-bbox` to skip diagram cropping and Cloudinary upload
- `--chunk-only` to reuse cached extraction outputs and jump straight to chunking

All experiments also support:

- `--stop-after-texts` to stop after chunk creation and writing
  `embedding_texts.jsonl`/`chunks.json`, skipping final chunk embedding and
  Pinecone upsert stages


- **`-provenance-only` flag** (in `pipeline.py`)
- Available on all three experiments. When set it forces `chunk_only=True` (skip render/extract/crop for Gemini experiments — reuses the cached `page_*.json` extractions) and `stop_after_texts=True` (skip embedding + Pinecone upsert).
- Net effect: rebuilds `chunks.json`, `embedding_texts.jsonl`, and the `_provenance.jsonl` files cheaply, with no Gemini calls and no vector writes. Run e.g. `python -m experiments.b2 --pdf ... --semester 2 --provenance-only`.

## Output

Each run writes its artefacts into the cache directory configured for the
experiment, including rendered pages, extraction JSON, OCR text, chunks, logs,
and embedding text caches.

For offline inspection/debugging, all experiments write:

- `chunks.json`
- `embedding_texts.jsonl`

If `--stop-after-texts` is set, the run exits after these artefacts are
written (before final Gemini chunk embedding and Pinecone upsert).

When multiple experiment cache directories were generated for the same PDF and
semester, the pipeline automatically reuses matching upstream artefacts from
sibling caches before rerunning expensive stages such as rendering, OCR, and
Gemini extraction.

## Notes

- The OCR baselines require `pytesseract` in Python and the Tesseract binary on
  the system path.
- The Gemini-based pipelines require valid Google Cloud credentials and a
  Vertex AI-enabled project.
- Pinecone index names are isolated per experiment so results do not collide.


rag-refactored/
├── ragkit/                   # The REUSABLE TOOLKIT (package)
│   ├── __init__.py           # Package definition
│   ├── config.py             # Configuration settings (all the "knobs")
│   ├── cache.py              # Disk storage organization
│   ├── pipeline.py           # The main orchestrator (glues everything)
│   ├── render.py             # PDF → PNG images
│   ├── extract/              # Text extraction methods
│   │   ├── gemini.py         # Google Gemini extractor
│   │   └── tesseract.py      # OCR extractor
│   ├── represent.py          # Format text for embedding
│   ├── chunk/                # Chunking strategies
│   │   ├── fixed.py          # Fixed-size chunks
│   │   └── pedagogical.py    # Smart chunks (by topic/section)
│   ├── embed.py              # Convert text → vectors
│   └── index.py              # Store vectors in Pinecone
│
├── experiments/              # THIN EXPERIMENT SCRIPTS (use the toolkit)
│   ├── proposed.py           # Proposed method: Gemini + pedagogical
│   ├── b1.py                 # Baseline 1: Tesseract OCR + fixed chunks
│   └── b2.py                 # Baseline 2: Gemini + fixed chunks
│
├── cache/                    # Cached outputs from runs
│   ├── chunks.json           # Final chunks
│   ├── embedding_texts.jsonl # Text for embeddings
│   ├── pages/                # Rendered PNG images
│   ├── extractions/          # Page-by-page structured data
│   ├── bboxes/               # Diagram locations
│   └── diagrams/             # Cropped diagram images
│
└── requirements.txt          # List of Python libraries to install