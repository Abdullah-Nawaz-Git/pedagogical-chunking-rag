# ragkit

`ragkit` is a small toolkit for reproducing and comparing three RAG ingestion
pipelines over Arabic high-school mathematics PDFs:

- `proposed`: VLM structured extraction + pedagogical chunking
- `b1`: Tesseract OCR + fixed-window chunking
- `b2`: VLM structured extraction + fixed-window chunking

The code keeps rendering, embedding, and Pinecone upsert logic shared so the
experiments differ only in the variables under study.

## Layout

- `ragkit/` contains the reusable pipeline components.
- `experiments/` contains the runnable entry points for the three pipelines, plus the evaluation and reporting scripts.
- `extraction_experiments/` contains smaller extraction-focused experiments and sampling utilities.
- `qa_dataset/` contains generated QA dataset artifacts, configs, and summaries.
- `judge_eval/` contains judge-run outputs, score files, and analysis reports.
- `retrieval_eval/` contains retrieval evaluation artifacts and comparison reports.
- `cache/`, `cache_b1/`, and `cache_b2/` hold cached run outputs for the main and baseline pipelines.
- `requirements.txt` lists the Python dependencies, and `credentials.json` stores local credentials for the workspace.

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
- `--embed-only` to skip render/extract/chunk stages and jump straight to
  embedding + Pinecone upsert using cached `chunks.json` and
  `embedding_texts.jsonl`

## Retrieval-results tables and figures

After running the three retrieval evaluations, generate the reporting bundle
from their existing artifacts with:

```bash
python -m experiments.retrieval_results_report
```

It reads the fixed inputs in `retrieval_eval/` (`config_used_*`,
`retrieval_summary_*`, `retrieval_records_*.jsonl`, and
`retrieval_comparison.json`) and writes CSV/Markdown tables plus PNG/SVG figures
to `retrieval_eval/analysis/`. Use `--input-dir` or `--output-dir` to point the
script at another artifact bundle or destination.


- **`--provenance-only` flag** (in `pipeline.py`)
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


project-root/
├── ragkit/                         # Reusable pipeline components
│   ├── __init__.py
│   ├── cache.py
│   ├── config.py
│   ├── embed.py
│   ├── index.py
│   ├── pipeline.py
│   ├── render.py
│   ├── represent.py
│   ├── chunk/                      # Chunking strategies
│   │   ├── __init__.py
│   │   ├── fixed.py
│   │   └── pedagogical.py
│   └── extract/                    # Extraction backends
│       ├── __init__.py
│       ├── gemini.py
│       └── tesseract.py
├── experiments/                    # Main experiment entry points and reports
│   ├── __init__.py
│   ├── b1.py
│   ├── b2.py
│   ├── judge.py
│   ├── judge_b1.py
│   ├── judge_b2.py
│   ├── judge_proposed.py
│   ├── judge_results_report.py
│   ├── proposed.py
│   ├── qa_dataset.py
│   ├── retrieval_b1.py
│   ├── retrieval_b2.py
│   ├── retrieval_proposed.py
│   └── retrieval_results_report.py
├── extraction_experiments/         # Smaller extraction-focused experiments
│   ├── CER.py
│   ├── Random_Blocks_Sample.py
│   ├── random_diagrams_sample.csv
│   ├── random_diagrams_sample.py
│   └── random_formula_sample.py
├── qa_dataset/                     # QA dataset artifacts and summaries
├── judge_eval/                     # Judge outputs and analysis reports
├── retrieval_eval/                 # Retrieval evaluation outputs and analysis
├── cache/                          # Main pipeline cache artifacts
├── cache_b1/                       # Cache artifacts for baseline 1
├── cache_b2/                       # Cache artifacts for baseline 2
├── credentials.json                # Local credentials file
└── requirements.txt                # Python dependencies
