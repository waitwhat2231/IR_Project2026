# Information Retrieval System — IR 2026

**Damascus University · Faculty of Information Technology Engineering · Information Retrieval Course (2026)**

A modular, service-oriented information retrieval (IR) engine that indexes a large
document collection and answers free-text queries using four classical and neural
retrieval models — **BM25**, **TF-IDF**, **SBERT**, and **Word2Vec** — together with
two **hybrid fusion** strategies. The system additionally provides query refinement,
unsupervised document clustering for corpus visualisation, and a standards-based
ranking-evaluation harness (MAP, Recall, P@10, nDCG).

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [System Overview](#2-system-overview)
3. [Environment & Dependencies](#3-environment--dependencies)
4. [Repository Layout](#4-repository-layout)
5. [The Dataset](#5-the-dataset)
6. [Offline Pipeline (Index/Model Construction)](#6-offline-pipeline-indexmodel-construction)
7. [Online Architecture (Serving)](#7-online-architecture-serving)
8. [Retrieval Models — Theory & Implementation](#8-retrieval-models--theory--implementation)
9. [Query Processing & Refinement](#9-query-processing--refinement)
10. [Document Clustering](#10-document-clustering)
11. [Evaluation Methodology & Results](#11-evaluation-methodology--results)
12. [Setup & Execution Guide](#12-setup--execution-guide)
13. [API Reference](#13-api-reference)
14. [Data Artifact Reference](#14-data-artifact-reference)
15. [Glossary](#15-glossary)

---

## 1. Abstract

This project implements an end-to-end information retrieval system over the
**BEIR / Webis-Touché 2020** argument-retrieval collection (~382K documents). The
design separates an **offline construction phase** — where the corpus is downloaded,
preprocessed, indexed, and embedded into several representations — from an **online
serving phase** — where a FastAPI gateway loads the prebuilt artifacts and answers
queries in real time.

The retrieval layer is deliberately pluralistic: it exposes a _sparse lexical_ family
(BM25, TF-IDF), a _dense semantic_ family (SBERT, Word2Vec), and _hybrid_ combinations
of the two via score-level fusion (parallel) and cascade reranking (serial). A separate
evaluation service grades every model against expert relevance judgments using the
standard IR metrics required by the course (MAP, Recall, Precision@10, nDCG). The system
is rounded out by a query-refinement module (spell-correction, synonym expansion, search
history) and an unsupervised clustering module that projects the corpus into 2-D space for
visual exploration.

---

## 2. System Overview

The system is organised around a clean separation between **building** the search
indexes (slow, run once) and **serving** queries against them (fast, run continuously).

```
                          OFFLINE (build once)                          ONLINE (serve)
  ┌──────────────┐   ┌───────────────────────────────────────┐   ┌───────────────────────┐
  │  ir_datasets │   │ step1 download → step2 preprocess →    │   │  Streamlit / React UI │
  │ (Webis-Touché│──▶│ step3 inverted index → step4 TF-IDF →  │   │            │          │
  │     2020)    │   │ step5 SBERT → step6 Word2Vec →         │   │            ▼          │
  └──────────────┘   │ step7 BM25 → step9 MongoDB →           │   │   FastAPI Gateway     │
                     │ step10 clustering                       │   │  (Services/gateway)   │
                     └───────────────────┬───────────────────┘   │            │          │
                                         │ writes artifacts to     │            ▼          │
                                         ▼                         │   SearchPipeline      │
                              ┌────────────────────┐               │   ├─ Preprocessor     │
                              │  data/ (pkl, npy,  │◀──────────────┤   ├─ QueryRefiner      │
                              │  npz, faiss, json) │   loaded by    │   ├─ HybridRetriever  │
                              │  + MongoDB         │   the gateway  │   │   ├─ BM25          │
                              └────────────────────┘               │   │   ├─ TF-IDF        │
                                                                   │   │   ├─ SBERT (FAISS) │
                                                                   │   │   └─ Word2Vec      │
                                                                   │   └─ ClusterManager    │
                                                                   └───────────────────────┘
```

**Design principles**

- **Service-oriented architecture (SOA).** Each capability (preprocessing, indexing,
  retrieval, query refinement, evaluation, clustering) lives in its own package under
  `Services/`, with a single FastAPI _gateway_ acting as the public entry point.
- **Offline/online split.** Expensive work (encoding 382K documents with SBERT, training
  Word2Vec, building the inverted index) happens once and is persisted to `data/`. The
  online path only _loads_ these artifacts.
- **Memory-safe streaming.** The corpus is too large to hold raw + processed + indexed in
  RAM simultaneously, so preprocessing writes a _multi-block pickle stream_ that every
  downstream step consumes chunk-by-chunk (`shared/pickle_stream.py`).
- **Separation of scores and content.** Retrievers return only `(doc_id, score)` pairs;
  the original document title/text is fetched separately from MongoDB at display time.

---

## 3. Environment & Dependencies

### Platform

| Component            | Recommendation                                                        |
| -------------------- | --------------------------------------------------------------------- |
| OS                   | Windows 10/11, Linux, or macOS (developed on Windows 10 + PowerShell) |
| Python               | 3.10 (Conda environment named `ir_project` is assumed by the docs)    |
| RAM                  | ≥ 8 GB (offline peaks ~4–5 GB; SBERT encoding benefits from a GPU)    |
| MongoDB              | 7.x (via Docker, or a local install)                                  |
| Frontend — Streamlit | Python only, via `frontend/requirements.txt` (no extra runtime)       |
| Frontend — React     | Node.js, in `web/` (see `web/README.md`)                              |

### Python packages

The committed root `requirements.txt` pins the **complete** dependency set — gateway runtime,
offline pipeline, and every retrieval model — at the exact versions the system is built and
tested against. The Streamlit frontend has its own, lighter `frontend/requirements.txt`; the
React frontend's dependencies live in `web/package.json` (see `web/README.md`).

```text
# Core dependencies (requirements.txt)
# Install with: pip install -r requirements.txt
--extra-index-url https://download.pytorch.org/whl/cpu

en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
faiss-cpu==1.7.4
fastapi==0.137.1
gensim==4.3.2
httpx==0.27.0
ir-datasets==0.5.9
matplotlib==3.10.9
nltk==3.8.1
numpy==1.26.4
pandas==2.3.3
pydantic==2.13.4
pymongo==4.6.3
pyspellchecker==0.8.1
rank-bm25==0.2.2
scikit-learn==1.7.2
scipy==1.12.0
seaborn==0.13.2
sentence-transformers==5.6.0
spacy==3.7.2
streamlit==1.58.0
torch==2.12.0+cpu
tqdm==4.68.2
umap-learn==0.5.12
uvicorn==0.49.0
plotly==6.8.0

# Streamlit frontend (frontend/requirements.txt)
streamlit>=1.32.0
requests>=2.31.0
urllib3>=2.0.0
```

> **NLTK data.** First run requires downloading several NLTK corpora — not installable via pip.
> From Python:
>
> ```python
> import nltk
> for p in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger", "omw-1.4"]:
>     nltk.download(p)
> ```

### Configuration

All shared constants live in `shared/config.py`:

```12:43:shared/config.py
# Datasets
DATASETS = {
    "webis-touche2020": "beir/webis-touche2020/v2"
}

# MongoDB
MONGO_URI  = "mongodb://localhost:27017"    # local dev
MONGO_URI_DOCKER = "mongodb://mongodb:27017"  # inside Docker
MONGO_DB   = "ir_system"
MONGO_COLL = "documents"

# BM25 defaults
BM25_K1 = 1.5
BM25_B  = 0.75

# Embedding model
SBERT_MODEL = "all-MiniLM-L6-v2"

# Service ports
PORTS = {
    "gateway":          8000,
    "preprocessing":    8001,
    "indexing":         8002,
    "retrieval":        8003,
    "query_refinement": 8004,
}

TOP_K          = 10
CANDIDATE_SIZE = 100   # hybrid serial first stage
```

Importing `shared.config` also **creates** the `data/raw`, `data/processed`,
`data/indexes`, and `data/models` directories if they do not exist.

---

## 4. Repository Layout

```
IR_Project2026/
├── shared/                            # cross-cutting utilities
│   ├── config.py                      # paths, dataset map, ports, BM25 defaults
│   ├── database.py                    # DocumentDatabase — MongoDB wrapper
│   ├── models.py                      # shared Pydantic/dataclass models
│   ├── pickle_stream.py               # streaming reader for multi-block pickles
│   └── utils.py                       # misc shared helpers
│
├── offline/                           # the build pipeline (run once, in order)
│   ├── step1_download.py              # download docs/queries/qrels via ir_datasets
│   ├── step2_preprocess.py            # clean + tokenise + stem (streaming)
│   ├── step3_build_inverted_index.py  # term → {doc_id: tf} index for BM25
│   ├── step4_train_tfidf.py           # TF-IDF matrix
│   ├── step5_train_sbert.py           # SBERT embeddings + FAISS index
│   ├── step6_train_word2vec.py        # Word2Vec model + doc vectors
│   ├── step7_train_bm25.py            # BM25 config + sensitivity report
│   ├── step8_verify_hybrid.py         # smoke test of the hybrid retriever
│   ├── step9_load_to_mongodb.py       # load original text into MongoDB
│   ├── step10_cluster.py              # KMeans clustering + 2-D scatter
│   ├── interactive_search.py          # terminal REPL search tool
│   └── test_my_index.py               # ad-hoc inverted-index sanity check
│
├── Services/                          # service-oriented application code
│   ├── gateway/                       # FastAPI public API — the live entry point
│   │   ├── main.py                    # endpoints
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   └── search_pipeline.py         # orchestrates preprocess → retrieve → fetch
│   ├── api_gateway/                   # legacy gateway package, superseded by gateway/
│   │   └── main.py
│   ├── PreprocessingService/
│   │   ├── main.py                    # standalone service entry point (port 8001)
│   │   └── preprocessor.py            # TextPreprocessor (NLP pipeline)
│   ├── indexing_service/
│   │   ├── main.py                    # standalone service entry point (port 8002)
│   │   └── inverted_index.py          # InvertedIndexManager
│   ├── retrieval_service/
│   │   ├── main.py                    # standalone service entry point (port 8003)
│   │   ├── bm25_retriever.py          # Okapi BM25
│   │   ├── tfidf_retriever.py         # sparse VSM
│   │   ├── sbert_retriever.py         # dense SBERT + FAISS
│   │   ├── word2vec_retriever.py      # dense Word2Vec
│   │   ├── hybrid_retriever.py        # parallel fusion + serial cascade
│   │   └── query_refiner.py           # spell-check, synonyms, history (in-process)
│   ├── query_refinement_service/
│   │   ├── main.py                    # standalone service entry point (port 8004)
│   │   ├── query_history.py           # search_history.json read/write
│   │   ├── spell_corrector.py         # pyspellchecker-based correction
│   │   └── synonym_expander.py        # WordNet synonym expansion
│   ├── clustering_service/
│   │   └── clusterer.py               # ClusterManager (serves cluster artifacts)
│   └── ranking_evaluation_service/
│       ├── scorer.py                  # metric math (pure functions)
│       ├── evaluator.py               # builds runs, scores, saves reports
│       └── main.py                    # CLI entry point
│
├── web/                               # React 19 + TypeScript UI (see web/README.md)
│
├── frontend/                          # Streamlit UI (see frontend/README.md)
│
├── docs/
│   └── postman/
│       └── IR_2026_Search_API.postman_collection.json
│
├── tests/                             # ad-hoc / unit tests for the offline models
│   ├── evaluate_tfidf.py
│   ├── test_bm25.py
│   ├── test_indexing.py
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   ├── test_tfidf_quick.py
│   ├── test_word2vec.py
│   └── testfile.py
│
├── docker-compose.yml                 # MongoDB container
├── start_all_services.sh              # launches the gateway + standalone microservices together
└── requirements.txt                   # full pinned dependency set (see §3)
```

> **Not shown above.** `web/` (React 19 + TypeScript UI) has its own `web/README.md` and is
> omitted from this tree for that reason — see §7 and §12.4 for how it fits into the system.
> `data/` is git-ignored and generated by the offline pipeline, so it won't exist in a fresh
> checkout (see §14 for its layout once populated).

> **Note on duplicates.** There are two gateway-like packages: `Services/gateway` is the live
> one; `Services/api_gateway` is legacy. The active code paths are the ones described in this
> guide.

---

## 5. The Dataset

The system is configured for **Webis-Touché 2020** (`beir/webis-touche2020/v2`), an
argument-retrieval benchmark from the BEIR suite. It comprises:

- **~382,000 documents** — argumentative passages on controversial topics
  (e.g. teacher tenure, e-cigarettes, climate policy).
- **49 evaluation queries** with relevance judgments.
- **Graded relevance qrels** — each (query, doc) pair is labelled `0` (not relevant),
  `1` (relevant), or `2` (highly relevant).

`offline/step1_download.py` retrieves three artifacts per dataset via `ir_datasets`:

| File                           | Contents                                |
| ------------------------------ | --------------------------------------- |
| `data/raw/<name>/docs.pkl`     | `{doc_id: {"title": str, "text": str}}` |
| `data/raw/<name>/queries.json` | `{query_id: "query text"}`              |
| `data/raw/<name>/qrels.json`   | `{query_id: {doc_id: relevance_grade}}` |

Adding another dataset is a one-line change to the `DATASETS` map in `shared/config.py`,
after which the whole pipeline can be re-run for it.

---

## 6. Offline Pipeline (Index/Model Construction)

The offline scripts must be run **in order** from the project root. Each step is
idempotent — it skips work if its output already exists — so re-running the pipeline is
safe. The data dependencies are:

```
step1 (raw docs/queries/qrels)
  ├─▶ step2 (processed_docs.pkl stream, processed_queries.json)
  │     ├─▶ step3 (inverted index)  ─▶ step7 (BM25 config)
  │     ├─▶ step4 (TF-IDF matrix)
  │     └─▶ step6 (Word2Vec)
  ├─▶ step5 (SBERT embeddings + FAISS)  ─▶ step10 (clustering, also needs step2)
  └─▶ step9 (MongoDB original text)
                          step8 (verify hybrid) needs steps 4–7
```

### Step-by-step

| Step | Script                          | Purpose                                                                                | Reads                                     | Writes                                                                         |
| ---- | ------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------ |
| 1    | `step1_download.py`             | Download corpus from BEIR                                                              | (network)                                 | `data/raw/<name>/{docs.pkl, queries.json, qrels.json}`                         |
| 2    | `step2_preprocess.py`           | Normalise → tokenise → stopword-remove → Porter-stem; streams output in 20K-doc blocks | `raw/docs.pkl`, `raw/queries.json`        | `processed/<name>/processed_docs.pkl` (multi-block), `processed_queries.json`  |
| 3    | `step3_build_inverted_index.py` | Build `term → {doc_id: tf}` postings + `df`, `N`, `avg_dl`, doc lengths                | `processed_docs.pkl`                      | `indexes/<name>_inverted.pkl`                                                  |
| 4    | `step4_train_tfidf.py`          | Fit `TfidfVectorizer`, build sparse doc matrix                                         | `processed_docs.pkl` (`processed_str`)    | `models/tfidf_<name>_matrix.npz`, `_meta.pkl`                                  |
| 5    | `step5_train_sbert.py`          | Encode raw `title+text` (truncated ~512 chars) with SBERT, build FAISS index           | `raw/docs.pkl`                            | `models/sbert_<name>/{doc_embeddings.npy, faiss.index, meta.pkl}`              |
| 6    | `step6_train_word2vec.py`       | Train Word2Vec on token streams, precompute mean doc vectors                           | `processed_docs.pkl` (`processed_tokens`) | `models/word2vec_<name>/{word2vec.model, doc_embeddings.npy, doc_ids.pkl}`     |
| 7    | `step7_train_bm25.py`           | Lock `k1`/`b`, verify index, parameter-sensitivity report                              | `indexes/<name>_inverted.pkl`             | `models/bm25_<name>/{bm25_config.pkl, parameter_sensitivity.json}`             |
| 8    | `step8_verify_hybrid.py`        | Smoke-test parallel + serial hybrid retrieval                                          | all model dirs                            | (nothing — prints only)                                                        |
| 9    | `step9_load_to_mongodb.py`      | Upsert original document text into MongoDB for display                                 | `raw/docs.pkl`                            | MongoDB `ir_system.documents`                                                  |
| 10   | `step10_cluster.py`             | MiniBatchKMeans over SBERT embeddings, top-terms, 2-D scatter (UMAP/PCA)               | `sbert_<name>/`, `processed_docs.pkl`     | `models/clusters_<name>/{clusters.json, doc_cluster_map.pkl, scatter_2d.json}` |

### Key implementation detail — memory-safe streaming

The corpus cannot be held in memory in all its forms at once. `step2` therefore writes
`processed_docs.pkl` as a _sequence_ of pickled blocks (≤ 20,000 documents each). Downstream
steps consume it via `shared/pickle_stream.py`, which repeatedly calls `pickle.load()` until
`EOFError`, yielding one block at a time so only a single chunk lives in RAM:

```python
from shared.pickle_stream import stream_chunks
for chunk in stream_chunks(processed_docs_path):   # chunk = {doc_id: {...}}
    ...   # process ~20K docs, then the chunk is garbage-collected
```

---

## 7. Online Architecture (Serving)

### The Gateway (`Services/gateway/main.py`)

A FastAPI application is the single public entry point. On startup (`lifespan`) it connects
to MongoDB; retrievers and cluster data are **lazy-loaded** per dataset on first use. CORS is
fully open so any frontend can call it.

Endpoints:

| Method | Path                         | Purpose                                         |
| ------ | ---------------------------- | ----------------------------------------------- |
| GET    | `/health`                    | Liveness + Mongo connectivity + loaded clusters |
| GET    | `/api/v1/datasets`           | Datasets with doc counts and model readiness    |
| GET    | `/api/v1/options`            | Supported modes + default parameters            |
| POST   | `/api/v1/search`             | Execute a search                                |
| GET    | `/api/v1/suggestions`        | Autocomplete from search history                |
| GET    | `/api/v1/evaluation`         | One phase's evaluation summary                  |
| GET    | `/api/v1/evaluation/compare` | baseline vs. enhanced side by side              |
| GET    | `/api/v1/clusters`           | Cluster summaries (sizes, top terms, reps)      |
| GET    | `/api/v1/clusters/scatter`   | ~10K-point 2-D sample for visualisation         |

### The Search Pipeline (`Services/gateway/search_pipeline.py`)

`SearchPipeline` is the orchestration core. A single search request flows through:

1. **Preprocess query.** In `basic` mode the query is run through the same `TextPreprocessor`
   used for documents. In `enhanced` mode it is additionally spell-corrected, saved to
   history, and expanded with WordNet synonyms before tokenisation.
2. **Retrieve.** The chosen `retrieval_mode` selects an algorithm on the per-dataset
   `HybridRetriever` (BM25 / TF-IDF / SBERT / Word2Vec / hybrid_parallel / hybrid_serial),
   returning ranked `(doc_id, score)` pairs.
3. **Hydrate.** The result `doc_id`s are looked up in MongoDB to attach `title`/`text`
   snippets. If Mongo is down, results still return (IDs + scores only).
4. **Annotate (optional).** If clustering has been loaded, each result is tagged with its
   `cluster_id`.
5. **Respond.** A structured `SearchResponse` is returned, exposing the full query-processing
   trace (original vs. processed query, spell-correction flag, expansion terms, tokens) so the
   pipeline is fully transparent.

### Frontends

Two interchangeable UIs talk to the same gateway, and both respect the same hard rule: never let a
search fire while `/api/v1/datasets` is still loading the corpus and every representation into
memory, since that is the single most reliable way to exhaust RAM mid-demo.

- **Streamlit** (`frontend/app.py`) — a polished console with a typed HTTP client
  (`frontend/api_client.py`), organised into `tabs/` (clusters, evaluation) and shared `ui/`
  building blocks. Good for quick demos. Run with `streamlit run frontend/app.py`.

- **React 19 + TypeScript** (`web/`) — a three-view SPA (Vite, Tailwind v4, TanStack Query,
  Zustand, Motion, Lucide, Recharts) built around what the grader needs to verify, not just what
  looks polished:

  - **Search** — a search-engine-style home screen that compresses into a results layout on the
    first query. Nothing `/api/v1/search` returns is hidden or summarised: the raw `doc_id` is shown
    verbatim and copyable (for cross-checking against the dataset's qrels file by hand), the score is
    rendered at full floating-point precision, the `cluster_id` badge appears wherever clustering has
    annotated a result, and a collapsible panel exposes the entire `query_processing` trace (original
    vs. processed query, spell-correction flag, expansion terms, tokens). Every result list also
    carries a raw-JSON toggle that dumps the literal API response. Model configuration — execution
    mode, retrieval mode, BM25 `k1`/`b`, hybrid sparse/dense legs, `alpha`, `cascade_top_n`, `top_k`,
    snippet length — lives in a slide-out settings panel so it never clutters the results themselves.
  - **Evaluation** — calls `/api/v1/evaluation/compare` on demand (never automatically) and puts
    MAP and nDCG front and center per the course's grading emphasis, with every other reported metric
    (Recall, P@10, …) available as a chart toggle. Baseline and enhanced phases are plotted side by
    side per model, so the required before/after comparison is one screen instead of a manual diff
    between two JSON files.
  - **Clusters** — Browse and Map tabs sharing one selection state: a card grid (size, % of corpus,
    top discriminative terms, representative `doc_id`s) and a 2-D UMAP scatter
    (`/api/v1/clusters/scatter`) colour-coded by cluster, where selecting a card highlights its points
    on the map and vice versa. The heavier ~10K-point scatter sample is fetched only once the Map tab
    is actually opened, while the lightweight cluster summaries load as soon as the dataset is ready.

  See `web/README.md` for the architecture write-up — the service-layer pattern mirroring the
  backend's SOA boundaries, the dataset-loading hard-gate, and the state-management choices.

---

## 8. Retrieval Models — Theory & Implementation

The system implements two **representation families** required by the course — _sparse
lexical_ and _dense semantic_ — plus two ways of combining them.

### 8.1 BM25 (Okapi, probabilistic)

The default sparse model. Built on the inverted index (`InvertedIndexManager`), it scores
each document for a query as a sum over query terms:

```
score(q, d) = Σ_t  IDF(t) · [ tf(t,d) · (k1 + 1) ] / [ tf(t,d) + k1 · (1 − b + b · |d|/avgdl) ]

IDF(t) = log( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

- `k1` (default **1.5**) controls term-frequency saturation; `b` (default **0.75**) controls
  document-length normalisation. Both are exposed as live UI sliders and applied dynamically
  in `HybridRetriever._compute_custom_bm25`.
- Implemented purely with the postings, document frequencies, and average document length
  stored in the inverted index — no learned weights.

### 8.2 TF-IDF (sparse vector space model)

A classic VSM using scikit-learn's `TfidfVectorizer` over the pre-stemmed corpus, with
`sublinear_tf=True`, `max_df=0.85`, `min_df=3`, `max_features=150,000`, and L2 normalisation.
Retrieval is **cosine similarity** between the query vector and the sparse document matrix
(`scipy` CSR), with top-k selected via `np.argpartition`. Persisted as a `.npz` matrix plus a
`_meta.pkl` holding the fitted vectorizer and ordered `doc_ids`.

### 8.3 SBERT (dense semantic, neural)

Sentence-BERT (`all-MiniLM-L6-v2`, 384-dim) encodes raw `title+text` into L2-normalised
embeddings. Search uses a **FAISS** index with inner-product metric (equivalent to cosine on
unit vectors). Three index types are supported — exact `flat`, `ivf` (k-means coarse
quantiser), and `hnsw` (graph ANN, the default) — trading recall for speed. Unlike the sparse
models, SBERT consumes the _original_ query text, not stemmed tokens, because the transformer
performs its own subword tokenisation.

### 8.4 Word2Vec (dense, corpus-trained)

A `gensim` Word2Vec model (200-dim, window 5, min_count 3, 5 epochs) trained on the
preprocessed token stream. Each document is represented by the **L2-normalised mean** of its
in-vocabulary word vectors; retrieval is a matrix–vector dot product (cosine). It operates on
preprocessed `processed_tokens`.

### 8.5 Hybrid — Parallel Fusion

`HybridRetriever.retrieve_hybrid` runs a sparse and a dense retriever _independently_, then
fuses their result lists. Two fusion modes exist:

- **Min-max (default).** Each list's scores are min-max normalised to `[0, 1]`, then combined
  as `α · sparse + (1 − α) · dense`. `α` is a UI slider (`0` = pure dense, `1` = pure sparse).
- **Reciprocal Rank Fusion (RRF).** Rank-based fusion `Σ 1/(60 + rank)`, robust to score-scale
  differences.

### 8.6 Hybrid — Serial Cascade

`HybridRetriever.retrieve_serial` is a two-stage **retrieve-then-rerank** pipeline: a fast
sparse model (BM25 or TF-IDF) selects the top `cascade_top_n` candidates (default 200), which
are then reordered by a dense model (SBERT or Word2Vec). This pairs lexical recall with
semantic precision at lower cost than scoring the whole corpus densely.

| Mode              | Sparse stage | Dense stage | Combination            |
| ----------------- | ------------ | ----------- | ---------------------- |
| `bm25`            | BM25         | —           | lexical only           |
| `tfidf`           | TF-IDF       | —           | lexical only           |
| `sbert`           | —            | SBERT       | semantic only          |
| `word2vec`        | —            | Word2Vec    | semantic only          |
| `hybrid_parallel` | BM25/TF-IDF  | SBERT/W2V   | weighted score fusion  |
| `hybrid_serial`   | BM25/TF-IDF  | SBERT/W2V   | candidate set → rerank |

---

## 9. Query Processing & Refinement

### Preprocessing (`Services/PreprocessingService/preprocessor.py`)

`TextPreprocessor` applies an identical pipeline to documents and queries (critical for
matching): Unicode→ASCII normalisation, lowercasing, URL/HTML/number/punctuation removal,
NLTK tokenisation, stopword + short-token filtering, and **Porter stemming**. It returns both
a joined string (`processed_str`, for TF-IDF) and a token list (`processed_tokens`, for
BM25/Word2Vec).

### Refinement (`Services/retrieval_service/query_refiner.py`) — `enhanced` mode

Active only when `execution_mode="enhanced"`:

- **Spell correction** via `pyspellchecker`, with a protected list of common
  stopwords/pronouns to avoid mangling valid short words.
- **Synonym expansion** via WordNet, restricted to academically relevant lexical categories
  (e.g. `noun.cognition`, `noun.communication`, `adj.all`) and excluding multi-word lemmas, to
  keep expansions index-compatible.
- **Search history** persisted to `data/search_history.json` (last 20 queries), powering the
  `/api/v1/suggestions` autocomplete endpoint.

---

## 10. Document Clustering

`offline/step10_cluster.py` performs **unsupervised clustering** over the SBERT document
embeddings using **MiniBatchKMeans** (default `k=20`). For each cluster it computes:

- size and percentage of the corpus,
- the top-10 **discriminative** stemmed terms (frequent within the cluster but rare elsewhere),
- 3 representative documents (closest to the cluster centroid), and
- a stratified ~10K-document **2-D projection** (UMAP, falling back to PCA) for plotting.

At serving time, `Services/clustering_service/clusterer.py` (`ClusterManager`) loads these
precomputed artifacts and provides O(1) lookups. The gateway uses it to (a) tag each search
result with its `cluster_id` and (b) serve the `/api/v1/clusters` and
`/api/v1/clusters/scatter` endpoints for corpus visualisation.

---

## 11. Evaluation Methodology & Results

The **Ranking & Evaluation Service** (`Services/ranking_evaluation_service/`) grades every
retrieval model against the dataset's qrels using standard IR metrics. It is split into three
files by responsibility:

| File           | Role                                                      |
| -------------- | --------------------------------------------------------- |
| `scorer.py`    | Pure metric math (no I/O): P@k, Recall@k, AP, nDCG@k      |
| `evaluator.py` | Builds per-model runs, scores them, writes reports        |
| `main.py`      | CLI entry point — loads models, evaluates, prints + saves |

### Metrics

| Metric          | Question it answers                                               |
| --------------- | ----------------------------------------------------------------- |
| **MAP**         | Are relevant docs ranked near the top, averaged over all queries? |
| **Recall@1000** | What fraction of all relevant docs were retrieved?                |
| **P@10**        | Of the top 10 results, how many are relevant?                     |
| **nDCG@10**     | Like P@10 but rewards placing the _most_ relevant docs first.     |

A document counts as relevant when its qrel grade is ≥ 1; nDCG uses the graded values directly.
Each metric is computed per query and then averaged across the 49 queries.

### Phases (before/after)

The `--phase` flag (`baseline` | `enhanced`) lets the same harness measure the system _before_
and _after_ enabling extra features, saving each into its own folder so the gateway's
`/api/v1/evaluation/compare` endpoint can show them side by side.

### Reported baseline results (49 queries)

| model           | MAP        | Recall@1000 | P@10       | nDCG@10    |
| --------------- | ---------- | ----------- | ---------- | ---------- |
| tfidf           | 0.0520     | 0.7330      | 0.0673     | 0.0566     |
| **bm25**        | **0.2194** | 0.8726      | **0.2898** | **0.3172** |
| sbert           | 0.1312     | 0.7706      | 0.1673     | 0.1709     |
| word2vec        | 0.1017     | 0.7441      | 0.1347     | 0.1561     |
| hybrid_parallel | 0.2137     | **0.8784**  | 0.2653     | 0.2903     |
| hybrid_serial   | 0.1431     | 0.8708      | 0.1755     | 0.1781     |

**Interpretation.** On this argument-retrieval collection BM25 is the strongest single model
across MAP/P@10/nDCG, while hybrid_parallel achieves the highest total recall — consistent with
lexical methods being well-suited to keyword-heavy debate text. Full per-query breakdowns are
written to `data/evaluation/<dataset>/<phase>/`. See `docs/evaluation_service.md` for an
extended, beginner-friendly walkthrough.

### Running the evaluation

```powershell
python Services/ranking_evaluation_service/main.py --phase baseline --top_k 1000
python Services/ranking_evaluation_service/main.py --models tfidf,bm25,sbert
```

---

## 12. Setup & Execution Guide

### 12.1 Install

```powershell
# 1. Create and activate the environment (Conda assumed by the docs)
conda create -n ir_project python=3.10 -y
conda activate ir_project

# 2. Install dependencies
pip install -r requirements.txt             # gateway + offline pipeline + all retrieval models
pip install -r frontend/requirements.txt    # for the Streamlit UI

# 3. Download NLTK data
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','averaged_perceptron_tagger','omw-1.4']]"
```

### 12.2 Start MongoDB

```powershell
docker compose up -d      # starts mongo:7 on localhost:27017
docker compose ps         # verify "healthy"
```

### 12.3 Build the artifacts (offline, run once, in order)

```powershell
python offline/step1_download.py
python offline/step2_preprocess.py
python offline/step3_build_inverted_index.py
python offline/step4_train_tfidf.py
python offline/step5_train_sbert.py        # slow on CPU (45–90 min); GPU strongly preferred
python offline/step6_train_word2vec.py
python offline/step7_train_bm25.py
python offline/step8_verify_hybrid.py      # optional smoke test
python offline/step9_load_to_mongodb.py    # needs MongoDB running
python offline/step10_cluster.py           # optional: clustering + visualisation
```

### 12.4 Serve

```powershell
# Terminal 1 — API gateway
uvicorn Services.gateway.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Streamlit UI
streamlit run frontend/app.py
#   → open http://localhost:8501

# (or) the React UI
cd web && npm install && npm run dev
```

Interactive docs are auto-generated at **http://localhost:8000/docs** (Swagger).

### 12.5 Optional: terminal search

```powershell
python offline/interactive_search.py
```

---

## 13. API Reference

### `POST /api/v1/search`

**Request body** (Pydantic `SearchRequest`):

```json
{
  "dataset": "webis-touche2020",
  "query": "should teachers get tenure",
  "execution_mode": "basic", // "basic" | "enhanced"
  "retrieval_mode": "hybrid_parallel", // bm25|tfidf|sbert|word2vec|hybrid_parallel|hybrid_serial
  "sparse_method": "bm25", // "bm25" | "tfidf"  (hybrid only)
  "dense_method": "sbert", // "sbert" | "word2vec" (hybrid only)
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "alpha": 0.5, // parallel fusion weight (sparse↔dense)
  "cascade_top_n": 200, // serial cascade candidate count
  "top_k": 10,
  "include_snippet_chars": 300
}
```

**Response** (`SearchResponse`): echoes the request mode, a full `query_processing` trace
(original query, processed query, `spell_corrected`, `expanded_terms`, `query_tokens`), and a
`results` array of `{rank, doc_id, score, title, text, cluster_id?}`.

A ready-to-import Postman collection is provided at
`docs/postman/IR_2026_Search_API.postman_collection.json`.

---

## 14. Data Artifact Reference

All generated artifacts live under `data/` (git-ignored). Layout:

```
data/
├── raw/<dataset>/
│   ├── docs.pkl                  # {doc_id: {title, text}}
│   ├── queries.json              # {qid: text}
│   └── qrels.json                # {qid: {doc_id: grade}}
├── processed/<dataset>/
│   ├── processed_docs.pkl        # multi-block stream of {doc_id: {processed_str, processed_tokens}}
│   └── processed_queries.json    # {qid: {original, processed_str, processed_tokens}}
├── indexes/
│   └── <dataset>_inverted.pkl    # InvertedIndexManager (postings, df, N, avg_dl, doc_lengths)
├── models/
│   ├── tfidf_<dataset>_matrix.npz + _meta.pkl
│   ├── sbert_<dataset>/{doc_embeddings.npy, faiss.index, meta.pkl}
│   ├── word2vec_<dataset>/{word2vec.model, doc_embeddings.npy, doc_ids.pkl}
│   ├── bm25_<dataset>/{bm25_config.pkl, parameter_sensitivity.json}
│   └── clusters_<dataset>/{clusters.json, doc_cluster_map.pkl, scatter_2d.json}
├── evaluation/<dataset>/<phase>/{metrics_summary.json, comparison.csv, report.txt, per_query/*.json}
└── search_history.json           # last 20 queries (refinement / autocomplete)
```

Original document **text is stored in MongoDB**, not on disk, and is fetched at query time by
`shared/database.py` (`DocumentDatabase`), keyed by `doc_id` (the Mongo `_id`).

---

## 15. Glossary

| Term                          | Meaning                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **BM25**                      | Okapi BM25, a probabilistic lexical ranking function with TF saturation (`k1`) and length normalisation (`b`). |
| **TF-IDF**                    | Term Frequency–Inverse Document Frequency weighting in a vector space model.                                   |
| **SBERT**                     | Sentence-BERT; produces sentence-level dense embeddings for semantic search.                                   |
| **Word2Vec**                  | Neural word-embedding model; documents represented as the mean of word vectors.                                |
| **FAISS**                     | Facebook AI Similarity Search — fast (approximate) nearest-neighbour index for dense vectors.                  |
| **Inverted index**            | Map from each term to the documents (and frequencies) in which it appears.                                     |
| **qrels**                     | Query relevance judgments — the human-labelled answer key for evaluation.                                      |
| **MAP / nDCG / P@k / Recall** | Standard IR effectiveness metrics (see §11).                                                                   |
| **Parallel fusion**           | Combining independent sparse + dense result lists by weighted/RRF score fusion.                                |
| **Serial cascade**            | Retrieve-then-rerank: a cheap model selects candidates, a richer model reorders them.                          |
| **Hydration**                 | Attaching original title/text (from MongoDB) to retrieved `doc_id`s for display.                               |
| **SOA**                       | Service-Oriented Architecture — each capability isolated behind a service boundary.                            |

---

_Damascus University · Information Retrieval 2026 · This guide documents the system as
implemented in this repository. §3 lists the complete, pinned dependency set needed to
reproduce the full pipeline._
