"""
API Gateway — Requirement 9 backend (no UI).

Run from project root:
    uvicorn Services.gateway.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health
    GET  /api/v1/datasets
    GET  /api/v1/options
    POST /api/v1/search
    GET  /api/v1/suggestions
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import BM25_B, BM25_K1, DATASETS, PORTS, TOP_K  # noqa: E402
from Services.gateway.schemas import (  # noqa: E402
    DatasetInfo,
    DatasetsResponse,
    RetrievalOptions,
    SearchRequest,
    SearchResponse,
    SuggestionsResponse,
)
from Services.gateway.search_pipeline import SearchPipeline  # noqa: E402

pipeline = SearchPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.connect_mongo()
    yield


app = FastAPI(
    title="IR 2026 Search API",
    description="Backend for Requirement 9 — dataset selection, search modes, hybrid models, BM25 tuning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    mongo_ok = pipeline._mongo_connected
    return {
        "status": "ok",
        "mongodb": mongo_ok,
        "datasets_configured": list(DATASETS.keys()),
    }


@app.get("/api/v1/datasets", response_model=DatasetsResponse)
def list_datasets():
    """List datasets available for search (Req 9: select dataset before querying)."""
    items = []
    for name, ir_key in DATASETS.items():
        items.append(DatasetInfo(
            name=name,
            ir_dataset_key=ir_key,
            document_count=pipeline.document_count(name),
            models_ready=pipeline.models_ready(name),
        ))
    return DatasetsResponse(datasets=items)


@app.get("/api/v1/options", response_model=RetrievalOptions)
def retrieval_options():
    """Describe execution modes, retrieval modes, and default parameters for the UI."""
    return RetrievalOptions(
        defaults={
            "execution_mode": "basic",
            "retrieval_mode": "hybrid_parallel",
            "sparse_method": "bm25",
            "dense_method": "sbert",
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "alpha": 0.5,
            "cascade_top_n": 200,
            "top_k": TOP_K,
        },
    )


@app.post("/api/v1/search", response_model=SearchResponse)
def search(body: SearchRequest):
    """
    Execute a search (Req 9).

    - execution_mode=basic: preprocessing only (core pipeline).
    - execution_mode=enhanced: spell correction, synonym expansion, search history.
    - retrieval_mode: bm25 | tfidf | sbert | word2vec | hybrid_parallel | hybrid_serial.
    - bm25_k1 / bm25_b: probabilistic model parameters (Req 2).
    - alpha / cascade_top_n / sparse_method / dense_method: hybrid controls (Req 4).
    """
    try:
        payload = pipeline.search(
            dataset=body.dataset,
            query=body.query,
            execution_mode=body.execution_mode,
            retrieval_mode=body.retrieval_mode,
            sparse_method=body.sparse_method,
            dense_method=body.dense_method,
            bm25_k1=body.bm25_k1,
            bm25_b=body.bm25_b,
            alpha=body.alpha,
            cascade_top_n=body.cascade_top_n,
            top_k=body.top_k,
            include_snippet_chars=body.include_snippet_chars,
        )
        return SearchResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@app.get("/api/v1/suggestions", response_model=SuggestionsResponse)
def query_suggestions(
    q: str = Query("", description="Partial query for autocomplete"),
    limit: int = Query(5, ge=1, le=20),
):
    """Query suggestions from search history (Requirement 5)."""
    suggestions = pipeline.suggestions(q, limit=limit) if q else []
    return SuggestionsResponse(query_prefix=q, suggestions=suggestions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "Services.gateway.main:app",
        host="0.0.0.0",
        port=PORTS["gateway"],
        reload=True,
    )
