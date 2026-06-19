"""
API Gateway — Requirement 9 backend.

Run from project root:
    uvicorn Services.gateway.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health
    GET  /api/v1/datasets
    GET  /api/v1/options
    POST /api/v1/search
    GET  /api/v1/suggestions
    GET  /api/v1/evaluation
    GET  /api/v1/evaluation/compare
    GET  /api/v1/clusters                 <- NEW
    GET  /api/v1/clusters/scatter         <- NEW
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import BM25_B, BM25_K1, DATA_DIR, DATASETS, MODEL_DIR, PORTS, TOP_K
from Services.gateway.schemas import (
    ClusterScatterResponse,
    ClustersResponse,
    DatasetInfo,
    DatasetsResponse,
    EvaluationCompareResponse,
    EvaluationResponse,
    ModelEvaluation,
    RetrievalOptions,
    ScatterPoint,
    SearchRequest,
    SearchResponse,
    SuggestionsResponse,
)
from Services.gateway.search_pipeline import SearchPipeline
from Services.clustering_service.clusterer import ClusterManager

pipeline = SearchPipeline()

# One ClusterManager per dataset — lazy-loaded on first cluster request
_cluster_managers: Dict[str, ClusterManager] = {}


def _get_cluster_manager(dataset: str) -> ClusterManager:
    """
    Load the ClusterManager for `dataset` on first access and cache it.
    Returns a ready manager, or raises HTTPException 404 if step10 hasn't run.
    """
    if dataset not in _cluster_managers:
        manager = ClusterManager()
        cluster_dir = MODEL_DIR / f"clusters_{dataset}"
        try:
            manager.load(dataset, cluster_dir)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No cluster data found for dataset '{dataset}'. "
                    f"Run:  python offline/step10_cluster.py"
                ),
            ) from exc
        _cluster_managers[dataset] = manager
    return _cluster_managers[dataset]


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.connect_mongo()
    yield


app = FastAPI(
    title="IR 2026 Search API",
    description=(
        "Backend for Requirement 9 — search, evaluation charts, "
        "and document cluster visualisation."
    ),
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


# ── Existing endpoints (unchanged) ────────────────────────────────────────────

@app.get("/health")
def health():
    mongo_ok = pipeline._mongo_connected
    cluster_status = {
        name: _cluster_managers[name].is_ready()
        for name in _cluster_managers
    }
    return {
        "status": "ok",
        "mongodb": mongo_ok,
        "datasets_configured": list(DATASETS.keys()),
        "clusters_loaded": cluster_status,
    }


@app.get("/api/v1/datasets", response_model=DatasetsResponse)
def list_datasets():
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
    return RetrievalOptions(
        defaults={
            "execution_mode":  "basic",
            "retrieval_mode":  "hybrid_parallel",
            "sparse_method":   "bm25",
            "dense_method":    "sbert",
            "bm25_k1":         BM25_K1,
            "bm25_b":          BM25_B,
            "alpha":           0.5,
            "cascade_top_n":   200,
            "top_k":           TOP_K,
        },
    )


@app.post("/api/v1/search", response_model=SearchResponse)
def search(body: SearchRequest):
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

        # Annotate results with cluster_id if clustering has been run
        # This adds a cluster badge to each result card in the UI
        if body.dataset in _cluster_managers:
            manager = _cluster_managers[body.dataset]
            doc_ids = [r["doc_id"] for r in payload["results"]]
            cluster_map = manager.annotate_doc_ids(doc_ids)
            for result in payload["results"]:
                result["cluster_id"] = cluster_map.get(result["doc_id"])

        return SearchResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@app.get("/api/v1/suggestions", response_model=SuggestionsResponse)
def query_suggestions(
    q:     str = Query("", description="Partial query for autocomplete"),
    limit: int = Query(5, ge=1, le=20),
):
    suggestions = pipeline.suggestions(q, limit=limit) if q else []
    return SuggestionsResponse(query_prefix=q, suggestions=suggestions)


# ── Evaluation endpoints ───────────────────────────────────────────────────────

def _load_evaluation_phase(
    dataset: str,
    phase:   str,
    include_per_query: bool,
) -> Optional[EvaluationResponse]:
    phase_dir    = DATA_DIR / "evaluation" / dataset / phase
    summary_path = phase_dir / "metrics_summary.json"

    if not summary_path.exists():
        return None

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    per_query_dir = phase_dir / "per_query"
    models = []

    for model_name, model_data in summary.get("models", {}).items():
        per_query_data = None
        if include_per_query:
            safe_name = model_name.replace(" ", "_")
            pq_path   = per_query_dir / f"{safe_name}.json"
            if pq_path.exists():
                with open(pq_path, encoding="utf-8") as f:
                    per_query_data = json.load(f)

        models.append(ModelEvaluation(
            name=model_name,
            aggregate=model_data.get("aggregate", {}),
            per_query=per_query_data,
            elapsed_sec=model_data.get("elapsed_sec"),
        ))

    return EvaluationResponse(
        dataset=summary["dataset"],
        phase=summary["phase"],
        num_queries=summary["num_queries"],
        top_k=summary["top_k"],
        generated_at=summary["generated_at"],
        models=models,
    )


@app.get("/api/v1/evaluation", response_model=EvaluationResponse)
def get_evaluation(
    dataset: str  = Query(..., description="Dataset name"),
    phase:   str  = Query("baseline", pattern="^(baseline|enhanced)$"),
    include_per_query: bool = Query(False),
):
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")

    result = _load_evaluation_phase(dataset, phase, include_per_query)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation results for dataset='{dataset}' phase='{phase}'.",
        )
    return result


@app.get("/api/v1/evaluation/compare", response_model=EvaluationCompareResponse)
def get_evaluation_compare(
    dataset: str  = Query(...),
    include_per_query: bool = Query(True),
):
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")

    baseline = _load_evaluation_phase(dataset, "baseline", include_per_query)
    enhanced = _load_evaluation_phase(dataset, "enhanced", include_per_query)

    if baseline is None and enhanced is None:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation results found for dataset='{dataset}'.",
        )
    return EvaluationCompareResponse(dataset=dataset, baseline=baseline, enhanced=enhanced)


# ── NEW: Cluster endpoints ─────────────────────────────────────────────────────

@app.get(
    "/api/v1/clusters",
    response_model=ClustersResponse,
    summary="All cluster summaries",
    description=(
        "Returns metadata and summaries for all document clusters: size, percentage "
        "of corpus, top discriminative terms, and 3 representative document IDs per "
        "cluster. Use this to build the cluster legend and the cluster-browse panel "
        "in the UI. Does NOT include the 2D scatter coordinates — call /clusters/scatter "
        "for those."
    ),
)
def get_clusters(
    dataset: str = Query(..., description="Dataset name, e.g. 'webis-touche2020'"),
):
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")

    manager = _get_cluster_manager(dataset)    # raises 404 if step10 not run
    summary = manager.get_summary()
    clusters = manager.get_all_clusters()

    return ClustersResponse(
        dataset=summary["dataset"],
        n_clusters=summary["n_clusters"],
        n_docs=summary["n_docs"],
        embedding_source=summary["embedding_source"],
        generated_at=summary["generated_at"],
        clusters=clusters,
    )


@app.get(
    "/api/v1/clusters/scatter",
    response_model=ClusterScatterResponse,
    summary="2D scatter data for corpus visualisation",
    description=(
        "Returns a stratified ~10K-point sample of the corpus in 2D space "
        "(UMAP projection of SBERT embeddings), each point labelled with its "
        "cluster_id for colour-coding.\n\n"
        "Also returns all cluster summaries so the UI can render the colour legend "
        "alongside the chart in a single request.\n\n"
        "**Chart type**: scatter plot where each point is a document, colour = cluster. "
        "Hovering a point shows the cluster's top_terms. "
        "Clicking a cluster in the legend highlights only that cluster's points."
    ),
)
def get_cluster_scatter(
    dataset: str = Query(..., description="Dataset name"),
):
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")

    manager  = _get_cluster_manager(dataset)
    summary  = manager.get_summary()
    clusters = manager.get_all_clusters()
    raw_pts  = manager.get_scatter_data()

    points = [
        ScatterPoint(
            doc_id=p["doc_id"],
            x=p["x"],
            y=p["y"],
            cluster_id=p["cluster_id"],
        )
        for p in raw_pts
    ]

    return ClusterScatterResponse(
        dataset=summary["dataset"],
        n_clusters=summary["n_clusters"],
        n_points=len(points),
        clusters=clusters,
        points=points,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "Services.gateway.main:app",
        host="0.0.0.0",
        port=PORTS["gateway"],
        reload=False,
    )