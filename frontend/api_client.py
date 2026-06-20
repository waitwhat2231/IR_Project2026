"""
HTTP client for the IR 2026 API Gateway.

All raw JSON is parsed here and returned as typed dataclasses.
No JSON ever leaks into the UI layer.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field, fields
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────── Response dataclasses ──────────────


@dataclass
class DatasetInfo:
    name: str
    ir_dataset_key: str
    document_count: int | None = None
    models_ready: bool = False


@dataclass
class QueryProcessingInfo:
    original_query: str
    processed_query: str
    spell_corrected: bool = False
    expanded_terms: list[str] = field(default_factory=list)
    query_tokens: list[str] = field(default_factory=list)


@dataclass
class SearchResultItem:
    rank: int
    doc_id: str
    score: float
    title: str | None = None
    text: str | None = None
    cluster_id: int | None = None  # populated when ClusterManager is loaded


@dataclass
class SearchResponse:
    dataset: str
    execution_mode: str
    retrieval_mode: str
    query_processing: QueryProcessingInfo
    results: list[SearchResultItem]
    total_results: int


@dataclass
class RetrievalOptions:
    execution_modes: list[str]
    retrieval_modes: list[str]
    sparse_methods: list[str]
    dense_methods: list[str]
    defaults: dict[str, Any]


# ─────────────────────────────────────── Evaluation dataclasses ────────────


@dataclass
class ModelEvaluation:
    """Aggregate + optional per-query metrics for one retrieval model."""

    name: str
    aggregate: dict[str, float]
    per_query: dict[str, dict[str, float]] | None = None
    elapsed_sec: float | None = None


@dataclass
class EvaluationResponse:
    dataset: str
    phase: str
    num_queries: int
    top_k: int
    generated_at: str
    models: list[ModelEvaluation]


@dataclass
class EvaluationCompareResponse:
    dataset: str
    baseline: EvaluationResponse | None = None
    enhanced: EvaluationResponse | None = None


# ─────────────────────────────────────── Cluster dataclasses ───────────────


@dataclass
class ClusterInfo:
    id: int
    size: int
    pct: float
    top_terms: list[str]
    representative_doc_ids: list[str]


@dataclass
class ClustersResponse:
    dataset: str
    n_clusters: int
    n_docs: int
    embedding_source: str
    generated_at: str
    clusters: list[ClusterInfo]


@dataclass
class ScatterPoint:
    doc_id: str
    x: float
    y: float
    cluster_id: int


@dataclass
class ClusterScatterResponse:
    dataset: str
    n_clusters: int
    n_points: int
    clusters: list[ClusterInfo]
    points: list[ScatterPoint]


# ─────────────────────────────────────── Exception ─────────────────────────


class IRAPIError(Exception):
    """
    Raised for any failure communicating with the API Gateway.

    Covers both HTTP errors (non-2xx responses) and connection errors
    so callers never need to import ``requests`` directly.
    """


# ─────────────────────────────────────── Helpers ───────────────────────────


def _to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """
    Build a dataclass instance from a raw API dict, tolerant of extra keys.

    The gateway is under active development, so it may add new response
    fields before this client is updated to match. Unknown keys are
    dropped instead of raising ``TypeError``. Missing *required* fields
    still fail loudly, but as a clear ``IRAPIError`` instead of a raw
    ``TypeError`` leaking out of the client.
    """
    known_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    try:
        return cls(**filtered)
    except TypeError as exc:
        raise IRAPIError(
            f"Gateway response for {cls.__name__} is missing required field(s): {exc}"
        ) from exc


# ─────────────────────────────────────── Response builders ─────────────────


def _build_eval_response(data: dict[str, Any]) -> EvaluationResponse:
    """Construct EvaluationResponse from a raw gateway dict."""
    models = [
        ModelEvaluation(
            name=m["name"],
            aggregate=m.get("aggregate", {}),
            per_query=m.get("per_query"),
            elapsed_sec=m.get("elapsed_sec"),
        )
        for m in data.get("models", [])
    ]
    return EvaluationResponse(
        dataset=data["dataset"],
        phase=data["phase"],
        num_queries=data["num_queries"],
        top_k=data["top_k"],
        generated_at=data["generated_at"],
        models=models,
    )


def _build_cluster_info(c: dict[str, Any]) -> ClusterInfo:
    return ClusterInfo(
        id=c["id"],
        size=c["size"],
        pct=c["pct"],
        top_terms=c.get("top_terms", []),
        representative_doc_ids=c.get("representative_doc_ids", []),
    )


# ─────────────────────────────────────── Client ────────────────────────────


def _build_session() -> requests.Session:
    """Return a Session with a conservative retry strategy."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[502, 503, 504],
        # GET is retried by default; POST is opted in explicitly because
        # /api/v1/search and /api/v1/suggestions are read-only on the
        # backend, so retrying a transient 502/503/504 is safe.
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


class IRAPIClient:
    """
    Thin, typed HTTP wrapper around the IR 2026 API Gateway.

    Usage::

        client = IRAPIClient("http://127.0.0.1:8000")
        health  = client.health()
        datasets = client.list_datasets()
        resp = client.search(dataset="webis-touche2020", query="teacher tenure")

    Can also be used as a context manager to ensure the underlying
    session is closed::

        with IRAPIClient() as client:
            ...
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = _build_session()

    def __enter__(self) -> "IRAPIClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._session.close()

    # ── internal ────────────────────────────────────────────────────────────

    def _get(self, path: str, *, timeout: int = 10) -> Any:
        try:
            resp = self._session.get(f"{self.base_url}{path}", timeout=timeout)
            resp.raise_for_status()
            if not resp.content:
                # Defensive retry once for an unexpectedly empty (but 2xx)
                # body, e.g. a transient proxy hiccup.
                resp = self._session.get(f"{self.base_url}{path}", timeout=timeout)
                resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise IRAPIError(
                f"Cannot reach {self.base_url} — is the gateway running?"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None:
                raise IRAPIError(
                    f"[{exc.response.status_code}] {exc.response.text}"
                ) from exc
            raise IRAPIError(str(exc)) from exc
        except requests.exceptions.RequestException as exc:
            raise IRAPIError(str(exc)) from exc
        except Exception as exc:
            raise IRAPIError(f"Unexpected error talking to gateway: {exc}") from exc

    def _get_once(self, path: str) -> Any:
        """
        Send exactly ONE request: no retries, no timeout.

        Reserved for endpoints whose backend work is expensive and
        non-idempotent (e.g. ``/api/v1/datasets``, which loads every
        retrieval model for every dataset into RAM). ``self._session``
        is NOT used here on purpose — it has a retry-enabled adapter
        mounted, and urllib3's ``Retry`` silently retries on read
        timeouts (not just the configured status codes) whenever
        ``total`` is set. For a normal endpoint that's harmless. For
        this one, a client-side timeout firing early just means the
        SAME expensive, RAM-hungry work gets re-triggered on the
        backend while the first attempt is still running — multiple
        times, invisibly, before anything finishes loading.

        DO NOT add a ``timeout=`` value or route this through
        ``self._session`` without understanding the above — that is
        precisely the bug this method exists to avoid.
        """
        try:
            resp = requests.get(f"{self.base_url}{path}", timeout=None)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise IRAPIError(
                f"Cannot reach {self.base_url} — is the gateway running?"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None:
                raise IRAPIError(
                    f"[{exc.response.status_code}] {exc.response.text}"
                ) from exc
            raise IRAPIError(str(exc)) from exc
        except requests.exceptions.RequestException as exc:
            raise IRAPIError(str(exc)) from exc
        except Exception as exc:
            raise IRAPIError(f"Unexpected error talking to gateway: {exc}") from exc

    def _post(self, path: str, payload: dict, *, timeout: int = 60) -> Any:
        try:
            resp = self._session.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=timeout,
            )
            if not resp.ok:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise IRAPIError(f"[{resp.status_code}] {detail}")
            return resp.json()
        except IRAPIError:
            raise
        except requests.exceptions.ConnectionError as exc:
            raise IRAPIError(
                f"Cannot reach {self.base_url} — is the gateway running?"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise IRAPIError(str(exc)) from exc
        except Exception as exc:
            # Catches e.g. JSONDecodeError on a malformed 200 response —
            # previously this leaked out unwrapped, unlike _get's behavior.
            raise IRAPIError(f"Unexpected error talking to gateway: {exc}") from exc

    # ── Health / Discovery ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Return gateway status and MongoDB connectivity."""
        return self._get("/health", timeout=5)

    def list_datasets(self) -> list[DatasetInfo]:
        """
        Return all configured datasets with doc counts and model readiness.

        This is the FIRST request the UI sends on startup, and on the
        backend it triggers loading every retrieval model for every
        dataset into RAM — a slow, non-idempotent, expensive operation.
        It is sent via ``_get_once()`` (no retries, no timeout) instead
        of the normal ``_get()``/shared-session path on purpose. See
        ``_get_once`` for why.
        """
        data = self._get_once("/api/v1/datasets")
        return [_to_dataclass(DatasetInfo, d) for d in data["datasets"]]

    def get_options(self) -> RetrievalOptions:
        """Return supported modes and default parameter values."""
        data = self._get("/api/v1/options", timeout=5)
        return _to_dataclass(RetrievalOptions, data)

    # ── Search ──────────────────────────────────────────────────────────────

    def search(
        self,
        *,
        dataset: str,
        query: str,
        execution_mode: str = "basic",
        retrieval_mode: str = "hybrid_parallel",
        sparse_method: str = "bm25",
        dense_method: str = "sbert",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        alpha: float = 0.5,
        cascade_top_n: int = 200,
        top_k: int = 10,
        include_snippet_chars: int = 2000,
    ) -> SearchResponse:
        """
        Execute a search and return a fully-typed ``SearchResponse``.

        Parameters mirror the gateway's ``SearchRequest`` schema exactly.
        ``include_snippet_chars=2000`` is the max the gateway supports; the UI
        can trim further without re-fetching.

        Raises
        ------
        IRAPIError
            On any network or HTTP error.
        """
        payload = {
            "dataset": dataset,
            "query": query,
            "execution_mode": execution_mode,
            "retrieval_mode": retrieval_mode,
            "sparse_method": sparse_method,
            "dense_method": dense_method,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "alpha": alpha,
            "cascade_top_n": cascade_top_n,
            "top_k": top_k,
            "include_snippet_chars": include_snippet_chars,
        }
        data = self._post("/api/v1/search", payload)
        return SearchResponse(
            dataset=data["dataset"],
            execution_mode=data["execution_mode"],
            retrieval_mode=data["retrieval_mode"],
            query_processing=_to_dataclass(
                QueryProcessingInfo, data["query_processing"]
            ),
            results=[_to_dataclass(SearchResultItem, r) for r in data["results"]],
            total_results=data["total_results"],
        )

    # ── Query refinement ─────────────────────────────────────────────────────

    def get_suggestions(self, prefix: str, limit: int = 5) -> list[str]:
        """Return autocomplete suggestions from search history."""
        qs = urllib.parse.urlencode({"q": prefix, "limit": limit})
        data = self._get(f"/api/v1/suggestions?{qs}", timeout=5)
        return data.get("suggestions", [])

    # ── Evaluation ───────────────────────────────────────────────────────────

    def get_evaluation(
        self,
        dataset: str,
        phase: str = "baseline",
        include_per_query: bool = False,
    ) -> EvaluationResponse:
        """
        Fetch aggregate (and optionally per-query) evaluation metrics for
        one phase (baseline or enhanced).

        Raises
        ------
        IRAPIError
            If no evaluation data exists yet for this dataset/phase.
        """
        qs = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "phase": phase,
                "include_per_query": str(include_per_query).lower(),
            }
        )
        data = self._get(f"/api/v1/evaluation?{qs}", timeout=30)
        return _build_eval_response(data)

    def get_evaluation_compare(
        self,
        dataset: str,
        include_per_query: bool = True,
    ) -> EvaluationCompareResponse:
        """
        Fetch both baseline and enhanced evaluation in one call.
        Either field may be None if that phase hasn't been run yet.
        """
        qs = urllib.parse.urlencode(
            {"dataset": dataset, "include_per_query": str(include_per_query).lower()}
        )
        data = self._get(f"/api/v1/evaluation/compare?{qs}", timeout=30)
        baseline = (
            _build_eval_response(data["baseline"]) if data.get("baseline") else None
        )
        enhanced = (
            _build_eval_response(data["enhanced"]) if data.get("enhanced") else None
        )
        return EvaluationCompareResponse(
            dataset=data["dataset"],
            baseline=baseline,
            enhanced=enhanced,
        )

    # ── Clustering ───────────────────────────────────────────────────────────

    def get_clusters(self, dataset: str) -> ClustersResponse:
        """
        Fetch all cluster summaries (top_terms, sizes, representative docs).
        Does NOT include 2D scatter points — call get_cluster_scatter() for those.
        """
        qs = urllib.parse.urlencode({"dataset": dataset})
        data = self._get(f"/api/v1/clusters?{qs}", timeout=30)
        return ClustersResponse(
            dataset=data["dataset"],
            n_clusters=data["n_clusters"],
            n_docs=data["n_docs"],
            embedding_source=data["embedding_source"],
            generated_at=data["generated_at"],
            clusters=[_build_cluster_info(c) for c in data["clusters"]],
        )

    def get_cluster_scatter(self, dataset: str) -> ClusterScatterResponse:
        """
        Fetch the stratified ~10K-point 2D scatter sample for the corpus
        visualisation chart (UMAP projection of SBERT embeddings).
        """
        qs = urllib.parse.urlencode({"dataset": dataset})
        data = self._get(f"/api/v1/clusters/scatter?{qs}", timeout=60)
        return ClusterScatterResponse(
            dataset=data["dataset"],
            n_clusters=data["n_clusters"],
            n_points=data["n_points"],
            clusters=[_build_cluster_info(c) for c in data["clusters"]],
            points=[
                ScatterPoint(
                    doc_id=p["doc_id"],
                    x=p["x"],
                    y=p["y"],
                    cluster_id=p["cluster_id"],
                )
                for p in data["points"]
            ],
        )
