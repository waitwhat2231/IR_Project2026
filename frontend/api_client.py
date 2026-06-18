"""
HTTP client for the IR 2026 API Gateway.

All raw JSON is parsed here and returned as typed dataclasses.
No JSON ever leaks into the UI layer.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
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


# ─────────────────────────────────────── Exception ─────────────────────────

class IRAPIError(Exception):
    """
    Raised for any failure communicating with the API Gateway.

    Covers both HTTP errors (non-2xx responses) and connection errors
    so callers never need to import ``requests`` directly.
    """


# ─────────────────────────────────────── Client ────────────────────────────

def _build_session() -> requests.Session:
    """Return a Session with a conservative retry strategy."""
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3, status_forcelist=[502, 503, 504])
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
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = _build_session()

    # ── internal ────────────────────────────────────────────────────────────

    def _get(self, path: str, *, timeout: int = 10) -> Any:
        try:
            resp = self._session.get(f"{self.base_url}{path}", timeout=timeout)
            resp.raise_for_status()
            # 304 Not Modified → body is empty; treat as a successful cached response
            # by re-fetching without the cache headers.
            if resp.status_code == 304 or not resp.content:
                self._session.headers.pop("If-None-Match", None)
                self._session.headers.pop("If-Modified-Since", None)
                resp = self._session.get(f"{self.base_url}{path}", timeout=timeout)
                resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise IRAPIError(f"Cannot reach {self.base_url} — is the gateway running?") from exc
        except requests.exceptions.HTTPError as exc:
            raise IRAPIError(f"[{exc.response.status_code}] {exc.response.text}") from exc
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
            raise IRAPIError(f"Cannot reach {self.base_url} — is the gateway running?") from exc
        except requests.exceptions.RequestException as exc:
            raise IRAPIError(str(exc)) from exc

    # ── Health / Discovery ──────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Return gateway status and MongoDB connectivity."""
        return self._get("/health", timeout=5)

    def list_datasets(self) -> list[DatasetInfo]:
        """Return all configured datasets with doc counts and model readiness."""
        data = self._get("/api/v1/datasets")
        return [DatasetInfo(**d) for d in data["datasets"]]

    def get_options(self) -> RetrievalOptions:
        """Return supported modes and default parameter values."""
        data = self._get("/api/v1/options", timeout=5)
        return RetrievalOptions(**data)

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
            query_processing=QueryProcessingInfo(**data["query_processing"]),
            results=[SearchResultItem(**r) for r in data["results"]],
            total_results=data["total_results"],
        )

    # ── Query refinement ─────────────────────────────────────────────────────

    def get_suggestions(self, prefix: str, limit: int = 5) -> list[str]:
        """Return autocomplete suggestions from search history."""
        qs = urllib.parse.urlencode({"q": prefix, "limit": limit})
        data = self._get(f"/api/v1/suggestions?{qs}", timeout=5)
        return data.get("suggestions", [])
