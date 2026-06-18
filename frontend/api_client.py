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
        """Return all configured datasets with doc counts and model readiness."""
        data = self._get("/api/v1/datasets")
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
