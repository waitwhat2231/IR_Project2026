# IR 2026 Search Engine — Streamlit Frontend

A Streamlit UI for exploring and evaluating multiple information-retrieval
models side by side, built as part of the **IR_Project2026** system. This is
the lightweight Python frontend; a separate React/TypeScript frontend talks
to the same API gateway for a more polished, production-style UI.

## What it does

- **Search** across multiple retrieval models — sparse (BM25, TF-IDF), dense
  (SBERT, Word2Vec), and hybrid fusion (parallel or serial cascade) — with
  tunable parameters (BM25 `k1`/`b`, fusion `alpha`, cascade `top_n`).
- **Evaluation** dashboard comparing baseline vs. enhanced retrieval phases,
  with per-model and per-query metric breakdowns.
- **Clusters** view: corpus-wide 2D scatter plot (SBERT embeddings →
  MiniBatchKMeans → UMAP projection) plus per-cluster summary cards with top
  terms and representative documents.

This app is a thin presentation layer — all retrieval, clustering, and
evaluation logic lives behind a FastAPI gateway. The frontend only talks to
it through `api_client.py`; no raw JSON or HTTP details leak into the UI
code.

## Running it

```bash
streamlit run frontend/app.py
```

Requirements: a running instance of the API gateway (default
`http://127.0.0.1:8000`), plus:

```bash
pip install streamlit requests plotly
```

`plotly` is optional — charts gracefully degrade to a "Plotly not
installed" notice if it's missing, everything else still works.

The gateway URL can be changed at runtime from the sidebar; no restart
needed.

## Project structure

```
frontend/
├── app.py              # Entry point: bootstrap gate, page header,
│                        # tab navigation, search tab
├── bootstrap.py          # Full-screen "loading models" overlay shown once
│                          # per session, plus the connection-error recovery
│                          # screen
├── sidebar.py              # API connection + search configuration controls
├── api_client.py            # Typed HTTP client — all gateway communication
│                             # and JSON parsing happens here
└── ui/
    ├── constants.py            # API defaults, retrieval-mode labels,
    │                            # color palettes, metric-key labels
    ├── icons.py                  # Hand-built inline SVG icon system
    │                              # (no emoji, no external icon fonts)
    ├── styles.py                   # Global CSS
    ├── helpers.py                    # URL normalization, client factory,
    │                                  # text/chip helpers
    ├── components.py                   # Result card + query-insights
    │                                    # renderers
    ├── charts.py                         # All Plotly chart builders —
    │                                      # the only module that imports
    │                                      # plotly directly
    ├── evaluation.py                       # Evaluation tab
    └── clusters.py                           # Clusters tab
```

Originally a single ~2,000-line `app.py`; split into the structure above
for maintainability. `ui/charts.py` is intentionally the sole owner of the
`plotly` import — every chart elsewhere is built by calling a function here,
not by touching `go` directly.

## Design notes worth knowing before you touch this code

- **The very first request the app makes (`list_datasets()` on startup)
  is deliberately retry-free and timeout-free.** On the backend, this
  endpoint loads every retrieval model for every dataset into RAM — slow
  and non-idempotent. Every other endpoint goes through the normal
  retry-enabled session in `api_client.py`; this one calls `_get_once()`
  instead, which bypasses that session entirely. **Do not** route it
  through `_get()` or add a `timeout=` value back — a premature
  client-side timeout there previously caused the same expensive
  model-loading work to silently re-trigger 2–3× in parallel via
  urllib3's automatic retry-on-timeout behavior, which was destroying
  available RAM on startup. See the docstring on `_get_once()` for the
  full explanation.
- **Plotly is an optional import**, guarded with a `TYPE_CHECKING` block
  so static type checkers (Pyright/Pylance) see it as always available
  for type-hinting purposes, while the actual runtime import stays
  inside a `try`/`except ImportError`. Every call site checks `_PLOTLY`
  before touching a chart-builder function.
- **The bootstrap screen blocks the rest of the UI** until `datasets`
  is loaded into `st.session_state` exactly once per session — sidebar,
  tabs, etc. never call `list_datasets()` again after that.
- Icons are hand-rolled inline SVG (`ui/icons.py`) for anything rendered
  via `unsafe_allow_html=True` (chips, headers). Native Streamlit widgets
  (buttons, tabs, alerts) use Google's `:material/...` Symbol icons
  instead, since Streamlit doesn't allow raw HTML in those labels.
