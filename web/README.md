# IR Search Console — Frontend

React 19 + TypeScript frontend for the `Services/gateway` FastAPI backend (Requirement 9 UI). Replaces the Streamlit prototype in `frontend/`.

## Stack

- **React 19** + **TypeScript** (strict)
- **Vite 6** — build tooling
- **Tailwind CSS v4** — via `@tailwindcss/vite`, CSS-first config in `src/index.css`
- **TanStack Query v5** — all server state (health, datasets, options, search, suggestions)
- **Zustand v5** — client state only (search config, active view), persisted to `localStorage`
- **Motion** (`motion/react`) — used sparingly: page-load stagger, drawer slide, result-card stagger
- **Lucide React** — icons

## Getting started

```bash
cd web
npm install
cp .env.example .env   # set VITE_GATEWAY_URL if not running on 127.0.0.1:8000
npm run dev
```

Requires the gateway running (`uvicorn Services.gateway.main:app --host 0.0.0.0 --port 8000`).

```bash
npm run lint        # ESLint (flat config, typescript-eslint strict + stylistic)
npm run typecheck   # tsc --noEmit
npm run build        # production build
```

## Architecture

```
src/
├── api/
│   ├── client.ts            # fetch wrapper: timeout, abort, typed ApiError
│   ├── types.ts              # mirrors Services/gateway/schemas.py exactly
│   └── endpoints/
│       ├── system.ts         # /health, /api/v1/datasets, /api/v1/options
│       ├── search.ts         # POST /api/v1/search
│       └── suggestions.ts    # /api/v1/suggestions
├── hooks/                    # one hook per endpoint, wraps TanStack Query
├── stores/
│   ├── searchConfigStore.ts  # model/params config — persisted
│   └── uiStore.ts             # active view, panel open state — not persisted
├── constants/
│   └── retrievalModels.ts    # UI labels + derived conditions (isHybridMode, usesBm25Params)
├── components/
│   ├── ui/                   # Button, IconButton, Badge, Slider, Segmented, ...
│   ├── layout/                # Header, AppShell, BootGate
│   ├── search/                 # SearchBar, SettingsPanel, QueryProcessingPanel, MetaStrip
│   ├── results/                # ResultCard, ResultsList, RawJsonView, EmptyState
│   ├── evaluation/             # EvaluationView (placeholder — endpoint not built yet)
│   └── views/
│       └── SearchView.tsx     # composes the search page
├── App.tsx
└── main.tsx
```

### Service-layer pattern (mirrors backend SOA)

The frontend can't literally *be* SOA (it's one bundle in one tab), but `src/api/endpoints/` mirrors the backend's service boundaries: one module per responsibility (`system`, `search`, `suggestions`), each owning its own types and error handling, talking only to the Gateway — never assuming knowledge of individual backend services. This is a frontend service-layer pattern, called out honestly as that rather than claimed as literal SOA.

### Why `/api/v1/datasets` hard-gates the whole app

That endpoint loads the dataset and every representation (TF-IDF, BM25, SBERT, Word2Vec) into the backend's memory on first hit and can be slow. `BootGate` blocks all interaction — not just a disabled button — until it resolves, to avoid a search firing concurrently and doubling memory pressure on the same backend process. `useDatasets` sets `staleTime: Infinity` so it's fetched exactly once per app load.

### Raw data philosophy

Per the grading requirement (qrels cross-referencing by `doc_id`), nothing returned by `/api/v1/search` is hidden or cleaned for display:
- Every result shows its raw `doc_id` (copyable) and full-precision `score`.
- `text` is never silently truncated — long bodies get a "Show full text" expand.
- `query_processing` (original vs. processed query, spell correction, expansions, tokens) is shown in full.
- A "Raw response JSON" toggle on every search result dumps the literal API response.

### Evaluation

No evaluation endpoint exists yet — `components/evaluation/EvaluationView.tsx` is an intentional placeholder, styled to match the rest of the app, ready to wire up once that endpoint exists. It will be fetched on-demand only (explicit "Run Evaluation" button), never automatically, consistent with the dataset-loading caution above.

## Environment

| Variable | Default | Description |
|---|---|---|
| `VITE_GATEWAY_URL` | `http://127.0.0.1:8000` | Base URL of `Services/gateway` |
