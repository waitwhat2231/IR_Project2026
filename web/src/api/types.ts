/**
 * Types mirror Services/gateway/schemas.py exactly. Keep them in sync with
 * the backend — these are not guessed shapes, they're transcribed from the
 * Pydantic models that FastAPI enforces as response_model.
 */

export type ExecutionMode = "basic" | "enhanced";

export type RetrievalMode =
  | "bm25"
  | "tfidf"
  | "sbert"
  | "word2vec"
  | "hybrid_parallel"
  | "hybrid_serial";

export type SparseMethod = "bm25" | "tfidf";
export type DenseMethod = "sbert" | "word2vec";

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------
export interface HealthResponse {
  status: string;
  mongodb: boolean;
  datasets_configured: string[];
}

// ---------------------------------------------------------------------------
// GET /api/v1/datasets
// ---------------------------------------------------------------------------
export interface DatasetInfo {
  name: string;
  ir_dataset_key: string;
  document_count: number | null;
  models_ready: boolean;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
}

// ---------------------------------------------------------------------------
// GET /api/v1/options
// ---------------------------------------------------------------------------
export interface RetrievalOptionsDefaults {
  execution_mode: ExecutionMode;
  retrieval_mode: RetrievalMode;
  sparse_method: SparseMethod;
  dense_method: DenseMethod;
  bm25_k1: number;
  bm25_b: number;
  alpha: number;
  cascade_top_n: number;
  top_k: number;
}

export interface RetrievalOptionsResponse {
  execution_modes: ExecutionMode[];
  retrieval_modes: RetrievalMode[];
  sparse_methods: SparseMethod[];
  dense_methods: DenseMethod[];
  defaults: RetrievalOptionsDefaults;
}

// ---------------------------------------------------------------------------
// POST /api/v1/search
// ---------------------------------------------------------------------------
export interface SearchRequestBody {
  dataset: string;
  query: string;
  execution_mode: ExecutionMode;
  retrieval_mode: RetrievalMode;
  sparse_method: SparseMethod;
  dense_method: DenseMethod;
  bm25_k1: number;
  bm25_b: number;
  alpha: number;
  cascade_top_n: number;
  top_k: number;
  include_snippet_chars: number;
}

export interface QueryProcessingInfo {
  original_query: string;
  processed_query: string;
  spell_corrected: boolean;
  expanded_terms: string[];
  query_tokens: string[];
}

export interface SearchResultItem {
  rank: number;
  doc_id: string;
  score: number;
  title: string | null;
  text: string | null;
}

export interface SearchResponse {
  dataset: string;
  execution_mode: ExecutionMode;
  retrieval_mode: RetrievalMode;
  query_processing: QueryProcessingInfo;
  results: SearchResultItem[];
  total_results: number;
}

// ---------------------------------------------------------------------------
// GET /api/v1/suggestions
// ---------------------------------------------------------------------------
export interface SuggestionsResponse {
  query_prefix: string;
  suggestions: string[];
}
