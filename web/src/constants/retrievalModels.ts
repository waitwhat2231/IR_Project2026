import type { RetrievalMode } from "@/api/types";

interface RetrievalModeMeta {
  label: string;
  description: string;
}

export const RETRIEVAL_MODE_META: Record<RetrievalMode, RetrievalModeMeta> = {
  bm25: {
    label: "BM25",
    description: "Probabilistic ranking over term statistics",
  },
  tfidf: {
    label: "TF-IDF",
    description: "Classic vector-space term-weighting model",
  },
  sbert: {
    label: "SBERT",
    description: "Dense sentence-transformer embeddings",
  },
  word2vec: {
    label: "Word2Vec",
    description: "Dense averaged word-vector embeddings",
  },
  hybrid_parallel: {
    label: "Hybrid — Parallel",
    description: "Sparse + dense scores fused together",
  },
  hybrid_serial: {
    label: "Hybrid — Serial",
    description: "Sparse retrieves candidates, dense reranks them",
  },
};

export const RETRIEVAL_MODES: RetrievalMode[] = [
  "bm25",
  "tfidf",
  "sbert",
  "word2vec",
  "hybrid_parallel",
  "hybrid_serial",
];

export function isHybridMode(mode: RetrievalMode): boolean {
  return mode === "hybrid_parallel" || mode === "hybrid_serial";
}

/** Whether BM25's k1/b sliders are relevant for the current configuration. */
export function usesBm25Params(
  mode: RetrievalMode,
  sparseMethod: "bm25" | "tfidf",
): boolean {
  if (mode === "bm25") return true;
  return isHybridMode(mode) && sparseMethod === "bm25";
}
