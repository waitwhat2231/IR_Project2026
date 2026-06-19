import { apiRequest } from "@/api/client";
import type {
  EvaluationCompareResponse,
  EvaluationPhase,
  EvaluationResponse,
} from "@/api/types";

/**
 * Service-layer module for the Ranking & Evaluation service, surfaced
 * through the Gateway. These reads are cheap (pre-computed JSON on disk —
 * no model loading), so unlike /api/v1/datasets they don't need a boot gate.
 * Still fetched on demand only, never automatically, per the same caution.
 */
export const evaluationService = {
  getEvaluation: (
    dataset: string,
    phase: EvaluationPhase,
    includePerQuery: boolean,
    signal?: AbortSignal,
  ): Promise<EvaluationResponse> =>
    apiRequest<EvaluationResponse>("/api/v1/evaluation", {
      searchParams: {
        dataset,
        phase,
        include_per_query: String(includePerQuery),
      },
      signal,
    }),

  getCompare: (
    dataset: string,
    includePerQuery: boolean,
    signal?: AbortSignal,
  ): Promise<EvaluationCompareResponse> =>
    apiRequest<EvaluationCompareResponse>("/api/v1/evaluation/compare", {
      searchParams: {
        dataset,
        include_per_query: String(includePerQuery),
      },
      signal,
    }),
};
