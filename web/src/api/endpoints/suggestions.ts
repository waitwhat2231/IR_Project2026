import { apiRequest } from "@/api/client";
import type { SuggestionsResponse } from "@/api/types";

/** Service-layer module for query refinement (Requirement 5 — history-based autocomplete). */
export const suggestionsService = {
  getSuggestions: (
    prefix: string,
    limit: number,
    signal?: AbortSignal,
  ): Promise<SuggestionsResponse> =>
    apiRequest<SuggestionsResponse>("/api/v1/suggestions", {
      searchParams: { q: prefix, limit },
      signal,
      timeoutMs: 5_000,
    }),
};
