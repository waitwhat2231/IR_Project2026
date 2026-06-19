import { apiRequest } from "@/api/client";
import type { SearchRequestBody, SearchResponse } from "@/api/types";

/** Service-layer module for the retrieval endpoint. */
export const searchService = {
  search: (
    body: SearchRequestBody,
    signal?: AbortSignal,
  ): Promise<SearchResponse> =>
    apiRequest<SearchResponse>("/api/v1/search", {
      method: "POST",
      body,
      signal,
      timeoutMs: 60_000, // cold model passes can be slow
    }),
};
