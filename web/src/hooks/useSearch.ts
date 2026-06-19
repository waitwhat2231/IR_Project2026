import { useMutation } from "@tanstack/react-query";
import { searchService } from "@/api/endpoints/search";
import type { SearchRequestBody } from "@/api/types";

/**
 * Search is a mutation, not a query: it's an explicit, user-triggered
 * action with side effects worth tracking (server-side history), not a
 * cacheable GET keyed by params.
 */
export function useSearch() {
  return useMutation({
    mutationFn: (body: SearchRequestBody) => searchService.search(body),
  });
}
