import { useQuery } from "@tanstack/react-query";
import { suggestionsService } from "@/api/endpoints/suggestions";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";

const SUGGESTION_LIMIT = 6;
const DEBOUNCE_MS = 220;
const MIN_PREFIX_LENGTH = 2;

/**
 * Requirement 5 — query suggestions drawn from server-side search history.
 * Debounced and length-gated so it doesn't fire on every keystroke.
 */
export function useSuggestions(prefix: string) {
  const debouncedPrefix = useDebouncedValue(prefix.trim(), DEBOUNCE_MS);
  const enabled = debouncedPrefix.length >= MIN_PREFIX_LENGTH;

  return useQuery({
    queryKey: ["suggestions", debouncedPrefix],
    queryFn: ({ signal }) =>
      suggestionsService.getSuggestions(
        debouncedPrefix,
        SUGGESTION_LIMIT,
        signal,
      ),
    enabled,
    staleTime: 10_000,
    retry: false,
  });
}
