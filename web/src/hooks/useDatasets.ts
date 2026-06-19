import { useQuery } from "@tanstack/react-query";
import { systemService } from "@/api/endpoints/system";

/**
 * This endpoint loads the dataset and every representation (TF-IDF, BM25,
 * SBERT, Word2Vec) into memory on the backend the first time it's hit, so
 * it can be slow. It is fetched exactly once per app load (staleTime:
 * Infinity, no background refetch) and the UI hard-gates search until it
 * resolves — see <BootGate />.
 */
export function useDatasets() {
  return useQuery({
    queryKey: ["datasets"],
    queryFn: ({ signal }) => systemService.listDatasets(signal),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1,
  });
}
