import { useQuery } from "@tanstack/react-query";
import { evaluationService } from "@/api/endpoints/evaluation";

/**
 * `enabled` is driven by the caller (a "Run Evaluation" button), not by
 * mount — evaluation scans the whole query set against qrels per model,
 * so it should never fire implicitly.
 */
export function useEvaluationCompare(dataset: string, enabled: boolean) {
  return useQuery({
    queryKey: ["evaluation-compare", dataset],
    queryFn: ({ signal }) =>
      evaluationService.getCompare(dataset, true, signal),
    enabled,
    staleTime: 60_000,
    retry: false,
  });
}
