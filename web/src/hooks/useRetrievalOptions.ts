import { useQuery } from "@tanstack/react-query";
import { systemService } from "@/api/endpoints/system";

export function useRetrievalOptions() {
  return useQuery({
    queryKey: ["retrieval-options"],
    queryFn: ({ signal }) => systemService.getRetrievalOptions(signal),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    retry: 1,
  });
}
