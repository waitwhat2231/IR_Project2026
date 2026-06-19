import { useQuery } from '@tanstack/react-query';
import { clustersService } from '@/api/endpoints/clusters';

export function useClusterScatter(dataset: string, enabled: boolean) {
  return useQuery({
    queryKey: ['cluster-scatter', dataset],
    queryFn: ({ signal }) => clustersService.getScatter(dataset, signal),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}
