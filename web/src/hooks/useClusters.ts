import { useQuery } from '@tanstack/react-query';
import { clustersService } from '@/api/endpoints/clusters';

export function useClusters(dataset: string, enabled: boolean) {
  return useQuery({
    queryKey: ['clusters', dataset],
    queryFn: ({ signal }) => clustersService.getClusters(dataset, signal),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}
