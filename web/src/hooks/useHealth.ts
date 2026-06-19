import { useQuery } from '@tanstack/react-query';
import { systemService } from '@/api/endpoints/system';

/**
 * Health is intentionally decoupled from the dataset-loading gate: it's a
 * cheap, fast endpoint that never touches the heavy in-memory models, so it
 * polls quietly in the background to drive a status indicator only.
 */
export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => systemService.getHealth(signal),
    staleTime: 15_000,
    refetchInterval: 300_000,
    retry: 1,
  });
}
