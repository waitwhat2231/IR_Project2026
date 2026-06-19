import { apiRequest } from '@/api/client';
import type {
  DatasetsResponse,
  HealthResponse,
  RetrievalOptionsResponse,
} from '@/api/types';

/**
 * Service-layer module for the Gateway's discovery surface (/health,
 * /api/v1/datasets, /api/v1/options). Kept isolated from the search and
 * suggestions modules so each mirrors a distinct backend responsibility —
 * the frontend equivalent of the service boundaries enforced server-side.
 */
export const systemService = {
  getHealth: (signal?: AbortSignal): Promise<HealthResponse> =>
    apiRequest<HealthResponse>('/health', { signal }),

  listDatasets: (signal?: AbortSignal): Promise<DatasetsResponse> =>
    apiRequest<DatasetsResponse>('/api/v1/datasets', {
      signal,
      timeoutMs: 2147483647,
    }),

  getRetrievalOptions: (
    signal?: AbortSignal,
  ): Promise<RetrievalOptionsResponse> =>
    apiRequest<RetrievalOptionsResponse>('/api/v1/options', { signal }),
};
