import { apiRequest } from "@/api/client";
import type { ClusterScatterResponse, ClustersResponse } from "@/api/types";

/**
 * Service-layer module for the Clustering service, surfaced through the
 * Gateway. The ClusterManager lazy-loads on first request per dataset and
 * is cached server-side afterwards, so only the very first call pays the
 * load cost — still kept on-demand (not auto-fetched) to stay consistent
 * with how every other heavier read is gated in this app.
 */
export const clustersService = {
  getClusters: (dataset: string, signal?: AbortSignal): Promise<ClustersResponse> =>
    apiRequest<ClustersResponse>("/api/v1/clusters", {
      searchParams: { dataset },
      signal,
      timeoutMs: 30_000,
    }),

  getScatter: (
    dataset: string,
    signal?: AbortSignal,
  ): Promise<ClusterScatterResponse> =>
    apiRequest<ClusterScatterResponse>("/api/v1/clusters/scatter", {
      searchParams: { dataset },
      signal,
      timeoutMs: 30_000, // ~10K-point payload
    }),
};
