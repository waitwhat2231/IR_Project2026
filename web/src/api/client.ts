const GATEWAY_URL: string =
  import.meta.env.VITE_GATEWAY_URL ?? 'http://127.0.0.1:8000';

const DEFAULT_TIMEOUT_MS = 20_000;

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

interface RequestOptions {
  method?: 'GET' | 'POST';
  body?: unknown;
  searchParams?: Record<string, string | number | undefined>;
  timeoutMs?: number;
  signal?: AbortSignal;
}

function buildUrl(
  path: string,
  searchParams?: Record<string, string | number | undefined>,
): string {
  const url = new URL(path, GATEWAY_URL);
  if (searchParams) {
    for (const [key, value] of Object.entries(searchParams)) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

/**
 * Thin, dependency-free fetch wrapper shared by every per-service API
 * module. Native fetch is intentional here — for a request surface this
 * small, a wrapper library like axios adds bundle weight with no real
 * benefit over fetch + AbortController.
 */
export async function apiRequest<TResponse>(
  path: string,
  options: RequestOptions = {},
): Promise<TResponse> {
  const {
    method = 'GET',
    body,
    searchParams,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    signal,
  } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  // Allow an external signal (e.g. from TanStack Query) to also abort.
  signal?.addEventListener('abort', () => {
    controller.abort();
  });

  try {
    const response = await fetch(buildUrl(path, searchParams), {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail: unknown;
      try {
        detail = (await response.json()) as unknown;
      } catch {
        detail = undefined;
      }
      const message =
        typeof detail === 'object' &&
        detail !== null &&
        'detail' in detail &&
        typeof (detail as { detail?: unknown }).detail === 'string'
          ? (detail as { detail: string }).detail
          : `Request to ${path} failed with status ${String(response.status)}`;
      throw new ApiError(message, response.status, detail);
    }

    return (await response.json()) as TResponse;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(
        `Request to ${path} timed out or was cancelled`,
        0,
        error,
      );
    }
    throw new ApiError(
      error instanceof Error ? error.message : 'Network error',
      0,
      error,
    );
  } finally {
    clearTimeout(timeoutId);
  }
}
