/**
 * API Client - Uses Next.js API Proxy
 * 
 * All API calls go through Next.js server-side proxy:
 * Browser → /api/* → Next.js Server → localhost:8000 (Python)
 * 
 * Benefits:
 * - No public URL detection needed
 * - No CORS issues (same origin)
 * - Python backend stays internal (not exposed publicly)
 * - Works with ANY cloud provider automatically
 */

/**
 * Get API base URL - always use relative path (same origin)
 */
export function getApiUrl(): string {
  return ''  // Empty = same origin, requests go to /api/*
}

/** For backwards compatibility */
export const API_URL = ''

/** Initialize - no-op since we use same origin */
export async function initApiUrl(): Promise<string> {
  return ''
}

/**
 * Make API call through Next.js proxy.
 * Waits for URL detection to complete on first call.
 */
export async function apiFetch(endpoint: string, options?: RequestInit): Promise<Response> {
  const baseUrl = await initApiUrl()
  return fetch(`${baseUrl}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
}

/**
 * Helper to make API calls and parse JSON response.
 */
export async function apiJson<T = unknown>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await apiFetch(endpoint, options)
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}
