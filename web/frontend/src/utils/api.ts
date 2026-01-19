/**
 * API URL Configuration
 * 
 * 1. Frontend uses hostname-based fallback for FIRST request
 * 2. First request to backend triggers URL detection (from Host header)
 * 3. Backend stores detected URL in memory, serves via /config endpoint
 * 4. Frontend caches URL permanently after first successful fetch
 */

// PERMANENT cache - set once, never changes
let _backendUrl: string | null = null
let _initPromise: Promise<string> | null = null

/**
 * Fallback URL based on hostname - used for FIRST request before config is loaded
 */
function getFallbackUrl(): string {
  if (typeof window === 'undefined') return ''
  const h = window.location.hostname
  const p = window.location.protocol
  // Cloud providers with port in hostname (RunPod, Vast.ai, etc.)
  if (h.includes('-3000')) return `${p}//${h.replace(/-3000/g, '-8000')}`
  // Local development
  if (h === 'localhost' || h === '127.0.0.1') return 'http://localhost:8000'
  // Default: same host, port 8000
  return `${p}//${h}:8000`
}

/**
 * Initialize: Fetch backend URL from /config endpoint ONCE, cache forever.
 */
export async function initApiUrl(): Promise<string> {
  if (_backendUrl !== null) return _backendUrl
  if (_initPromise !== null) return _initPromise
  
  _initPromise = (async (): Promise<string> => {
    // Use fallback URL to make first request to backend
    const fallback = getFallbackUrl()
    try {
      console.log('[API] Fetching config from backend (ONE TIME ONLY)...')
      const res = await fetch(`${fallback}/config`)
      if (!res.ok) throw new Error('Config endpoint failed')
      const cfg = await res.json()
      // Backend returns detected URL, or we keep using fallback
      _backendUrl = cfg.backendUrl || fallback
      console.log('[API] Backend URL (cached permanently):', _backendUrl)
    } catch {
      _backendUrl = fallback
      console.log('[API] Using fallback (cached permanently):', _backendUrl)
    }
    return _backendUrl as string
  })()
  
  return _initPromise
}

/**
 * Get cached backend URL. Returns fallback if not yet initialized.
 */
export function getApiUrl(): string {
  if (_backendUrl !== null) return _backendUrl
  if (typeof window !== 'undefined') initApiUrl()
  return getFallbackUrl()
}

/** Cached API URL for components */
export const API_URL = typeof window !== 'undefined' ? getApiUrl() : ''

/**
 * Make API call - waits for URL init on first call.
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
