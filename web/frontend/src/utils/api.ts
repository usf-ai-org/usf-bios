/**
 * API URL Detection Utility
 * 
 * Universal backend URL detection that works with ANY cloud provider.
 * Detects the correct backend URL based on the current browser hostname.
 * 
 * IMPORTANT: localhost from browser != localhost in container
 * On cloud providers, the browser runs on user's machine, not in the container.
 */

// Cached API URL - detected once at startup, used for entire runtime
let _cachedApiUrl: string | null = null
let _detectionPromise: Promise<string> | null = null

/**
 * Get the primary backend URL based on current hostname.
 * This is the MOST LIKELY correct URL for the current environment.
 */
function getPrimaryBackendUrl(): string {
  if (typeof window === 'undefined') return ''
  
  const hostname = window.location.hostname
  const protocol = window.location.protocol
  
  // RunPod, Vast.ai, Lambda, etc. - port is in hostname
  // xxx-3000.proxy.runpod.net -> xxx-8000.proxy.runpod.net
  if (hostname.includes('-3000')) {
    return `${protocol}//${hostname.replace(/-3000/g, '-8000')}`
  }
  
  // Local development only
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000'
  }
  
  // Azure, AWS, GCP, etc. - same hostname, different port
  return `${protocol}//${hostname}:8000`
}

/**
 * Generate candidate API URLs to try, ordered by likelihood.
 * NEVER includes localhost for cloud environments (browser != container).
 */
function getCandidateUrls(): string[] {
  if (typeof window === 'undefined') return []
  
  const hostname = window.location.hostname
  const protocol = window.location.protocol
  const primary = getPrimaryBackendUrl()
  
  const candidates: string[] = [primary]
  
  // For cloud environments, also try same host with port 8000
  if (hostname.includes('-3000')) {
    // Already have the -8000 version, also try direct hostname:8000
    candidates.push(`${protocol}//${hostname}:8000`)
  }
  
  // For local dev, try both localhost and 127.0.0.1
  if (hostname === 'localhost') {
    candidates.push('http://127.0.0.1:8000')
  } else if (hostname === '127.0.0.1') {
    candidates.push('http://localhost:8000')
  }
  
  // Same host without port (reverse proxy routing)
  candidates.push('')
  
  // Remove duplicates while preserving order
  return Array.from(new Set(candidates))
}

/**
 * Probe a URL to check if it's the working backend.
 * Uses the /health endpoint which should always be available.
 */
async function probeUrl(baseUrl: string): Promise<boolean> {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 3000) // 3 second timeout
    
    const response = await fetch(`${baseUrl}/health`, {
      method: 'GET',
      signal: controller.signal,
    })
    
    clearTimeout(timeout)
    return response.ok
  } catch {
    return false
  }
}

/**
 * Detect the working API URL by probing candidates.
 * This runs ONCE at startup and caches the result.
 */
async function detectApiUrl(): Promise<string> {
  const candidates = getCandidateUrls()
  
  console.log('[API] Detecting backend URL from candidates:', candidates)
  
  // Try each candidate in order
  for (const url of candidates) {
    const isWorking = await probeUrl(url)
    if (isWorking) {
      console.log('[API] Backend detected at:', url)
      return url
    }
  }
  
  // If no probe succeeded, fall back to first candidate
  // (might be a timing issue where backend isn't ready yet)
  const fallback = candidates[0] || ''
  console.warn('[API] No backend detected, using fallback:', fallback)
  return fallback
}

/**
 * Initialize API URL detection. Call this once at app startup.
 * Returns a promise that resolves to the detected URL.
 */
export async function initApiUrl(): Promise<string> {
  if (_cachedApiUrl !== null) {
    return _cachedApiUrl
  }
  
  // Prevent multiple simultaneous detections
  if (_detectionPromise === null) {
    _detectionPromise = detectApiUrl().then(url => {
      _cachedApiUrl = url
      return url
    })
  }
  
  return _detectionPromise
}

/**
 * Get the cached API URL synchronously.
 * Returns primary backend URL if not yet detected.
 * Use initApiUrl() for async initialization.
 */
export function getApiUrl(): string {
  if (_cachedApiUrl !== null) {
    return _cachedApiUrl
  }
  
  // Start detection in background if not already started
  if (_detectionPromise === null && typeof window !== 'undefined') {
    initApiUrl()
  }
  
  // Return primary URL synchronously while detection runs
  // This is the best guess based on hostname - NEVER returns localhost on cloud
  return getPrimaryBackendUrl()
}

/**
 * Pre-computed API URL for use in components.
 * This starts detection and returns best guess synchronously.
 */
export const API_URL = typeof window !== 'undefined' ? getApiUrl() : ''

/**
 * Helper to make API calls with the correct base URL.
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
