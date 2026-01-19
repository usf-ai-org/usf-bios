/**
 * API URL Detection Utility
 * 
 * Automatically detects the correct backend API URL based on the current hostname.
 * Works with all major cloud providers and GPU platforms.
 */

/**
 * Detects the correct API URL at runtime based on the current browser hostname.
 * 
 * Supported platforms:
 * - RunPod: xxx-3000.proxy.runpod.net -> xxx-8000.proxy.runpod.net
 * - Vast.ai: xxx-3000.direct.vast.ai -> xxx-8000.direct.vast.ai
 * - Lambda Labs: Similar port-based patterns
 * - Azure Container Apps: Uses same hostname, different port or path
 * - AWS (ECS/EKS): Uses same hostname pattern
 * - GCP (Cloud Run/GKE): Uses same hostname pattern
 * - Local development: localhost:3000 -> localhost:8000
 */
export function getApiUrl(): string {
  if (typeof window === 'undefined') return ''
  
  const hostname = window.location.hostname
  const protocol = window.location.protocol
  const port = window.location.port
  
  // RunPod: xxx-3000.proxy.runpod.net -> xxx-8000.proxy.runpod.net
  if (hostname.includes('.proxy.runpod.net')) {
    return `${protocol}//${hostname.replace('-3000.', '-8000.')}`
  }
  
  // Vast.ai: xxx-3000.direct.vast.ai -> xxx-8000.direct.vast.ai
  if (hostname.includes('.direct.vast.ai') || hostname.includes('.vast.ai')) {
    return `${protocol}//${hostname.replace('-3000.', '-8000.')}`
  }
  
  // Lambda Labs: xxx-3000.cloud.lambdalabs.com -> xxx-8000.cloud.lambdalabs.com
  if (hostname.includes('.lambdalabs.com') || hostname.includes('.lambda.cloud')) {
    return `${protocol}//${hostname.replace('-3000.', '-8000.')}`
  }
  
  // Paperspace: Similar pattern
  if (hostname.includes('.paperspace.com') || hostname.includes('.gradient.run')) {
    return `${protocol}//${hostname.replace('-3000.', '-8000.')}`
  }
  
  // CoreWeave
  if (hostname.includes('.coreweave.com') || hostname.includes('.ord1.coreweave.cloud')) {
    return `${protocol}//${hostname.replace('-3000.', '-8000.')}`
  }
  
  // Generic port-based pattern (catches most GPU cloud providers)
  // Pattern: xxx-3000.xxx -> xxx-8000.xxx
  if (hostname.includes('-3000.') || hostname.includes('-3000-')) {
    const newHostname = hostname.replace('-3000.', '-8000.').replace('-3000-', '-8000-')
    return `${protocol}//${newHostname}`
  }
  
  // Azure Container Apps: Uses environment variable or same host
  // Pattern: xxx.azurecontainerapps.io (frontend and backend on same host, different paths)
  if (hostname.includes('.azurecontainerapps.io') || hostname.includes('.azure')) {
    // Azure typically uses same hostname with /api prefix routed to backend
    // If port 3000 is in URL, replace with 8000
    if (port === '3000') {
      return `${protocol}//${hostname}:8000`
    }
    // Otherwise assume same host (API gateway routing)
    return ''
  }
  
  // AWS (ECS, EKS, App Runner): Similar patterns
  if (hostname.includes('.amazonaws.com') || hostname.includes('.aws')) {
    if (port === '3000') {
      return `${protocol}//${hostname}:8000`
    }
    return ''
  }
  
  // GCP (Cloud Run, GKE): Similar patterns
  if (hostname.includes('.run.app') || hostname.includes('.googleusercontent.com') || hostname.includes('.gcp')) {
    if (port === '3000') {
      return `${protocol}//${hostname}:8000`
    }
    return ''
  }
  
  // DigitalOcean App Platform
  if (hostname.includes('.ondigitalocean.app')) {
    if (port === '3000') {
      return `${protocol}//${hostname}:8000`
    }
    return ''
  }
  
  // Render
  if (hostname.includes('.onrender.com')) {
    if (port === '3000') {
      return `${protocol}//${hostname}:8000`
    }
    return ''
  }
  
  // Railway
  if (hostname.includes('.railway.app')) {
    if (port === '3000') {
      return `${protocol}//${hostname}:8000`
    }
    return ''
  }
  
  // Local development
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000'
  }
  
  // Docker Compose / Kubernetes internal: same container, different port
  // If we're on port 3000, backend is on 8000
  if (port === '3000') {
    return `${protocol}//${hostname}:8000`
  }
  
  // Default: assume same host with port 8000
  // This covers most deployment scenarios where frontend and backend
  // are on the same machine/container
  return `${protocol}//${hostname}:8000`
}

/**
 * Pre-computed API URL for use in components.
 * This is evaluated once when the module loads.
 */
export const API_URL = typeof window !== 'undefined' ? getApiUrl() : ''

/**
 * Helper to make API calls with the correct base URL.
 * 
 * @param endpoint - The API endpoint (e.g., '/api/system/status')
 * @param options - Fetch options
 * @returns Fetch promise
 */
export async function apiFetch(endpoint: string, options?: RequestInit): Promise<Response> {
  const url = `${getApiUrl()}${endpoint}`
  return fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
}

/**
 * Helper to make API calls and parse JSON response.
 * 
 * @param endpoint - The API endpoint (e.g., '/api/system/status')
 * @param options - Fetch options
 * @returns Parsed JSON response
 */
export async function apiJson<T = unknown>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await apiFetch(endpoint, options)
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}
