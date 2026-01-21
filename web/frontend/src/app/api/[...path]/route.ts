/**
 * Next.js API Proxy Route
 * 
 * All API calls from the browser go through this route.
 * This runs on the Next.js SERVER (inside the container),
 * so it can access Python backend via localhost:8000.
 * 
 * Browser → Next.js (port 3000) → Python (localhost:8000/api/*)
 * 
 * Benefits:
 * - No public URL detection needed
 * - No CORS issues (same origin)
 * - Python backend stays internal
 */

const BACKEND_URL = 'http://localhost:8000'

async function proxyRequest(request: Request, path: string): Promise<Response> {
  // Extract query string from original request and forward it
  const requestUrl = new URL(request.url)
  const queryString = requestUrl.search // includes the '?' if present
  const url = `${BACKEND_URL}/api/${path}${queryString}`
  
  // Log the request for debugging
  console.log(`[API Proxy] ${request.method} ${path}${queryString}`)
  
  // Forward headers (except host)
  const headers = new Headers()
  request.headers.forEach((value, key) => {
    if (key.toLowerCase() !== 'host') {
      headers.set(key, value)
    }
  })
  
  // Build fetch options
  const fetchOptions: RequestInit = {
    method: request.method,
    headers,
    // Disable caching for real-time data
    cache: 'no-store',
  }
  
  // Forward body for POST/PUT/PATCH
  if (['POST', 'PUT', 'PATCH'].includes(request.method)) {
    fetchOptions.body = await request.text()
  }
  
  try {
    const response = await fetch(url, fetchOptions)
    
    // Forward response headers
    const responseHeaders = new Headers()
    response.headers.forEach((value, key) => {
      // Skip headers that Next.js handles
      if (!['content-encoding', 'transfer-encoding'].includes(key.toLowerCase())) {
        responseHeaders.set(key, value)
      }
    })
    
    // Add cache control headers to prevent caching
    responseHeaders.set('Cache-Control', 'no-store, no-cache, must-revalidate')
    
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    })
  } catch (error) {
    console.error('[API Proxy] Error:', error)
    return Response.json(
      { error: 'Backend unavailable', details: 'Unable to reach backend service' },
      { status: 502 }
    )
  }
}

export async function GET(
  request: Request,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params
  return proxyRequest(request, path.join('/'))
}

export async function POST(
  request: Request,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params
  return proxyRequest(request, path.join('/'))
}

export async function PUT(
  request: Request,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params
  return proxyRequest(request, path.join('/'))
}

export async function DELETE(
  request: Request,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params
  return proxyRequest(request, path.join('/'))
}

export async function PATCH(
  request: Request,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params
  return proxyRequest(request, path.join('/'))
}
