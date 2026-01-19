/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  
  // API routes in app/api/[...path]/route.ts handle proxying to backend
  // No rewrites needed - they conflict with API routes in standalone builds
}

module.exports = nextConfig
