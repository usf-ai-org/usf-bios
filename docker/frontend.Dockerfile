# USF BIOS Frontend - Docker Image
FROM node:18-alpine

LABEL maintainer="US Inc <support@us.inc>"
LABEL description="USF BIOS Frontend UI"
LABEL version="1.0.10"

# Set working directory
WORKDIR /app

# Copy package files
COPY web/frontend/package*.json /app/

# Install dependencies
RUN npm ci --only=production 2>/dev/null || npm install

# Copy frontend code
COPY web/frontend /app

# Build the Next.js application
RUN npm run build

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Start the application
CMD ["npm", "start"]
