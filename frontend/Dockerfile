# ==============================================================================
# Multi-stage Production Dockerfile for Frontend (TanStack Start / Nitro)
# ==============================================================================

# Stage 1: Build stage
FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies first for layer caching
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and build optimized Nitro bundles
COPY frontend/ ./
ENV NODE_ENV=production
ENV FASTAPI_URL=http://backend:8000
RUN npm run build

# Stage 2: Minimal Production Runtime
FROM node:22-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000 \
    HOST=0.0.0.0 \
    FASTAPI_URL=http://backend:8000

# Install curl for container health check
RUN apk add --no-cache curl

# Copy only the compiled server and client static output
COPY --from=builder /app/.output ./.output

EXPOSE 3000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# Start Nitro production server
CMD ["node", ".output/server/index.mjs"]
