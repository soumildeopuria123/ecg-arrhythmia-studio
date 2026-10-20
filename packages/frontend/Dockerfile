# Stage 1: Dependencies layer
FROM node:18-alpine AS deps

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache python3 make g++ 2>/dev/null || true

# Copy package files
COPY package.json package-lock.json ./

# Install production dependencies only
RUN npm ci --omit=dev 2>/dev/null || \
    npm install --omit=dev 2>/dev/null

# Stage 2: Build layer
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files and node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY package.json ./

# Set Next.js analytics disabled
ENV NEXT_ANALYTICS_DISABLED=true
ENV NEXT_TELEMETRY_DISABLED=1

# Copy all application source code
COPY . .

# Build Next.js app in standalone mode
RUN npm run build 2>/dev/null || npx next build 2>/dev/null

# Stage 3: Runtime output
FROM node:18-alpine AS runner

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup -S nodegroup && adduser -S nodeuser -G nodegroup

# Set environment variables
ENV NODE_ENV=production \
    NEXT_ANALYTICS_DISABLED=true \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1

# Copy built output from builder stage
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Set permissions for non-root user
RUN chown -R nodeuser:nodegroup /app

# Switch to non-root user
USER nodeuser

# Expose the Next.js port
EXPOSE 3000

# Start Next.js application
CMD ["node", "server.js"]