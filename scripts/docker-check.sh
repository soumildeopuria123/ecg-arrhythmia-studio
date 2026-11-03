#!/bin/bash
# docker-check.sh - Verify Docker compose setup and endpoint health

set -e

echo "=== Building and Starting Docker Containers ==="
docker compose up --build -d

echo "=== Waiting for Backend Healthcheck ==="
# Wait for backend to pass healthcheck (max 60 seconds)
for i in $(seq 1 20); do
    if docker inspect --format='{{.State.Health.Status}}' ecg-backend 2>/dev/null | grep -q "healthy"; then
        echo "Backend is healthy!"
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "WARNING: Backend healthcheck timed out after 60 seconds"
        docker inspect --format='{{.State.Health.Status}}' ecg-backend
        exit 1
    fi
    echo "Waiting for backend healthcheck... (${i}/20)"
    sleep 3
done

echo "=== Running Test pings ==="

# Test Backend (FastAPI) endpoint
echo "Testing FastAPI /health endpoint..."
BACKEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/health)
if [ "$BACKEND_RESPONSE" = "200" ]; then
    echo "✅ FastAPI backend is responding (HTTP 200)"
else
    echo "❌ FastAPI backend health check failed (HTTP $BACKEND_RESPONSE)"
    exit 1
fi

# Test Frontend (Next.js) endpoint
echo "Testing Next.js UI..."
FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$FRONTEND_RESPONSE" = "200" ]; then
    echo "✅ Next.js frontend is responding (HTTP 200)"
else
    echo "❌ Next.js frontend health check failed (HTTP $FRONTEND_RESPONSE)"
    exit 1
fi

echo "=== All Checks Passed ==="
echo "FastAPI Backend: http://localhost:8000/api/v1"
echo "Next.js Frontend: http://localhost:3000"
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop: docker compose down"