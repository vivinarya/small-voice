#!/bin/bash
# scripts/build_orin_frontend.sh
# Automates setting the Vite build environment variable VITE_WS_URL and building the frontend.
# Usage: ./scripts/build_orin_frontend.sh wss://your-websocket-tunnel.ngrok-free.app

set -e

if [ -z "$1" ]; then
  echo "Error: Please provide your ngrok WebSocket URL (e.g. wss://xxx.ngrok-free.app)"
  echo "Usage: $0 wss://<ngrok-websocket-subdomain>.ngrok-free.app"
  exit 1
fi

WS_URL="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building frontend with VITE_WS_URL=$WS_URL..."

cd "$PROJECT_ROOT/frontend"

# Export the variable so Vite picks it up during compilation
export VITE_WS_URL="$WS_URL"

# Build static files
npm install
npm run build

echo "Frontend built successfully in frontend/dist/ with WebSocket destination $WS_URL"
