#!/usr/bin/env bash
# scripts/download_models.sh
# Download all model files needed for the Orin Nano Super 8GB setup.
# Run from the project root: bash scripts/download_models.sh
#
# Requirements: wget, ~2 GB free disk space
# Internet connection required (models are cached after first download).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[download]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

# ── LLM: Qwen2.5-1.5B Instruct Q4_K_M (~1 GB) ───────────────────────────────
log "Creating models/ directory..."
mkdir -p models

GGUF_PATH="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

if [ -f "$GGUF_PATH" ]; then
    warn "GGUF model already exists at $GGUF_PATH — skipping."
else
    log "Downloading Qwen2.5-1.5B Q4_K_M GGUF (~1 GB)..."
    wget --progress=bar:force -O "$GGUF_PATH" "$GGUF_URL"
    log "GGUF model saved to $GGUF_PATH"
fi

# ── TTS: Piper en_US-lessac-medium ───────────────────────────────────────────
log "Creating assets/piper_voices/ directory..."
mkdir -p assets/piper_voices

PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
PIPER_ONNX="assets/piper_voices/en_US-lessac-medium.onnx"
PIPER_JSON="assets/piper_voices/en_US-lessac-medium.onnx.json"

if [ -f "$PIPER_ONNX" ]; then
    warn "Piper voice already exists — skipping."
else
    log "Downloading Piper en_US-lessac-medium voice (~60 MB)..."
    wget --progress=bar:force -O "$PIPER_ONNX"      "$PIPER_BASE/en_US-lessac-medium.onnx"
    wget --progress=bar:force -O "$PIPER_JSON"       "$PIPER_BASE/en_US-lessac-medium.onnx.json"
    log "Piper voice saved to $PIPER_ONNX"
fi

# ── Wake word: hey_jarvis ─────────────────────────────────────────────────────
log "Creating assets/wakeword_models/ directory..."
mkdir -p assets/wakeword_models

WW_PATH="assets/wakeword_models/hey_jarvis_v0.1.onnx"
WW_URL="https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx"

if [ -f "$WW_PATH" ]; then
    warn "Wake word model already exists — skipping."
else
    log "Downloading hey_jarvis wake word model (~400 KB)..."
    wget --progress=bar:force -O "$WW_PATH" "$WW_URL"
    log "Wake word model saved to $WW_PATH"
fi

# ── Apply Orin config ─────────────────────────────────────────────────────────
if [ -f "config-orin.yaml" ] && [ ! -f "config.yaml" ]; then
    log "Copying config-orin.yaml to config.yaml..."
    cp config-orin.yaml config.yaml
elif [ -f "config-orin.yaml" ]; then
    warn "config.yaml already exists — not overwriting. To switch to Orin config:"
    warn "  cp config-orin.yaml config.yaml"
fi

echo ""
log "=== Download complete ==="
log "  GGUF model : $GGUF_PATH"
log "  Piper voice: $PIPER_ONNX"
log "  Wake word  : $WW_PATH"
echo ""
log "Next: run 'python src/main.py' to start the assistant."
