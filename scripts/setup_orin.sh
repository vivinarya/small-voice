#!/usr/bin/env bash
# scripts/setup_orin.sh
#
# Full automated setup for Jetson Orin Nano Super 8GB (JetPack 6.x).
# Run once after cloning the project to the Orin:
#
#   cd ~/small-voice-main
#   bash scripts/setup_orin.sh
#
# What this script does:
#   1. Installs system apt packages
#   2. Creates the Python virtual environment
#   3. Upgrades pip
#   4. Detects the JetPack CUDA version and installs the matching PyTorch Jetson wheel
#   5. Installs the matching onnxruntime-gpu Jetson wheel
#   6. Builds llama-cpp-python from source with CUDA support
#   7. Installs the remaining Python packages (requirements-orin.txt)
#   8. Downloads all model files
#   9. Copies config-orin.yaml to config.yaml
#  10. Installs and enables the jarvis systemd service
#
# Total estimated time: 20–30 minutes (mostly compilation of llama-cpp-python)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

VENV="$HOME/jarvis-env"
PYTHON="$VENV/bin/python3"
PIP="$VENV/bin/pip"

# ── Step 1: System packages ──────────────────────────────────────────────────
log "Step 1/10 — Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential cmake git wget curl ninja-build \
    python3-pip python3-dev python3-venv \
    portaudio19-dev libsndfile1 ffmpeg \
    libgomp1 espeak-ng nodejs npm

# ── Step 2: Python venv ──────────────────────────────────────────────────────
log "Step 2/10 — Creating Python virtual environment at $VENV..."
python3 -m venv "$VENV"
# Add activation to .bashrc if not already there
if ! grep -q "jarvis-env" ~/.bashrc 2>/dev/null; then
    echo "source $VENV/bin/activate" >> ~/.bashrc
fi
source "$VENV/bin/activate"

# ── Step 3: Upgrade pip ──────────────────────────────────────────────────────
log "Step 3/10 — Upgrading pip..."
"$PIP" install --upgrade pip setuptools wheel

# ── Step 4: PyTorch Jetson wheel ─────────────────────────────────────────────
log "Step 4/10 — Installing PyTorch (NVIDIA Jetson wheel for JetPack 6)..."

# Detect JetPack version from dpkg
JP_VER=$(dpkg -l | grep -i "jetpack" | awk '{print $3}' | head -1 | cut -d. -f1-2 || echo "6.1")
log "  Detected JetPack version: $JP_VER"

# JetPack 6.x PyTorch wheel (CUDA 12.6, Python 3.10, aarch64)
TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    warn "  PyTorch with CUDA already installed — skipping."
else
    log "  Downloading and installing PyTorch Jetson wheel (~1.5 GB, takes a few minutes)..."
    "$PIP" install --no-cache-dir "$TORCH_WHEEL"
fi

"$PYTHON" -c "import torch; print('  PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ── Step 5: ONNX Runtime GPU ─────────────────────────────────────────────────
log "Step 5/10 — Installing onnxruntime-gpu (NVIDIA Jetson wheel)..."

ORT_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/onnxruntime/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl"

if "$PYTHON" -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" 2>/dev/null; then
    warn "  onnxruntime-gpu already installed — skipping."
else
    "$PIP" uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    "$PIP" install --no-cache-dir "$ORT_WHEEL"
fi

"$PYTHON" -c "import onnxruntime as ort; print('  ORT providers:', ort.get_available_providers())"

# ── Step 6: llama-cpp-python with CUDA ──────────────────────────────────────
log "Step 6/10 — Building llama-cpp-python with CUDA (takes ~10-15 minutes)..."

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda

if "$PYTHON" -c "from llama_cpp import Llama; print('  llama-cpp-python OK')" 2>/dev/null; then
    warn "  llama-cpp-python already installed — skipping. To rebuild: FORCE_CMAKE=1 CMAKE_ARGS=\"-DGGML_CUDA=ON\" pip install llama-cpp-python --force-reinstall"
else
    CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda" \
    FORCE_CMAKE=1 \
    "$PIP" install llama-cpp-python \
        --no-cache-dir \
        --force-reinstall \
        --upgrade
    log "  llama-cpp-python built with CUDA support."
fi

# ── Step 7: Python requirements ──────────────────────────────────────────────
log "Step 7/10 — Installing Python requirements (requirements-orin.txt)..."
"$PIP" install --no-cache-dir -r requirements-orin.txt

# ── Step 8: Download models ──────────────────────────────────────────────────
log "Step 8/10 — Downloading model files..."
bash scripts/download_models.sh

# ── Step 9: Apply Orin config ─────────────────────────────────────────────────
log "Step 9/10 — Applying Orin configuration..."
if [ ! -f config.yaml ] || ! grep -q "llama_cpp" config.yaml; then
    cp config-orin.yaml config.yaml
    log "  config-orin.yaml copied to config.yaml"
else
    warn "  config.yaml already has llama_cpp backend — not overwriting."
fi

# ── Step 10: Build frontend ──────────────────────────────────────────────────
log "Step 10/10 — Building frontend..."
if [ -d frontend ]; then
    cd frontend
    npm install --silent
    npm run build
    cd "$PROJECT_ROOT"
    log "  Frontend built to frontend/dist/"
else
    warn "  frontend/ directory not found — skipping."
fi

# ── Optional: install systemd service ────────────────────────────────────────
INSTALL_SERVICE=${INSTALL_SERVICE:-""}
if [ -n "$INSTALL_SERVICE" ]; then
    log "Installing systemd service (jarvis.service)..."
    sudo tee /etc/systemd/system/jarvis.service > /dev/null << EOF
[Unit]
Description=JARVIS Edge Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$VENV/bin:/usr/local/cuda/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
ExecStart=$PYTHON src/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable jarvis
    log "  Service installed. Start with: sudo systemctl start jarvis"
fi

echo ""
echo -e "${GREEN}=== Setup complete ===${NC}"
echo ""
echo "  To start JARVIS:"
echo "    source ~/jarvis-env/bin/activate"
echo "    cd $PROJECT_ROOT"
echo "    python src/main.py"
echo ""
echo "  To enable auto-start on boot:"
echo "    INSTALL_SERVICE=1 bash scripts/setup_orin.sh"
echo ""
echo "  To access from Windows via SSH port forward:"
echo "    ssh -L 3000:localhost:3000 -L 8765:localhost:8765 jarvis@<ORIN_IP>"
echo "    Then open: http://localhost:3000"
