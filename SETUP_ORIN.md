# Jetson Orin Nano Super 8GB — Complete Setup Guide
# (Updated: ChromaDB persistent RAG + Ollama LLM backend)

This guide takes you from a freshly flashed Jetson Orin Nano Super 8GB to a
fully running JARVIS / Baymax edge assistant with persistent ChromaDB RAG,
Ollama LLM, and Piper TTS — all running 100% offline.

---

## Architecture Overview

```
Microphone → Wake Word (OpenWakeWord) → STT (faster-whisper)
         → ChromaDB RAG (school knowledge) → Ollama LLM (Qwen2.5)
         → TTS (Piper) → Speaker → Robot (optional)
```

**Key design:**  ChromaDB stores all school/event info as a persistent SQLite
database (`data/chroma/`). Ollama just receives a text prompt with relevant
context pre-injected — no tight coupling, no rebuild on reboot.

---

## Hardware Requirements

| Item | Spec |
|---|---|
| Board | NVIDIA Jetson Orin Nano Super Developer Kit 8GB |
| Storage | 32 GB+ microSD **or** 256 GB+ NVMe M.2 (NVMe strongly recommended) |
| Power | Included 19V DC barrel supply |
| Network | Ethernet (recommended during setup) or Wi-Fi 802.11ac |
| Audio | USB microphone + 3.5mm speaker (for on-device voice) |
| Display | HDMI monitor for first boot only — RustDesk takes over after |

---

## Part 0 — Flash JetPack 6.x

> **Critical:** Use JetPack **6.x** (CUDA 12.6, Ubuntu 22.04).
> Do **not** use JetPack 7.x — the Python AI ecosystem (PyTorch wheels, Ollama,
> llama-cpp-python CUDA builds) all target JetPack 6 at this time.

### Option A — microSD (simplest)
1. Download **Balena Etcher**: https://etcher.balena.io
2. Download the Jetson Orin Nano SD card image from NVIDIA:
   https://developer.nvidia.com/embedded/jetpack (choose JetPack 6.x, SD Card image)
3. Flash the image to a 32 GB+ microSD with Etcher.
4. Insert microSD, connect HDMI + keyboard + ethernet, power on.

### Option B — NVMe (faster, recommended for production)
1. Use NVIDIA SDK Manager on a host Ubuntu PC:
   https://developer.nvidia.com/sdk-manager
2. Install JetPack 6.x to the Orin via USB-C flashing.

### First-boot setup
- Set username `jarvis`, hostname `jarvis-orin` when prompted.
- Connect to your Wi-Fi network.
- **Enable max performance mode:**
  ```bash
  sudo nvpmodel -m 0          # MAXN power mode (25W, full GPU/CPU clocks)
  sudo jetson_clocks          # lock clocks at max
  ```
- Note the Orin's IP address:
  ```bash
  ip addr show | grep "inet "
  ```

---

## Part 1 — Install RustDesk for remote access

### On the Jetson (server side)
```bash
# Download the latest RustDesk .deb for ARM64
# Check https://github.com/rustdesk/rustdesk/releases for the exact filename
wget https://github.com/rustdesk/rustdesk/releases/latest/download/rustdesk-<version>-aarch64.deb
sudo dpkg -i rustdesk-*.deb
sudo apt-get install -f -y

# Enable as a systemd service so it auto-starts on boot
sudo systemctl enable rustdesk
sudo systemctl start rustdesk

# Get the RustDesk ID and password
rustdesk --get-id
rustdesk --password
```

### On your Windows PC (client side)
1. Download RustDesk: https://github.com/rustdesk/rustdesk/releases
2. Install, open it, enter the Jetson's ID and password.
3. Connect — you now have a full desktop session on the Orin.

> **Tip:** Connect both devices to the same router via Ethernet for LAN
> direct mode — much lower latency.

---

## Part 2 — System packages and Python environment

Run all commands **in a terminal on the Orin** (via RustDesk or SSH).

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install build tools and audio libraries
sudo apt-get install -y \
    build-essential cmake git wget curl \
    python3-pip python3-dev python3-venv \
    portaudio19-dev libsndfile1 ffmpeg \
    libgomp1 espeak-ng

# Create a Python virtual environment for the project
python3 -m venv ~/jarvis-env
echo 'source ~/jarvis-env/bin/activate' >> ~/.bashrc
source ~/jarvis-env/bin/activate
```

---

## Part 3 — Copy the project to the Orin

### Option A — SCP from your Windows PC
On your **Windows PC**, open PowerShell:
```powershell
# Replace 192.168.1.x with the Orin's IP address
scp -r "C:\Users\vivin\OneDrive\Desktop\small-voice-main" jarvis@192.168.1.x:~/small-voice-main
```

### Option B — Git clone (if repo is on GitHub)
```bash
cd ~
git clone https://github.com/<your-org>/small-voice-main.git
cd small-voice-main
```

---

## Part 4 — Install Ollama (LLM backend)

> Ollama is the simplest way to run Qwen2.5 on the Orin. It bundles its own
> CUDA runtime — no system CUDA toolkit needed for the LLM.

```bash
# Install Ollama (official installer, ARM64 supported)
curl -fsSL https://ollama.com/install.sh | sh

# Verify Ollama is running
ollama --version

# Pull the model (choose one):
ollama pull qwen2.5:3b        # ~2 GB — recommended (better quality)
# ollama pull qwen2.5:1.5b    # ~1 GB — faster, still good

# Test it works
ollama run qwen2.5:3b "Hello, are you working?"
# Type /bye to exit
```

> Ollama runs as a background service on port 11434. It auto-starts on boot.
> The model stays loaded in VRAM between queries — zero reload latency.

---

## Part 5 — Install PyTorch (NVIDIA Jetson wheel)

> **Important:** The standard `pip install torch` fetches an x86-64 wheel.
> You MUST use the NVIDIA-provided aarch64 Jetson wheel for GPU support.

```bash
source ~/jarvis-env/bin/activate
cd ~/small-voice-main

pip install --upgrade pip

# Install the NVIDIA Jetson PyTorch wheel for JetPack 6 / CUDA 12 / Python 3.10
pip install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Verify GPU is visible to PyTorch
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
# Expected: CUDA available: True  |  Device: Orin
```

> If the wheel URL is stale, find the latest at:
> https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/

---

## Part 6 — Install ONNX Runtime GPU (Jetson wheel)

> Standard `pip install onnxruntime` installs CPU-only.
> Piper TTS and OpenWakeWord need the CUDA-enabled aarch64 wheel.

```bash
pip install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v61/onnxruntime/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

# Verify
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

> If the URL is stale, find the latest at:
> https://developer.download.nvidia.com/compute/redist/jp/v61/onnxruntime/

---

## Part 7 — Install Python dependencies

```bash
cd ~/small-voice-main

# Install Orin-specific requirements
# (litert-lm excluded — no aarch64 CUDA wheel; Ollama replaces it)
pip install --no-cache-dir -r requirements-orin.txt
```

---

## Part 7.5 — Install ChromaDB (persistent RAG database)

> ChromaDB replaces FAISS as the vector store. It's a pure-Python SQLite-
> backed database that works on aarch64 JetPack 6 with no compilation step.
> **No server process needed — it's fully embedded and runs offline.**

```bash
source ~/jarvis-env/bin/activate

pip install --no-cache-dir "chromadb>=0.5.0"

# Verify
python3 -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"
# Expected: ChromaDB version: 0.5.x or higher
```

---

## Part 8 — Download Piper voice model

```bash
mkdir -p ~/small-voice-main/assets/piper_voices

# en_US-lessac-medium (natural sounding, ~60 MB)
wget -O ~/small-voice-main/assets/piper_voices/en_US-lessac-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx

wget -O ~/small-voice-main/assets/piper_voices/en_US-lessac-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

> Higher quality options (download both `.onnx` + `.onnx.json` and update `config.yaml`):
> - `en_US-lessac-high` (~120 MB) — more natural delivery
> - `en_US-libritts-high` (~120 MB) — LibriTTS-trained, very natural

---

## Part 9 — Download wake word model

```bash
mkdir -p ~/small-voice-main/assets/wakeword_models

wget -O ~/small-voice-main/assets/wakeword_models/hey_jarvis_v0.1.onnx \
    https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx
```

---

## Part 10 — Download sentence-transformers embedding model *(offline cache)*

> The ChromaDB RAG embedder (`all-MiniLM-L6-v2`) must be cached locally
> before running offline. This one-time download takes ~90 MB.

```bash
source ~/jarvis-env/bin/activate

# This downloads and caches the model to ~/.cache/huggingface/
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Embedding model cached OK — dim:', model.get_sentence_embedding_dimension())
"
# Expected: Embedding model cached OK — dim: 384
```

> After this step, the embedder loads instantly from local cache with no
> internet. The `TRANSFORMERS_OFFLINE=1` env var (set in `embedder.py`) 
> prevents any accidental network calls at runtime.

---

## Part 11 — Apply the Orin config

The Orin-specific config (`config-orin.yaml`) is already set up with:
- `engine.backend: ollama` — uses your Ollama install
- `retrieval.backend: chroma` — uses ChromaDB
- `retrieval.index_dir: data/chroma` — persistent database folder
- `stt.compute_type: int8` — CPU STT (stable, no CUDA dependency for STT)

```bash
cd ~/small-voice-main

# Apply the Orin config
cp config-orin.yaml config.yaml

# Verify the config looks correct
cat config.yaml
```

The key sections should show:
```yaml
engine:
  backend: ollama
  model_path: qwen2.5:3b

retrieval:
  backend: chroma
  index_dir: data/chroma
  k: 5
```

---

## Part 12 — Build the frontend

```bash
cd ~/small-voice-main/frontend
npm install
npm run build       # builds to frontend/dist/

# Or run the dev server for development:
# npm run dev       # serves on http://localhost:3000
```

---

## Part 13 — Build the ChromaDB knowledge index *(run once)*

> This loads all your school/event PDFs into the persistent ChromaDB database.
> **Run this once** — the database survives all reboots with no rebuild needed.

### Step 1 — Copy your PDFs to the Orin

From your **Windows PC**, copy your school knowledge documents:
```powershell
# Replace with your Orin's IP
scp "C:\path\to\hacknexus_brochure.pdf" jarvis@192.168.1.x:~/small-voice-main/data/docs/
scp "C:\path\to\nps_school_info.pdf" jarvis@192.168.1.x:~/small-voice-main/data/docs/
# Add as many PDFs as you want — any topic, any layout
```

Or on the Orin directly:
```bash
mkdir -p ~/small-voice-main/data/docs
# Copy PDFs here via any method
```

### Step 2 — Build the index

```bash
cd ~/small-voice-main
source ~/jarvis-env/bin/activate

python scripts/build_index.py --src data/docs --out data/chroma

# Expected output:
# [INFO] RAG Index Builder — backend: CHROMA
# [INFO]   Source  : /home/jarvis/small-voice-main/data/docs
# [INFO]   Output  : /home/jarvis/small-voice-main/data/chroma
# [INFO] Found N PDF file(s).
# [INFO] ChromaDB index built successfully!
# [INFO]   Chunks  : NNN
# [INFO] The database persists automatically. Restart the assistant
# [INFO] and it will load instantly — no rebuild needed on reboot.
```

### Step 3 — Verify

```bash
ls -lh data/chroma/
# You should see: chroma.sqlite3 (and some UUID subdirectories)

# Quick sanity check — count stored chunks
python3 -c "
import sys; sys.path.insert(0, 'src')
import chromadb
client = chromadb.PersistentClient(path='data/chroma')
col = client.get_collection('jarvis_rag')
print(f'Chunks in DB: {col.count()}')
"
```

> **Adding more PDFs later:** Drop new PDFs into `data/docs/` and re-run
> `build_index.py`. The existing database is replaced with a fresh full build.

---

## Part 14 — Run the assistant

```bash
cd ~/small-voice-main
source ~/jarvis-env/bin/activate

# Make sure Ollama is running (it auto-starts, but just in case)
ollama list        # should show qwen2.5:3b

# Start JARVIS
python src/main.py
```

### What you should see in the logs:

```
[INFO] Config loaded: engine=ollama stt=faster_whisper/small.en tts=piper robot=True(192.168.1.22)
[INFO] Building ChromaRetrievalService from 'data/chroma'
[INFO] ChromaIndex loaded: NNN chunks from 'data/chroma'
[INFO] ChromaRetrievalService ready: NNN chunks, 0 cache entries, min_score=0.25
[INFO] OllamaEngine ready (model=qwen2.5:3b, host=http://localhost:11434)
[INFO] WebSocket server started on ws://localhost:8765
```

Open the frontend:
```
http://localhost:3000      # if running npm run dev
# or open frontend/dist/index.html directly in a browser
```

> **If the robot (Reachy Mini) is not connected:** The assistant still works
> fully — ChromaDB + Ollama + STT + TTS all run independently of the robot.
> Set `robot.enabled: false` in `config.yaml` to suppress connection warnings.

---

## Part 15 — Auto-start on boot (optional)

```bash
# Create systemd services for Ollama (usually auto-installed) and JARVIS

# JARVIS backend service
sudo tee /etc/systemd/system/jarvis.service > /dev/null << 'EOF'
[Unit]
Description=JARVIS Edge Voice Assistant
After=network.target sound.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=jarvis
WorkingDirectory=/home/jarvis/small-voice-main
Environment=PATH=/home/jarvis/jarvis-env/bin:/usr/local/cuda/bin:/usr/bin:/bin
ExecStart=/home/jarvis/jarvis-env/bin/python src/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jarvis
sudo systemctl start jarvis

# View live logs
journalctl -u jarvis -f
```

---

## Part 16 — Access from Windows

### Option A — RustDesk desktop (simplest)
Connect via RustDesk and open the browser on the Orin:
`http://localhost:3000`

### Option B — SSH port forward (lower latency for the web UI)
On your Windows PC:
```powershell
# Replace IP with the Orin's IP
ssh -L 3000:localhost:3000 -L 8765:localhost:8765 jarvis@192.168.1.x
```
Then open `http://localhost:3000` in your **Windows browser** — the UI runs
natively but all AI compute stays on the Orin.

---

## Memory Budget (Orin Nano Super 8GB)

| Component | VRAM / RAM |
|---|---|
| Qwen2.5-3B (Ollama, Q4) | ~2.0 GB VRAM |
| STT faster-whisper small.en | ~0.3 GB RAM |
| ChromaDB + MiniLM embedder | ~0.4 GB RAM |
| Piper TTS | ~0.1 GB RAM |
| System + OS | ~1.5 GB RAM |
| **Free** | **~3.7 GB ✓** |

```bash
# Verify at runtime
sudo tegrastats          # shows live GPU + RAM usage
nvidia-smi               # shows VRAM breakdown
free -h                  # shows system RAM
```

---

## Expected Performance (Orin Nano Super 8GB)

| Step | Time |
|---|---|
| Wake word detection | always-on, ~10 ms |
| STT (small.en, int8 CPU) | ~400–600 ms |
| ChromaDB retrieval (cosine search) | ~5–15 ms |
| LLM TTFT (Qwen2.5-3B, Ollama) | ~1.5–3 s |
| TTS synthesis (one sentence, Piper) | ~200–400 ms |
| **Total first response** | **~2.5–4.5 s** |

---

## Troubleshooting

### `ChromaDB database not found` at startup
```bash
# The index hasn't been built yet — run:
cd ~/small-voice-main
source ~/jarvis-env/bin/activate
python scripts/build_index.py --src data/docs --out data/chroma
```

### `ChromaRetrievalService ready: 0 chunks` (empty index)
```bash
# No PDFs in data/docs — add PDFs and rebuild:
ls data/docs/       # should show your PDF files
python scripts/build_index.py --src data/docs --out data/chroma
```

### `Ollama connection refused` (port 11434)
```bash
# Check Ollama is running
systemctl status ollama
# Start it if not running
sudo systemctl start ollama
# Pull the model again if needed
ollama pull qwen2.5:3b
```

### `CUDA not available` in PyTorch
```bash
nvcc --version      # should show CUDA 12.x
python3 -c "import torch; print(torch.version.cuda)"
# If wrong, reinstall the Jetson-specific PyTorch wheel (Part 5)
```

### `onnxruntime provider mismatch` (Piper uses CPU only)
```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install <the aarch64 GPU wheel URL from Part 6>
```

### Audio device not found (USB microphone)
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
# Set device index in config.yaml under audio.device_index if needed
```

### Robot not connected warning at startup
```bash
# Edit config.yaml and set:
# robot:
#   enabled: false
# The assistant works fully without the robot.
```

### RustDesk drops during index build
Index building can peg the CPU for 30–60 seconds. This is normal — the index
is built once and cached permanently. All subsequent starts are instant.
