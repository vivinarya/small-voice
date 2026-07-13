# Jetson Orin Nano Super 8GB — Complete Setup Guide

This guide takes you from a freshly flashed Jetson Orin Nano Super 8GB to a
fully running JARVIS edge assistant, accessible remotely via RustDesk.

---

## Hardware requirements

| Item | Spec |
|---|---|
| Board | NVIDIA Jetson Orin Nano Super Developer Kit 8GB |
| Storage | 32 GB+ microSD **or** 256 GB+ NVMe M.2 (NVMe strongly recommended) |
| Power | Included 19V DC barrel supply |
| Network | Ethernet (recommended during setup) or Wi-Fi 802.11ac |
| Audio (optional) | USB microphone + 3.5mm speaker (for on-device voice) |
| Display | HDMI monitor for first boot only — RustDesk takes over after |

---

## Part 0 — Flash JetPack 6.x  *(do this once, on the Orin itself)*

> **Critical:** Use JetPack **6.x** (CUDA 12.6, Ubuntu 22.04).
> Do **not** use JetPack 7.x — the Python AI ecosystem (PyTorch wheels, Ollama,
> llama-cpp-python CUDA builds) all target JetPack 6 at this time.

### Option A — microSD (simplest)
1. On a Windows/Mac/Linux PC, download **Balena Etcher** (https://etcher.balena.io).
2. Download the Jetson Orin Nano Developer Kit SD card image from NVIDIA:
   https://developer.nvidia.com/embedded/jetpack (choose JetPack 6.x, SD Card image).
3. Flash the image to a 32 GB+ microSD with Etcher.
4. Insert microSD into the board, connect HDMI + keyboard + ethernet, power on.

### Option B — NVMe (faster, recommended for production)
1. Follow the NVIDIA SDK Manager guide on a host Ubuntu PC:
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
  You'll need this for RustDesk and to access the web frontend.

---

## Part 1 — Install RustDesk for remote access

### On the Jetson (server side)
```bash
# Download the latest RustDesk .deb for ARM64
wget https://github.com/rustdesk/rustdesk/releases/latest/download/rustdesk-<version>-aarch64.deb
# Check https://github.com/rustdesk/rustdesk/releases for the exact filename
sudo dpkg -i rustdesk-*.deb
sudo apt-get install -f -y        # fix any missing deps

# Enable as a systemd service so it auto-starts on boot
sudo systemctl enable rustdesk
sudo systemctl start rustdesk

# Get the RustDesk ID and one-time password
rustdesk --get-id
rustdesk --password
```

### On your Windows PC (client side)
1. Download RustDesk for Windows: https://github.com/rustdesk/rustdesk/releases
2. Install and open it.
3. Enter the Jetson's RustDesk ID and the password you noted above.
4. Connect — you now have a full desktop session on the Orin.

> **Tip:** For lower-latency remote sessions, connect the Orin via Ethernet and
> your Windows PC to the same router.  RustDesk will use LAN direct mode
> automatically.

---

## Part 2 — System packages and Python environment

Run all commands below **in a terminal on the Orin** (via RustDesk or SSH).

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

## Part 3 — Clone / copy the project

### Option A — Copy from your Windows PC via SCP
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

## Part 4 — Install PyTorch (NVIDIA Jetson wheel — NOT standard pip)

> **This is the most important step.**  The standard `pip install torch` fetches
> an x86-64 wheel that either fails or runs CPU-only.  The NVIDIA-provided
> Jetson wheel links against the JetPack CUDA/cuDNN libraries.

```bash
source ~/jarvis-env/bin/activate
cd ~/small-voice-main

# Install the NVIDIA Jetson PyTorch wheel for JetPack 6 / CUDA 12 / Python 3.10
pip install --upgrade pip
pip install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Verify GPU is visible to PyTorch
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
# Expected: CUDA available: True  |  Device: Orin
```

> If the above wheel URL is stale, find the latest at:
> https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/

---

## Part 5 — Install ONNX Runtime GPU (Jetson wheel)

> Standard `pip install onnxruntime` installs a CPU-only build.
> Piper TTS and OpenWakeWord need the CUDA-enabled aarch64 wheel.

```bash
# Install onnxruntime-gpu for JetPack 6 / CUDA 12 / Python 3.10
pip install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v61/onnxruntime/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

# Verify
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

> If the URL is stale, find the latest at:
> https://developer.download.nvidia.com/compute/redist/jp/v61/onnxruntime/

---

## Part 6 — Install llama-cpp-python with CUDA (build from source)

> There is no pre-built CUDA wheel for aarch64.
> This must be compiled against your JetPack CUDA installation.
> Build time: ~10-15 minutes on the Orin.

```bash
# Ensure CUDA is on PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build and install llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87" \
FORCE_CMAKE=1 \
pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

# Verify GPU offload is available
python3 -c "from llama_cpp import Llama; print('llama-cpp-python installed OK')"
```

> `CUDA_ARCHITECTURES=87` is the Orin's Ampere compute capability.

---

## Part 7 — Install the rest of the Python dependencies

```bash
cd ~/small-voice-main

# Install Orin-specific requirements (skips litert-lm, uses llama-cpp-python instead)
pip install --no-cache-dir -r requirements-orin.txt
```

---

## Part 8 — Download the LLM model (GGUF)

```bash
mkdir -p ~/small-voice-main/models

# Option A: Qwen2.5-1.5B Instruct Q4_K_M — fast, ~1 GB, excellent for this use case
wget -O ~/small-voice-main/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Option B: Qwen2.5-3B Instruct Q4_K_M — better quality, ~2 GB, still fits in 8 GB
# wget -O ~/small-voice-main/models/qwen2.5-3b-instruct-q4_k_m.gguf \
#     https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

---

## Part 9 — Download Piper voice model

```bash
mkdir -p ~/small-voice-main/assets/piper_voices

# Download en_US-lessac-medium voice (same as Windows setup)
wget -O ~/small-voice-main/assets/piper_voices/en_US-lessac-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx

wget -O ~/small-voice-main/assets/piper_voices/en_US-lessac-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

---

## Part 10 — Download wake word model

```bash
mkdir -p ~/small-voice-main/assets/wakeword_models

wget -O ~/small-voice-main/assets/wakeword_models/hey_jarvis_v0.1.onnx \
    https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx
```

---

## Part 11 — Configure for Orin (switch to llama_cpp backend)

The Orin config file is already present at `config-orin.yaml`.
Copy it over the default config:

```bash
cd ~/small-voice-main
cp config-orin.yaml config.yaml
```

Or set via environment variable if you want to keep both configs:
```bash
export ENGINE_BACKEND=llama_cpp
export MODEL_PATH=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
export STT_COMPUTE_TYPE=float16      # uses Tensor cores on Orin GPU
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

## Part 13 — Run the assistant

```bash
cd ~/small-voice-main
source ~/jarvis-env/bin/activate

# First run — also builds the FAISS embedding index if data/docs has PDFs
python src/main.py
```

The backend WebSocket starts on `ws://localhost:8765`.
Open the frontend in a browser on the Orin (or forwarded via RustDesk):
```
http://localhost:3000      # if running npm run dev
# or open frontend/dist/index.html directly
```

---

## Part 14 — Auto-start on boot (optional)

```bash
# Create a systemd service for the JARVIS backend
sudo tee /etc/systemd/system/jarvis.service > /dev/null << 'EOF'
[Unit]
Description=JARVIS Edge Voice Assistant
After=network.target sound.target

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

## Part 15 — Access from Windows via RustDesk

Since the assistant's web frontend runs on the Orin, you have two options:

### Option A — Use RustDesk desktop (simplest)
Connect via RustDesk and open the browser on the Orin desktop pointing to
`http://localhost:3000`.

### Option B — Port forward via SSH (lower latency for the web UI)
On your Windows PC:
```powershell
# Replace IP with the Orin's IP
ssh -L 3000:localhost:3000 -L 8765:localhost:8765 jarvis@192.168.1.x
```
Then open `http://localhost:3000` in your Windows browser — the UI runs
locally but talks to the Orin's backend.  This gives native browser performance
with all compute staying on the Orin.

---

## Memory budget verification

After startup, run this on the Orin to confirm VRAM/RAM usage:
```bash
# Show GPU memory usage
sudo tegrastats
# or
nvidia-smi      # shows VRAM used by the GGUF model layers

# Show system RAM
free -h
```

Expected at idle (model loaded, no query):
- GPU memory: ~1.1 GB (Qwen2.5-1.5B Q4 fully offloaded)
- System RAM: ~1.5–2 GB
- Free RAM: ~5.5–6 GB  ✓

---

## Troubleshooting

### `CUDA not available` in PyTorch
```bash
# Verify CUDA install
nvcc --version                    # should show CUDA 12.x
python3 -c "import torch; print(torch.version.cuda)"
# If wrong, reinstall the Jetson-specific PyTorch wheel (Part 4)
```

### `No GPU layers offloaded` in llama-cpp-python
```bash
# Rebuild with explicit CUDA flags
export CUDA_HOME=/usr/local/cuda
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda" \
FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall
```

### `onnxruntime provider mismatch` (Piper uses CPU only)
```bash
# Reinstall the GPU wheel (Part 5)
pip uninstall onnxruntime onnxruntime-gpu -y
pip install <the aarch64 GPU wheel URL from Part 5>
```

### Audio device not found (USB microphone)
```bash
# List audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
# Set the device index in config.yaml under audio.device_index if needed
```

### RustDesk connection drops during index build
Index building (first PDF upload) can peg the CPU for 30–60 seconds, causing
brief UI lag over RustDesk. This is expected — the index is built once and
cached. Subsequent uploads are faster.

---

## Expected performance (Orin Nano Super 8GB)

| Step | Time |
|---|---|
| Wake word detection | always-on, ~10ms |
| STT (small.en, float16 GPU) | ~300ms |
| FAISS retrieval (426 chunks) | ~20ms |
| LLM TTFT (Qwen2.5-1.5B, Q4, full GPU) | ~1–3s |
| TTS synthesis (one sentence, Piper) | ~200–400ms |
| **Total first response** | **~2–4s** |
