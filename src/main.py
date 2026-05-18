import asyncio
import sys
import os
import time
import warnings
import numpy as np
import sounddevice as sd

# Force standard streams to support UTF-8 encoding to prevent Unicode errors on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Suppress deep C++ logging from TensorFlow, LiteRT, ONNX, and Whisper
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings('ignore')

# Redirect stderr to suppress C-level startup warnings from PortAudio/LiteRT/Whisper
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

import logging

# Configure logging to write ONLY to a file to keep the console completely clean
log_file = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding='utf-8')]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio.recorder import AudioRecorder
from audio.processor import AudioProcessor
from audio.wakeword import WakeWordDetector
from inference.engine import GemmaEngine
from synthesis.tts_stream import TTSStreamer

# Restore stderr after libraries are loaded
sys.stderr.close()
sys.stderr = stderr_backup

# ── Tuning knobs ──────────────────────────────────────────────────────────────
WAKEWORD_COOLDOWN  = 2.0   
LISTEN_TIMEOUT_S   = 5.0   
SILENCE_CHUNKS_END = 15    # 15 × 80ms = 1200ms silence → utterance done
MIN_SPEECH_BYTES   = 16000 # ~0.5s — discard shorter captures

# Shutdown keywords
_SHUTDOWN_KW = {"shutdown", "shut down", "power off", "turn off", "exit", "quit", "goodbye"}

def _is_shutdown(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _SHUTDOWN_KW)

# States
IDLE      = "IDLE"      
LISTENING = "LISTENING"  
CAPTURING = "CAPTURING"  
SPEAKING  = "SPEAKING"

# ── CLI Interface Helpers ─────────────────────────────────────────────────────
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_header():
    clear_terminal()
    print("=" * 60)
    print("       J A R V I S   E D G E   V O I C E   A S S I S T A N T")
    print("=" * 60)
    print("  [100% Local]  [Privacy First]  [Low-Latency CPU Pipeline]")
    print("-" * 60)

def show_status(state: str, details: str = ""):
    status_icons = {
        IDLE: "IDLE",
        LISTENING: "LISTENING",
        CAPTURING: "CAPTURING",
        SPEAKING: "SPEAKING"
    }
    icon = status_icons.get(state, "LOADING")
    
    # Save cursor, clear status line, write colored status, and restore cursor
    sys.stdout.write("\033[s") # Save cursor position
    sys.stdout.write(f"\033[H\033[2K") # Go to top line and clear
    sys.stdout.write(f"\r\033[1;36mSTATUS: [{icon}] \033[0;37m{details}\033[0m\n")
    sys.stdout.write("\033[u") # Restore cursor position
    sys.stdout.flush()

# ── Main Event Loop ───────────────────────────────────────────────────────────
async def main_loop() -> None:
    draw_header()
    print("\n[SYSTEM] Loading offline AI models into memory. Please wait...", flush=True)

    # Load models
    import whisper
    stt_model = whisper.load_model("base.en")
    
    recorder  = AudioRecorder(samplerate=16000, blocksize=1280)  
    processor = AudioProcessor()
    wakeword  = WakeWordDetector(
        model_paths=["assets/wakeword_models/hey_jarvis_v0.1.onnx"]
    )
    tts = TTSStreamer()

    model_path = "assets/gemma-4-e2b-it.litertlm"
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found at '{model_path}'. Please check README.")
        return

    engine = GemmaEngine(model_path=model_path)

    recorder.start()
    
    # Redraw the clean interface
    draw_header()
    print("\n" * 2) 
    show_status(IDLE, "Say 'Hey Jarvis' to wake me up.")

    state              = IDLE
    buffer: list[bytes] = []
    silence_chunks     = 0
    last_ww_time       = 0.0
    activation_time    = 0.0
    response_task: asyncio.Task | None = None
    shutdown_event     = asyncio.Event()

    def _interrupt() -> None:
        nonlocal response_task
        sd.stop()                                
        if response_task and not response_task.done():
            response_task.cancel()
        response_task = None

    def _activate() -> None:
        nonlocal state, buffer, silence_chunks, activation_time
        _interrupt()
        recorder.clear_queue()
        buffer          = []
        silence_chunks  = 0
        activation_time = time.monotonic()
        state           = LISTENING
        show_status(LISTENING, "I'm listening. Ask me anything!")

    try:
        while not shutdown_event.is_set():
            chunk = recorder.get_audio_chunk()

            if chunk is None:
                await asyncio.sleep(0.005)
                continue
            detected, name = wakeword.check(chunk)
            
            if state != CAPTURING:
                elapsed_since_ww = time.monotonic() - last_ww_time
                if elapsed_since_ww >= WAKEWORD_COOLDOWN and detected:
                    last_ww_time = time.monotonic()
                    show_status(LISTENING, "Wake word detected!")
                    _activate()
                    await asyncio.sleep(0.005)
                    continue

            #State handlers
            if state == IDLE:
                pass

            elif state == LISTENING:
                # Timeout if user never speaks
                if (time.monotonic() - activation_time) > LISTEN_TIMEOUT_S:
                    show_status(IDLE, "Say 'Hey Jarvis' to wake me up.")
                    state = IDLE
                    recorder.clear_queue()
                    continue

                if processor.is_speech(chunk, threshold=300):
                    show_status(CAPTURING, "Recording speech...")
                    state          = CAPTURING
                    silence_chunks = 0
                    buffer         = [processor.process_for_inference(chunk)]

            elif state == CAPTURING:
                if processor.is_speech(chunk, threshold=300):
                    silence_chunks = 0
                else:
                    silence_chunks += 1
                buffer.append(processor.process_for_inference(chunk))

                if silence_chunks >= SILENCE_CHUNKS_END:
                    audio_data     = b"".join(buffer)
                    buffer         = []
                    silence_chunks = 0
                    state          = SPEAKING

                    if len(audio_data) < MIN_SPEECH_BYTES:
                        show_status(IDLE, "Discarded short audio clip.")
                        recorder.clear_queue()
                        state = IDLE
                    else:
                        show_status(SPEAKING, "Processing transcription & thoughts...")
                        response_task = asyncio.create_task(
                            _handle_response(audio_data, engine, tts, stt_model, shutdown_event)
                        )

                        def _on_done(fut: asyncio.Task) -> None:
                            nonlocal state, silence_chunks, activation_time, buffer
                            if fut.cancelled():
                                state = IDLE
                                show_status(IDLE, "Say 'Hey Jarvis' to wake me up.")
                                return
                            elif fut.exception():
                                state = IDLE
                                show_status(IDLE, "Say 'Hey Jarvis' to wake me up.")
                                return
                                
                            try:
                                response_text = fut.result()
                                if response_text and response_text.strip().endswith('?'):
                                    state = LISTENING
                                    silence_chunks = 0
                                    buffer = []
                                    activation_time = time.monotonic()
                                    recorder.clear_queue()
                                    show_status(LISTENING, "Awaiting follow-up response...")
                                    return
                            except Exception:
                                pass

                            state = IDLE
                            show_status(IDLE, "Say 'Hey Jarvis' to wake me up.")

                        response_task.add_done_callback(_on_done)

            elif state == SPEAKING:
                pass

            await asyncio.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        _interrupt()
        recorder.stop()
        print("\n\n[SYSTEM] Goodbye!")


async def _handle_response(
    audio_data: bytes,
    engine: GemmaEngine,
    tts: TTSStreamer,
    stt_model,
    shutdown_event: asyncio.Event,
) -> str:
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Transcribe offline
    result = await asyncio.to_thread(stt_model.transcribe, audio_np, fp16=False)
    text = result.get("text", "").strip()

    if not text:
        return ""

    # Display what the user asked
    print(f"\nUser: {text}")

    # Shutdown keyword check
    if _is_shutdown(text):
        print("Jarvis: Shutting down. Goodbye.")
        await asyncio.to_thread(tts.speak, "Shutting down. Goodbye.")
        shutdown_event.set()
        return "Shutting down."

    # Process thoughts
    show_status(SPEAKING, "Generating reply...")
    stream = engine.get_stream(None, text)
    
    # Stream the text and trigger asynchronous speech
    print("Jarvis: ", end="", flush=True)
    full_text = await asyncio.to_thread(tts.stream_text, stream)
    
    await asyncio.sleep(0.3)
    return full_text


if __name__ == "__main__":
    # Enable colored ANSI escape codes in Windows Terminal
    if os.name == 'nt':
        os.system('color')
    asyncio.run(main_loop())

