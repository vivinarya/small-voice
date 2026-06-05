import asyncio
import sys
import os
import time
import warnings
import numpy as np
import sounddevice as sd

# Enable UTF-8 encoding on standard streams to support colored characters on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Suppress underlying C-level warnings/logging from TensorFlow, LiteRT, and Whisper
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings('ignore')

# Temporarily redirect stderr to suppress library load warnings from PortAudio & Whisper
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

import logging

# Configure file-only logger to prevent debug traces from corrupting the clean console CLI
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

# Restore stderr after libraries are fully imported
sys.stderr.close()
sys.stderr = stderr_backup

# DSP and silence detection thresholds
WAKEWORD_COOLDOWN  = 2.0   
LISTEN_TIMEOUT_S   = 5.0   
SILENCE_CHUNKS_END = 15    # 15 * 80ms chunks = 1.2s silence to detect end of query
MIN_SPEECH_BYTES   = 16000 # Minimum ~0.5s recording threshold to filter noise

# Shutdown triggers
_SHUTDOWN_KW = {"shutdown", "shut down", "power off", "turn off", "exit", "quit", "goodbye"}

def _is_shutdown(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _SHUTDOWN_KW)

# State Machine States
IDLE      = "IDLE"      
LISTENING = "LISTENING"  
CAPTURING = "CAPTURING"  
SPEAKING  = "SPEAKING"

# CLI Helpers
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
    
    # Write status to the top line using ANSI cursor positions
    sys.stdout.write("\033[s") 
    sys.stdout.write(f"\033[H\033[2K") 
    sys.stdout.write(f"\r\033[1;36mSTATUS: [{icon}] \033[0;37m{details}\033[0m\n")
    sys.stdout.write("\033[u") 
    sys.stdout.flush()

# Main Thread Loop
async def main_loop() -> None:
    draw_header()
    print("\n[SYSTEM] Loading offline AI models into memory. Please wait...", flush=True)

    import whisper
    stt_model = whisper.load_model("base.en")
    
    recorder  = AudioRecorder(samplerate=16000, blocksize=1280)  
    processor = AudioProcessor()
    wakeword  = WakeWordDetector(
        model_paths=["assets/wakeword_models/hey_jarvis_v0.1.onnx"]
    )
    tts = TTSStreamer()

    model_path = "assets/gemma-4-E4B-it.litertlm"
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found at '{model_path}'. Please check README.")
        return

    engine = GemmaEngine(model_path=model_path)
    recorder.start()
    
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

            # State Machine Handlers
            if state == IDLE:
                pass

            elif state == LISTENING:
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
                                txt_lower = response_text.strip().lower() if response_text else ""
                                is_followup = (
                                    txt_lower.endswith('?') or 
                                    "tell me" in txt_lower or 
                                    "i need" in txt_lower or
                                    "please" in txt_lower[-30:]
                                )
                                if response_text and is_followup:
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
    t_stt_start = time.perf_counter()
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    result = await asyncio.to_thread(stt_model.transcribe, audio_np, fp16=False)
    text = result.get("text", "").strip()
    stt_ms = int((time.perf_counter() - t_stt_start) * 1000)

    from knowledge.graph import autocorrect_stt
    text = autocorrect_stt(text)

    if not text:
        return ""

    print(f"\nUser: {text}")

    if _is_shutdown(text):
        print("Jarvis: Shutting down. Goodbye.")
        await asyncio.to_thread(tts.speak, "Shutting down. Goodbye.")
        shutdown_event.set()
        return "Shutting down."

    # Inject wiki context if applicable
    from knowledge.graph import fast_wiki_router
    from knowledge.search import ActiveWebUpdater
    
    wiki_context = fast_wiki_router(text)
    prompt_text = text
    
    # Check explicitly for update commands first!
    update_triggers = ["update yourself", "update your self", "learn about", "search the internet and update", "update it"]
    if any(trigger in text.lower() for trigger in update_triggers):
        show_status(SPEAKING, "Searching the internet and updating my vault...")
        updater = ActiveWebUpdater()
        web_context = await asyncio.to_thread(updater.search_and_stage, text)
        
        from knowledge.vault_compiler import WikiCustodian, LocalCompilerClient
        compiler_client = LocalCompilerClient(engine)
        
        def run_compiler():
            WikiCustodian().process_incoming_web_logs(compiler_client)
            
        asyncio.create_task(asyncio.to_thread(run_compiler))
        
        # Bypass the LLM for immediate 0ms TTFT response
        def hardcoded_stream():
            yield "I am researching that topic on the internet right now, and compiling the facts into my permanent knowledge vault!"
        
        print("\nJarvis: ", end="", flush=True)
        return await asyncio.to_thread(tts.stream_text, hardcoded_stream())
    elif wiki_context:
        prompt_text = f"Context from the knowledge vault:\n{wiki_context}\n\nUser: {text}"
    elif any(trigger in text.lower() for trigger in ["search", "look up", "what is the current", "who won", "internet"]):
        show_status(SPEAKING, "Searching the web...")
        updater = ActiveWebUpdater()
        web_context = await asyncio.to_thread(updater.search_and_stage, text)
        prompt_text = f"{web_context}\n\nUser: {text}"

    t_llm_start = time.perf_counter()
    show_status(SPEAKING, "Generating reply...")
    stream = engine.get_stream(None, prompt_text)
    
    def latency_wrapper():
        first = True
        for chunk in stream:
            if first:
                ttft_ms = int((time.perf_counter() - t_llm_start) * 1000)
                print(f"\n\033[2m[TTFT (Time-to-First-Token): {ttft_ms}ms]\033[0m")
                first = False
            yield chunk

    print("Jarvis: ", end="", flush=True)
    full_text = await asyncio.to_thread(tts.stream_text, latency_wrapper())
    total_generation_ms = int((time.perf_counter() - t_llm_start) * 1000)
    
    print(f"\n\033[2m[Latency Profile -> STT: {stt_ms}ms | Total Gen+Speech: {total_generation_ms}ms]\033[0m")
    
    await asyncio.sleep(0.3)
    return full_text


if __name__ == "__main__":
    if os.name == 'nt':
        os.system('color')
    asyncio.run(main_loop())
