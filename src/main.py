import asyncio
import logging
import sys
import os
import io
import wave
import platform
import time
import numpy as np
import sounddevice as sd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio.recorder import AudioRecorder
from audio.processor import AudioProcessor
from audio.wakeword import WakeWordDetector
from inference.engine import GemmaEngine
from synthesis.tts_stream import TTSStreamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

#Tuning knobs
WAKEWORD_COOLDOWN  = 2.0   # seconds between wake word triggers
LISTEN_TIMEOUT_S   = 5.0   # seconds to wait for speech after activation
SILENCE_CHUNKS_END = 15     # 15 × 80ms = 1200ms silence → utterance done
MIN_SPEECH_BYTES   = 16000 # ~0.5s — discard shorter captures

#Shutdown keywords
_SHUTDOWN_KW = {"shutdown", "shut down", "power off", "turn off", "exit", "quit", "goodbye"}

def _is_shutdown(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _SHUTDOWN_KW)


# States
IDLE      = "IDLE"      
LISTENING = "LISTENING"  
CAPTURING = "CAPTURING"  
SPEAKING  = "SPEAKING"


async def main_loop() -> None:
    logger.info("Initializing Gemma Edge Assistant...")

    import whisper
    logger.info("Initializing Local Whisper STT model...")
    stt_model = whisper.load_model("base.en")
    logger.info("Whisper model loaded.")

    recorder  = AudioRecorder(samplerate=16000, blocksize=1280)  # 80 ms chunks
    processor = AudioProcessor()
    wakeword  = WakeWordDetector(
        model_paths=["assets/wakeword_models/hey_jarvis_v0.1.onnx"]
    )
    tts = TTSStreamer()

    model_path = "assets/gemma-4-e2b-it.litertlm"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at '{model_path}'.")
        return

    engine = GemmaEngine(model_path=model_path)

    recorder.start()
    logger.info("Ready — say 'Hey Jarvis' to activate.")

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
        logger.info("Listening...")

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
                    logger.info(f"[{state}] Wake word '{name}' — activating (barge-in).")
                    _activate()
                    await asyncio.sleep(0.005)
                    continue

            #State handlers
            if state == IDLE:
                pass

            elif state == LISTENING:
                # Timeout if user never speaks
                if (time.monotonic() - activation_time) > LISTEN_TIMEOUT_S:
                    logger.info("No speech — back to idle.")
                    state = IDLE
                    recorder.clear_queue()
                    continue

                if processor.is_speech(chunk, threshold=300):
                    logger.info("Speech detected — capturing.")
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
                        logger.info(f"Too short ({len(audio_data)} B) — discarding.")
                        recorder.clear_queue()
                        state = IDLE
                    else:
                        logger.info("Utterance complete — processing...")
                        response_task = asyncio.create_task(
                            _handle_response(audio_data, engine, tts, stt_model, shutdown_event)
                        )

                        def _on_done(fut: asyncio.Task) -> None:
                            nonlocal state, silence_chunks, activation_time, buffer
                            if fut.cancelled():
                                logger.info("Response cancelled (barge-in).")
                                state = IDLE
                                logger.info("Ready — say 'Hey Jarvis'.")
                                return
                            elif fut.exception():
                                logger.error(f"Response task error: {fut.exception()}")
                                state = IDLE
                                logger.info("Ready — say 'Hey Jarvis'.")
                                return
                                
                            try:
                                response_text = fut.result()
                                if response_text and response_text.strip().endswith('?'):
                                    logger.info("Ended with a question — keeping microphone open.")
                                    state = LISTENING
                                    silence_chunks = 0
                                    buffer = []
                                    activation_time = time.monotonic()
                                    recorder.clear_queue()
                                    return
                            except Exception as e:
                                logger.error(f"Failed to get result: {e}")

                            state = IDLE
                            logger.info("Ready — say 'Hey Jarvis'.")

                        response_task.add_done_callback(_on_done)

            elif state == SPEAKING:
                pass

            await asyncio.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    finally:
        _interrupt()
        recorder.stop()
        logger.info("Session ended.")


async def _handle_response(
    audio_data: bytes,
    engine: GemmaEngine,
    tts: TTSStreamer,
    stt_model,
    shutdown_event: asyncio.Event,
) -> str:
    """STT -> shutdown check -> LLM -> TTS pipeline, fully cancellable."""

    # STT
    t_stt = time.perf_counter()
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Transcribe offline
    result = await asyncio.to_thread(stt_model.transcribe, audio_np, fp16=False)
    text = result.get("text", "").strip()
    stt_ms = int((time.perf_counter() - t_stt) * 1000)
    logger.info(f"Local STT transcribed in {stt_ms} ms: '{text}'")

    if not text:
        return ""

    #Shutdown
    if _is_shutdown(text):
        await asyncio.to_thread(tts.speak, "Shutting down. Goodbye.")
        shutdown_event.set()
        return "Shutting down."

    #LLM inference & TTS Streaming
    t0 = time.perf_counter()
    stream = engine.get_stream(None, text)
    
    full_text = await asyncio.to_thread(tts.stream_text, stream)
    
    llm_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"Total Response & TTS time: {llm_ms} ms.")

    await asyncio.sleep(0.3)
    return full_text


if __name__ == "__main__":
    asyncio.run(main_loop())
