import asyncio
import sys
import os
import time
import warnings
import http
import mimetypes
import pathlib
import numpy as np
import sounddevice as sd
import websockets
import json

CONNECTED_CLIENTS = set()
ui_wake_event = asyncio.Event()
ui_stop_event = asyncio.Event()

# Shared index-status holder so ws_handler can respond to get_index_status
# requests. Populated by main_loop after build_retrieval; read-only in ws_handler.
_index_status_ref: dict = {}

# Shared mutable app state — allows ws_handler to hot-swap the retrieval
# reference after a successful rebuild_index, closing the build-once gap
# (Req 2.11). Keys: "retrieval" (current RetrievalService), "cfg" (AppConfig).
_app_state: dict = {}

async def broadcast(message_dict):
    if CONNECTED_CLIENTS:
        msg = json.dumps(message_dict)
        websockets.broadcast(CONNECTED_CLIENTS, msg)

async def ws_handler(websocket):
    remote = getattr(websocket, 'remote_address', 'unknown')
    print(f"[WS] Client connected from {remote}", flush=True)
    CONNECTED_CLIENTS.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "get_graph":
                wiki_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "vault", "wiki", "entities")
                nodes = []
                if os.path.exists(wiki_dir):
                    for fn in os.listdir(wiki_dir):
                        if fn.endswith(".md"):
                            path = os.path.join(wiki_dir, fn)
                            with open(path, "r", encoding="utf-8") as f:
                                content = f.read()
                            nodes.append({
                                "id": fn.replace(".md", ""),
                                "label": fn.replace(".md", "").replace("-", " ").title(),
                                "desc": content
                            })
                await websocket.send(json.dumps({"type": "graph_data", "nodes": nodes}))
            elif data.get("type") == "update_node":
                node_id = data.get("id")
                content = data.get("content")
                if node_id and content:
                    wiki_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "vault", "wiki", "entities")
                    path = os.path.join(wiki_dir, node_id + ".md")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"\n[WS] Updated node {node_id}")
            elif data.get("type") == "start_listening":
                ui_wake_event.set()
            elif data.get("type") == "stop_listening":
                ui_stop_event.set()
            elif data.get("type") == "ping":
                # Heartbeat from browser — keep connection alive through ngrok idle timeout
                await websocket.send(json.dumps({"type": "pong"}))
            elif data.get("type") == "browser_audio":
                # Audio captured by the browser mic and sent as base64 WebM/WAV.
                # Decode, convert to float32 PCM, run STT + full response pipeline.
                import base64 as _b64, io as _io
                import numpy as _np
                audio_b64 = data.get("audio_b64", "")
                if not audio_b64:
                    await websocket.send(json.dumps({"type": "error", "message": "No audio data received."}))
                else:
                    try:
                        raw_bytes = _b64.b64decode(audio_b64)
                        # Convert browser audio (WebM/Opus/OGG) to WAV via ffmpeg,
                        # then decode with soundfile. soundfile cannot read WebM directly.
                        import subprocess as _sp, tempfile as _tf, os as _os
                        with _tf.NamedTemporaryFile(suffix=".webm", delete=False) as _f:
                            _f.write(raw_bytes)
                            _tmp_in = _f.name
                        _tmp_out = _tmp_in.replace(".webm", ".wav")
                        try:
                            _sp.run(
                                ["ffmpeg", "-y", "-i", _tmp_in,
                                 "-ar", "16000", "-ac", "1", "-f", "wav", _tmp_out],
                                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, check=True
                            )
                            import soundfile as _sf
                            audio_np, sr = _sf.read(_tmp_out, dtype="float32", always_2d=False)
                        finally:
                            _os.unlink(_tmp_in)
                            if _os.path.exists(_tmp_out):
                                _os.unlink(_tmp_out)
                        import numpy as _np
                        # Normalise
                        peak = _np.abs(audio_np).max()
                        if peak > 0:
                            audio_np = audio_np / peak
                        await broadcast({"type": "state", "state": "processing"})
                        active_engine   = _app_state.get("engine",   None)
                        active_tts      = _app_state.get("tts",      None)
                        active_stt      = _app_state.get("stt",      None)
                        active_retrieval = _app_state.get("retrieval", None)
                        if active_engine is None or active_tts is None or active_stt is None or active_retrieval is None:
                            await websocket.send(json.dumps({"type": "error", "message": "Backend models not ready yet."}))
                        else:
                            shutdown_ev = asyncio.Event()
                            asyncio.create_task(
                                _handle_browser_response(
                                    audio_np, active_engine, active_tts,
                                    active_stt, active_retrieval, shutdown_ev
                                )
                            )
                    except Exception as exc:
                        await websocket.send(json.dumps({"type": "error", "message": f"Audio processing error: {exc}"}))
            elif data.get("type") == "get_index_status":
                # Respond with the current index status (additive, does not change
                # any existing message type — Req 3.3, 3.8, 3.9 preserved).
                # _index_status_ref is populated by main_loop after build_retrieval.
                status = _index_status_ref.get("status", {"built": False, "chunk_count": 0})
                await websocket.send(json.dumps({"type": "index_status", **status}))
            elif data.get("type") == "upload_pdf":
                import base64
                filename = data.get("filename", "document.pdf")
                content_b64 = data.get("content_b64", "")
                class_num = data.get("class_num")
                subject = data.get("subject")
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                # Flat upload — all PDFs go into data/docs/ regardless of type
                dest_dir = os.path.join(project_root, "data", "docs")
                os.makedirs(dest_dir, exist_ok=True)
                safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
                if not safe_name.endswith(".pdf"):
                    safe_name += ".pdf"
                dest_path = os.path.join(dest_dir, safe_name)
                try:
                    pdf_bytes = base64.b64decode(content_b64)
                    with open(dest_path, "wb") as f:
                        f.write(pdf_bytes)
                    await websocket.send(json.dumps({
                        "type": "upload_result", "success": True,
                        "path": dest_path,
                        "message": f"Saved '{safe_name}' to knowledge base"
                    }))
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "upload_result", "success": False, "message": str(exc)}))
            elif data.get("type") == "rebuild_index":
                import sys as _sys
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                src_dir = os.path.join(project_root, "data", "docs")
                out_dir = os.path.join(project_root, "data", "index")
                script = os.path.join(project_root, "scripts", "build_index.py")
                await websocket.send(json.dumps({"type": "index_progress", "message": "Starting index build..."}))
                try:
                    proc = await asyncio.create_subprocess_exec(
                        _sys.executable, script,
                        "--src", src_dir, "--out", out_dir, "--embed", "minilm",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=project_root,
                    )
                    captured_output_lines: list[str] = []
                    if proc.stdout is not None:
                        async for line in proc.stdout:
                            msg = line.decode("utf-8", errors="replace").strip()
                            if msg:
                                captured_output_lines.append(msg)
                                await websocket.send(json.dumps({"type": "index_progress", "message": msg}))
                    await proc.wait()
                    success = proc.returncode == 0
                    if success:
                        done_message = "Index built successfully!"
                    else:
                        # Check captured output for known missing optional dependencies
                        # and surface an actionable install message to the user.
                        combined_output = "\n".join(captured_output_lines).lower()
                        raw_combined = "\n".join(captured_output_lines)
                        missing_deps: list[str] = []
                        # sentence-transformers
                        if (
                            "sentence-transformers" in raw_combined
                            or "sentence_transformers" in raw_combined
                            or "no module named 'sentence_transformers'" in raw_combined.lower()
                            or (
                                ("modulenotfounderror" in combined_output or "importerror" in combined_output)
                                and "sentence" in combined_output
                            )
                        ):
                            missing_deps.append("sentence-transformers")
                        # faiss-cpu
                        if (
                            "faiss" in combined_output
                            and "no module named 'faiss'" in combined_output
                        ) or ("faiss" in combined_output and (
                            "modulenotfounderror" in combined_output or "importerror" in combined_output
                        )):
                            missing_deps.append("faiss-cpu")
                        # pypdf
                        if (
                            "pypdf" in combined_output
                            or "no module named 'pypdf'" in combined_output
                            or "no module named 'pypdf2'" in combined_output
                        ):
                            missing_deps.append("pypdf")
                        if len(missing_deps) == 1:
                            dep = missing_deps[0]
                            failure_message = (
                                f"Missing dependency: {dep}. "
                                f"Install with: pip install {dep}"
                            )
                        elif len(missing_deps) > 1:
                            failure_message = (
                                "Index build failed. Check app.log. "
                                "You may need: pip install "
                                + " ".join(missing_deps)
                            )
                        else:
                            failure_message = (
                                "Index build failed. Check app.log. "
                                "You may need: pip install sentence-transformers faiss-cpu pypdf"
                            )
                        done_message = failure_message
                    await websocket.send(json.dumps({
                        "type": "index_done", "success": success,
                        "message": done_message
                    }))
                    await broadcast({"type": "refresh_ncert_graph"})
                    # Broadcast updated indexed_docs so TextbooksView refreshes immediately
                    if success:
                        try:
                            import json as _json2, hashlib as _hl2
                            _chunks_p = os.path.join(project_root, "data", "index", "chunks.jsonl")
                            if os.path.exists(_chunks_p):
                                _doc_stats: dict = {}
                                with open(_chunks_p, "r", encoding="utf-8") as _fds:
                                    for _ln in _fds:
                                        _ln = _ln.strip()
                                        if not _ln:
                                            continue
                                        _ch = _json2.loads(_ln)
                                        _src = _ch.get("source") or _ch.get("chapter") or _ch.get("subject") or "Unknown"
                                        _pg = _ch.get("page", 1)
                                        if _src not in _doc_stats:
                                            _doc_stats[_src] = {"source": _src, "chunk_count": 0, "max_page": 0}
                                        _doc_stats[_src]["chunk_count"] += 1
                                        if _pg > _doc_stats[_src]["max_page"]:
                                            _doc_stats[_src]["max_page"] = _pg
                                _docs_out = [
                                    {"source": d["source"], "page_count": d["max_page"], "chunk_count": d["chunk_count"]}
                                    for d in _doc_stats.values()
                                ]
                                await broadcast({"type": "indexed_docs", "docs": _docs_out})
                        except Exception:
                            pass
                    # Hot-reload: after a successful build, swap in a fresh
                    # retrieval service so the running process uses the new
                    # index without requiring a restart (Req 2.11).
                    if success and "cfg" in _app_state:
                        try:
                            new_retrieval = build_retrieval(_app_state["cfg"])
                            _app_state["retrieval"] = new_retrieval
                            new_status = _get_index_status(new_retrieval)
                            _index_status_ref["status"] = new_status
                            await broadcast({"type": "index_status", **new_status})
                        except Exception as swap_exc:
                            # Fallback arm: hot-swap failed — instruct user to restart.
                            await broadcast({
                                "type": "index_status",
                                "built": False,
                                "chunk_count": 0,
                                "restart_required": True,
                                "message": f"Index built but reload failed — please restart: {swap_exc}"
                            })
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "index_done", "success": False, "message": str(exc)}))
            elif data.get("type") == "delete_doc":
                source_name = data.get("source", "")
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                docs_dir = os.path.join(project_root, "data", "docs")
                target_path = os.path.join(docs_dir, source_name)
                # Prevent path traversal
                if os.path.exists(target_path) and os.path.abspath(target_path).startswith(os.path.abspath(docs_dir)):
                    try:
                        os.unlink(target_path)
                        await websocket.send(json.dumps({
                            "type": "delete_result",
                            "success": True,
                            "message": f"Successfully deleted {source_name}"
                        }))
                        # Rebuild index automatically after delete
                        script = os.path.join(project_root, "scripts", "build_index.py")
                        out_dir = os.path.join(project_root, "data", "index")
                        await websocket.send(json.dumps({"type": "index_progress", "message": "Rebuilding index after document deletion..."}))
                        proc = await asyncio.create_subprocess_exec(
                            sys.executable, script,
                            "--src", docs_dir, "--out", out_dir, "--embed", "minilm",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                            cwd=project_root,
                        )
                        if proc.stdout is not None:
                            async for line in proc.stdout:
                                msg = line.decode("utf-8", errors="replace").strip()
                                if msg:
                                    await websocket.send(json.dumps({"type": "index_progress", "message": msg}))
                        await proc.wait()
                        success = proc.returncode == 0
                        await websocket.send(json.dumps({
                            "type": "index_done", "success": success,
                            "message": "Index rebuilt successfully after deletion!" if success else "Index rebuild failed after deletion."
                        }))
                        await broadcast({"type": "refresh_ncert_graph"})
                        # Broadcast fresh doc list
                        try:
                            import json as _json2
                            _chunks_p = os.path.join(project_root, "data", "index", "chunks.jsonl")
                            if os.path.exists(_chunks_p):
                                _doc_stats: dict = {}
                                with open(_chunks_p, "r", encoding="utf-8") as _fds:
                                    for _ln in _fds:
                                        _ln = _ln.strip()
                                        if not _ln:
                                            continue
                                        _ch = _json2.loads(_ln)
                                        _src = _ch.get("source") or _ch.get("chapter") or _ch.get("subject") or "Unknown"
                                        _pg = _ch.get("page", 1)
                                        if _src not in _doc_stats:
                                            _doc_stats[_src] = {"source": _src, "chunk_count": 0, "max_page": 0}
                                        _doc_stats[_src]["chunk_count"] += 1
                                        if _pg > _doc_stats[_src]["max_page"]:
                                            _doc_stats[_src]["max_page"] = _pg
                                _docs_out = [
                                    {"source": d["source"], "page_count": d["max_page"], "chunk_count": d["chunk_count"]}
                                    for d in _doc_stats.values()
                                ]
                                await broadcast({"type": "indexed_docs", "docs": _docs_out})
                            else:
                                await broadcast({"type": "indexed_docs", "docs": []})
                        except Exception:
                            pass
                        # Hot-swap index
                        if success and "cfg" in _app_state:
                            try:
                                new_retrieval = build_retrieval(_app_state["cfg"])
                                _app_state["retrieval"] = new_retrieval
                                new_status = _get_index_status(new_retrieval)
                                _index_status_ref["status"] = new_status
                                await broadcast({"type": "index_status", **new_status})
                            except Exception:
                                pass
                    except Exception as e:
                        await websocket.send(json.dumps({
                            "type": "delete_result",
                            "success": False,
                            "message": f"Error deleting: {e}"
                        }))
                else:
                    await websocket.send(json.dumps({
                        "type": "delete_result",
                        "success": False,
                        "message": f"Document not found: {source_name}"
                    }))
            elif data.get("type") == "get_indexed_docs":
                # Return a list of all indexed documents with per-doc stats.
                # Reads chunks.jsonl and aggregates by source name.
                # Response: {"type": "indexed_docs", "docs": [{"source", "page_count", "chunk_count"}]}
                import json as _json, hashlib as _hl
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                chunks_path = os.path.join(project_root, "data", "index", "chunks.jsonl")
                if not os.path.exists(chunks_path):
                    await websocket.send(_json.dumps({"type": "indexed_docs", "docs": []}))
                else:
                    doc_stats: dict[str, dict] = {}
                    try:
                        with open(chunks_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                chunk = _json.loads(line)
                                source = chunk.get("source") or chunk.get("chapter") or chunk.get("subject") or "Unknown"
                                page = chunk.get("page", 1)
                                if source not in doc_stats:
                                    doc_stats[source] = {"source": source, "chunk_count": 0, "max_page": 0}
                                doc_stats[source]["chunk_count"] += 1
                                if page > doc_stats[source]["max_page"]:
                                    doc_stats[source]["max_page"] = page
                        docs_list = [
                            {"source": d["source"], "page_count": d["max_page"], "chunk_count": d["chunk_count"]}
                            for d in doc_stats.values()
                        ]
                        await websocket.send(_json.dumps({"type": "indexed_docs", "docs": docs_list}))
                    except Exception as exc:
                        await websocket.send(_json.dumps({"type": "indexed_docs", "docs": [], "error": str(exc)}))
            elif data.get("type") == "get_ncert_graph":
                import json as _json, hashlib as _hl
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                chunks_path = os.path.join(project_root, "data", "index", "chunks.jsonl")
                if not os.path.exists(chunks_path):
                    await websocket.send(_json.dumps({"type": "ncert_graph_data", "nodes": [], "edges": []}))
                else:
                    seen_docs = {}
                    try:
                        with open(chunks_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                chunk = _json.loads(line)
                                # Support both new schema (source) and legacy (subject/chapter)
                                source = chunk.get("source") or chunk.get("chapter") or chunk.get("subject") or "Unknown"
                                doc_hash = _hl.md5(source.encode()).hexdigest()[:8]
                                doc_id = f"doc_{doc_hash}"
                                if doc_id not in seen_docs:
                                    page_count = chunk.get("page", 1)
                                    seen_docs[doc_id] = {
                                        "id": doc_id,
                                        "label": source[:30],
                                        "desc": source,
                                        "source": source,
                                        "pages": page_count,
                                        "nodeType": "subject",  # reuse subject colour slot
                                    }
                                else:
                                    # Track max page as proxy for document size
                                    pg = chunk.get("page", 1)
                                    if pg > seen_docs[doc_id]["pages"]:
                                        seen_docs[doc_id]["pages"] = pg

                        # Enrich desc with page count
                        for doc in seen_docs.values():
                            doc["desc"] = f"{doc['source']} ({doc['pages']} pages)"

                        all_nodes = [{"id": "jarvis", "label": "JARVIS", "desc": f"Knowledge base: {len(seen_docs)} document(s) indexed."}]
                        all_nodes += list(seen_docs.values())
                        all_edges = [{"source": "jarvis", "target": d["id"]} for d in seen_docs.values()]
                        await websocket.send(_json.dumps({"type": "ncert_graph_data", "nodes": all_nodes, "edges": all_edges}))
                    except Exception as exc:
                        await websocket.send(_json.dumps({"type": "ncert_graph_data", "nodes": [], "edges": [], "error": str(exc)}))
    except Exception as e:
        print(f"[WS] Handler error: {e}", flush=True)
    finally:
        print(f"[WS] Client disconnected", flush=True)
        CONNECTED_CLIENTS.discard(websocket)

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
from config import load_config
from factories import build_engine, build_stt, build_tts, build_retrieval
from inference.base import BaseEngine
from stt.base import BaseSTT
from synthesis.base import BaseTTS
from retrieval.service import NullRetrievalService

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


def _get_index_status(retrieval) -> dict:
    """Return index status dict: {built: bool, chunk_count: int}.

    Works for both NullRetrievalService (chunk_count=0) and FAISSRetrievalService
    (chunk_count from the underlying FAISS index ntotal).
    """
    if isinstance(retrieval, NullRetrievalService):
        return {"built": False, "chunk_count": 0}
    index = getattr(retrieval, "_index", None)
    ntotal = getattr(index, "ntotal", 0) if index is not None else 0
    return {"built": ntotal > 0, "chunk_count": ntotal}

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
    print("       B A Y M A X   E D G E   V O I C E   A S S I S T A N T")
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

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast({"type": "state", "state": state.lower(), "details": details}))
    except Exception:
        pass

# ── Combined HTTP + WebSocket request handler (websockets 12+) ────────────────
# Serves the frontend dist/ folder for regular HTTP GET requests so that both
# the frontend and the WebSocket share a single port (8080).  This means the
# free ngrok plan only needs ONE tunnel:  ngrok http 8080
#   • Browser opens https://<ngrok>.ngrok-free.app  → gets the React UI
#   • React connects to  wss://<ngrok>.ngrok-free.app  → same tunnel, WS upgrade
_DIST_DIR: str = ""

# Import the websockets 12 Response/Headers types used in _process_request.
from websockets.http11 import Response as _WsResponse          # noqa: E402
from websockets.datastructures import Headers as _WsHeaders    # noqa: E402


def _make_http_response(
    status_code: int,
    reason: str,
    body: bytes,
    content_type: str = "application/octet-stream",
) -> "_WsResponse":
    """Build a websockets 12 Response object for serving static files."""
    headers = _WsHeaders([
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-cache"),
    ])
    return _WsResponse(status_code, reason, headers, body)


async def _process_request(connection, request):
    """Intercept plain HTTP GET requests and serve frontend static files.

    WebSocket upgrade requests ALWAYS contain a 'Sec-WebSocket-Key' header.
    Regular browser HTTP requests (for HTML/CSS/JS assets) never have this.
    Using Sec-WebSocket-Key is more reliable than checking the Upgrade header
    because some proxies (including ngrok) may normalise or forward headers
    differently.
    """
    # If this is a real WebSocket upgrade, let websockets handle it
    ws_key = (request.headers.get("Sec-WebSocket-Key")
              or request.headers.get("sec-websocket-key")
              or request.headers.get("SEC-WEBSOCKET-KEY"))
    if ws_key:
        return None  # proceed with WebSocket handshake as normal

    if not _DIST_DIR or not os.path.isdir(_DIST_DIR):
        body = b"Frontend not built. Run: cd frontend && npm run build"
        return _make_http_response(200, "OK", body, "text/plain; charset=utf-8")

    # Strip query string and resolve path
    clean_path = request.path.split("?")[0].lstrip("/")
    file_path = pathlib.Path(_DIST_DIR) / (clean_path or "index.html")

    # SPA fallback: unknown paths → index.html so React Router works
    if not file_path.exists() or not file_path.is_file():
        file_path = pathlib.Path(_DIST_DIR) / "index.html"

    try:
        content = file_path.read_bytes()
        mime, _ = mimetypes.guess_type(str(file_path))
        return _make_http_response(200, "OK", content, mime or "application/octet-stream")
    except Exception as exc:
        body = f"Error serving file: {exc}".encode()
        return _make_http_response(500, "Internal Server Error", body, "text/plain")


# Main Thread Loop
async def main_loop() -> None:
    global _DIST_DIR
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DIST_DIR = os.path.join(_project_root, "frontend", "dist")

    # Start WebSocket Server
    try:
        # Port 8765: local WebSocket for same-machine access (no process_request needed)
        await websockets.serve(
            ws_handler, "localhost", 8765, max_size=200 * 1024 * 1024
        )
        # Port 8080: combined HTTP + WebSocket server for ngrok access.
        # Regular HTTP GET → serves frontend dist/
        # WebSocket Upgrade → handled by ws_handler
        # Free ngrok plan:  ngrok http 8080  (single tunnel covers everything)
        await websockets.serve(
            ws_handler, "0.0.0.0", 8080,
            max_size=200 * 1024 * 1024,
            process_request=_process_request,
            ping_interval=None,   # disable built-in WS pings; browser handles keepalive
        )
        if os.path.isdir(_DIST_DIR):
            print("[HTTP] Frontend + WS on http://0.0.0.0:8080  (ngrok: ngrok http 8080)", flush=True)
        else:
            print("[HTTP] Frontend dist/ not found — run: cd frontend && npm run build", flush=True)
            print("[WS]   WebSocket on ws://0.0.0.0:8080  (ngrok: ngrok http 8080)", flush=True)
    except Exception as e:
        print(f"[WS] Failed to start server: {e}")

    draw_header()
    print("\n[SYSTEM] Loading offline AI models into memory. Please wait...", flush=True)

    cfg = load_config("config.yaml")
    stt = build_stt(cfg)
    tts = build_tts(cfg)
    engine = build_engine(cfg)
    retrieval = build_retrieval(cfg)

    # Store in shared app state so ws_handler can hot-swap retrieval after a
    # successful rebuild_index (closes the build-once gap — Req 2.11).
    _app_state["retrieval"] = retrieval
    _app_state["cfg"] = cfg
    _app_state["engine"] = engine
    _app_state["tts"] = tts
    _app_state["stt"] = stt

    # Detect index readiness immediately after building retrieval
    _index_status = _get_index_status(retrieval)
    _index_status_ref["status"] = _index_status  # expose for ws_handler get_index_status
    if not _index_status["built"]:
        print("[KNOWLEDGE BASE] NOT built — answers will be ungrounded. Upload a PDF and rebuild.", flush=True)
    else:
        print(f"[KNOWLEDGE BASE] Ready — {_index_status['chunk_count']} chunks indexed.", flush=True)

    recorder  = AudioRecorder(samplerate=16000, blocksize=1280)  
    processor = AudioProcessor()
    wakeword  = WakeWordDetector(
        model_paths=["assets/wakeword_models/hey_jarvis_v0.1.onnx"]
    )
    recorder.start()

    print("\n[SYSTEM] Warming up inference engine...", flush=True)
    engine.warmup()
    if getattr(engine, "warmup_done", False):
        print("[SYSTEM] Warmup complete.", flush=True)
    else:
        print("[SYSTEM] Warmup failed or was skipped — first turn may be slower.", flush=True)
    
    # Broadcast index_status over WebSocket now that the server is ready
    await broadcast({"type": "index_status", **_index_status})

    draw_header()
    print("\n" * 2) 
    show_status(IDLE, "Say 'Baymax' to wake me up.")

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
                
            if ui_stop_event.is_set():
                ui_stop_event.clear()
                _interrupt()
                state = IDLE
                recorder.clear_queue()
                show_status(IDLE, "Interrupted by UI. Say 'Baymax' to wake me up.")
                continue
                
            detected, name = wakeword.check(chunk)
            
            if state != CAPTURING:
                elapsed_since_ww = time.monotonic() - last_ww_time
                if (elapsed_since_ww >= WAKEWORD_COOLDOWN and detected) or ui_wake_event.is_set():
                    ui_wake_event.clear()
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
                    show_status(IDLE, "Say 'Baymax' to wake me up.")
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
                            _handle_response(audio_data, engine, tts, stt, retrieval, shutdown_event)
                        )

                        def _on_done(fut: asyncio.Task) -> None:
                            nonlocal state, silence_chunks, activation_time, buffer
                            if fut.cancelled():
                                state = IDLE
                                show_status(IDLE, "Say 'Baymax' to wake me up.")
                                return
                            elif fut.exception():
                                state = IDLE
                                show_status(IDLE, "Say 'Baymax' to wake me up.")
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
                            show_status(IDLE, "Say 'Baymax' to wake me up.")

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


def _pcm_to_wav_bytes(pcm_np: "np.ndarray", samplerate: int = 22050) -> bytes:
    """Convert a int16 PCM numpy array to an in-memory WAV file (bytes)."""
    import io, wave as _wave
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(pcm_np.tobytes())
    return buf.getvalue()


async def _handle_browser_response(
    audio_np,
    engine: "BaseEngine",
    tts: "BaseTTS",
    stt: "BaseSTT",
    retrieval,
    shutdown_event: asyncio.Event,
) -> None:
    import traceback
    try:
        await _handle_browser_response_inner(
            audio_np, engine, tts, stt, retrieval, shutdown_event
        )
    except Exception as exc:
        print(f"\n[ERROR] in _handle_browser_response: {exc}", flush=True)
        traceback.print_exc()
        try:
            await broadcast({"type": "error", "message": f"Server error: {exc}"})
            await broadcast({"type": "state", "state": "idle"})
        except Exception:
            pass

async def _handle_browser_response_inner(
    audio_np,
    engine: "BaseEngine",
    tts: "BaseTTS",
    stt: "BaseSTT",
    retrieval,
    shutdown_event: asyncio.Event,
) -> None:
    """Handle audio captured by the browser mic.

    STT + LLM run on the Orin; TTS audio is returned to the browser
    as base64-encoded WAV chunks so the browser plays it locally —
    no ALSA/sounddevice output needed on the Orin when using ngrok.
    """
    import time as _time, base64 as _b64
    t_stt = _time.perf_counter()

    # Build STT prompt for browser response (biases Whisper to domain terms)
    import os, json as _json
    wiki_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "vault", "wiki", "entities")
    entity_names = ["Bangalore", "Whitefield", "Reachy Mini", "Dr. Anjali"]
    if os.path.exists(wiki_dir):
        from_files = [fn.replace(".md", "").replace("-", " ").title() for fn in os.listdir(wiki_dir) if fn.endswith(".md")]
        entity_names.extend(from_files)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_path = os.path.join(project_root, "data", "index", "chunks.jsonl")
    if os.path.exists(chunks_path):
        try:
            domain_terms: set[str] = set()
            with open(chunks_path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _chunk = _json.loads(_line)
                        subj = _chunk.get("subject", "")
                        chap = _chunk.get("chapter", "")
                        if subj:
                            domain_terms.add(subj.strip())
                        if chap:
                            chap_words = chap.strip().split()[:3]
                            domain_terms.add(" ".join(chap_words))
                    except Exception:
                        continue
            entity_names.extend(list(domain_terms)[:20])
        except Exception:
            pass

    stt_prompt = ", ".join(list(set(entity_names))[:30])
    text = await asyncio.to_thread(stt.transcribe, audio_np, initial_prompt=stt_prompt)
    stt_ms = int((_time.perf_counter() - t_stt) * 1000)
    if not text:
        await broadcast({"type": "state", "state": "idle", "details": "Couldn't hear anything — try again."})
        return
    print(f"\nUser (browser mic): {text}")
    await broadcast({"type": "text", "text": ""})  # clear previous

    try:
        from knowledge.graph import autocorrect_stt as _autocorrect_stt
        text = _autocorrect_stt(text)
    except ImportError:
        pass  # older graph.py without autocorrect_stt — skip correction

    from inference.prompt_builder import build_prompt, build_page_prompt
    from retrieval.query_router import (
        parse_page_number, is_book_list_query, extract_book_name_query,
        format_book_list, answer_book_query,
    )
    loop = asyncio.get_running_loop()
    active_retrieval = _app_state.get("retrieval", retrieval)
    assert active_retrieval is not None

    async def _speak_browser(answer: str) -> None:
        """Synthesize answer and send WAV audio back to all browser clients."""
        print(f"\nBaymax: {answer}")
        await broadcast({"type": "text", "text": answer})
        pcm = await asyncio.to_thread(tts._synthesize_to_pcm, answer)
        if pcm is not None:
            wav_bytes = _pcm_to_wav_bytes(pcm, tts.samplerate)
            audio_b64 = _b64.b64encode(wav_bytes).decode("utf-8")
            await broadcast({"type": "audio_out", "audio_b64": audio_b64})
        await broadcast({"type": "state", "state": "idle"})

    # Special query routing (book awareness / page lookup)
    try:
        sources = active_retrieval.list_sources()
    except Exception:
        sources = []

    if is_book_list_query(text):
        await _speak_browser(format_book_list(sources))
        return
    bq = extract_book_name_query(text)
    if bq is not None:
        await _speak_browser(answer_book_query(bq, sources))
        return
    page_no = parse_page_number(text)
    page_chunks = []
    if page_no is not None and sources:
        try:
            page_chunks = active_retrieval.get_page(page_no)
        except Exception:
            page_chunks = []
        if not page_chunks:
            await _speak_browser(f"I couldn't find page {page_no} in the uploaded documents.")
            return

    if page_no is not None and page_chunks:
        prompt_text = build_page_prompt(text, page_no, page_chunks)
    else:
        from knowledge.graph import fast_wiki_router
        wiki_context = fast_wiki_router(text)
        if wiki_context:
            from retrieval.base import Chunk, RetrievedChunk
            chunks = [
                RetrievedChunk(
                    chunk=Chunk(id="wiki_hit", text=wiki_context, source="Knowledge Vault", page=1),
                    score=1.0
                )
            ]
        else:
            chunks = await asyncio.to_thread(active_retrieval.retrieve, text, 3)
        prompt_text = build_prompt(text, chunks)

    # Generate and send one-shot "Thinking." voice chunk
    try:
        thinking_pcm = await asyncio.to_thread(tts._synthesize_to_pcm, "Thinking.")
        if thinking_pcm is not None:
            wav_bytes = _pcm_to_wav_bytes(thinking_pcm, tts.samplerate)
            audio_b64 = _b64.b64encode(wav_bytes).decode("utf-8")
            await broadcast({"type": "audio_out", "audio_b64": audio_b64})
    except Exception as e:
        print(f"Error generating thinking voice: {e}")

    # Set up terminal thinking spinner
    import threading, itertools
    spinner_stop = threading.Event()

    def _run_spinner():
        import time as _time_pkg, sys as _sys_pkg
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for f in itertools.cycle(frames):
            if spinner_stop.is_set():
                _sys_pkg.stdout.write("\r\033[2K")
                _sys_pkg.stdout.flush()
                break
            _sys_pkg.stdout.write(f"\r  {f}  \033[2mBaymax is thinking…\033[0m")
            _sys_pkg.stdout.flush()
            _time_pkg.sleep(0.09)

    spinner_thread = threading.Thread(target=_run_spinner, daemon=True)
    spinner_thread.start()

    t_llm = _time.perf_counter()
    try:
        engine.reset()
    except Exception:
        pass
    stream = engine.get_stream(prompt_text)

    # Collect all LLM tokens, broadcast text chunks live, then synthesize full
    # response as a single WAV and send back to browser for gapless playback.
    full_text_parts: list[str] = []
    first_token = True

    def _collect_stream():
        nonlocal first_token
        try:
            for chunk in stream:
                if first_token:
                    spinner_stop.set()
                    ttft_ms = int((_time.perf_counter() - t_llm) * 1000)
                    print(f"\r\033[2K\033[2m[TTFT: {ttft_ms}ms]\033[0m")
                    first_token = False
                    # Now set state to speaking in the browser to hide thinking UI
                    asyncio.run_coroutine_threadsafe(
                        broadcast({"type": "state", "state": "speaking"}), loop
                    )
                try:
                    asyncio.run_coroutine_threadsafe(
                        broadcast({"type": "text", "text": chunk}), loop
                    )
                except Exception:
                    pass
                full_text_parts.append(chunk)
                yield chunk
        finally:
            spinner_stop.set()

    print("Baymax: ", end="", flush=True)

    # Run the sentence-level TTS in a thread, but capture PCM chunks and
    # send them over the WebSocket instead of writing to sounddevice.
    def _stream_tts_to_ws():
        """Synthesize sentence by sentence; send each WAV chunk to the browser."""
        from synthesis.text_norm import extract_complete_sentences, normalize_for_tts
        import re as _re
        buffer = ""
        full_text = ""
        pending_short = ""
        SHORT_THRESH = 3

        def _synth_and_send(sentence: str):
            cleaned = sentence.strip()
            if not cleaned:
                return
            pcm = tts._synthesize_to_pcm(cleaned)
            if pcm is not None:
                wav_bytes = _pcm_to_wav_bytes(pcm, tts.samplerate)
                audio_b64 = _b64.b64encode(wav_bytes).decode("utf-8")
                asyncio.run_coroutine_threadsafe(
                    broadcast({"type": "audio_out", "audio_b64": audio_b64}), loop
                )

        for chunk in _collect_stream():
            if not chunk:
                continue
            print(chunk, end="", flush=True)
            buffer += chunk
            full_text += chunk
            results = extract_complete_sentences(buffer)
            if results:
                sentence, buffer = results[0]
                if sentence.strip():
                    wc = len(sentence.split())
                    if not pending_short and wc <= SHORT_THRESH:
                        pending_short = sentence
                    else:
                        to_speak = (pending_short + " " + sentence).strip() if pending_short else sentence
                        pending_short = ""
                        _synth_and_send(to_speak)

        remaining = (pending_short + " " + buffer).strip() if pending_short else buffer.strip()
        if remaining:
            _synth_and_send(remaining)
        print()
        return full_text.strip()

    full_text = await asyncio.to_thread(_stream_tts_to_ws)
    active_retrieval.cache_put(text, full_text)
    total_ms = int((_time.perf_counter() - t_llm) * 1000)
    print(f"\n\033[2m[Browser STT: {stt_ms}ms | Gen+Speech: {total_ms}ms]\033[0m")
    await broadcast({"type": "state", "state": "idle"})


async def _handle_response(
    audio_data: bytes,
    engine: BaseEngine,
    tts: BaseTTS,
    stt: BaseSTT,
    retrieval,
    shutdown_event: asyncio.Event,
) -> str:
    t_stt_start = time.perf_counter()
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    # Dynamically inject knowledge graph entities + NCERT domain vocabulary into
    # Whisper's initial_prompt to guide phonetics (Req 3.4 contract preserved —
    # only the contents are enriched, the hint mechanism is unchanged).
    import os, json as _json
    wiki_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge", "vault", "wiki", "entities")
    entity_names = ["Bangalore", "Whitefield", "Reachy Mini", "Dr. Anjali"]
    if os.path.exists(wiki_dir):
        from_files = [fn.replace(".md", "").replace("-", " ").title() for fn in os.listdir(wiki_dir) if fn.endswith(".md")]
        entity_names.extend(from_files)

    # Enrich with NCERT subject/chapter vocabulary for domain biasing (Task 9.2 / Req 2.13).
    # Gracefully skipped when chunks.jsonl does not exist — no error raised.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_path = os.path.join(project_root, "data", "index", "chunks.jsonl")
    if os.path.exists(chunks_path):
        try:
            domain_terms: set[str] = set()
            with open(chunks_path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _chunk = _json.loads(_line)
                        subj = _chunk.get("subject", "")
                        chap = _chunk.get("chapter", "")
                        if subj:
                            domain_terms.add(subj.strip())
                        if chap:
                            # Take first 3 words of chapter title for brevity
                            chap_words = chap.strip().split()[:3]
                            domain_terms.add(" ".join(chap_words))
                    except Exception:
                        continue
            entity_names.extend(list(domain_terms)[:20])  # cap at 20 domain terms
        except Exception:
            pass  # graceful — no domain terms if file unreadable

    stt_prompt = ", ".join(list(set(entity_names))[:30])  # cap total at 30 terms
    
    text = await asyncio.to_thread(stt.transcribe, audio_np, initial_prompt=stt_prompt)
    stt_ms = int((time.perf_counter() - t_stt_start) * 1000)

    try:
        from knowledge.graph import autocorrect_stt as _autocorrect_stt
        text = _autocorrect_stt(text)
    except ImportError:
        pass  # older graph.py without autocorrect_stt — skip correction

    if not text:
        return ""

    print(f"\nUser: {text}")

    if _is_shutdown(text):
        print("Baymax: Shutting down. Goodbye.")
        await asyncio.to_thread(tts.speak, "Shutting down. Goodbye.")
        shutdown_event.set()
        return "Shutting down."

    from inference.prompt_builder import build_prompt, build_page_prompt
    from retrieval.query_router import (
        parse_page_number,
        is_book_list_query,
        extract_book_name_query,
        book_query_has_match,
        format_book_list,
        answer_book_query,
    )
    loop = asyncio.get_running_loop()

    # Read the current retrieval service from shared app state so any
    # hot-swap performed by rebuild_index takes effect immediately (Req 2.11).
    # Fall back to the parameter if _app_state is not yet populated.
    active_retrieval = _app_state.get("retrieval", retrieval)
    assert active_retrieval is not None

    # ── Special-query routing: knowledge-base awareness + exact page lookup ──
    # These are answered deterministically from the index metadata rather than
    # via semantic search, so the assistant reliably knows which books it has.
    try:
        _sources = active_retrieval.list_sources()
    except Exception:
        _sources = []

    async def _speak_answer(answer: str) -> str:
        print(f"\nBaymax: {answer}")
        try:
            await broadcast({"type": "text", "text": answer})
        except Exception:
            pass
        await asyncio.to_thread(tts.speak, answer)
        return answer

    # (a) "What books / documents do you have?"
    if is_book_list_query(text):
        return await _speak_answer(format_book_list(_sources))

    # (b) "Do you have the <X> book?" — only short-circuit when it's a genuine
    # availability question that positively matches an indexed document.
    # Content questions fall through to semantic retrieval below, so the
    # assistant automatically pulls from whichever textbook is relevant.
    _book_q = extract_book_name_query(text)
    if _book_q is not None and book_query_has_match(_book_q, _sources):
        return await _speak_answer(answer_book_query(_book_q, _sources))

    # (c) "Read page N" / "what's on page N" — exact page lookup
    _page_no = parse_page_number(text)
    _page_chunks = []
    if _page_no is not None and _sources:
        try:
            _page_chunks = active_retrieval.get_page(_page_no)
        except Exception:
            _page_chunks = []
        if not _page_chunks:
            return await _speak_answer(
                f"I couldn't find page {_page_no} in the uploaded documents."
            )

    # Check answer cache first (instant replay for repeated questions)
    cached = active_retrieval.cache_get(text)
    if cached:
        show_status(SPEAKING, "Answering from cache...")
        print(f"\nBaymax: {cached.answer_text}")
        await asyncio.to_thread(tts.speak, cached.answer_text)
        return cached.answer_text

    if _page_no is not None and _page_chunks:
        # Exact-page lookup: build a page-summary prompt from the page's chunks.
        prompt_text = build_page_prompt(text, _page_no, _page_chunks)
    else:
        from knowledge.graph import fast_wiki_router
        wiki_context = fast_wiki_router(text)
        if wiki_context:
            from retrieval.base import Chunk, RetrievedChunk
            chunks = [
                RetrievedChunk(
                    chunk=Chunk(id="wiki_hit", text=wiki_context, source="Knowledge Vault", page=1),
                    score=1.0
                )
            ]
        else:
            # Retrieve relevant context via semantic search.
            chunks = await asyncio.to_thread(active_retrieval.retrieve, text, 3)
        prompt_text = build_prompt(text, chunks)

    t_llm_start = time.perf_counter()
    show_status(SPEAKING, "Generating reply...")
    # Reset conversation each turn so prefill stays small and TTFT does not grow
    # over a session. Each query is self-contained (its own retrieved/page
    # context is injected), so multi-turn memory is not needed here.
    try:
        engine.reset()
    except Exception:
        pass
    stream = engine.get_stream(prompt_text)
    
    def latency_wrapper():
        first = True
        for chunk in stream:
            if first:
                ttft_ms = int((time.perf_counter() - t_llm_start) * 1000)
                print(f"\n\033[2m[TTFT (Time-to-First-Token): {ttft_ms}ms]\033[0m")
                first = False
            try:
                asyncio.run_coroutine_threadsafe(broadcast({"type": "text", "text": chunk}), loop)
            except Exception:
                pass
            yield chunk

    print("Baymax: ", end="", flush=True)
    full_text = await asyncio.to_thread(tts.stream_text, latency_wrapper())
    active_retrieval.cache_put(text, full_text)  # Cache the answer for future instant replay
    total_generation_ms = int((time.perf_counter() - t_llm_start) * 1000)
    
    print(f"\n\033[2m[Latency Profile -> STT: {stt_ms}ms | Total Gen+Speech: {total_generation_ms}ms]\033[0m")
    
    await asyncio.sleep(0.3)
    return full_text


if __name__ == "__main__":
    if os.name == 'nt':
        os.system('color')
    asyncio.run(main_loop())
