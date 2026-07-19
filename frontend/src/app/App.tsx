import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Mic, Square } from "lucide-react";
import { KnowledgeGraph } from "./components/KnowledgeGraph";
import { TextbooksView } from "./components/TextbooksView";
import { TextEffect } from "./components/TextEffect";

type JarvisState = "idle" | "listening" | "processing" | "speaking";
type AppView = "speak" | "graph" | "textbooks";

const FONT = "'Plus Jakarta Sans', sans-serif";
const BG = "#eae9e4";
const TEXT = "#757068";
const MUTED = "#a8a09a";
const MINT = "#3DD68C";
const DARK = "#1e1d1b";

// ─── Shared WebSocket hook ─────────────────────────────────────────────────────
function getWsUrl(): string {
  // When accessed via ngrok or any remote host, connect WebSocket to port 8080.
  // When on localhost, use the standard port 8765.
  const { protocol, hostname } = window.location;
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("ws");
  if (fromQuery) return fromQuery;

  if (hostname !== "localhost" && hostname !== "127.0.0.1") {
    // Accessed remotely (e.g. via ngrok) — WebSocket is on port 8080 same host
    const wsProtocol = protocol === "https:" ? "wss:" : "ws:";
    return `${wsProtocol}//${hostname}:8080`;
  }
  return "ws://localhost:8765";
}

function useSharedWS(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    const connect = () => {
      const socket = new WebSocket(url);
      wsRef.current = socket;
      socket.onopen = () => {
        if (!cancelled) setWs(socket);
      };
      socket.onclose = () => {
        if (!cancelled) {
          wsRef.current = null;
          setWs(null);
          setTimeout(connect, 2000);
        }
      };
    };
    connect();
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [url]);

  return ws;
}

// ─── Nav pill ─────────────────────────────────────────────────────────────────
const NAV_LABELS: Record<AppView, string> = {
  speak: "Jarvis",
  graph: "Knowledge Graph",
  textbooks: "Textbooks",
};

function NavPill({ view, onChange }: { view: AppView; onChange: (v: AppView) => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      style={{
        display: "flex",
        padding: 4,
        background: "rgba(255,255,255,0.48)",
        backdropFilter: "blur(18px)",
        WebkitBackdropFilter: "blur(18px)",
        border: "1px solid rgba(255,255,255,0.85)",
        borderRadius: 100,
        boxShadow: "0 4px 24px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04)",
        position: "relative",
      }}
    >
      {(["speak", "graph", "textbooks"] as AppView[]).map((v) => {
        const active = view === v;
        return (
          <motion.button
            key={v}
            onClick={() => onChange(v)}
            style={{
              position: "relative",
              padding: "8px 22px",
              borderRadius: 100,
              background: "transparent",
              border: "none",
              cursor: "pointer",
              fontFamily: FONT,
              fontSize: 12.5,
              fontWeight: active ? 500 : 400,
              color: active ? TEXT : MUTED,
              letterSpacing: "0.015em",
              outline: "none",
              zIndex: 1,
              transition: "color 0.25s ease",
            }}
          >
            {active && (
              <motion.div
                layoutId="nav-pill-bg"
                style={{
                  position: "absolute",
                  inset: 0,
                  borderRadius: 100,
                  background: "rgba(0,0,0,0.055)",
                  zIndex: -1,
                }}
                transition={{ type: "spring", stiffness: 380, damping: 30 }}
              />
            )}
            {NAV_LABELS[v]}
          </motion.button>
        );
      })}
    </motion.div>
  );
}

// ─── Audio waveform bars ──────────────────────────────────────────────────────
function AudioWave() {
  const BARS = 9;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, height: 36 }}>
      {Array.from({ length: BARS }, (_, i) => {
        const norm = i / (BARS - 1);
        const peak = Math.sin(norm * Math.PI) * 22 + 7;
        return (
          <motion.div
            key={i}
            style={{ width: 3, borderRadius: 100, background: MINT }}
            animate={{
              height: [4, peak, peak * 0.35, peak * 0.82, 4],
              opacity: [0.3, 0.95, 0.45, 0.85, 0.3],
            }}
            transition={{
              duration: 1.05,
              delay: norm * 0.32,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        );
      })}
    </div>
  );
}

// ─── Glass pill controls ──────────────────────────────────────────────────────
function Controls({
  state,
  onStart,
  onStop,
}: {
  state: JarvisState;
  onStart: () => void;
  onStop: () => void;
}) {
  const isListening = state === "listening";
  const isIdle = state === "idle";
  const isBusy = state === "processing" || state === "speaking";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <AnimatePresence>
        {isListening && (
          <motion.button
            key="stop"
            initial={{ opacity: 0, scale: 0.85, x: 14 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            exit={{ opacity: 0, scale: 0.85, x: 14 }}
            transition={{ type: "spring", stiffness: 380, damping: 28 }}
            onClick={onStop}
            style={{
              display: "flex", alignItems: "center", gap: 9,
              padding: "15px 26px",
              background: DARK, border: "none", borderRadius: 100,
              cursor: "pointer", outline: "none",
            }}
          >
            <Square size={13} color="rgba(255,255,255,0.65)" strokeWidth={2.5} fill="rgba(255,255,255,0.65)" />
            <span style={{ fontFamily: FONT, fontSize: 13, fontWeight: 500, color: "rgba(255,255,255,0.65)", letterSpacing: "0.015em" }}>
              Stop
            </span>
          </motion.button>
        )}
      </AnimatePresence>

      <motion.button
        onClick={isIdle ? onStart : undefined}
        disabled={isBusy}
        style={{
          display: "flex", alignItems: "center", gap: 11,
          padding: "15px 28px",
          background: isListening ? MINT : "rgba(255,255,255,0.58)",
          backdropFilter: "blur(18px)",
          WebkitBackdropFilter: "blur(18px)",
          border: isListening ? "1px solid rgba(61,214,140,0.35)" : "1px solid rgba(255,255,255,0.85)",
          borderRadius: 100,
          cursor: isBusy ? "default" : isListening ? "default" : "pointer",
          boxShadow: isListening
            ? "0 6px 32px rgba(61,214,140,0.22), 0 1px 4px rgba(0,0,0,0.06)"
            : "0 4px 28px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.04)",
          outline: "none",
          position: "relative",
          overflow: "hidden",
        }}
        whileHover={isIdle ? { scale: 1.025, transition: { type: "spring", stiffness: 400, damping: 20 } } : {}}
        whileTap={isIdle ? { scale: 0.96 } : {}}
        animate={isListening ? { scale: [1, 1.018, 1] } : { scale: 1 }}
        transition={isListening ? { duration: 2.4, repeat: Infinity, ease: "easeInOut" } : { type: "spring", stiffness: 300, damping: 24 }}
      >
        <AnimatePresence>
          {state === "processing" && (
            <motion.div
              key="overlay"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              style={{
                position: "absolute", inset: 0, borderRadius: 100,
                display: "flex", alignItems: "center", justifyContent: "center",
                background: "rgba(255,255,255,0.58)", backdropFilter: "blur(18px)",
              }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                style={{ width: 16, height: 16, borderRadius: "50%", border: `2px solid ${MUTED}`, borderTopColor: "transparent" }}
              />
            </motion.div>
          )}
        </AnimatePresence>

        <motion.div animate={{ scale: isListening ? [1, 1.12, 1] : 1 }} transition={isListening ? { duration: 1.6, repeat: Infinity } : {}}>
          <Mic size={15} strokeWidth={2.2} color={isListening ? DARK : TEXT} />
        </motion.div>

        <AnimatePresence mode="wait">
          <motion.span
            key={state}
            initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }}
            transition={{ duration: 0.18 }}
            style={{ fontFamily: FONT, fontSize: 13.5, fontWeight: 500, color: isListening ? DARK : TEXT, letterSpacing: "0.012em", whiteSpace: "nowrap" }}
          >
            {isIdle ? "Tap to speak" : isListening ? "Listening…" : state === "processing" ? "Processing…" : "Speaking"}
          </motion.span>
        </AnimatePresence>
      </motion.button>
    </div>
  );
}

// ─── Speak view ───────────────────────────────────────────────────────────────
function SpeakView({ ws }: { ws: WebSocket | null }) {
  const [state, setState] = useState<JarvisState>("idle");
  const [streamedWords, setStreamedWords] = useState<string[]>([]);
  const [micError, setMicError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [streamedWords]);

  useEffect(() => {
    if (!ws) return;
    const handler = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "state") {
          const s = data.state;
          if (s === "idle") {
            setState("idle");
            setTimeout(() => setStreamedWords([]), 900);
          } else if (s === "listening" || s === "capturing") {
            setState("listening");
            setStreamedWords([]);
          } else if (s === "speaking") {
            setState("processing");
          } else if (s === "processing") {
            setState("processing");
          }
        } else if (data.type === "text") {
          setState("speaking");
          if (data.text) setStreamedWords((p) => [...p, data.text]);
        } else if (data.type === "error") {
          setState("idle");
          setMicError(data.message || "Error");
          setTimeout(() => setMicError(null), 4000);
        }
      } catch {}
    };
    ws.addEventListener("message", handler);
    return () => ws.removeEventListener("message", handler);
  }, [ws]);

  const handleStart = useCallback(async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    setMicError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      // Pick a supported audio format — WebM works in Chrome/Firefox; fallback to default
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "";
      const recorderOpts = mimeType ? { mimeType } : {};
      const recorder = new MediaRecorder(stream, recorderOpts);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = (reader.result as string).split(",")[1];
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "browser_audio", audio_b64: b64 }));
          }
        };
        reader.readAsDataURL(blob);
      };

      recorder.start();
      setState("listening");
    } catch (err) {
      setMicError("Microphone access denied. Please allow mic access in your browser.");
      setTimeout(() => setMicError(null), 5000);
    }
  }, [ws]);

  const handleStop = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setState("processing");
    } else if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop_listening" }));
    }
  }, [ws]);

  const showWords = streamedWords.length > 0 && state !== "idle";
  const showWave = state === "listening";
  const showHint = state === "idle" && streamedWords.length === 0;
  const showProcessing = state === "processing" && streamedWords.length === 0;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center" }}>
      {/* Ambient blob */}
      <motion.div
        style={{
          position: "absolute", borderRadius: "50%", pointerEvents: "none",
          width: 480, height: 480, top: "50%", left: "50%", x: "-50%", y: "-50%",
          background: "radial-gradient(circle, rgba(61,214,140,0.06) 0%, transparent 70%)",
        }}
        animate={{
          scale: state === "listening" ? [1, 1.1, 1] : [1, 1.04, 1],
          opacity: state === "listening" ? [0.6, 1, 0.6] : [0.2, 0.35, 0.2],
        }}
        transition={{ duration: state === "listening" ? 2 : 4, repeat: Infinity, ease: "easeInOut" }}
      />

      <main style={{
        flex: 1, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: streamedWords.length > 0 ? "flex-start" : "center",
        width: "100%", maxWidth: 860, padding: "48px 40px", boxSizing: "border-box",
        overflowY: "auto", overflowX: "hidden",
      }}>
        <AnimatePresence mode="wait">
          {showWave ? (
            <motion.div key="wave"
              initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.92 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}
            >
              <AudioWave />
              <motion.span
                animate={{ opacity: [0.45, 0.85, 0.45] }}
                transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
                style={{ fontSize: 13, fontFamily: FONT, color: MUTED, letterSpacing: "0.06em" }}
              >
                Listening…
              </motion.span>
            </motion.div>
          ) : showProcessing ? (
            <motion.div key="dots"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              style={{ display: "flex", gap: 8 }}
            >
              {[0, 1, 2].map((i) => (
                <motion.div key={i}
                  style={{ width: 7, height: 7, borderRadius: "50%", background: MUTED }}
                  animate={{ opacity: [0.25, 1, 0.25], y: [0, -6, 0] }}
                  transition={{ duration: 0.9, delay: i * 0.22, repeat: Infinity, ease: "easeInOut" }}
                />
              ))}
            </motion.div>
          ) : showWords ? (
            <motion.div key="words"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              exit={{ opacity: 0, filter: "blur(10px)" }}
              transition={{ duration: 0.6 }}
              style={{ width: "100%", textAlign: "center", fontFamily: FONT, fontSize: "clamp(1.55rem, 3.8vw, 3.1rem)", lineHeight: 1.3, color: TEXT, fontWeight: 400, padding: "0 24px", paddingBottom: "80px" }}
            >
              <TextEffect per="word" preset="blur">
                {streamedWords.join(" ")}
              </TextEffect>
              <div ref={messagesEndRef} style={{ height: 1 }} />
            </motion.div>
          ) : showHint ? (
            <motion.div key="hint"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.7 }}
            >
              <p style={{
                fontFamily: FONT, fontSize: "clamp(1.55rem, 3.8vw, 3.1rem)",
                color: "rgba(117,112,104,0.18)", fontWeight: 400, textAlign: "center", margin: 0, lineHeight: 1.3,
              }}>
                Say something…
              </p>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </main>

      <motion.footer
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
        style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 22, paddingBottom: 52 }}
      >
        {micError && (
          <motion.div
            initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{
              background: "rgba(230,80,80,0.1)", border: "1px solid rgba(230,80,80,0.25)",
              borderRadius: 10, padding: "8px 16px", maxWidth: 380, textAlign: "center",
              fontFamily: FONT, fontSize: 12, color: "#c04040",
            }}
          >
            {micError}
          </motion.div>
        )}
        <AnimatePresence>
          {state === "speaking" && (
            <motion.div
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }}
              transition={{ duration: 0.3 }}
              style={{ display: "flex", alignItems: "center", gap: 7 }}
            >
              <motion.div
                style={{ width: 6, height: 6, borderRadius: "50%", background: MINT }}
                animate={{ opacity: [0.4, 1, 0.4], scale: [0.85, 1.15, 0.85] }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
              />
              <span style={{ fontFamily: FONT, fontSize: 11, fontWeight: 500, letterSpacing: "0.2em", textTransform: "uppercase", color: MUTED }}>
                Speaking
              </span>
            </motion.div>
          )}
        </AnimatePresence>
        <Controls state={state} onStart={handleStart} onStop={handleStop} />
      </motion.footer>
    </div>
  );
}

// ─── App root ─────────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView] = useState<AppView>("speak");
  const ws = useSharedWS(getWsUrl());

  return (
    <div
      style={{
        minHeight: "100vh",
        height: "100vh",
        background: BG,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        fontFamily: FONT,
        overflow: "hidden",
      }}
    >
      {/* Header nav */}
      <header style={{ padding: "40px 0 0", flexShrink: 0 }}>
        <NavPill view={view} onChange={setView} />
      </header>

      {/* View content */}
      <AnimatePresence mode="wait">
        {view === "speak" ? (
          <motion.div
            key="speak"
            style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", width: "100%", position: "relative", minHeight: 0 }}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <SpeakView ws={ws} />
          </motion.div>
        ) : view === "graph" ? (
          <motion.div
            key="graph"
            style={{ flex: 1, display: "flex", flexDirection: "column", width: "100%", minHeight: 0 }}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <KnowledgeGraph ws={ws} />
          </motion.div>
        ) : (
          <motion.div
            key="textbooks"
            style={{ flex: 1, display: "flex", flexDirection: "column", width: "100%", minHeight: 0, overflowY: "auto" }}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <TextbooksView ws={ws} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
