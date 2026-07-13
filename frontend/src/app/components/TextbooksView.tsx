import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Upload, FileText, CheckCircle, XCircle, Loader, BookOpen, Hash } from "lucide-react";

const FONT = "'Plus Jakarta Sans', sans-serif";
const TEXT = "#757068";
const MUTED = "#a8a09a";
const MINT = "#3DD68C";

type UploadStatus =
  | { phase: "idle" }
  | { phase: "uploading"; filename: string }
  | { phase: "indexing"; lines: string[] }
  | { phase: "done"; success: boolean; message: string }
  | { phase: "error"; message: string };

interface IndexedDoc {
  source: string;
  page_count: number;
  chunk_count: number;
}

function ProgressLog({ lines }: { lines: string[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  return (
    <div
      style={{
        background: "rgba(30,29,27,0.04)",
        border: "1px solid rgba(0,0,0,0.07)",
        borderRadius: 10,
        padding: "12px 14px",
        maxHeight: 160,
        overflowY: "auto",
        fontFamily: "monospace",
        fontSize: 11,
        color: MUTED,
        lineHeight: 1.7,
      }}
    >
      {lines.map((l, i) => (
        <div
          key={i}
          style={{
            color: l.includes("ERROR") ? "#e05b5b" : l.includes("successfully") ? "#3DD68C" : MUTED,
          }}
        >
          {l}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

export function TextbooksView({ ws }: { ws: WebSocket | null }) {
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState<UploadStatus>({ phase: "idle" });
  const [indexedDocs, setIndexedDocs] = useState<IndexedDoc[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const connected = ws?.readyState === WebSocket.OPEN;

  // Fetch the already-indexed docs from the backend
  const fetchIndexedDocs = useCallback(() => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "get_indexed_docs" }));
    }
  }, [ws]);

  // Fetch whenever ws object changes (new connection established)
  useEffect(() => {
    if (!ws) return;
    // ws just became a new open socket — request the doc list
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "get_indexed_docs" }));
    } else {
      const onOpen = () => ws.send(JSON.stringify({ type: "get_indexed_docs" }));
      ws.addEventListener("open", onOpen);
      return () => ws.removeEventListener("open", onOpen);
    }
  }, [ws]);

  useEffect(() => {
    if (!ws) return;

    // Reset stuck upload state on reconnect
    setStatus((prev) => (prev.phase === "uploading" ? { phase: "idle" } : prev));

    const handler = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);

        if (data.type === "indexed_docs") {
          // Full replacement — backend is the source of truth
          setIndexedDocs(data.docs ?? []);
          return;
        }

        if (data.type === "upload_result") {
          clearTimeout((ws as any)._uploadTimeout);
          if (data.success) {
            setStatus({ phase: "indexing", lines: ["Upload complete. Building knowledge index…"] });
            ws.send(JSON.stringify({ type: "rebuild_index" }));
          } else {
            setStatus({ phase: "error", message: data.message });
          }
          return;
        }

        if (data.type === "index_progress") {
          setStatus((prev) =>
            prev.phase === "indexing"
              ? { phase: "indexing", lines: [...prev.lines, data.message] }
              : { phase: "indexing", lines: [data.message] }
          );
          return;
        }

        if (data.type === "index_done") {
          setStatus({ phase: "done", success: data.success, message: data.message });
          if (data.success) {
            // Backend will push indexed_docs broadcast — but also request explicitly
            // in case the broadcast races with the component mounting
            setTimeout(fetchIndexedDocs, 500);
          }
          setTimeout(() => setStatus({ phase: "idle" }), 5000);
          return;
        }
      } catch {}
    };

    ws.addEventListener("message", handler);
    return () => ws.removeEventListener("message", handler);
  }, [ws, fetchIndexedDocs]);

  const processFile = useCallback(
    (file: File) => {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setStatus({ phase: "error", message: "Only PDF files are supported." });
        return;
      }
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        setStatus({ phase: "error", message: "Not connected — backend is still starting. Try again in a moment." });
        return;
      }
      setStatus({ phase: "uploading", filename: file.name });
      const reader = new FileReader();
      reader.onload = (ev) => {
        const b64 = (ev.target?.result as string).split(",")[1];
        ws.send(JSON.stringify({ type: "upload_pdf", filename: file.name, content_b64: b64 }));

        // Timeout: if no upload_result within 15s, show error
        const timeout = setTimeout(() => {
          setStatus((prev) =>
            prev.phase === "uploading"
              ? { phase: "error", message: "Upload timed out. Check that the backend is running and try again." }
              : prev
          );
        }, 15000);
        (ws as any)._uploadTimeout = timeout;
      };
      reader.readAsDataURL(file);
    },
    [ws]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = "";
  };

  const isBusy = status.phase === "uploading" || status.phase === "indexing";
  const isDisabled = isBusy || !connected;

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "40px 24px 48px",
        overflowY: "auto",
        gap: 24,
      }}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        style={{ textAlign: "center", maxWidth: 480 }}
      >
        <p
          style={{
            fontFamily: FONT,
            fontSize: 13,
            color: MUTED,
            margin: 0,
            lineHeight: 1.6,
            letterSpacing: "0.01em",
          }}
        >
          Upload any PDF — textbooks, notes, manuals, research papers — and Jarvis
          will index every page so you can ask questions about it.
        </p>

        {/* Connection badge */}
        <div
          style={{
            marginTop: 10,
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            padding: "4px 10px",
            borderRadius: 100,
            background: connected ? "rgba(61,214,140,0.08)" : "rgba(230,80,80,0.08)",
            border: `1px solid ${connected ? "rgba(61,214,140,0.2)" : "rgba(230,80,80,0.2)"}`,
          }}
        >
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: connected ? MINT : "#e05b5b",
            }}
          />
          <span
            style={{
              fontFamily: FONT,
              fontSize: 10.5,
              letterSpacing: "0.05em",
              color: connected ? "#3aa870" : "#c04040",
            }}
          >
            {connected ? "Backend connected" : "Backend offline — retrying…"}
          </span>
        </div>
      </motion.div>

      {/* Upload card */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        style={{
          width: "100%",
          maxWidth: 480,
          background: "rgba(255,255,255,0.52)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: "1px solid rgba(255,255,255,0.88)",
          borderRadius: 18,
          boxShadow: "0 4px 32px rgba(0,0,0,0.06)",
          padding: "24px",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        {/* Drop zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => !isDisabled && fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${
              !connected
                ? "rgba(168,160,154,0.2)"
                : dragOver
                ? MINT
                : "rgba(168,160,154,0.35)"
            }`,
            borderRadius: 12,
            padding: "40px 24px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
            cursor: isDisabled ? "not-allowed" : "pointer",
            background: !connected
              ? "rgba(0,0,0,0.01)"
              : dragOver
              ? "rgba(61,214,140,0.04)"
              : "rgba(0,0,0,0.015)",
            transition: "border-color 0.2s, background 0.2s",
            opacity: !connected ? 0.5 : 1,
          }}
        >
          <motion.div
            animate={dragOver ? { scale: 1.12 } : { scale: 1 }}
            transition={{ type: "spring", stiffness: 400, damping: 20 }}
          >
            <Upload size={24} color={dragOver ? MINT : MUTED} strokeWidth={1.8} />
          </motion.div>
          <div style={{ textAlign: "center" }}>
            <p
              style={{
                fontFamily: FONT,
                fontSize: 13,
                color: TEXT,
                margin: "0 0 3px",
                fontWeight: 500,
              }}
            >
              Drop any PDF here
            </p>
            <p style={{ fontFamily: FONT, fontSize: 11.5, color: MUTED, margin: 0 }}>
              Textbooks · Notes · Manuals · Research papers · Anything
            </p>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          style={{ display: "none" }}
          onChange={onFileChange}
        />

        {/* Status area */}
        <AnimatePresence mode="wait">
          {status.phase === "uploading" && (
            <motion.div
              key="uploading"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              style={{ display: "flex", alignItems: "center", gap: 9 }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <Loader size={14} color={MUTED} />
              </motion.div>
              <span style={{ fontFamily: FONT, fontSize: 12.5, color: MUTED }}>
                Uploading {status.filename}…
              </span>
            </motion.div>
          )}
          {status.phase === "indexing" && (
            <motion.div
              key="indexing"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{ display: "flex", flexDirection: "column", gap: 10 }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
                >
                  <Loader size={14} color={MINT} />
                </motion.div>
                <span
                  style={{
                    fontFamily: FONT,
                    fontSize: 12.5,
                    color: TEXT,
                    fontWeight: 500,
                  }}
                >
                  Building knowledge index…
                </span>
              </div>
              <ProgressLog lines={status.lines} />
            </motion.div>
          )}
          {status.phase === "done" && (
            <motion.div
              key="done"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              style={{ display: "flex", alignItems: "center", gap: 9 }}
            >
              {status.success ? (
                <CheckCircle size={15} color={MINT} />
              ) : (
                <XCircle size={15} color="#e05b5b" />
              )}
              <span
                style={{
                  fontFamily: FONT,
                  fontSize: 12.5,
                  color: status.success ? MINT : "#e05b5b",
                  fontWeight: 500,
                }}
              >
                {status.message}
              </span>
            </motion.div>
          )}
          {status.phase === "error" && (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{ display: "flex", alignItems: "center", gap: 9 }}
            >
              <XCircle size={15} color="#e05b5b" />
              <span style={{ fontFamily: FONT, fontSize: 12.5, color: "#e05b5b" }}>
                {status.message}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Indexed documents — loaded from backend, survives navigation */}
      <AnimatePresence>
        {indexedDocs.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            style={{ width: "100%", maxWidth: 480 }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 10,
              }}
            >
              <p
                style={{
                  fontFamily: FONT,
                  fontSize: 11,
                  color: MUTED,
                  letterSpacing: "0.06em",
                  margin: 0,
                }}
              >
                KNOWLEDGE BASE
              </p>
              <span
                style={{
                  fontFamily: FONT,
                  fontSize: 10,
                  color: "rgba(61,214,140,0.8)",
                  background: "rgba(61,214,140,0.08)",
                  border: "1px solid rgba(61,214,140,0.2)",
                  padding: "2px 8px",
                  borderRadius: 100,
                  letterSpacing: "0.04em",
                }}
              >
                {indexedDocs.length} {indexedDocs.length === 1 ? "document" : "documents"}
              </span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {indexedDocs.map((doc) => (
                <motion.div
                  key={doc.source}
                  layout
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "12px 14px",
                    background: "rgba(255,255,255,0.45)",
                    backdropFilter: "blur(12px)",
                    border: "1px solid rgba(255,255,255,0.82)",
                    borderRadius: 12,
                  }}
                >
                  <FileText size={14} color={MINT} strokeWidth={1.8} />

                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p
                      style={{
                        fontFamily: FONT,
                        fontSize: 12.5,
                        fontWeight: 500,
                        color: TEXT,
                        margin: "0 0 4px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {doc.source}
                    </p>
                    <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                      {/* Page count badge */}
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 4,
                        }}
                      >
                        <BookOpen size={10} color={MUTED} strokeWidth={2} />
                        <span
                          style={{
                            fontFamily: FONT,
                            fontSize: 10.5,
                            color: MUTED,
                          }}
                        >
                          {doc.page_count} {doc.page_count === 1 ? "page" : "pages"}
                        </span>
                      </div>

                      {/* Chunk count badge */}
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 4,
                        }}
                      >
                        <Hash size={10} color={MUTED} strokeWidth={2} />
                        <span
                          style={{
                            fontFamily: FONT,
                            fontSize: 10.5,
                            color: MUTED,
                          }}
                        >
                          {doc.chunk_count} chunks indexed
                        </span>
                      </div>
                    </div>
                  </div>

                  <CheckCircle size={13} color="rgba(61,214,140,0.7)" />
                </motion.div>
              ))}
            </div>

            {/* Total stats footer */}
            <p
              style={{
                fontFamily: FONT,
                fontSize: 10.5,
                color: "rgba(168,160,154,0.55)",
                margin: "10px 0 0 2px",
                letterSpacing: "0.03em",
              }}
            >
              {indexedDocs.reduce((s, d) => s + d.chunk_count, 0).toLocaleString()} total chunks ·{" "}
              {indexedDocs.reduce((s, d) => s + d.page_count, 0)} total pages indexed
            </p>
          </motion.div>
        )}

        {/* Empty state — connected but no docs yet */}
        {indexedDocs.length === 0 && connected && status.phase === "idle" && (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ delay: 0.3 }}
            style={{ textAlign: "center" }}
          >
            <p
              style={{
                fontFamily: FONT,
                fontSize: 12,
                color: "rgba(168,160,154,0.45)",
                margin: 0,
              }}
            >
              No documents indexed yet — upload a PDF to get started
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
