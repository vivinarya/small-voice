import { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";

const FONT = "'Plus Jakarta Sans', sans-serif";
const TEXT = "#757068";
const MUTED = "#a8a09a";
const MINT = "#3DD68C";
const DARK = "#1e1d1b";

interface GNode {
  id: string;
  label: string;
  desc: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

interface GEdge {
  source: string;
  target: string;
}

let NODES_RAW = [
  { id: "jarvis",      label: "JARVIS",            desc: "Core intelligence. Orchestrates all subsystems and manages real-time interaction." },
  { id: "voice",       label: "Voice Recognition", desc: "Real-time voice pattern analysis. Speech-to-intent conversion at 99.2% accuracy." },
  { id: "neural",      label: "Neural Networks",   desc: "Deep learning inference for reasoning, prediction, and language understanding." },
  { id: "memory",      label: "Memory Banks",      desc: "Episodic and semantic memory. Full context retained across every session." },
  { id: "security",    label: "Security",          desc: "Multi-layer threat detection. Zero breaches in 847 operational days." },
  { id: "env",         label: "Environmental Scan",desc: "Sensor fusion monitoring ambient conditions, spatial awareness, threat vectors." },
  { id: "quantum",     label: "Quantum Core",      desc: "Quantum co-processor handling parallel computation on complex optimizations." },
  { id: "user",        label: "User Profile",       desc: "Behavioral model, preferences, and biometric data. Updated continuously." },
  { id: "calendar",    label: "Scheduling",         desc: "Predictive scheduling engine with conflict resolution and priority balancing." },
  { id: "research",    label: "Research DB",        desc: "Indexed knowledge base with real-time integration and source verification." },
  { id: "comms",       label: "Communications",     desc: "Message routing, drafting assistance, and relationship graph management." },
  { id: "diagnostics", label: "Diagnostics",        desc: "Real-time performance monitoring. All systems currently nominal." },
  { id: "weather",     label: "Weather",            desc: "Hyperlocal atmospheric modeling. 72-hour predictive accuracy at 94.7%." },
  { id: "encryption",  label: "Encryption",         desc: "AES-256 and quantum-resistant cryptographic protocols on all channels." },
];

let EDGES: GEdge[] = [
  { source: "jarvis",      target: "voice" },
  { source: "jarvis",      target: "neural" },
  { source: "jarvis",      target: "security" },
  { source: "jarvis",      target: "user" },
  { source: "jarvis",      target: "diagnostics" },
  { source: "jarvis",      target: "env" },
  { source: "neural",      target: "memory" },
  { source: "neural",      target: "research" },
  { source: "neural",      target: "voice" },
  { source: "neural",      target: "quantum" },
  { source: "memory",      target: "user" },
  { source: "memory",      target: "research" },
  { source: "security",    target: "encryption" },
  { source: "security",    target: "comms" },
  { source: "env",         target: "quantum" },
  { source: "env",         target: "weather" },
  { source: "user",        target: "calendar" },
  { source: "user",        target: "comms" },
  { source: "calendar",    target: "comms" },
  { source: "diagnostics", target: "quantum" },
  { source: "weather",     target: "user" },
  { source: "research",    target: "comms" },
];

function countDegrees() {
  const d = new Map<string, number>();
  EDGES.forEach(e => {
    d.set(e.source, (d.get(e.source) || 0) + 1);
    d.set(e.target, (d.get(e.target) || 0) + 1);
  });
  return d;
}

function buildNodes(W: number, H: number): GNode[] {
  const cx = W / 2, cy = H / 2;
  const deg = countDegrees();
  return NODES_RAW.map((d, i) => {
    const angle = (i / NODES_RAW.length) * Math.PI * 2 - Math.PI / 2;
    const spread = 155 + (Math.random() - 0.5) * 30;
    const connections = deg.get(d.id) || 1;
    return {
      ...d,
      x: d.id === "jarvis" ? cx : cx + Math.cos(angle) * spread,
      y: d.id === "jarvis" ? cy : cy + Math.sin(angle) * spread,
      vx: 0,
      vy: 0,
      r: d.id === "jarvis" ? 13 : Math.min(4.5 + connections * 1.1, 10),
    };
  });
}

function tickForces(nodes: GNode[], edges: GEdge[], W: number, H: number, heat: number) {
  const cx = W / 2, cy = H / 2;

  nodes.forEach(n => { n.vx *= 0.87; n.vy *= 0.87; });

  // Repulsion
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x || 0.01;
      const dy = b.y - a.y || 0.01;
      const d2 = Math.max(dx * dx + dy * dy, 1);
      const d = Math.sqrt(d2);
      const f = (12000 / d2) * heat;
      a.vx -= (dx / d) * f;
      a.vy -= (dy / d) * f;
      b.vx += (dx / d) * f;
      b.vy += (dy / d) * f;
    }
  }

  // Spring along edges
  const nMap = new Map(nodes.map(n => [n.id, n]));
  edges.forEach(e => {
    const a = nMap.get(e.source), b = nMap.get(e.target);
    if (!a || !b) return;
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
    const f = 0.048 * (d - 135) * heat;
    a.vx += (dx / d) * f; a.vy += (dy / d) * f;
    b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
  });

  // Center gravity
  nodes.forEach(n => {
    n.vx += (cx - n.x) * 0.016 * heat;
    n.vy += (cy - n.y) * 0.016 * heat;
  });

  // Integrate + bounds
  const pad = 55;
  nodes.forEach(n => {
    n.x = Math.max(pad, Math.min(W - pad, n.x + n.vx));
    n.y = Math.max(pad, Math.min(H - pad, n.y + n.vy));
  });
}

function render(
  ctx: CanvasRenderingContext2D,
  nodes: GNode[],
  edges: GEdge[],
  selected: string | null,
  hovered: string | null,
  W: number,
  H: number,
  t: number
) {
  ctx.clearRect(0, 0, W, H);

  const nMap = new Map(nodes.map(n => [n.id, n]));

  const connectedTo = (id: string) =>
    edges.some(e => (e.source === selected && e.target === id) || (e.target === selected && e.source === id));

  // ── Edges ──────────────────────────────────────────────────
  edges.forEach(e => {
    const a = nMap.get(e.source), b = nMap.get(e.target);
    if (!a || !b) return;
    const lit = selected && (e.source === selected || e.target === selected);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = lit ? "rgba(61,214,140,0.42)" : "rgba(117,112,104,0.11)";
    ctx.lineWidth = lit ? 1.3 : 0.65;
    ctx.stroke();
  });

  // ── Nodes ──────────────────────────────────────────────────
  nodes.forEach(n => {
    const isSel = n.id === selected;
    const isHov = n.id === hovered;
    const isConn = !!selected && connectedTo(n.id);
    const dim = !!selected && !isSel && !isConn;
    const nr = isHov && !isSel ? n.r + 2 : n.r;

    ctx.save();
    ctx.globalAlpha = dim ? 0.22 : 1;

    // Jarvis pulse ring
    if (n.id === "jarvis") {
      const pulse = Math.sin(t * 0.0025) * 0.5 + 0.5;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 7 + pulse * 5, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(61,214,140,${0.08 + pulse * 0.07})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (isSel) { ctx.shadowColor = MINT; ctx.shadowBlur = 16; }

    ctx.beginPath();
    ctx.arc(n.x, n.y, nr, 0, Math.PI * 2);

    if (isSel)                  ctx.fillStyle = MINT;
    else if (n.id === "jarvis") ctx.fillStyle = "rgba(110,104,98,0.9)";
    else if (isHov)             ctx.fillStyle = "rgba(100,95,88,0.92)";
    else if (isConn)            ctx.fillStyle = "rgba(80,130,105,0.68)";
    else                        ctx.fillStyle = "rgba(160,152,144,0.68)";

    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(255,255,255,0.88)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.restore();
  });

  // ── Labels (second pass so they render on top) ─────────────
  nodes.forEach(n => {
    const isSel = n.id === selected;
    const isHov = n.id === hovered;
    const isConn = !!selected && connectedTo(n.id);
    const dim = !!selected && !isSel && !isConn;
    const nr = isHov && !isSel ? n.r + 2 : n.r;

    const alpha = dim ? 0.06 : isSel ? 0.95 : isHov || isConn ? 0.7 : 0.55;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = `${isSel ? 500 : 400} ${isSel ? 11 : 10}px ${FONT}`;
    ctx.fillStyle = isSel ? DARK : TEXT;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(n.label, n.x, n.y + nr + 5);
    ctx.restore();
  });
}

export function KnowledgeGraph() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<GNode[]>([]);
  const selRef = useRef<string | null>(null);
  const hovRef = useRef<string | null>(null);
  const dragRef = useRef<string | null>(null);
  const rafRef = useRef<number>(0);
  const dimRef = useRef({ W: 0, H: 0 });
  const wsRef = useRef<WebSocket | null>(null);
  const [selNode, setSelNode] = useState<GNode | null>(null);
  const [forceRender, setForceRender] = useState(0);

  useEffect(() => {
    const wrap = wrapRef.current!;
    const canvas = canvasRef.current!;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth;
    const H = wrap.clientHeight;
    dimRef.current = { W, H };
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);

    const reinitGraph = () => {
      nodesRef.current = buildNodes(W, H);
      for (let i = 0; i < 260; i++) {
        tickForces(nodesRef.current, EDGES, W, H, 1 - i / 300);
      }
      setForceRender(Date.now());
    };

    reinitGraph();

    // Connect to WebSocket to fetch real nodes
    wsRef.current = new WebSocket("ws://localhost:8765");
    wsRef.current.onopen = () => {
      wsRef.current?.send(JSON.stringify({ type: "get_graph" }));
    };
    wsRef.current.onmessage = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "graph_data" && data.nodes && data.nodes.length > 0) {
           NODES_RAW = data.nodes;
           if (!NODES_RAW.find(n => n.id === "jarvis")) {
              NODES_RAW.unshift({ id: "jarvis", label: "JARVIS", desc: "Core Intelligence Vault." });
           }
           EDGES = NODES_RAW.filter(n => n.id !== "jarvis").map(n => ({ source: "jarvis", target: n.id }));
           reinitGraph();
        }
      } catch (err) {}
    };

    // Live loop
    const frame = (t: number) => {
      tickForces(nodesRef.current, EDGES, W, H, 0.12);
      render(ctx, nodesRef.current, EDGES, selRef.current, hovRef.current, W, H, t);
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);

    // ── Mouse events ──────────────────────────────────────
    const hitTest = (mx: number, my: number) =>
      nodesRef.current.find((n: GNode) => {
        const dx = n.x - mx, dy = n.y - my;
        return Math.sqrt(dx * dx + dy * dy) <= n.r + 7;
      });

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      if (dragRef.current) {
        const n = nodesRef.current.find((nd: GNode) => nd.id === dragRef.current);
        if (n) { n.x = mx; n.y = my; n.vx = 0; n.vy = 0; }
        return;
      }
      const hit = hitTest(mx, my);
      hovRef.current = hit?.id || null;
      canvas.style.cursor = hit ? "pointer" : "default";
    };

    const onDown = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
      if (hit) dragRef.current = hit.id;
    };

    const onUp = () => { dragRef.current = null; };

    const onClick = (e: MouseEvent) => {
      if (dragRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
      if (hit) {
        const next = hit.id === selRef.current ? null : hit.id;
        selRef.current = next;
        setSelNode(next ? hit : null);
      } else {
        selRef.current = null;
        setSelNode(null);
      }
    };

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("click", onClick);

    return () => {
      cancelAnimationFrame(rafRef.current);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("click", onClick);
    };
  }, []);

  const connCount = selNode
    ? EDGES.filter(e => e.source === selNode.id || e.target === selNode.id).length
    : 0;

  return (
    <div
      ref={wrapRef}
      style={{ position: "relative", flex: 1, overflow: "hidden", minHeight: 0 }}
    >
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      />

      {/* Hint */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8, duration: 0.7 }}
        style={{
          position: "absolute",
          top: 20,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: FONT,
          fontSize: 11,
          color: "rgba(168,160,154,0.55)",
          pointerEvents: "none",
          letterSpacing: "0.06em",
        }}
      >
        Click a node to explore · Drag to rearrange
      </motion.p>

      {/* Node count */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.7 }}
        style={{
          position: "absolute",
          bottom: 24,
          right: 28,
          fontFamily: FONT,
          fontSize: 10,
          color: "rgba(168,160,154,0.4)",
          pointerEvents: "none",
          letterSpacing: "0.05em",
        }}
      >
        {NODES_RAW.length} nodes · {EDGES.length} connections
      </motion.p>

      {/* Selected node panel */}
      <AnimatePresence>
        {selNode && (
          <motion.div
            key={selNode.id}
            initial={{ opacity: 0, y: 10, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.97 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            style={{
              position: "absolute",
              bottom: 28,
              left: 28,
              maxWidth: 248,
              padding: "16px 18px",
              background: "rgba(255,255,255,0.62)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              border: "1px solid rgba(255,255,255,0.88)",
              borderRadius: 14,
              boxShadow: "0 4px 28px rgba(0,0,0,0.07)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 8 }}>
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: MINT, flexShrink: 0 }} />
              <p style={{ fontFamily: FONT, fontSize: 12, fontWeight: 600, color: TEXT, margin: 0, letterSpacing: "0.01em" }}>
                {selNode.label}
              </p>
            </div>
            <textarea
              key={selNode.id}
              defaultValue={selNode.desc}
              style={{ fontFamily: FONT, fontSize: 11, fontWeight: 400, color: MUTED, margin: "0 0 10px", lineHeight: 1.6, width: "100%", minHeight: 120, background: "rgba(0,0,0,0.03)", border: "1px solid rgba(0,0,0,0.08)", borderRadius: 6, padding: "8px", boxSizing: "border-box", resize: "none" }}
              onBlur={(e: React.FocusEvent<HTMLTextAreaElement>) => {
                 if (e.target.value !== selNode.desc && wsRef.current && selNode.id !== "jarvis") {
                    wsRef.current.send(JSON.stringify({ type: "update_node", id: selNode.id, content: e.target.value }));
                    selNode.desc = e.target.value;
                 }
              }}
            />
            <p style={{ fontFamily: FONT, fontSize: 10, color: "rgba(168,160,154,0.6)", margin: 0, letterSpacing: "0.05em" }}>
              {connCount} connection{connCount !== 1 ? "s" : ""}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
