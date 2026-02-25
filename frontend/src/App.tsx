import { useMemo, useState } from "react";
import { ConnectionIndicator } from "./components/ConnectionIndicator";
import { SubtitleOverlay } from "./components/SubtitleOverlay";
import { WebcamCapture } from "./components/WebcamCapture";
import { useASLWebSocket } from "./hooks/useASLWebSocket";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/asl-stream";

export default function App() {
  const [latestLandmarks, setLatestLandmarks] = useState<Float32Array | null>(null);
  const { connectionState, gloss, confidence, alternatives, rawTranscript, refinedText } = useASLWebSocket(
    latestLandmarks,
    WS_URL
  );

  const websocketHelp = useMemo(() => {
    return "Landmarks are extracted directly in-browser and streamed to backend over WebSocket.";
  }, []);

  return (
    <main className="app-shell">
      <header className="top-bar">
        <h1>ASL Meeting Copilot</h1>
        <ConnectionIndicator state={connectionState} />
      </header>

      <section className="layout-grid">
        <div className="video-card">
          <WebcamCapture onLandmarks={setLatestLandmarks} />
          <SubtitleOverlay text={refinedText || gloss} confidence={confidence ?? null} />
        </div>

        <aside className="control-card">
          <h2>Live Recognition</h2>

          <div className="meta-block">
            <p>
              <strong>Latest gloss:</strong> {gloss || "UNSURE"}
            </p>
            <p>
              <strong>Confidence:</strong> {(confidence * 100).toFixed(1)}%
            </p>
            <p>
              <strong>Raw transcript:</strong> {rawTranscript || "n/a"}
            </p>
            <p>
              <strong>Alternatives:</strong> {alternatives.length ? alternatives.join(", ") : "n/a"}
            </p>
            <p className="helper-text">{websocketHelp}</p>
          </div>
        </aside>
      </section>
    </main>
  );
}

