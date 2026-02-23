import { useEffect, useMemo, useRef, useState } from "react";
import { ConnectionIndicator } from "./components/ConnectionIndicator";
import { SubtitleOverlay } from "./components/SubtitleOverlay";
import { useWebSocket } from "./hooks/useWebSocket";
import { StreamMessage } from "./types/stream";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/asl-stream";

function isPrediction(message: StreamMessage | null): message is Extract<StreamMessage, { type: "prediction" }> {
  return message?.type === "prediction";
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [subtitle, setSubtitle] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [lastRawPrediction, setLastRawPrediction] = useState("n/a");
  const [lastDetail, setLastDetail] = useState("Waiting for stream data...");

  const { connectionState, latestMessage, sendJson } = useWebSocket({ url: WS_URL });

  useEffect(() => {
    let stream: MediaStream | null = null;
    async function startWebcam() {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, frameRate: 30 },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    }
    startWebcam().catch(() => {
      setLastDetail("Webcam access failed. Please allow camera permission.");
    });
    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!latestMessage) return;
    if (latestMessage.type === "buffering") {
      setLastDetail(latestMessage.detail);
      return;
    }
    if (latestMessage.type === "error") {
      setLastDetail("Backend stream error.");
      return;
    }
    if (latestMessage.type === "connection") {
      setLastDetail(`WebSocket ${latestMessage.status}`);
      return;
    }
    if (!isPrediction(latestMessage)) return;

    setSubtitle(latestMessage.refined_text || latestMessage.raw_prediction);
    setConfidence(latestMessage.confidence);
    setLastRawPrediction(latestMessage.raw_prediction);
    setLastDetail(`Frame ${latestMessage.frame_index} @ ${latestMessage.timestamp}`);

    if (speechEnabled && latestMessage.audio_url) {
      const src = latestMessage.audio_url.startsWith("http")
        ? latestMessage.audio_url
        : `${API_BASE_URL}${latestMessage.audio_url}`;
      const audio = new Audio(src);
      void audio.play().catch(() => undefined);
    }
  }, [latestMessage, speechEnabled]);

  const websocketHelp = useMemo(() => {
    return "Use ml/infer.py to send MediaPipe landmarks to this backend WebSocket.";
  }, []);

  const sendTestFrame = () => {
    const landmarks = Array.from({ length: 63 }, () => Math.random() * 2 - 1);
    const sent = sendJson({
      type: "landmarks",
      landmarks,
      refine_text: true,
      speak: speechEnabled
    });
    if (!sent) {
      setLastDetail("WebSocket is not open yet.");
    }
  };

  return (
    <main className="app-shell">
      <header className="top-bar">
        <h1>ASL Meeting Copilot</h1>
        <ConnectionIndicator state={connectionState} />
      </header>

      <section className="layout-grid">
        <div className="video-card">
          <video ref={videoRef} autoPlay muted playsInline className="video-preview" />
          <SubtitleOverlay text={subtitle} confidence={confidence} />
        </div>

        <aside className="control-card">
          <h2>Session Controls</h2>
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={speechEnabled}
              onChange={(event) => setSpeechEnabled(event.target.checked)}
            />
            <span>Enable speech playback</span>
          </label>

          <button className="btn" onClick={sendTestFrame}>
            Send test frame to backend
          </button>

          <div className="meta-block">
            <p>
              <strong>Last raw prediction:</strong> {lastRawPrediction}
            </p>
            <p>
              <strong>Status:</strong> {lastDetail}
            </p>
            <p className="helper-text">{websocketHelp}</p>
          </div>
        </aside>
      </section>
    </main>
  );
}

