import { useEffect, useMemo, useRef, useState } from "react";
import { ConnectionIndicator } from "./components/ConnectionIndicator";
import { PiPView } from "./components/PiPView";
import { SourceSelector, type CaptureSourceMode } from "./components/SourceSelector";
import { SubtitleOverlay } from "./components/SubtitleOverlay";
import { WebcamCapture } from "./components/WebcamCapture";
import { useASLWebSocket } from "./hooks/useASLWebSocket";
import { useScreenCapture } from "./hooks/useScreenCapture";
import { cropToParticipant, type GridPosition } from "./utils/streamHelpers";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/asl-stream";

export default function App() {
  const [sourceMode, setSourceMode] = useState<CaptureSourceMode>("camera");
  const [latestLandmarks, setLatestLandmarks] = useState<Float32Array | null>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const [cropEnabled, setCropEnabled] = useState(false);
  const [gridPosition, setGridPosition] = useState<GridPosition>("top-left");
  const [screenSourceStream, setScreenSourceStream] = useState<MediaStream | null>(null);
  const cropCleanupRef = useRef<(() => void) | null>(null);

  const requiresCamera = sourceMode === "camera" || sourceMode === "both";
  const requiresScreen = sourceMode === "screen" || sourceMode === "both";
  const screenCapture = useScreenCapture(requiresScreen);
  const singleModeEnabled = sourceMode !== "both";

  useEffect(() => {
    let cancelled = false;
    const startCamera = async () => {
      try {
        setCameraError(null);
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 }
          },
          audio: false
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        setCameraStream((previous) => {
          previous?.getTracks().forEach((track) => track.stop());
          return stream;
        });
      } catch {
        if (!cancelled) {
          setCameraError("Camera permission denied or unavailable.");
          setCameraStream((previous) => {
            previous?.getTracks().forEach((track) => track.stop());
            return null;
          });
        }
      }
    };

    if (requiresCamera && !cameraStream) {
      void startCamera();
    }
    if (!requiresCamera && cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      setCameraStream(null);
    }

    return () => {
      cancelled = true;
    };
  }, [cameraStream, requiresCamera]);

  useEffect(() => {
    cropCleanupRef.current?.();
    cropCleanupRef.current = null;

    const base = screenCapture.stream;
    if (!base) {
      setScreenSourceStream(null);
      return;
    }

    if (!cropEnabled) {
      setScreenSourceStream(base);
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const cropped = await cropToParticipant(base, gridPosition);
        if (cancelled) {
          cropped.cleanup();
          return;
        }
        cropCleanupRef.current = cropped.cleanup;
        setScreenSourceStream(cropped.stream);
      } catch {
        setScreenSourceStream(base);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [cropEnabled, gridPosition, screenCapture.stream]);

  useEffect(
    () => () => {
      cropCleanupRef.current?.();
      cropCleanupRef.current = null;
      cameraStream?.getTracks().forEach((track) => track.stop());
    },
    [cameraStream]
  );

  const { connectionState, gloss, confidence, alternatives, rawTranscript, refinedText } = useASLWebSocket(
    singleModeEnabled ? latestLandmarks : null,
    WS_URL,
    { enabled: singleModeEnabled }
  );

  const websocketHelp = useMemo(() => {
    return "Landmarks are extracted directly in-browser and streamed to backend over WebSocket.";
  }, []);

  const selectedStream = sourceMode === "camera" ? cameraStream : screenSourceStream;
  const selectedError = sourceMode === "camera" ? cameraError : screenCapture.error;

  return (
    <main className="app-shell">
      <header className="top-bar">
        <h1>ASL Meeting Copilot</h1>
        {sourceMode === "both" ? (
          <span className="helper-text">Dual stream mode: local + remote</span>
        ) : (
          <ConnectionIndicator state={connectionState} />
        )}
      </header>
      <SourceSelector value={sourceMode} onChange={setSourceMode} />

      <section className="layout-grid">
        <div className="video-card">
          {sourceMode === "both" ? (
            <PiPView
              localStream={cameraStream}
              remoteStream={screenSourceStream}
              wsUrl={WS_URL}
              localError={cameraError}
              remoteError={screenCapture.error}
            />
          ) : (
            <>
              <WebcamCapture
                stream={selectedStream}
                onLandmarks={setLatestLandmarks}
                label={sourceMode === "camera" ? "LOCAL" : "REMOTE"}
                externalError={selectedError}
              />
              <SubtitleOverlay text={refinedText || gloss} confidence={confidence ?? null} />
            </>
          )}
        </div>

        <aside className="control-card">
          <h2>Live Recognition</h2>

          {(sourceMode === "screen" || sourceMode === "both") && (
            <div className="screen-controls">
              <button type="button" className="btn" onClick={screenCapture.requestCapture}>
                Select Zoom/Meet Window
              </button>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={cropEnabled}
                  onChange={(event) => setCropEnabled(event.target.checked)}
                />
                <span>Crop to participant tile</span>
              </label>
              <label className="select-row">
                <span>Tile quadrant</span>
                <select value={gridPosition} onChange={(event) => setGridPosition(event.target.value as GridPosition)}>
                  <option value="top-left">Top Left</option>
                  <option value="top-right">Top Right</option>
                  <option value="bottom-left">Bottom Left</option>
                  <option value="bottom-right">Bottom Right</option>
                </select>
              </label>
            </div>
          )}

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

