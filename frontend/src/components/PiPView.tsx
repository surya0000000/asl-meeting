import { useCallback, useMemo, useState } from "react";
import { useASLWebSocket, type ASLPredictionEvent } from "../hooks/useASLWebSocket";
import { WebcamCapture } from "./WebcamCapture";
import { ConnectionIndicator } from "./ConnectionIndicator";

interface TranscriptEntry {
  id: string;
  source: "LOCAL" | "REMOTE";
  gloss: string;
  refinedText: string;
  confidence: number;
  timestamp: number;
}

interface Props {
  localStream: MediaStream | null;
  remoteStream: MediaStream | null;
  wsUrl: string;
  localError?: string | null;
  remoteError?: string | null;
}

export function PiPView({ localStream, remoteStream, wsUrl, localError = null, remoteError = null }: Props) {
  const [localLandmarks, setLocalLandmarks] = useState<Float32Array | null>(null);
  const [remoteLandmarks, setRemoteLandmarks] = useState<Float32Array | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);

  const appendEntry = useCallback((source: "LOCAL" | "REMOTE", payload: ASLPredictionEvent) => {
    setTranscript((previous) => {
      const next: TranscriptEntry = {
        id: `${source}-${payload.timestamp}-${Math.random().toString(36).slice(2, 8)}`,
        source,
        gloss: payload.gloss,
        refinedText: payload.refinedText,
        confidence: payload.confidence,
        timestamp: payload.timestamp
      };
      const merged = [...previous, next].sort((a, b) => a.timestamp - b.timestamp);
      return merged.slice(-40);
    });
  }, []);

  const localWs = useASLWebSocket(localLandmarks, wsUrl, {
    onPrediction: (payload) => appendEntry("LOCAL", payload)
  });
  const remoteWs = useASLWebSocket(remoteLandmarks, wsUrl, {
    onPrediction: (payload) => appendEntry("REMOTE", payload)
  });

  const latestSummary = useMemo(() => {
    const latest = transcript[transcript.length - 1];
    if (!latest) {
      return "Waiting for local/remote sign input...";
    }
    return `[${latest.source}] ${latest.refinedText || latest.gloss}`;
  }, [transcript]);

  return (
    <div className="pip-layout">
      <div className="pip-stage">
        <WebcamCapture
          stream={remoteStream}
          onLandmarks={setRemoteLandmarks}
          className="pip-remote"
          label="REMOTE"
          externalError={remoteError}
        />
        <div className="pip-local-wrapper">
          <WebcamCapture
            stream={localStream}
            onLandmarks={setLocalLandmarks}
            className="pip-local"
            label="LOCAL"
            externalError={localError}
          />
        </div>
      </div>

      <div className="pip-meta">
        <div className="pip-connections">
          <ConnectionIndicator state={localWs.connectionState} />
          <ConnectionIndicator state={remoteWs.connectionState} />
        </div>
        <p className="pip-latest">
          <strong>Latest:</strong> {latestSummary}
        </p>
        <div className="pip-transcript">
          {transcript.map((entry) => (
            <div key={entry.id} className={`pip-transcript-line ${entry.source === "LOCAL" ? "local" : "remote"}`}>
              <span className="pip-source">[{entry.source}]</span>
              <span>{entry.refinedText || entry.gloss}</span>
              <span className="pip-confidence">{(entry.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

