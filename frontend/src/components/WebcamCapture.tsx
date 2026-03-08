import { useEffect, useMemo } from "react";
import { useFrameExtractor } from "../hooks/useFrameExtractor";
import { useHandLandmarks } from "../hooks/useHandLandmarks";
import { LandmarkOverlay } from "./LandmarkOverlay";

interface Props {
  stream: MediaStream | null;
  onLandmarks?: (landmarks: Float32Array | null) => void;
  className?: string;
  label?: string;
  externalError?: string | null;
}

export function WebcamCapture({ stream, onLandmarks, className, label, externalError }: Props) {
  const { processFrame, landmarks, fps: landmarkFps, handsDetected, left, right, leftConfidence, rightConfidence } =
    useHandLandmarks();
  const { videoRef, fps: captureFps, error: extractorError } = useFrameExtractor(stream, processFrame, 15);

  useEffect(() => {
    onLandmarks?.(landmarks);
  }, [landmarks, onLandmarks]);

  const mergedError = externalError || extractorError || (!stream ? "No media stream selected." : null);
  const displayFps = useMemo(() => Math.min(captureFps || 0, landmarkFps || 0), [captureFps, landmarkFps]);

  return (
    <div className={`webcam-capture ${className ?? ""}`.trim()}>
      <video ref={videoRef} autoPlay muted playsInline className="video-preview" />
      <LandmarkOverlay
        videoRef={videoRef}
        left={left}
        right={right}
        leftConfidence={leftConfidence}
        rightConfidence={rightConfidence}
      />

      <div className="capture-indicators">
        {label ? <span className="capture-badge">{label}</span> : null}
        <span className="capture-badge">FPS: {displayFps.toFixed(0)}</span>
        <span className="capture-badge">Hands detected: {handsDetected}</span>
      </div>

      {mergedError ? <div className="capture-error">{mergedError}</div> : null}
    </div>
  );
}

