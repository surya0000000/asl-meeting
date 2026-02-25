import { useEffect, useRef, useState } from "react";
import { useHandLandmarks } from "../hooks/useHandLandmarks";
import { LandmarkOverlay } from "./LandmarkOverlay";

interface Props {
  onLandmarks?: (landmarks: Float32Array | null) => void;
}

export function WebcamCapture({ onLandmarks }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const { landmarks, fps, handsDetected, left, right, leftConfidence, rightConfidence } =
    useHandLandmarks(videoRef);

  useEffect(() => {
    let stream: MediaStream | null = null;
    const setupWebcam = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 }
          },
          audio: false
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch {
        setCameraError("Unable to access webcam. Please allow camera permissions.");
      }
    };

    void setupWebcam();
    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    onLandmarks?.(landmarks);
  }, [landmarks, onLandmarks]);

  return (
    <div className="webcam-capture">
      <video ref={videoRef} autoPlay muted playsInline className="video-preview" />
      <LandmarkOverlay
        videoRef={videoRef}
        left={left}
        right={right}
        leftConfidence={leftConfidence}
        rightConfidence={rightConfidence}
      />

      <div className="capture-indicators">
        <span className="capture-badge">FPS: {fps.toFixed(0)}</span>
        <span className="capture-badge">Hands detected: {handsDetected}</span>
      </div>

      {cameraError ? <div className="capture-error">{cameraError}</div> : null}
    </div>
  );
}

