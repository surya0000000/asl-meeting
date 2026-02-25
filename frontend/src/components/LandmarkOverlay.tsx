import { RefObject, useCallback, useEffect, useRef } from "react";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { HAND_CONNECTIONS } from "@mediapipe/hands";

interface Props {
  videoRef: RefObject<HTMLVideoElement>;
  left: Float32Array | null;
  right: Float32Array | null;
  leftConfidence?: number;
  rightConfidence?: number;
}

interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

function parseHandLandmarks(flat: Float32Array | null): LandmarkPoint[] {
  if (!flat || flat.length < 63) {
    return [];
  }
  const points: LandmarkPoint[] = [];
  for (let index = 0; index < 63; index += 3) {
    points.push({
      x: flat[index],
      y: flat[index + 1],
      z: flat[index + 2]
    });
  }
  const nonZero = points.some((point) => Math.abs(point.x) + Math.abs(point.y) + Math.abs(point.z) > 1e-6);
  return nonZero ? points : [];
}

export function LandmarkOverlay({
  videoRef,
  left,
  right,
  leftConfidence = 0.65,
  rightConfidence = 0.65
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) {
      return;
    }
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    if (canvas.width !== width) {
      canvas.width = width;
    }
    if (canvas.height !== height) {
      canvas.height = height;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const leftLandmarks = parseHandLandmarks(left);
    const rightLandmarks = parseHandLandmarks(right);

    if (leftLandmarks.length > 0) {
      const opacity = Math.min(1, Math.max(0.25, leftConfidence));
      drawConnectors(ctx, leftLandmarks, HAND_CONNECTIONS, {
        color: `rgba(59, 130, 246, ${opacity})`,
        lineWidth: 2
      });
      drawLandmarks(ctx, leftLandmarks, {
        color: `rgba(96, 165, 250, ${opacity})`,
        lineWidth: 1,
        radius: 3
      });
    }

    if (rightLandmarks.length > 0) {
      const opacity = Math.min(1, Math.max(0.25, rightConfidence));
      drawConnectors(ctx, rightLandmarks, HAND_CONNECTIONS, {
        color: `rgba(34, 197, 94, ${opacity})`,
        lineWidth: 2
      });
      drawLandmarks(ctx, rightLandmarks, {
        color: `rgba(74, 222, 128, ${opacity})`,
        lineWidth: 1,
        radius: 3
      });
    }
  }, [left, leftConfidence, right, rightConfidence, videoRef]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  useEffect(() => {
    const onResize = () => drawOverlay();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
    };
  }, [drawOverlay]);

  return <canvas ref={canvasRef} className="landmark-overlay" aria-hidden="true" />;
}

