import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface LandmarkWorkerResponse {
  type: "landmarks";
  timestamp: number;
  left: number[] | null;
  right: number[] | null;
  combined: number[];
  leftConfidence?: number;
  rightConfidence?: number;
  handsDetected?: number;
}

interface LandmarkWorkerError {
  type: "error";
  timestamp: number;
  detail: string;
}

type WorkerMessage = LandmarkWorkerResponse | LandmarkWorkerError;

export interface HandLandmarkState {
  landmarks: Float32Array | null;
  fps: number;
  handsDetected: number;
  left: Float32Array | null;
  right: Float32Array | null;
  leftConfidence: number;
  rightConfidence: number;
}

export interface HandLandmarkController extends HandLandmarkState {
  processFrame: (frame: ImageBitmap) => void;
}

export function useHandLandmarks(): HandLandmarkController {
  const [landmarks, setLandmarks] = useState<Float32Array | null>(null);
  const [left, setLeft] = useState<Float32Array | null>(null);
  const [right, setRight] = useState<Float32Array | null>(null);
  const [leftConfidence, setLeftConfidence] = useState(0);
  const [rightConfidence, setRightConfidence] = useState(0);
  const [handsDetected, setHandsDetected] = useState(0);
  const [fps, setFps] = useState(0);

  const workerRef = useRef<Worker | null>(null);
  const frameTimestampsRef = useRef<number[]>([]);
  const processFrame = useCallback((frame: ImageBitmap) => {
    const worker = workerRef.current;
    if (!worker) {
      frame.close();
      return;
    }
    worker.postMessage(
      {
        type: "frame",
        timestamp: performance.now(),
        frame
      },
      [frame]
    );
  }, []);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/landmarkWorker.ts", import.meta.url), {
      type: "module"
    });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const payload = event.data;
      if (!payload || payload.type === "error") {
        return;
      }

      const timestamp = Number(payload.timestamp || performance.now());
      const currentTimestamps = frameTimestampsRef.current.filter((t) => timestamp - t <= 1000);
      currentTimestamps.push(timestamp);
      frameTimestampsRef.current = currentTimestamps;
      setFps(currentTimestamps.length);

      setLandmarks(new Float32Array(payload.combined));
      setLeft(payload.left ? new Float32Array(payload.left) : null);
      setRight(payload.right ? new Float32Array(payload.right) : null);
      setLeftConfidence(Number(payload.leftConfidence ?? 0));
      setRightConfidence(Number(payload.rightConfidence ?? 0));
      if (typeof payload.handsDetected === "number") {
        setHandsDetected(payload.handsDetected);
      } else {
        setHandsDetected(Number(Boolean(payload.left)) + Number(Boolean(payload.right)));
      }
    };

    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  return useMemo(
    () => ({
      processFrame,
      landmarks,
      fps,
      handsDetected,
      left,
      right,
      leftConfidence,
      rightConfidence
    }),
    [fps, handsDetected, landmarks, left, leftConfidence, processFrame, right, rightConfidence]
  );
}

