/// <reference lib="webworker" />

import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import type { Results } from "@mediapipe/hands";

interface WorkerFrameInput {
  type: "frame";
  timestamp: number;
  frame: ImageBitmap;
}

interface WorkerLandmarkOutput {
  type: "landmarks";
  timestamp: number;
  left: number[] | null;
  right: number[] | null;
  combined: number[];
  leftConfidence: number;
  rightConfidence: number;
  handsDetected: number;
}

const EMPTY_HAND = new Array<number>(63).fill(0);

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

let activeTimestamp = 0;
let processing = false;
let queuedMessage: WorkerFrameInput | null = null;
const offscreenCanvas = new OffscreenCanvas(1, 1);
const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: false });

function flattenLandmarks(landmarks: Results["multiHandLandmarks"][number]): number[] {
  const output = new Array<number>(63);
  let offset = 0;
  for (const lm of landmarks) {
    output[offset++] = lm.x;
    output[offset++] = lm.y;
    output[offset++] = lm.z;
  }
  return output;
}

hands.onResults((results: Results) => {
  let left: number[] | null = null;
  let right: number[] | null = null;
  let leftConfidence = 0;
  let rightConfidence = 0;

  const handedness = results.multiHandedness ?? [];
  const handLandmarks = results.multiHandLandmarks ?? [];

  for (let i = 0; i < handLandmarks.length; i += 1) {
    const landmarks = handLandmarks[i];
    const classification = handedness[i]?.classification?.[0];
    const label = (classification?.label || "").toLowerCase();
    const score = Number(classification?.score ?? 0.0);
    const flat = flattenLandmarks(landmarks);

    if (label === "left") {
      if (left === null || score >= leftConfidence) {
        left = flat;
        leftConfidence = score;
      }
      continue;
    }
    if (label === "right") {
      if (right === null || score >= rightConfidence) {
        right = flat;
        rightConfidence = score;
      }
      continue;
    }

    // Fallback when handedness is missing/ambiguous.
    if (left === null) {
      left = flat;
      leftConfidence = score;
    } else if (right === null) {
      right = flat;
      rightConfidence = score;
    } else if (score > leftConfidence || score > rightConfidence) {
      if (leftConfidence <= rightConfidence) {
        left = flat;
        leftConfidence = score;
      } else {
        right = flat;
        rightConfidence = score;
      }
    }
  }

  const combined = [...(left ?? EMPTY_HAND), ...(right ?? EMPTY_HAND)];
  const output: WorkerLandmarkOutput = {
    type: "landmarks",
    timestamp: activeTimestamp,
    left,
    right,
    combined,
    leftConfidence,
    rightConfidence,
    handsDetected: Number(left !== null) + Number(right !== null)
  };
  self.postMessage(output);
});

async function processFrame(message: WorkerFrameInput): Promise<void> {
  processing = true;
  activeTimestamp = message.timestamp;
  const frame = message.frame;

  try {
    if (!offscreenCtx) {
      throw new Error("Unable to obtain OffscreenCanvas context for landmark worker.");
    }
    offscreenCanvas.width = frame.width;
    offscreenCanvas.height = frame.height;
    offscreenCtx.drawImage(frame, 0, 0, frame.width, frame.height);
    await hands.send({ image: offscreenCanvas });
  } catch (error) {
    self.postMessage({
      type: "error",
      timestamp: message.timestamp,
      detail: error instanceof Error ? error.message : "Unknown worker error"
    });
  } finally {
    frame.close();
    processing = false;
    if (queuedMessage) {
      const next = queuedMessage;
      queuedMessage = null;
      await processFrame(next);
    }
  }
}

self.onmessage = (event: MessageEvent<WorkerFrameInput>) => {
  const payload = event.data;
  if (!payload || payload.type !== "frame") {
    return;
  }
  if (processing) {
    if (queuedMessage?.frame) {
      queuedMessage.frame.close();
    }
    queuedMessage = payload;
    return;
  }
  void processFrame(payload);
};

// Keep imports used to prevent aggressive tree-shaking in some build paths.
void HAND_CONNECTIONS;

export {};

