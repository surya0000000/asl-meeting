import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ASLPredictionState,
  ConnectionState,
  LandmarkSequenceMessage,
  PredictionMessage,
  StreamMessage
} from "../types/stream";

const BUFFER_SIZE = 30;
const LANDMARK_DIM = 126;
const SEND_INTERVAL_MS = 100;
const MAX_BACKOFF_MS = 30_000;

function randomSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
}

function parsePrediction(message: PredictionMessage): ASLPredictionState {
  const gloss = message.gloss || message.raw_prediction || "UNSURE";
  const refined = message.refinedText || message.refined_text || gloss;
  const rawTranscript = message.rawTranscript || message.raw_transcript || gloss;
  return {
    gloss,
    confidence: Number(message.confidence ?? 0),
    alternatives: Array.isArray(message.alternatives) ? message.alternatives : [],
    rawTranscript,
    refinedText: refined
  };
}

export function useASLWebSocket(
  landmarks: Float32Array | null,
  url: string = "ws://localhost:8000/ws/asl-stream"
) {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const sequenceBufferRef = useRef<Float32Array[]>(
    Array.from({ length: BUFFER_SIZE }, () => new Float32Array(LANDMARK_DIM))
  );
  const headRef = useRef(0);
  const countRef = useRef(0);
  const dirtyRef = useRef(false);
  const manualCloseRef = useRef(false);
  const backoffMsRef = useRef(1000);
  const sessionIdRef = useRef(randomSessionId());

  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting");
  const [prediction, setPrediction] = useState<ASLPredictionState>({
    gloss: "UNSURE",
    confidence: 0,
    alternatives: [],
    rawTranscript: "",
    refinedText: ""
  });

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  const pushLandmarks = useCallback((frame: Float32Array) => {
    const normalizedFrame = new Float32Array(LANDMARK_DIM);
    if (frame.length >= LANDMARK_DIM) {
      normalizedFrame.set(frame.subarray(0, LANDMARK_DIM));
    } else {
      normalizedFrame.set(frame);
    }

    sequenceBufferRef.current[headRef.current] = normalizedFrame;
    headRef.current = (headRef.current + 1) % BUFFER_SIZE;
    countRef.current = Math.min(BUFFER_SIZE, countRef.current + 1);
    dirtyRef.current = true;
  }, []);

  const buildSequence = useCallback((): number[][] => {
    const count = countRef.current;
    const sequence: number[][] = [];
    const zeroFrame = new Array<number>(LANDMARK_DIM).fill(0);

    // Left pad with zeros until the temporal window is full.
    const padCount = Math.max(0, BUFFER_SIZE - count);
    for (let index = 0; index < padCount; index += 1) {
      sequence.push([...zeroFrame]);
    }

    if (count > 0) {
      const start = (headRef.current - count + BUFFER_SIZE) % BUFFER_SIZE;
      for (let offset = 0; offset < count; offset += 1) {
        const bufferIndex = (start + offset) % BUFFER_SIZE;
        sequence.push(Array.from(sequenceBufferRef.current[bufferIndex]));
      }
    }

    return sequence;
  }, []);

  const sendSequence = useCallback(() => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    if (!dirtyRef.current && countRef.current < BUFFER_SIZE) {
      return false;
    }
    const payload: LandmarkSequenceMessage = {
      type: "landmark_sequence",
      sequence: buildSequence(),
      session_id: sessionIdRef.current
    };
    socket.send(JSON.stringify(payload));
    dirtyRef.current = false;
    return true;
  }, [buildSequence]);

  useEffect(() => {
    if (!landmarks) {
      return;
    }
    pushLandmarks(landmarks);
    if (countRef.current >= BUFFER_SIZE) {
      void sendSequence();
    }
  }, [landmarks, pushLandmarks, sendSequence]);

  useEffect(() => {
    manualCloseRef.current = false;
    let heartbeatTimer: number | null = null;

    const scheduleReconnect = () => {
      clearReconnectTimer();
      const delay = backoffMsRef.current;
      reconnectTimerRef.current = window.setTimeout(() => {
        connect();
      }, delay);
      backoffMsRef.current = Math.min(backoffMsRef.current * 2, MAX_BACKOFF_MS);
    };

    const connect = () => {
      setConnectionState("connecting");
      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        sessionIdRef.current = randomSessionId();
        backoffMsRef.current = 1000;
        setConnectionState("connected");
        if (heartbeatTimer !== null) {
          window.clearInterval(heartbeatTimer);
        }
        heartbeatTimer = window.setInterval(() => {
          void sendSequence();
        }, SEND_INTERVAL_MS);
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as StreamMessage;
          if (payload.type === "prediction") {
            setPrediction(parsePrediction(payload));
          }
        } catch {
          setConnectionState("error");
        }
      };

      socket.onerror = () => {
        setConnectionState("error");
      };

      socket.onclose = () => {
        if (heartbeatTimer !== null) {
          window.clearInterval(heartbeatTimer);
          heartbeatTimer = null;
        }
        if (manualCloseRef.current) {
          setConnectionState("disconnected");
          return;
        }
        setConnectionState("disconnected");
        scheduleReconnect();
      };
    };

    connect();

    return () => {
      manualCloseRef.current = true;
      clearReconnectTimer();
      if (heartbeatTimer !== null) {
        window.clearInterval(heartbeatTimer);
      }
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [sendSequence, url]);

  return useMemo(
    () => ({
      connectionState,
      gloss: prediction.gloss,
      confidence: prediction.confidence,
      alternatives: prediction.alternatives,
      rawTranscript: prediction.rawTranscript,
      refinedText: prediction.refinedText
    }),
    [connectionState, prediction]
  );
}

