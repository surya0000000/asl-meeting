export type ConnectionState = "connecting" | "connected" | "disconnected" | "error";

export interface PredictionMessage {
  type: "prediction";
  raw_prediction: string;
  refined_text: string;
  confidence: number;
  audio_url?: string | null;
  frame_index: number;
  timestamp: string;
}

export interface ConnectionMessage {
  type: "connection";
  status: string;
}

export interface BufferingMessage {
  type: "buffering";
  detail: string;
}

export interface ErrorMessage {
  type: "error";
  detail: unknown;
}

export type StreamMessage =
  | PredictionMessage
  | ConnectionMessage
  | BufferingMessage
  | ErrorMessage;

