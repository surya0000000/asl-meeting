export type ConnectionState = "connecting" | "connected" | "disconnected" | "error";

export interface PredictionMessage {
  type: "prediction";
  gloss?: string;
  raw_prediction?: string;
  refined_text?: string;
  refinedText?: string;
  rawTranscript?: string;
  raw_transcript?: string;
  alternatives?: string[];
  confidence?: number;
  audio_url?: string | null;
  frame_index?: number;
  timestamp?: string;
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

export interface LandmarkSequenceMessage {
  type: "landmark_sequence";
  sequence: number[][];
  session_id: string;
}

export interface ASLPredictionState {
  gloss: string;
  confidence: number;
  alternatives: string[];
  rawTranscript: string;
  refinedText: string;
}

export type StreamMessage =
  | PredictionMessage
  | ConnectionMessage
  | BufferingMessage
  | ErrorMessage;

