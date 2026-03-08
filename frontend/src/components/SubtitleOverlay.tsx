interface Props {
  text: string;
  confidence: number | null;
}

export function SubtitleOverlay({ text, confidence }: Props) {
  return (
    <div className="subtitle-overlay" role="status" aria-live="polite">
      <div className="subtitle-text">{text || "Waiting for ASL gesture predictions..."}</div>
      {confidence !== null ? (
        <div className="subtitle-confidence">Confidence: {(confidence * 100).toFixed(1)}%</div>
      ) : null}
    </div>
  );
}

