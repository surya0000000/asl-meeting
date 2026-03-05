export type CaptureSourceMode = "camera" | "screen" | "both";

interface Props {
  value: CaptureSourceMode;
  onChange: (mode: CaptureSourceMode) => void;
}

const OPTIONS: Array<{ key: CaptureSourceMode; label: string }> = [
  { key: "camera", label: "My Camera" },
  { key: "screen", label: "Screen / Zoom Window" },
  { key: "both", label: "Both" }
];

export function SourceSelector({ value, onChange }: Props) {
  return (
    <div className="source-selector" role="tablist" aria-label="Capture source selector">
      {OPTIONS.map((option) => (
        <button
          key={option.key}
          type="button"
          role="tab"
          aria-selected={value === option.key}
          className={`source-selector-btn ${value === option.key ? "active" : ""}`}
          onClick={() => onChange(option.key)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

