import { ConnectionState } from "../types/stream";

interface Props {
  state: ConnectionState;
}

const stateToColor: Record<ConnectionState, string> = {
  connecting: "#f59e0b",
  connected: "#22c55e",
  disconnected: "#64748b",
  error: "#ef4444"
};

export function ConnectionIndicator({ state }: Props) {
  return (
    <div className="connection-indicator">
      <span
        className="connection-dot"
        style={{ backgroundColor: stateToColor[state] }}
        aria-hidden="true"
      />
      <span>Backend: {state}</span>
    </div>
  );
}

