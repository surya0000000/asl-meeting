import { useCallback, useEffect, useRef, useState } from "react";
import { ConnectionState, StreamMessage } from "../types/stream";

interface UseWebSocketOptions {
  url: string;
  reconnectDelayMs?: number;
}

export function useWebSocket(options: UseWebSocketOptions) {
  const { url, reconnectDelayMs = 2000 } = options;
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const isManuallyClosed = useRef(false);

  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting");
  const [latestMessage, setLatestMessage] = useState<StreamMessage | null>(null);

  const clearReconnectTimer = () => {
    if (reconnectTimer.current !== null) {
      window.clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
  };

  const connect = useCallback(() => {
    clearReconnectTimer();
    setConnectionState("connecting");

    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => {
      setConnectionState("connected");
    };

    socket.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as StreamMessage;
        setLatestMessage(parsed);
      } catch {
        setLatestMessage({ type: "error", detail: "Invalid message from server" });
      }
    };

    socket.onerror = () => {
      setConnectionState("error");
    };

    socket.onclose = () => {
      if (isManuallyClosed.current) {
        setConnectionState("disconnected");
        return;
      }
      setConnectionState("disconnected");
      reconnectTimer.current = window.setTimeout(() => {
        connect();
      }, reconnectDelayMs);
    };
  }, [reconnectDelayMs, url]);

  useEffect(() => {
    isManuallyClosed.current = false;
    connect();
    return () => {
      isManuallyClosed.current = true;
      clearReconnectTimer();
      socketRef.current?.close();
    };
  }, [connect]);

  const sendJson = useCallback((payload: unknown) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    socket.send(JSON.stringify(payload));
    return true;
  }, []);

  return {
    connectionState,
    latestMessage,
    sendJson,
  };
}

