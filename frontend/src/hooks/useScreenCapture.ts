import { useCallback, useEffect, useMemo, useState } from "react";

interface ScreenCaptureState {
  stream: MediaStream | null;
  error: string | null;
  isCapturing: boolean;
  requestCapture: () => Promise<void>;
  stopCapture: () => void;
}

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

export function useScreenCapture(active: boolean): ScreenCaptureState {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);

  const requestCapture = useCallback(async () => {
    try {
      setError(null);
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 15 },
        audio: false
      });
      setStream((previous) => {
        stopStream(previous);
        return displayStream;
      });
      const [track] = displayStream.getVideoTracks();
      if (track) {
        track.onended = () => {
          setStream((previous) => {
            stopStream(previous);
            return null;
          });
        };
      }
    } catch (captureError) {
      const domError = captureError as DOMException;
      if (domError?.name === "NotAllowedError") {
        setError("Screen capture permission denied. Please allow window sharing and select your Zoom/Meet window.");
      } else if (domError?.name === "NotFoundError") {
        setError("No screen/window source available for capture.");
      } else {
        setError("Failed to start screen capture. Please retry and select a meeting window.");
      }
      setStream((previous) => {
        stopStream(previous);
        return null;
      });
    }
  }, []);

  const stopCapture = useCallback(() => {
    setStream((previous) => {
      stopStream(previous);
      return null;
    });
  }, []);

  useEffect(() => {
    if (active && !stream) {
      void requestCapture();
    }
    if (!active && stream) {
      stopCapture();
    }
  }, [active, requestCapture, stopCapture, stream]);

  return useMemo(
    () => ({
      stream,
      error,
      isCapturing: Boolean(stream),
      requestCapture,
      stopCapture
    }),
    [error, requestCapture, stopCapture, stream]
  );
}

