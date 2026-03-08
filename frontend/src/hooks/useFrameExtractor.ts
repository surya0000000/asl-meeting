import { RefObject, useEffect, useMemo, useRef, useState } from "react";

interface FrameExtractorState {
  videoRef: RefObject<HTMLVideoElement>;
  fps: number;
  error: string | null;
}

const DEFAULT_TARGET_FPS = 15;

export function useFrameExtractor(
  stream: MediaStream | null,
  onFrame: (frame: ImageBitmap) => void,
  targetFps: number = DEFAULT_TARGET_FPS
): FrameExtractorState {
  const videoRef = useRef<HTMLVideoElement>(null);
  const rafRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);
  const lastFrameSentMs = useRef(0);
  const frameTimesRef = useRef<number[]>([]);

  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    if (!stream) {
      setFps(0);
    }
    video.srcObject = stream;
    void video.play().catch(() => undefined);
    return () => {
      video.srcObject = null;
    };
  }, [stream]);

  useEffect(() => {
    const intervalMs = 1000 / Math.max(1, targetFps);
    let stopped = false;

    const captureLoop = async (now: number) => {
      if (stopped) {
        return;
      }
      const video = videoRef.current;
      if (
        stream &&
        video &&
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        !video.paused &&
        !video.ended &&
        !inFlightRef.current &&
        now - lastFrameSentMs.current >= intervalMs
      ) {
        try {
          lastFrameSentMs.current = now;
          inFlightRef.current = true;
          const frame = await createImageBitmap(video);
          onFrame(frame);

          const timestamps = frameTimesRef.current.filter((t) => now - t <= 1000);
          timestamps.push(now);
          frameTimesRef.current = timestamps;
          setFps(timestamps.length);
          setError(null);
        } catch {
          setError("Unable to extract frames from selected media source.");
        } finally {
          inFlightRef.current = false;
        }
      }
      rafRef.current = requestAnimationFrame(captureLoop);
    };

    rafRef.current = requestAnimationFrame(captureLoop);

    return () => {
      stopped = true;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [onFrame, stream, targetFps]);

  return useMemo(
    () => ({
      videoRef,
      fps,
      error
    }),
    [error, fps]
  );
}

