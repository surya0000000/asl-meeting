export type GridPosition = "top-left" | "top-right" | "bottom-left" | "bottom-right";

export interface CroppedStreamResult {
  stream: MediaStream;
  cleanup: () => void;
}

function getCropRect(width: number, height: number, position: GridPosition) {
  const halfWidth = Math.floor(width / 2);
  const halfHeight = Math.floor(height / 2);

  switch (position) {
    case "top-right":
      return { x: halfWidth, y: 0, width: halfWidth, height: halfHeight };
    case "bottom-left":
      return { x: 0, y: halfHeight, width: halfWidth, height: halfHeight };
    case "bottom-right":
      return { x: halfWidth, y: halfHeight, width: halfWidth, height: halfHeight };
    case "top-left":
    default:
      return { x: 0, y: 0, width: halfWidth, height: halfHeight };
  }
}

/**
 * Crop a Zoom/Meet grid stream to one participant quadrant.
 *
 * This utility creates a canvas-based cropped stream and returns a cleanup function.
 */
export async function cropToParticipant(stream: MediaStream, gridPosition: GridPosition): Promise<CroppedStreamResult> {
  const video = document.createElement("video");
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;

  await video.play();
  await new Promise<void>((resolve) => {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      resolve();
      return;
    }
    video.onloadedmetadata = () => resolve();
  });

  const sourceWidth = video.videoWidth || 1280;
  const sourceHeight = video.videoHeight || 720;
  const crop = getCropRect(sourceWidth, sourceHeight, gridPosition);

  const canvas = document.createElement("canvas");
  canvas.width = crop.width;
  canvas.height = crop.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Unable to initialize crop canvas context.");
  }

  let rafId: number | null = null;
  let stopped = false;

  const render = () => {
    if (stopped) {
      return;
    }
    ctx.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);
    rafId = requestAnimationFrame(render);
  };
  rafId = requestAnimationFrame(render);

  const croppedStream = canvas.captureStream(15);
  const cleanup = () => {
    stopped = true;
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
    }
    croppedStream.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  };

  return { stream: croppedStream, cleanup };
}

