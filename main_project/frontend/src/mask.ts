export type Tool = "brush" | "erase" | "pan";

export type Stroke = {
  tool: Exclude<Tool, "pan">;
  points: number[];
  width: number;
};

export function clampPoint(x: number, y: number, width: number, height: number): [number, number] {
  return [Math.max(0, Math.min(width, x)), Math.max(0, Math.min(height, y))];
}

export function drawStroke(context: CanvasRenderingContext2D, stroke: Stroke): void {
  if (stroke.points.length < 2) return;
  context.save();
  context.strokeStyle = stroke.tool === "brush" ? "#ffffff" : "#000000";
  context.lineWidth = stroke.width;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(stroke.points[0], stroke.points[1]);
  for (let index = 2; index < stroke.points.length; index += 2) {
    context.lineTo(stroke.points[index], stroke.points[index + 1]);
  }
  context.stroke();
  context.restore();
}

export async function buildMaskBlob(
  initialMask: HTMLImageElement,
  width: number,
  height: number,
  strokes: Stroke[],
): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: false });
  if (!context) throw new Error("Canvas is unavailable in this browser");
  context.fillStyle = "#000000";
  context.fillRect(0, 0, width, height);
  context.drawImage(initialMask, 0, 0, width, height);
  strokes.forEach((stroke) => drawStroke(context, stroke));
  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the approved mask"))), "image/png");
  });
}
