import { describe, expect, it, vi } from "vitest";
import { clampPoint, drawStroke } from "./mask";

describe("mask geometry", () => {
  it("clamps pointer coordinates to the native image", () => {
    expect(clampPoint(-4, 80, 100, 60)).toEqual([0, 60]);
    expect(clampPoint(33, 22, 100, 60)).toEqual([33, 22]);
    expect(clampPoint(180, -2, 100, 60)).toEqual([100, 0]);
  });

  it("draws brush and eraser strokes as white and black pixels", () => {
    const context = {
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      strokeStyle: "",
      lineWidth: 0,
      lineCap: "butt",
      lineJoin: "miter",
    } as unknown as CanvasRenderingContext2D;

    drawStroke(context, { tool: "brush", points: [1, 2, 3, 4], width: 18 });
    expect(context.strokeStyle).toBe("#ffffff");
    expect(context.lineWidth).toBe(18);
    expect(context.lineTo).toHaveBeenCalledWith(3, 4);

    drawStroke(context, { tool: "erase", points: [4, 5, 6, 7], width: 12 });
    expect(context.strokeStyle).toBe("#000000");
    expect(context.lineWidth).toBe(12);
  });
});
