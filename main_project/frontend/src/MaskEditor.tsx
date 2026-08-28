import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { ArrowsOut, Eraser, Hand, MagnifyingGlassMinus, MagnifyingGlassPlus, PaintBrush, ArrowCounterClockwise, ArrowClockwise } from "@phosphor-icons/react";
import Konva from "konva";
import { Group, Image as KonvaImage, Layer, Line, Stage } from "react-konva";
import { buildMaskBlob, clampPoint, type Stroke, type Tool } from "./mask";

export type MaskEditorHandle = {
  exportMask: () => Promise<Blob>;
};

type Props = {
  sourceUrl: string;
  maskUrl: string;
  maskOverlayUrl: string;
  overlayUrl: string;
  imageWidth: number;
  imageHeight: number;
};

function useImageElement(url: string): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    const next = new Image();
    next.onload = () => setImage(next);
    next.src = url;
    return () => {
      next.onload = null;
    };
  }, [url]);
  return image;
}

const MaskEditor = forwardRef<MaskEditorHandle, Props>(function MaskEditor(
  { sourceUrl, maskUrl, maskOverlayUrl, overlayUrl, imageWidth, imageHeight },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const groupRef = useRef<Konva.Group>(null);
  const source = useImageElement(sourceUrl);
  const evidence = useImageElement(overlayUrl);
  const initialMask = useImageElement(maskUrl);
  const maskOverlay = useImageElement(maskOverlayUrl);
  const [size, setSize] = useState({ width: 900, height: 620 });
  const [tool, setTool] = useState<Tool>("brush");
  const [brushSize, setBrushSize] = useState(28);
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [drawing, setDrawing] = useState(false);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [redo, setRedo] = useState<Stroke[]>([]);
  const [view, setView] = useState<"mask" | "evidence">("mask");

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: Math.max(320, entry.contentRect.width), height: Math.max(420, entry.contentRect.height) });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const fitScale = Math.min((size.width - 48) / imageWidth, (size.height - 48) / imageHeight);
  const scale = Math.max(0.01, fitScale * zoom);

  useEffect(() => {
    setPosition({ x: (size.width - imageWidth * scale) / 2, y: (size.height - imageHeight * scale) / 2 });
  }, [size.width, size.height, imageWidth, imageHeight, scale]);

  useImperativeHandle(ref, () => ({
    exportMask: async () => {
      if (!initialMask) throw new Error("The detected mask is still loading");
      return buildMaskBlob(initialMask, imageWidth, imageHeight, strokes);
    },
  }), [initialMask, imageWidth, imageHeight, strokes]);

  const pointer = () => {
    const point = groupRef.current?.getRelativePointerPosition();
    if (!point) return null;
    return clampPoint(point.x, point.y, imageWidth, imageHeight);
  };

  const startStroke = () => {
    if (tool === "pan" || view === "evidence") return;
    const point = pointer();
    if (!point) return;
    setDrawing(true);
    setRedo([]);
    setStrokes((current) => [...current, { tool, points: point, width: brushSize }]);
  };

  const extendStroke = () => {
    if (!drawing || tool === "pan") return;
    const point = pointer();
    if (!point) return;
    setStrokes((current) => {
      if (!current.length) return current;
      const next = [...current];
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, points: [...last.points, ...point] };
      return next;
    });
  };

  const undo = () => {
    setStrokes((current) => {
      if (!current.length) return current;
      const removed = current[current.length - 1];
      setRedo((items) => [...items, removed]);
      return current.slice(0, -1);
    });
  };

  const redoStroke = () => {
    setRedo((current) => {
      if (!current.length) return current;
      const restored = current[current.length - 1];
      setStrokes((items) => [...items, restored]);
      return current.slice(0, -1);
    });
  };

  return (
    <section className="editor-shell" aria-label="Redaction mask editor">
      <div className="editor-toolbar">
        <div className="segmented-control" aria-label="Canvas view">
          <button className={view === "mask" ? "active" : ""} onClick={() => setView("mask")}>Review mask</button>
          <button className={view === "evidence" ? "active" : ""} onClick={() => setView("evidence")}>Evidence</button>
        </div>
        <div className="tool-strip" aria-label="Editing tools">
          <button className={tool === "brush" ? "active" : ""} onClick={() => { setTool("brush"); setView("mask"); }} title="Brush"><PaintBrush /></button>
          <button className={tool === "erase" ? "active" : ""} onClick={() => { setTool("erase"); setView("mask"); }} title="Eraser"><Eraser /></button>
          <button className={tool === "pan" ? "active" : ""} onClick={() => setTool("pan")} title="Pan"><Hand /></button>
          <span className="tool-divider" />
          <button onClick={undo} disabled={!strokes.length} title="Undo"><ArrowCounterClockwise /></button>
          <button onClick={redoStroke} disabled={!redo.length} title="Redo"><ArrowClockwise /></button>
          <button onClick={() => { setStrokes([]); setRedo([]); }} disabled={!strokes.length} title="Reset corrections"><ArrowsOut /></button>
        </div>
        <label className="brush-control">
          <span>Brush</span>
          <input type="range" min="6" max="120" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
          <output>{brushSize}px</output>
        </label>
        <div className="zoom-control">
          <button onClick={() => setZoom((value) => Math.max(1, Number((value - 0.25).toFixed(2))))} aria-label="Zoom out"><MagnifyingGlassMinus /></button>
          <span>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom((value) => Math.min(3, Number((value + 0.25).toFixed(2))))} aria-label="Zoom in"><MagnifyingGlassPlus /></button>
        </div>
      </div>
      <div className={`canvas-frame tool-${tool}`} ref={containerRef}>
        {!source || !initialMask ? <div className="canvas-loading"><span />Preparing native-resolution canvas</div> : (
          <Stage
            width={size.width}
            height={size.height}
            onMouseDown={startStroke}
            onMouseMove={extendStroke}
            onMouseUp={() => setDrawing(false)}
            onMouseLeave={() => setDrawing(false)}
            onTouchStart={startStroke}
            onTouchMove={extendStroke}
            onTouchEnd={() => setDrawing(false)}
          >
            <Layer>
              <Group
                ref={groupRef}
                x={position.x}
                y={position.y}
                scaleX={scale}
                scaleY={scale}
                draggable={tool === "pan"}
                onDragEnd={(event) => setPosition({ x: event.target.x(), y: event.target.y() })}
                clip={{ x: 0, y: 0, width: imageWidth, height: imageHeight }}
              >
                <KonvaImage image={view === "evidence" && evidence ? evidence : source} width={imageWidth} height={imageHeight} listening={false} />
              </Group>
            </Layer>
            {view === "mask" && (
              <Layer>
                <Group x={position.x} y={position.y} scaleX={scale} scaleY={scale} listening={tool !== "pan"} clip={{ x: 0, y: 0, width: imageWidth, height: imageHeight }}>
                  {maskOverlay && <KonvaImage image={maskOverlay} width={imageWidth} height={imageHeight} listening={false} />}
                  {strokes.map((stroke, index) => (
                    <Line
                      key={index}
                      points={stroke.points}
                      stroke="rgba(49, 183, 166, 0.78)"
                      strokeWidth={stroke.width}
                      lineCap="round"
                      lineJoin="round"
                      globalCompositeOperation={stroke.tool === "erase" ? "destination-out" : "source-over"}
                      listening={false}
                    />
                  ))}
                </Group>
              </Layer>
            )}
          </Stage>
        )}
        <div className="canvas-readout"><span>{imageWidth} × {imageHeight}</span><span>{tool === "pan" ? "Drag to inspect" : tool === "erase" ? "Removing coverage" : "Adding coverage"}</span></div>
      </div>
    </section>
  );
});

export default MaskEditor;
