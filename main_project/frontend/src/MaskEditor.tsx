import { forwardRef, memo, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { ArrowClockwise, ArrowCounterClockwise, Broom, CornersOut, Eraser, Hand, MagnifyingGlass, MagnifyingGlassMinus, MagnifyingGlassPlus, PaintBrush } from "@phosphor-icons/react";
import type Konva from "konva";
import { Circle, Group, Image as KonvaImage, Layer, Line, Rect, Stage } from "react-konva";
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

type Point = { x: number; y: number };

/** Canvas drawing cannot use CSS variables directly, so read the design tokens once. */
function readTokens() {
  const style = getComputedStyle(document.documentElement);
  const token = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return { marker: token("--marker", "#ff6a3d"), paper: token("--paper", "#f1ebe0"), lightBox: token("--light-box", "#0e0d0c") };
}
const MIN_BRUSH = 6;
const MAX_BRUSH = 160;
const MAX_ZOOM = 8;
const LOUPE_RADIUS = 88;
// The cover is drawn nearly opaque so the preview reads as the saved black,
// while a trace of the photo stays visible for judging edges.
const COVER_OPACITY = 0.88;

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

/** Turns the server's tinted overlay into an opaque black stencil of the covered pixels. */
function useCoverStencil(overlay: HTMLImageElement | null): HTMLCanvasElement | null {
  return useMemo(() => {
    if (!overlay) return null;
    const canvas = document.createElement("canvas");
    canvas.width = overlay.naturalWidth;
    canvas.height = overlay.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(overlay, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    const words = new Uint32Array(pixels.data.buffer);
    for (let index = 0; index < words.length; index += 1) {
      // Little-endian RGBA: alpha is the high byte.
      words[index] = words[index] >>> 24 ? 0xff000000 : 0;
    }
    context.putImageData(pixels, 0, 0);
    return canvas;
  }, [overlay]);
}

/** A china-marker hatch tile, laid over the cover so it reads as "marked to hide". */
function makeHatch(color: string): HTMLCanvasElement | null {
  const size = 12;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.strokeStyle = color;
  context.globalAlpha = 0.72;
  context.lineWidth = 1.75;
  context.beginPath();
  for (const offset of [-size, 0, size]) {
    context.moveTo(offset, size);
    context.lineTo(offset + size, 0);
  }
  context.stroke();
  return canvas;
}

type CoverProps = { stencil: HTMLCanvasElement | null; strokes: Stroke[]; hatch: HTMLCanvasElement | null; width: number; height: number; patternScale: number };

// Memoised so pointer moves (brush ring, loupe) do not redraw every stroke.
const Cover = memo(function Cover({ stencil, strokes, hatch, width, height, patternScale }: CoverProps) {
  return (
    <>
      {stencil && <KonvaImage image={stencil} width={width} height={height} listening={false} />}
      {strokes.map((stroke, index) => (
        <Line
          key={index}
          points={stroke.points}
          stroke="#000"
          strokeWidth={stroke.width}
          lineCap="round"
          lineJoin="round"
          globalCompositeOperation={stroke.tool === "erase" ? "destination-out" : "source-over"}
          listening={false}
        />
      ))}
      {hatch && (
        <Rect
          width={width}
          height={height}
          fillPatternImage={hatch as unknown as HTMLImageElement}
          fillPatternScale={{ x: patternScale, y: patternScale }}
          fillPatternRepeat="repeat"
          globalCompositeOperation="source-atop"
          listening={false}
        />
      )}
    </>
  );
});

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName);
}

const TOOLS: { key: Tool; label: string; shortcut: string; icon: typeof PaintBrush }[] = [
  { key: "brush", label: "Brush", shortcut: "B", icon: PaintBrush },
  { key: "erase", label: "Eraser", shortcut: "E", icon: Eraser },
  { key: "loupe", label: "Loupe", shortcut: "L", icon: MagnifyingGlass },
  { key: "pan", label: "Move", shortcut: "H", icon: Hand },
];

const HINTS: Record<Tool, string> = {
  brush: "Paint to cover · [ ] resize",
  erase: "Paint to uncover · [ ] resize",
  loupe: "Move over the photo to look closely",
  pan: "Drag to move · scroll to zoom",
};

const MaskEditor = forwardRef<MaskEditorHandle, Props>(function MaskEditor(
  { sourceUrl, maskUrl, maskOverlayUrl, overlayUrl, imageWidth, imageHeight },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const drawing = useRef(false);
  const panStart = useRef<{ pointer: Point; origin: Point } | null>(null);
  const source = useImageElement(sourceUrl);
  const evidence = useImageElement(overlayUrl);
  const initialMask = useImageElement(maskUrl);
  const maskOverlay = useImageElement(maskOverlayUrl);
  const stencil = useCoverStencil(maskOverlay);
  const colors = useMemo(readTokens, []);
  const hatch = useMemo(() => makeHatch(colors.marker), [colors.marker]);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [tool, setTool] = useState<Tool>("brush");
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [panning, setPanning] = useState(false);
  const [brushSize, setBrushSize] = useState(32);
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState<Point>({ x: 0, y: 0 });
  const [pointer, setPointer] = useState<Point | null>(null);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [redo, setRedo] = useState<Stroke[]>([]);
  const [view, setView] = useState<"cover" | "found">("cover");

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: Math.round(entry.contentRect.width), height: Math.round(entry.contentRect.height) });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const pad = size.width < 640 ? 10 : 24;
  const fitScale = Math.max(0.01, Math.min((size.width - pad * 2) / imageWidth, (size.height - pad * 2) / imageHeight));
  const scale = fitScale * zoom;

  /** Centres any axis where the photo is smaller than the frame and stops panning past its edges. */
  const constrain = (next: Point, nextScale: number): Point => {
    const drawnWidth = imageWidth * nextScale;
    const drawnHeight = imageHeight * nextScale;
    const x = drawnWidth <= size.width - pad * 2
      ? (size.width - drawnWidth) / 2
      : Math.min(pad, Math.max(size.width - drawnWidth - pad, next.x));
    const y = drawnHeight <= size.height - pad * 2
      ? (size.height - drawnHeight) / 2
      : Math.min(pad, Math.max(size.height - drawnHeight - pad, next.y));
    return { x, y };
  };

  // Keep the photo framed when the frame resizes; fit mode always re-centres.
  useEffect(() => {
    setPosition((current) => constrain(zoom === 1 ? { x: 0, y: 0 } : current, scale));
  }, [size.width, size.height, imageWidth, imageHeight]);

  const zoomTo = (nextZoom: number, anchor?: Point | null) => {
    const clamped = Math.min(MAX_ZOOM, Math.max(1, Number(nextZoom.toFixed(3))));
    const focus = anchor ?? { x: size.width / 2, y: size.height / 2 };
    const nextScale = fitScale * clamped;
    const imageX = (focus.x - position.x) / scale;
    const imageY = (focus.y - position.y) / scale;
    setZoom(clamped);
    setPosition(constrain({ x: focus.x - imageX * nextScale, y: focus.y - imageY * nextScale }, nextScale));
  };

  useImperativeHandle(ref, () => ({
    exportMask: async () => {
      if (!initialMask) throw new Error("The detected cover is still loading");
      return buildMaskBlob(initialMask, imageWidth, imageHeight, strokes);
    },
  }), [initialMask, imageWidth, imageHeight, strokes]);

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

  const pickTool = (next: Tool) => {
    setTool(next);
    if (next === "brush" || next === "erase") setView("cover");
  };

  // Standard editor shortcuts. Ignored while typing in a form field.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTyping(event.target)) return;
      const key = event.key.toLowerCase();
      const command = event.ctrlKey || event.metaKey;
      if (command && key === "z") {
        event.preventDefault();
        if (event.shiftKey) redoStroke(); else undo();
        return;
      }
      if (command && key === "y") { event.preventDefault(); redoStroke(); return; }
      if (command || event.altKey) return;
      if (key === " " && !(event.target instanceof HTMLButtonElement)) {
        event.preventDefault();
        setSpaceHeld(true);
        return;
      }
      if (key === "b") pickTool("brush");
      else if (key === "e") pickTool("erase");
      else if (key === "l") pickTool("loupe");
      else if (key === "h") pickTool("pan");
      else if (key === "[") setBrushSize((value) => Math.max(MIN_BRUSH, value - 4));
      else if (key === "]") setBrushSize((value) => Math.min(MAX_BRUSH, value + 4));
      else if (key === "+" || key === "=") zoomTo(zoom * 1.25);
      else if (key === "-") zoomTo(zoom / 1.25);
      else if (key === "0") zoomTo(1);
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === " ") setSpaceHeld(false);
    };
    const onBlur = () => {
      setSpaceHeld(false);
      endGesture();
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  });

  const effectiveTool: Tool = spaceHeld ? "pan" : tool;
  const painting = (effectiveTool === "brush" || effectiveTool === "erase") && view === "cover";

  const stagePointer = (): Point | null => stageRef.current?.getPointerPosition() ?? null;
  const toImage = (point: Point) => clampPoint((point.x - position.x) / scale, (point.y - position.y) / scale, imageWidth, imageHeight);

  const onPointerDown = (event: Konva.KonvaEventObject<PointerEvent>) => {
    const point = stagePointer();
    if (!point) return;
    if (effectiveTool === "pan" || event.evt.button === 1) {
      panStart.current = { pointer: point, origin: position };
      setPanning(true);
      return;
    }
    if (!painting || event.evt.button > 0) return;
    const [x, y] = toImage(point);
    drawing.current = true;
    setRedo([]);
    // A repeated first point lets a single click leave a round dot.
    setStrokes((current) => [...current, { tool: effectiveTool as Stroke["tool"], points: [x, y, x, y], width: brushSize / scale }]);
  };

  const onPointerMove = () => {
    const point = stagePointer();
    setPointer(point);
    if (!point) return;
    if (panStart.current) {
      const { pointer: start, origin } = panStart.current;
      setPosition(constrain({ x: origin.x + point.x - start.x, y: origin.y + point.y - start.y }, scale));
      return;
    }
    if (!drawing.current) return;
    const [x, y] = toImage(point);
    setStrokes((current) => {
      if (!current.length) return current;
      const next = [...current];
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, points: [...last.points, x, y] };
      return next;
    });
  };

  const endGesture = () => {
    drawing.current = false;
    panStart.current = null;
    setPanning(false);
  };

  const onWheel = (event: Konva.KonvaEventObject<WheelEvent>) => {
    event.evt.preventDefault();
    const factor = event.evt.deltaY < 0 ? 1.15 : 1 / 1.15;
    zoomTo(zoom * factor, stagePointer());
  };

  const ready = Boolean(source && initialMask && size.width);
  const photo = view === "found" && evidence ? evidence : source;
  const mainTransform = { x: position.x, y: position.y, scaleX: scale, scaleY: scale };
  const imageClip = { x: 0, y: 0, width: imageWidth, height: imageHeight };

  const overImage = pointer
    ? pointer.x >= position.x && pointer.y >= position.y && pointer.x <= position.x + imageWidth * scale && pointer.y <= position.y + imageHeight * scale
    : false;
  const showLoupe = effectiveTool === "loupe" && pointer && overImage && !panning;
  const loupeScale = Math.min(Math.max(scale * 3, 1), scale * MAX_ZOOM);
  const loupeTransform = pointer
    ? {
        x: pointer.x - ((pointer.x - position.x) / scale) * loupeScale,
        y: pointer.y - ((pointer.y - position.y) / scale) * loupeScale,
        scaleX: loupeScale,
        scaleY: loupeScale,
      }
    : mainTransform;
  const loupeClip = (context: Konva.Context) => {
    if (!pointer) return;
    context.beginPath();
    context.arc(pointer.x, pointer.y, LOUPE_RADIUS, 0, Math.PI * 2);
    context.closePath();
  };

  const frameClass = [
    "canvas-frame",
    `tool-${effectiveTool}`,
    view === "found" ? "view-found" : "",
    panning ? "is-panning" : "",
  ].join(" ");

  return (
    <section className="editor" aria-label="Cover editor">
      <div className="editor-toolbar" role="group" aria-label="Editing tools">
        <div className="segmented" role="group" aria-label="What to show">
          <button aria-pressed={view === "cover"} onClick={() => setView("cover")}>Cover</button>
          <button aria-pressed={view === "found"} onClick={() => setView("found")} title="Show what the detectors found">Found</button>
        </div>
        <div className="tool-group" role="group" aria-label="Tools">
          {TOOLS.map(({ key, label, shortcut, icon: Icon }) => (
            <button key={key} className="tool" aria-pressed={tool === key} aria-label={label} title={`${label} (${shortcut})`} onClick={() => pickTool(key)}>
              <Icon aria-hidden="true" />
            </button>
          ))}
        </div>
        <div className="tool-group" role="group" aria-label="History">
          <button className="tool" onClick={undo} disabled={!strokes.length} aria-label="Undo" title="Undo (Ctrl+Z)"><ArrowCounterClockwise aria-hidden="true" /></button>
          <button className="tool" onClick={redoStroke} disabled={!redo.length} aria-label="Redo" title="Redo (Ctrl+Shift+Z)"><ArrowClockwise aria-hidden="true" /></button>
          <button className="tool" onClick={() => { setStrokes([]); setRedo([]); }} disabled={!strokes.length} aria-label="Discard my changes" title="Discard my changes"><Broom aria-hidden="true" /></button>
        </div>
        <label className="brush-size">
          <span>Size</span>
          <input type="range" aria-label="Brush size" min={MIN_BRUSH} max={MAX_BRUSH} value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
          <output className="data">{brushSize}</output>
        </label>
        <div className="tool-group zoom" role="group" aria-label="Zoom">
          <button className="tool" onClick={() => zoomTo(zoom / 1.25)} disabled={zoom <= 1} aria-label="Zoom out" title="Zoom out (−)"><MagnifyingGlassMinus aria-hidden="true" /></button>
          <output className="data zoom-level" aria-live="polite">{Math.round(zoom * 100)}%</output>
          <button className="tool" onClick={() => zoomTo(zoom * 1.25)} disabled={zoom >= MAX_ZOOM} aria-label="Zoom in" title="Zoom in (+)"><MagnifyingGlassPlus aria-hidden="true" /></button>
          <button className="tool" onClick={() => zoomTo(1)} disabled={zoom === 1} aria-label="Fit to frame" title="Fit to frame (0)"><CornersOut aria-hidden="true" /></button>
        </div>
      </div>

      <div className={frameClass} ref={containerRef} style={{ aspectRatio: `${imageWidth} / ${imageHeight}` }}>
        {!ready ? (
          <div className="canvas-loading"><span className="skeleton-bar" />Loading your photo</div>
        ) : (
          <Stage
            ref={stageRef}
            width={size.width}
            height={size.height}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endGesture}
            onPointerCancel={endGesture}
            onPointerLeave={() => { endGesture(); setPointer(null); }}
            onWheel={onWheel}
          >
            <Layer listening={false}>
              <Group {...mainTransform} clip={imageClip}>
                <KonvaImage image={photo ?? undefined} width={imageWidth} height={imageHeight} />
              </Group>
            </Layer>
            {view === "cover" && (
              <Layer listening={false} opacity={COVER_OPACITY}>
                <Group {...mainTransform} clip={imageClip}>
                  <Cover stencil={stencil} strokes={strokes} hatch={hatch} width={imageWidth} height={imageHeight} patternScale={1 / scale} />
                </Group>
              </Layer>
            )}
            {showLoupe && pointer && (
              <Layer listening={false}>
                <Circle x={pointer.x} y={pointer.y} radius={LOUPE_RADIUS} fill={colors.lightBox} />
                <Group clipFunc={loupeClip}>
                  <Group {...loupeTransform} clip={imageClip}>
                    <KonvaImage image={photo ?? undefined} width={imageWidth} height={imageHeight} />
                  </Group>
                </Group>
              </Layer>
            )}
            {showLoupe && pointer && view === "cover" && (
              <Layer listening={false} opacity={COVER_OPACITY}>
                <Group clipFunc={loupeClip}>
                  <Group {...loupeTransform} clip={imageClip}>
                    <Cover stencil={stencil} strokes={strokes} hatch={hatch} width={imageWidth} height={imageHeight} patternScale={1 / loupeScale} />
                  </Group>
                </Group>
              </Layer>
            )}
            <Layer listening={false}>
              {showLoupe && pointer && (
                <>
                  <Circle x={pointer.x} y={pointer.y} radius={LOUPE_RADIUS + 1} stroke="rgba(0,0,0,0.55)" strokeWidth={5} />
                  <Circle x={pointer.x} y={pointer.y} radius={LOUPE_RADIUS} stroke={colors.paper} strokeWidth={2} />
                  <Line points={[pointer.x - 7, pointer.y, pointer.x + 7, pointer.y]} stroke={colors.marker} strokeWidth={1.5} />
                  <Line points={[pointer.x, pointer.y - 7, pointer.x, pointer.y + 7]} stroke={colors.marker} strokeWidth={1.5} />
                </>
              )}
              {painting && pointer && !panning && (
                <>
                  <Circle x={pointer.x} y={pointer.y} radius={brushSize / 2} stroke="rgba(0,0,0,0.6)" strokeWidth={3} />
                  <Circle
                    x={pointer.x}
                    y={pointer.y}
                    radius={brushSize / 2}
                    stroke={effectiveTool === "erase" ? colors.paper : colors.marker}
                    strokeWidth={1.5}
                    dash={effectiveTool === "erase" ? [4, 3] : undefined}
                  />
                </>
              )}
            </Layer>
          </Stage>
        )}
        <div className="canvas-readout" aria-hidden="true">
          <span className="data">{imageWidth} × {imageHeight}</span>
          <span>{view === "found" ? "Showing what was found · switch to Cover to edit" : HINTS[effectiveTool]}</span>
        </div>
      </div>
    </section>
  );
});

export default MaskEditor;
