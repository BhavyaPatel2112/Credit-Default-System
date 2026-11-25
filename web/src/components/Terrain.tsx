import { useEffect, useRef } from "react";
import { contours } from "d3-contour";
import { createNoise3D } from "simplex-noise";

// A slowly shifting topographic map: contour lines of a noise "risk surface".
// Drawn on a canvas with d3-contour; pauses off-screen and stays still when
// the viewer prefers reduced motion.
export function Terrain({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const noise = createNoise3D(seeded(7));
    const NX = 120, NY = 68;
    const values = new Float64Array(NX * NY);
    const gen = contours().size([NX, NY]).thresholds(18);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let t = 0, raf = 0, last = 0, visible = true, w = 0, h = 0;

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = r.width; h = r.height;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    };

    const field = () => {
      for (let j = 0; j < NY; j++)
        for (let i = 0; i < NX; i++) {
          const x = i / NX, y = j / NY;
          // two octaves of noise plus a ridge that rises towards the right
          values[j * NX + i] =
            noise(x * 2.1, y * 1.6, t) * 0.62 + noise(x * 5.2, y * 4.3, t * 1.7 + 9) * 0.22 + x * 0.55 - y * 0.18;
        }
    };

    const draw = () => {
      if (!w) return;
      field();
      ctx.clearRect(0, 0, w, h);
      const sx = w / (NX - 1), sy = h / (NY - 1);
      const levels = gen(Array.from(values));
      levels.forEach((c, k) => {
        const strength = k / levels.length;
        ctx.beginPath();
        for (const poly of c.coordinates)
          for (const ring of poly)
            ring.forEach(([x, y], n) => (n ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)));
        const index = k % 4 === 0;
        ctx.strokeStyle = `rgba(232, 231, 224, ${(index ? 0.2 : 0.07) + strength * 0.14})`;
        ctx.lineWidth = index ? 1.1 : 0.7;
        ctx.stroke();
      });
    };

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      if (!visible || now - last < 50) return;
      last = now; t += 0.0022; draw();
    };

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting));
    io.observe(canvas);
    resize();
    if (!reduce) raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); };
  }, []);

  return <canvas ref={ref} aria-hidden className={`block h-full w-full ${className}`} />;
}

function seeded(s: number) {
  return () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
}
