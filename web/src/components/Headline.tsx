import { motion, useReducedMotion } from "motion/react";

// Two-tone serif headline. The first line is the context (grey), the second
// the point (ink). With `reveal`, words sharpen from a blur in sequence:
// the one orchestrated motion moment on the landing page.
export function Headline({
  quiet, loud, as: Tag = "h1", className = "", reveal = false, size = "xl",
}: {
  quiet: string; loud: string; as?: "h1" | "h2"; className?: string; reveal?: boolean; size?: "xl" | "lg" | "md";
}) {
  const reduce = useReducedMotion();
  const sizes = {
    xl: "text-[3.1rem] sm:text-[4.6rem] lg:text-[5.6rem]",
    lg: "text-[2.4rem] sm:text-[3.2rem] lg:text-[3.8rem]",
    md: "text-[2rem] sm:text-[2.5rem]",
  };
  const words = [...quiet.split(" ").map((w) => ({ w, q: true })), ...loud.split(" ").map((w) => ({ w, q: false }))];
  const breakAt = quiet.split(" ").length;
  const animate = reveal && !reduce;

  return (
    <Tag className={`display text-balance ${sizes[size]} ${className}`}>
      {words.map(({ w, q }, i) => (
        <span key={i}>
          {i === breakAt && <br />}
          <motion.span
            className={`inline-block ${q ? "text-mute" : "text-ink"}`}
            initial={animate ? { opacity: 0, filter: "blur(14px)", y: 12 } : false}
            animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
            transition={{ duration: 1.1, delay: 0.15 + i * 0.075, ease: [0.22, 1, 0.36, 1] }}
          >
            {w}
          </motion.span>
          {i < words.length - 1 && i + 1 !== breakAt ? " " : ""}
        </span>
      ))}
    </Tag>
  );
}
