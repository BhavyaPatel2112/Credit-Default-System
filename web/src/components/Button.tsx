import { motion } from "motion/react";
import { forwardRef, type ComponentProps } from "react";
import clsx from "clsx";
import { twMerge } from "tailwind-merge";

const cx = (...a: Parameters<typeof clsx>) => twMerge(clsx(...a));

type Variant = "primary" | "secondary" | "ghost" | "light";

const styles: Record<Variant, string> = {
  primary: "bg-ink text-stone hover:bg-[#2b2b29]",
  secondary: "bg-transparent text-ink ring-1 ring-inset ring-line hover:ring-ink/40 hover:bg-white/50",
  ghost: "bg-transparent text-soft hover:text-ink hover:bg-stone-2",
  light: "bg-stone text-ink hover:bg-white",
};

type Props = Omit<ComponentProps<typeof motion.button>, "ref"> & { variant?: Variant; size?: "sm" | "md" | "lg" };

// Press feedback is a small scale-down: motion that answers the click.
export const Button = forwardRef<HTMLButtonElement, Props>(function Button(
  { variant = "primary", size = "md", className, children, ...rest }, ref,
) {
  const sz = { sm: "h-8 px-3.5 text-[13px]", md: "h-10 px-5 text-sm", lg: "h-12 px-6 text-[15px]" }[size];
  return (
    <motion.button
      ref={ref}
      whileTap={{ scale: 0.97 }}
      transition={{ type: "spring", stiffness: 500, damping: 30 }}
      className={cx(
        "inline-flex select-none items-center justify-center gap-2 rounded-full font-medium transition-colors duration-200 disabled:pointer-events-none disabled:opacity-40",
        styles[variant], sz, className,
      )}
      {...rest}
    >
      {children}
    </motion.button>
  );
});

export function LinkButton({ href, variant = "primary", size = "md", className, children }: {
  href: string; variant?: Variant; size?: "sm" | "md" | "lg"; className?: string; children: React.ReactNode;
}) {
  const sz = { sm: "h-8 px-3.5 text-[13px]", md: "h-10 px-5 text-sm", lg: "h-12 px-6 text-[15px]" }[size];
  return (
    <motion.a
      href={href}
      whileTap={{ scale: 0.97 }}
      className={cx("inline-flex items-center justify-center gap-2 rounded-full font-medium transition-colors duration-200", styles[variant], sz, className)}
    >
      {children}
    </motion.a>
  );
}
