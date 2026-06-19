import { forwardRef } from "react";
import type { ButtonHTMLAttributes } from "react";
import clsx from "clsx";

type IconButtonVariant = "ghost" | "outline" | "solid";

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  "aria-label": string;
  variant?: IconButtonVariant;
  active?: boolean;
}

const VARIANT_CLASSES: Record<IconButtonVariant, string> = {
  ghost: "bg-transparent text-text-secondary hover:text-text hover:bg-surface-hover",
  outline:
    "bg-surface border border-border text-text-secondary hover:text-text hover:border-border-strong",
  solid: "bg-surface-elevated border border-border-strong text-text hover:bg-surface-hover",
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton(
    { variant = "ghost", active = false, className, ...props },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type="button"
        className={clsx(
          "inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-control",
          "transition-colors duration-150 ease-out cursor-pointer",
          "active:scale-[0.96] transition-transform",
          active && "text-accent bg-accent-subtle",
          VARIANT_CLASSES[variant],
          className,
        )}
        {...props}
      />
    );
  },
);
