import clsx from "clsx";

interface SpinnerProps {
  size?: number;
  className?: string;
}

export function Spinner({ size = 18, className }: SpinnerProps) {
  return (
    <svg
      role="status"
      aria-label="Loading"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={clsx("animate-spin", className)}
    >
      <circle
        cx="12"
        cy="12"
        r="9.5"
        stroke="currentColor"
        strokeOpacity="0.2"
        strokeWidth="3"
      />
      <path
        d="M21.5 12c0-5.25-4.25-9.5-9.5-9.5"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}
