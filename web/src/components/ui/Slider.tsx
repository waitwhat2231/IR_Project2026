import { useId } from "react";
import clsx from "clsx";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  formatValue?: (value: number) => string;
  description?: string;
  disabled?: boolean;
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  formatValue,
  description,
  disabled = false,
}: SliderProps) {
  const id = useId();
  const percent = ((value - min) / (max - min)) * 100;

  return (
    <div className={clsx("space-y-2", disabled && "opacity-40")}>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-sm font-medium text-text">
          {label}
        </label>
        <span className="font-mono text-sm tabular-nums text-accent-warm">
          {formatValue ? formatValue(value) : value}
        </span>
      </div>

      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => {
          onChange(Number(event.target.value));
        }}
        style={{
          background: `linear-gradient(to right, var(--color-accent) ${String(percent)}%, var(--color-border) ${String(percent)}%)`,
        }}
        className={clsx(
          "h-1.5 w-full appearance-none rounded-full outline-none",
          "disabled:cursor-not-allowed",
          !disabled && "cursor-pointer",
          "[&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4",
          "[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full",
          "[&::-webkit-slider-thumb]:bg-accent-warm [&::-webkit-slider-thumb]:border-2",
          "[&::-webkit-slider-thumb]:border-bg [&::-webkit-slider-thumb]:shadow-[0_0_0_1px_var(--color-accent)]",
          "[&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:duration-150",
          "[&:active::-webkit-slider-thumb]:scale-110",
          "[&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full",
          "[&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-bg [&::-moz-range-thumb]:bg-accent-warm",
        )}
      />

      {description && (
        <p className="text-xs leading-relaxed text-text-muted">
          {description}
        </p>
      )}
    </div>
  );
}
