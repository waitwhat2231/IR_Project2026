import clsx from 'clsx';

interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

interface SegmentedProps<T extends string> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  'aria-label': string;
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  disabled = false,
  'aria-label': ariaLabel,
}: SegmentedProps<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={clsx(
        'inline-flex w-full rounded-control border border-border bg-surface p-1',
        disabled && 'pointer-events-none opacity-40',
      )}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => {
              onChange(option.value);
            }}
            className={clsx(
              'flex-1 rounded-[calc(var(--radius-control)-4px)] px-3 py-1.5 text-sm font-medium',
              'transition-colors duration-150 ease-out cursor-pointer whitespace-nowrap',
              selected
                ? 'bg-accent text-[#140a04] shadow-sm'
                : 'text-text-secondary hover:text-text',
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
