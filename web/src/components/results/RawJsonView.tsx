import { useState } from "react";
import { Braces, ChevronDown } from "lucide-react";
import clsx from "clsx";

interface RawJsonViewProps {
  data: unknown;
}

export function RawJsonView({ data }: RawJsonViewProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-card border border-border-subtle bg-surface/60">
      <button
        type="button"
        onClick={() => {
          setOpen((value) => !value);
        }}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <div className="flex items-center gap-2">
          <Braces size={14} className="text-text-muted" />
          <span className="text-sm font-medium text-text">
            Raw response JSON
          </span>
        </div>
        <ChevronDown
          size={16}
          className={clsx(
            "text-text-muted transition-transform duration-200",
            open && "rotate-180",
          )}
        />
      </button>

      <div
        className={clsx(
          "grid transition-[grid-template-rows] duration-300 ease-out",
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        )}
      >
        <div className="overflow-hidden">
          <pre className="max-h-96 overflow-auto border-t border-border-subtle px-4 py-3 font-mono text-xs leading-relaxed text-text-secondary">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}
