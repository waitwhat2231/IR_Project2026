import { useState } from "react";
import { motion } from "motion/react";
import type { SearchResultItem } from "@/api/types";
import { CopyButton } from "@/components/ui/CopyButton";
import { formatScore } from "@/utils/format";

interface ResultCardProps {
  result: SearchResultItem;
  index: number;
}

const TEXT_PREVIEW_LENGTH = 420;

export function ResultCard({ result, index }: ResultCardProps) {
  const [expanded, setExpanded] = useState(false);
  const text = result.text ?? "";
  const isLong = text.length > TEXT_PREVIEW_LENGTH;
  const displayText = expanded || !isLong ? text : `${text.slice(0, TEXT_PREVIEW_LENGTH)}…`;

  return (
    <motion.article
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.35,
        ease: [0.16, 1, 0.3, 1],
        delay: Math.min(index, 8) * 0.035,
      }}
      className="group rounded-card border border-border-subtle bg-surface/60 p-4 transition-colors duration-150 hover:border-border"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex h-6 min-w-6 shrink-0 items-center justify-center rounded-full bg-surface-elevated px-1.5 font-mono text-[11px] font-semibold text-text-secondary">
            {result.rank}
          </span>
          {result.title ? (
            <h3 className="truncate text-[15px] font-semibold text-text">
              {result.title}
            </h3>
          ) : (
            <h3 className="truncate text-[15px] font-semibold italic text-text-muted">
              (no title field)
            </h3>
          )}
        </div>

        <span className="shrink-0 rounded-full border border-border bg-surface-elevated px-2 py-0.5 font-mono text-xs tabular-nums text-accent-warm">
          {formatScore(result.score)}
        </span>
      </div>

      <div className="mt-2 flex items-center gap-1.5">
        <span className="text-xs text-text-muted">doc_id</span>
        <code className="truncate rounded bg-surface-elevated px-1.5 py-0.5 font-mono text-xs text-text-secondary">
          {result.doc_id}
        </code>
        <CopyButton value={result.doc_id} label="doc_id" />
      </div>

      {result.text !== null && (
        <div className="mt-3">
          <p className="whitespace-pre-line text-sm leading-relaxed text-text-secondary">
            {displayText}
          </p>
          {isLong && (
            <button
              type="button"
              onClick={() => {
                setExpanded((value) => !value);
              }}
              className="mt-2 cursor-pointer text-xs font-medium text-accent-warm hover:text-accent"
            >
              {expanded ? "Show less" : "Show full text"}
            </button>
          )}
        </div>
      )}

      {result.text === null && (
        <p className="mt-3 text-xs italic text-text-muted">
          Text body omitted (include_snippet_chars = 0)
        </p>
      )}
    </motion.article>
  );
}
