import { useState } from 'react';
import type { ReactNode } from 'react';
import { ChevronDown, SpellCheck2 } from 'lucide-react';
import clsx from 'clsx';
import type { QueryProcessingInfo } from '@/api/types';
import { Badge } from '@/components/ui/Badge';

interface QueryProcessingPanelProps {
  data: QueryProcessingInfo;
}

export function QueryProcessingPanel({ data }: QueryProcessingPanelProps) {
  const [open, setOpen] = useState(true);
  const queryChanged = data.original_query !== data.processed_query;

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
          <span className="text-sm font-medium text-text">
            Query processing
          </span>
          {data.spell_corrected && (
            <Badge tone="info">
              <SpellCheck2 size={11} />
              Spell-corrected
            </Badge>
          )}
        </div>
        <ChevronDown
          size={16}
          className={clsx(
            'text-text-muted transition-transform duration-200',
            open && 'rotate-180',
          )}
        />
      </button>

      <div
        className={clsx(
          'grid transition-[grid-template-rows] duration-300 ease-out',
          open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]',
        )}
      >
        <div className="overflow-hidden">
          <div className="space-y-3 px-4 pb-4 text-sm">
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Original query">
                <span className="font-mono text-text">
                  {data.original_query}
                </span>
              </Field>
              <Field label="Processed query">
                <span
                  className={clsx(
                    'font-mono',
                    queryChanged ? 'text-accent-warm' : 'text-text',
                  )}
                >
                  {data.processed_query}
                </span>
              </Field>
            </div>

            {data.expanded_terms.length > 0 && (
              <Field
                label={`Expanded terms (${String(data.expanded_terms.length)})`}
              >
                <ChipList items={data.expanded_terms} />
              </Field>
            )}

            {data.query_tokens.length > 0 && (
              <Field
                label={`Query tokens (${String(data.query_tokens.length)})`}
              >
                <ChipList items={data.query_tokens} />
              </Field>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium uppercase tracking-wide text-text-muted">
        {label}
      </p>
      {children}
    </div>
  );
}

function ChipList({ items }: { items: string[] }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item, index) => (
        <span
          // Tokens/terms can repeat; index keeps keys stable without implying identity.
          key={`${item}-${String(index)}`}
          className="rounded-full border border-border bg-surface-elevated px-2 py-0.5 font-mono text-xs text-text-secondary"
        >
          {item}
        </span>
      ))}
    </div>
  );
}
