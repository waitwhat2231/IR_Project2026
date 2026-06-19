import { motion } from "motion/react";
import { FlaskConical, Gauge } from "lucide-react";

const PLANNED_METRICS = ["MAP", "Recall", "Precision@10", "nDCG"];

export function EvaluationView() {
  return (
    <div className="mx-auto max-w-3xl px-5 py-20">
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col items-center gap-5 rounded-card border border-dashed border-border bg-surface/40 px-8 py-16 text-center"
      >
        <div className="flex h-14 w-14 items-center justify-center rounded-full border border-border bg-surface text-accent">
          <FlaskConical size={24} />
        </div>

        <div className="space-y-2">
          <h1 className="text-lg font-semibold text-text">
            Evaluation isn't wired up yet
          </h1>
          <p className="mx-auto max-w-md text-sm leading-relaxed text-text-secondary">
            This view is reserved for the evaluation endpoint — running MAP,
            Recall, Precision@10, and nDCG against the qrels for each
            retrieval model, before and after the additional features.
            Plugging it in is a matter of wiring one more request; the
            layout is ready.
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
          {PLANNED_METRICS.map((metric) => (
            <span
              key={metric}
              className="rounded-full border border-border bg-surface-elevated px-3 py-1 font-mono text-xs text-text-muted"
            >
              {metric}
            </span>
          ))}
        </div>

        <div className="flex items-center gap-1.5 pt-1 text-xs text-text-muted">
          <Gauge size={13} />
          <span>Will run on demand — not fetched automatically</span>
        </div>
      </motion.div>
    </div>
  );
}
