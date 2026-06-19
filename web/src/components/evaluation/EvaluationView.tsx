import { useMemo, useState } from 'react';
import { motion } from 'motion/react';
import { AlertTriangle, FlaskConical, Gauge, RotateCw } from 'lucide-react';
import { useDatasets } from '@/hooks/useDatasets';
import { useEvaluationCompare } from '@/hooks/useEvaluationCompare';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { MetricsBarChart } from '@/components/evaluation/MetricsBarChart';
import { ModelEvaluationTable } from '@/components/evaluation/ModelEvaluationTable';
import { RawJsonView } from '@/components/results/RawJsonView';

export function EvaluationView() {
  const [triggered, setTriggered] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);

  const datasetsQuery = useDatasets();
  const dataset = datasetsQuery.data?.datasets[0];

  const compareQuery = useEvaluationCompare(
    dataset?.name ?? '',
    triggered && !!dataset,
  );

  const allModels = useMemo(
    () => [
      ...(compareQuery.data?.baseline?.models ?? []),
      ...(compareQuery.data?.enhanced?.models ?? []),
    ],
    [compareQuery.data],
  );

  const availableMetrics = useMemo(
    () =>
      Array.from(new Set(allModels.flatMap((m) => Object.keys(m.aggregate)))),
    [allModels],
  );

  const activeMetric =
    selectedMetric ??
    availableMetrics.find((k) => k.toLowerCase().includes('map')) ??
    availableMetrics.at(0) ??
    '';

  const runEvaluation = (): void => {
    if (!triggered) {
      setTriggered(true);
    } else {
      void compareQuery.refetch();
    }
  };

  if (!dataset) return null;

  if (!triggered) {
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
              Run the evaluation
            </h1>
            <p className="mx-auto max-w-md text-sm leading-relaxed text-text-secondary">
              Scores MAP, Recall, Precision@10, and nDCG for every retrieval
              model against the qrels for{' '}
              <span className="font-mono text-text-secondary">
                {dataset.name}
              </span>
              , for both the baseline and enhanced phases. This iterates the
              full test query set per model — it isn't fetched automatically.
            </p>
          </div>

          <Button onClick={runEvaluation}>
            <Gauge size={15} />
            Run evaluation
          </Button>
        </motion.div>
      </div>
    );
  }

  if (compareQuery.isPending) {
    return (
      <div className="mx-auto max-w-3xl px-5 py-20">
        <div className="flex flex-col items-center gap-4 rounded-card border border-border-subtle bg-surface/40 px-8 py-16 text-center">
          <FlaskConical size={24} className="animate-pulse text-accent" />
          <p className="text-sm text-text-secondary">
            Scoring every model against the qrels for {dataset.name}…
          </p>
        </div>
      </div>
    );
  }

  if (compareQuery.isError) {
    return (
      <div className="mx-auto max-w-2xl px-5 py-20">
        <div className="flex flex-col items-center gap-4 rounded-card border border-dashed border-border bg-surface/40 px-8 py-14 text-center">
          <AlertTriangle size={26} className="text-text-muted" />
          <div className="space-y-1.5">
            <p className="text-sm font-medium text-text">
              No evaluation results yet
            </p>
            <p className="text-xs leading-relaxed text-text-secondary">
              {compareQuery.error instanceof Error
                ? compareQuery.error.message
                : 'Run the offline RankingEvaluator for this dataset first.'}
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              void compareQuery.refetch();
            }}
          >
            <RotateCw size={14} />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  const { baseline, enhanced } = compareQuery.data;

  return (
    <div className="mx-auto max-w-4xl px-5 py-8">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <FlaskConical size={18} className="text-accent" />
          <h1 className="text-lg font-semibold text-text">
            Evaluation — {dataset.name}
          </h1>
        </div>
        <Button
          variant="secondary"
          size="sm"
          disabled={compareQuery.isFetching}
          onClick={runEvaluation}
        >
          <RotateCw
            size={14}
            className={compareQuery.isFetching ? 'animate-spin' : undefined}
          />
          Re-run
        </Button>
      </div>

      {availableMetrics.length > 0 && (
        <div className="mb-6 space-y-3">
          <div className="flex flex-wrap items-center gap-1.5">
            {availableMetrics.map((metric) => (
              <button
                key={metric}
                type="button"
                onClick={() => {
                  setSelectedMetric(metric);
                }}
                className={
                  metric === activeMetric
                    ? 'cursor-pointer rounded-full border border-accent/60 bg-accent-subtle px-2.5 py-1 font-mono text-xs text-text'
                    : 'cursor-pointer rounded-full border border-border bg-surface-elevated px-2.5 py-1 font-mono text-xs text-text-secondary hover:text-text'
                }
              >
                {metric}
              </button>
            ))}
          </div>

          <div className="rounded-card border border-border-subtle bg-surface/60 p-4">
            <MetricsBarChart
              metric={activeMetric}
              baselineModels={baseline?.models ?? []}
              enhancedModels={enhanced?.models ?? []}
            />
          </div>
        </div>
      )}

      <div className="space-y-5">
        <section className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge tone="info">Baseline</Badge>
            {baseline && (
              <span className="font-mono text-xs text-text-muted">
                {baseline.num_queries} queries · top_k={baseline.top_k} ·{' '}
                {baseline.generated_at}
              </span>
            )}
          </div>
          <ModelEvaluationTable
            title="Baseline"
            models={baseline?.models ?? []}
          />
        </section>

        <section className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge tone="accent">Enhanced</Badge>
            {enhanced && (
              <span className="font-mono text-xs text-text-muted">
                {enhanced.num_queries} queries · top_k={enhanced.top_k} ·{' '}
                {enhanced.generated_at}
              </span>
            )}
          </div>
          <ModelEvaluationTable
            title="Enhanced"
            models={enhanced?.models ?? []}
          />
        </section>

        <RawJsonView data={compareQuery.data} />
      </div>
    </div>
  );
}
