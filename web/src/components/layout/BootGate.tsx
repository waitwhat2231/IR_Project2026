import { useEffect } from "react";
import type { ReactNode } from "react";
import { motion } from "motion/react";
import { AlertTriangle, RotateCw, SearchCode } from "lucide-react";
import { useDatasets } from "@/hooks/useDatasets";
import { useRetrievalOptions } from "@/hooks/useRetrievalOptions";
import { useSearchConfigStore } from "@/stores/searchConfigStore";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";

interface BootGateProps {
  children: ReactNode;
}

/**
 * GET /api/v1/datasets loads the dataset and every representation
 * (TF-IDF, BM25, SBERT, Word2Vec) into the backend's memory on first hit.
 * Letting a search fire while that's in flight risks doubling memory
 * pressure on the same process, so this gate fully blocks interaction
 * (not just disables a button) until it resolves once.
 */
export function BootGate({ children }: BootGateProps) {
  const datasetsQuery = useDatasets();
  const optionsQuery = useRetrievalOptions();
  const hydrateDefaults = useSearchConfigStore((s) => s.hydrateDefaults);

  useEffect(() => {
    if (optionsQuery.data) {
      hydrateDefaults(optionsQuery.data.defaults);
    }
  }, [optionsQuery.data, hydrateDefaults]);

  if (datasetsQuery.isError) {
    return (
      <FullScreenState
        icon={<AlertTriangle size={28} className="text-danger" />}
        title="Couldn't reach the search backend"
        message={
          datasetsQuery.error instanceof Error
            ? datasetsQuery.error.message
            : "The gateway didn't respond. Confirm it's running on the configured URL."
        }
        action={
          <Button
            variant="secondary"
            onClick={() => {
              void datasetsQuery.refetch();
            }}
          >
            <RotateCw size={15} />
            Retry
          </Button>
        }
      />
    );
  }

  if (datasetsQuery.isPending) {
    return (
      <FullScreenState
        icon={<Spinner size={28} className="text-accent" />}
        title="Loading the dataset into memory"
        message="The first request spins up every representation — TF-IDF, BM25, SBERT, and Word2Vec — on the backend. This can take a while. Search will unlock automatically the moment it's ready."
      />
    );
  }

  return <>{children}</>;
}

interface FullScreenStateProps {
  icon: ReactNode;
  title: string;
  message: string;
  action?: ReactNode;
}

function FullScreenState({ icon, title, message, action }: FullScreenStateProps) {
  return (
    <div className="flex min-h-dvh items-center justify-center bg-bg px-6">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="flex max-w-md flex-col items-center gap-5 text-center"
      >
        <div className="flex items-center gap-2 text-text-muted">
          <SearchCode size={18} />
          <span className="font-mono text-xs uppercase tracking-[0.2em]">
            IR Search Console
          </span>
        </div>

        <div className="flex h-14 w-14 items-center justify-center rounded-full border border-border bg-surface">
          {icon}
        </div>

        <div className="space-y-2">
          <h1 className="text-lg font-semibold text-text">{title}</h1>
          <p className="text-sm leading-relaxed text-text-secondary">
            {message}
          </p>
        </div>

        {action}
      </motion.div>
    </div>
  );
}
