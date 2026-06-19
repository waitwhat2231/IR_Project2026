import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, SearchCode } from "lucide-react";
import { useSearch } from "@/hooks/useSearch";
import { useDatasets } from "@/hooks/useDatasets";
import { useSearchConfigStore } from "@/stores/searchConfigStore";
import { useUiStore } from "@/stores/uiStore";
import { RETRIEVAL_MODE_META } from "@/constants/retrievalModels";
import { SearchBar } from "@/components/search/SearchBar";
import { SettingsPanel } from "@/components/search/SettingsPanel";
import { MetaStrip } from "@/components/search/MetaStrip";
import { QueryProcessingPanel } from "@/components/search/QueryProcessingPanel";
import { ResultsList } from "@/components/results/ResultsList";
import { ResultsSkeleton } from "@/components/results/ResultsSkeleton";
import { EmptyState } from "@/components/results/EmptyState";
import { RawJsonView } from "@/components/results/RawJsonView";
import { Badge } from "@/components/ui/Badge";

const EXAMPLE_QUERIES = [
  "Should teachers get tenure?",
  "Is vaping with e-cigarettes safe?",
  "gun control laws",
  "artificial intelligence ethics",
];

export function SearchView() {
  const [inputValue, setInputValue] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);

  const datasetsQuery = useDatasets();
  const dataset = datasetsQuery.data?.datasets[0];

  const config = useSearchConfigStore();
  const openSettings = useUiStore((s) => s.toggleSettingsPanel);
  const searchMutation = useSearch();

  const runSearch = (query: string): void => {
    if (!dataset) return;
    setSubmittedQuery(query);
    searchMutation.mutate({
      dataset: dataset.name,
      query,
      execution_mode: config.executionMode,
      retrieval_mode: config.retrievalMode,
      sparse_method: config.sparseMethod,
      dense_method: config.denseMethod,
      bm25_k1: config.bm25K1,
      bm25_b: config.bm25B,
      alpha: config.alpha,
      cascade_top_n: config.cascadeTopN,
      top_k: config.topK,
      include_snippet_chars: config.includeSnippetChars,
    });
  };

  const handleSubmit = (query: string): void => {
    setInputValue(query);
    runSearch(query);
  };

  return (
    <>
      <SettingsPanel />

      <AnimatePresence mode="wait">
        {submittedQuery === null ? (
          <motion.div
            key="hero"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="flex min-h-[calc(100dvh-4rem)] flex-col items-center justify-center px-5"
          >
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              className="mb-6 flex items-center gap-3"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-subtle text-accent">
                <SearchCode size={24} strokeWidth={2.25} />
              </div>
              <h1 className="text-balance text-2xl font-semibold tracking-tight text-text">
                Search the corpus
              </h1>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
              className="w-full max-w-xl"
            >
              <SearchBar
                value={inputValue}
                onChange={setInputValue}
                onSubmit={handleSubmit}
                variant="hero"
                autoFocus
                onOpenSettings={openSettings}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.16, ease: [0.16, 1, 0.3, 1] }}
              className="mt-5 flex flex-wrap items-center justify-center gap-2"
            >
              {dataset && (
                <Badge tone="neutral" mono>
                  {dataset.name}
                </Badge>
              )}
              <Badge tone="accent">
                {RETRIEVAL_MODE_META[config.retrievalMode].label}
              </Badge>
              <Badge tone="neutral" className="capitalize">
                {config.executionMode}
              </Badge>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.24 }}
              className="mt-8 flex flex-wrap items-center justify-center gap-2"
            >
              {EXAMPLE_QUERIES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => {
                    handleSubmit(example);
                  }}
                  className="cursor-pointer rounded-full border border-border bg-surface px-3.5 py-1.5 text-xs text-text-secondary transition-colors duration-150 hover:border-border-strong hover:text-text"
                >
                  {example}
                </button>
              ))}
            </motion.div>
          </motion.div>
        ) : (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
            className="mx-auto max-w-3xl px-5 py-6"
          >
            <SearchBar
              value={inputValue}
              onChange={setInputValue}
              onSubmit={handleSubmit}
              variant="compact"
              isSearching={searchMutation.isPending}
              onOpenSettings={openSettings}
            />

            <div className="mt-5 space-y-5">
              {searchMutation.isPending && <ResultsSkeleton />}

              {searchMutation.isError && (
                <div className="flex items-start gap-3 rounded-card border border-danger/30 bg-danger-subtle px-4 py-3.5">
                  <AlertTriangle size={17} className="mt-0.5 shrink-0 text-danger" />
                  <div className="space-y-0.5">
                    <p className="text-sm font-medium text-text">
                      Search failed
                    </p>
                    <p className="text-xs text-text-secondary">
                      {searchMutation.error instanceof Error
                        ? searchMutation.error.message
                        : "Unknown error"}
                    </p>
                  </div>
                </div>
              )}

              {searchMutation.isSuccess && (
                <>
                  <MetaStrip response={searchMutation.data} />
                  <QueryProcessingPanel data={searchMutation.data.query_processing} />

                  {searchMutation.data.results.length === 0 ? (
                    <EmptyState query={submittedQuery} />
                  ) : (
                    <ResultsList results={searchMutation.data.results} />
                  )}

                  <RawJsonView data={searchMutation.data} />
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
