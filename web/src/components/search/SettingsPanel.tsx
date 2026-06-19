import { AnimatePresence, motion } from "motion/react";
import { Layers, RotateCcw, X } from "lucide-react";
import clsx from "clsx";
import { useUiStore } from "@/stores/uiStore";
import { useSearchConfigStore } from "@/stores/searchConfigStore";
import { useRetrievalOptions } from "@/hooks/useRetrievalOptions";
import {
  RETRIEVAL_MODE_META,
  RETRIEVAL_MODES,
  isHybridMode,
  usesBm25Params,
} from "@/constants/retrievalModels";
import { IconButton } from "@/components/ui/IconButton";
import { Segmented } from "@/components/ui/Segmented";
import { Slider } from "@/components/ui/Slider";
import { Button } from "@/components/ui/Button";

export function SettingsPanel() {
  const open = useUiStore((s) => s.settingsPanelOpen);
  const close = useUiStore((s) => s.closeSettingsPanel);
  const optionsQuery = useRetrievalOptions();

  const config = useSearchConfigStore();

  const hybrid = isHybridMode(config.retrievalMode);
  const showBm25Params = usesBm25Params(config.retrievalMode, config.sparseMethod);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={close}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-[2px]"
            aria-hidden="true"
          />

          <motion.aside
            key="panel"
            role="dialog"
            aria-modal="true"
            aria-label="Search and model settings"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-border-strong bg-surface shadow-2xl"
          >
            <div className="flex items-center justify-between border-b border-border-subtle px-5 py-4">
              <div className="flex items-center gap-2">
                <Layers size={16} className="text-accent" />
                <h2 className="text-sm font-semibold text-text">
                  Search &amp; model settings
                </h2>
              </div>
              <IconButton aria-label="Close settings" onClick={close}>
                <X size={17} />
              </IconButton>
            </div>

            <div className="flex-1 space-y-7 overflow-y-auto px-5 py-6">
              {/* Execution mode */}
              <section className="space-y-2.5">
                <SectionLabel
                  title="Execution mode"
                  hint="Enhanced adds spell correction, synonym expansion, and search history."
                />
                <Segmented
                  aria-label="Execution mode"
                  value={config.executionMode}
                  onChange={config.setExecutionMode}
                  options={[
                    { value: "basic", label: "Basic" },
                    { value: "enhanced", label: "Enhanced" },
                  ]}
                />
              </section>

              {/* Retrieval model */}
              <section className="space-y-2.5">
                <SectionLabel
                  title="Retrieval model"
                  hint="Representation used to match the query against the corpus."
                />
                <div className="grid grid-cols-2 gap-2">
                  {RETRIEVAL_MODES.map((mode) => {
                    const meta = RETRIEVAL_MODE_META[mode];
                    const selected = config.retrievalMode === mode;
                    return (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => {
                          config.setRetrievalMode(mode);
                        }}
                        aria-pressed={selected}
                        className={clsx(
                          "rounded-control border px-3 py-2.5 text-left transition-colors duration-150 cursor-pointer",
                          selected
                            ? "border-accent/60 bg-accent-subtle"
                            : "border-border bg-surface-elevated hover:border-border-strong",
                        )}
                      >
                        <p
                          className={clsx(
                            "text-sm font-semibold",
                            selected ? "text-accent-warm" : "text-text",
                          )}
                        >
                          {meta.label}
                        </p>
                        <p className="mt-0.5 text-xs leading-snug text-text-muted">
                          {meta.description}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </section>

              {/* Hybrid legs */}
              {hybrid && (
                <section className="space-y-4 rounded-card border border-border-subtle bg-bg/40 p-4">
                  <SectionLabel
                    title="Hybrid composition"
                    hint="Choose the sparse and dense representations being combined."
                  />
                  <div className="space-y-1.5">
                    <p className="text-xs font-medium text-text-secondary">
                      Sparse leg
                    </p>
                    <Segmented
                      aria-label="Sparse method"
                      value={config.sparseMethod}
                      onChange={config.setSparseMethod}
                      options={[
                        { value: "bm25", label: "BM25" },
                        { value: "tfidf", label: "TF-IDF" },
                      ]}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <p className="text-xs font-medium text-text-secondary">
                      Dense leg
                    </p>
                    <Segmented
                      aria-label="Dense method"
                      value={config.denseMethod}
                      onChange={config.setDenseMethod}
                      options={[
                        { value: "sbert", label: "SBERT" },
                        { value: "word2vec", label: "Word2Vec" },
                      ]}
                    />
                  </div>

                  {config.retrievalMode === "hybrid_parallel" && (
                    <Slider
                      label="Fusion weight (α)"
                      value={config.alpha}
                      min={0}
                      max={1}
                      step={0.05}
                      onChange={config.setAlpha}
                      formatValue={(v) => v.toFixed(2)}
                      description={`Sparse weight ${config.alpha.toFixed(2)} · dense weight ${(1 - config.alpha).toFixed(2)}`}
                    />
                  )}

                  {config.retrievalMode === "hybrid_serial" && (
                    <Slider
                      label="Cascade candidates"
                      value={config.cascadeTopN}
                      min={10}
                      max={5000}
                      step={10}
                      onChange={config.setCascadeTopN}
                      description="Candidates BM25 hands to the dense reranker before truncating to Top K."
                    />
                  )}
                </section>
              )}

              {/* BM25 parameters */}
              {showBm25Params && (
                <section className="space-y-4">
                  <SectionLabel
                    title="BM25 parameters"
                    hint="Probabilistic model tuning — adjustable per query."
                  />
                  <Slider
                    label="k1 (term frequency saturation)"
                    value={config.bm25K1}
                    min={0.1}
                    max={3}
                    step={0.1}
                    onChange={config.setBm25K1}
                    formatValue={(v) => v.toFixed(1)}
                    description="Higher values let repeated terms keep contributing to the score for longer."
                  />
                  <Slider
                    label="b (length normalization)"
                    value={config.bm25B}
                    min={0}
                    max={1}
                    step={0.05}
                    onChange={config.setBm25B}
                    formatValue={(v) => v.toFixed(2)}
                    description="0 disables document-length normalization entirely; 1 applies it fully."
                  />
                </section>
              )}

              {/* Result shaping */}
              <section className="space-y-4">
                <SectionLabel title="Results" />
                <Slider
                  label="Top K"
                  value={config.topK}
                  min={1}
                  max={100}
                  step={1}
                  onChange={config.setTopK}
                  description="Number of ranked documents returned."
                />
                <Slider
                  label="Snippet length"
                  value={config.includeSnippetChars}
                  min={0}
                  max={2000}
                  step={50}
                  onChange={config.setIncludeSnippetChars}
                  formatValue={(v) => (v === 0 ? "Omit text" : `${String(v)} chars`)}
                  description="0 omits the document body entirely; raise it for full, untruncated text."
                />
              </section>
            </div>

            <div className="border-t border-border-subtle px-5 py-4">
              <Button
                variant="ghost"
                size="sm"
                disabled={!optionsQuery.data}
                onClick={() => {
                  if (optionsQuery.data) {
                    config.resetToDefaults(optionsQuery.data.defaults);
                  }
                }}
                className="w-full"
              >
                <RotateCcw size={14} />
                Reset to server defaults
              </Button>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function SectionLabel({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="space-y-0.5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">
        {title}
      </h3>
      {hint && <p className="text-xs leading-relaxed text-text-muted">{hint}</p>}
    </div>
  );
}
