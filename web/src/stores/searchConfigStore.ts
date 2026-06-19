import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  DenseMethod,
  ExecutionMode,
  RetrievalMode,
  RetrievalOptionsDefaults,
  SparseMethod,
} from '@/api/types';

interface SearchConfigState {
  executionMode: ExecutionMode;
  retrievalMode: RetrievalMode;
  sparseMethod: SparseMethod;
  denseMethod: DenseMethod;
  bm25K1: number;
  bm25B: number;
  alpha: number;
  cascadeTopN: number;
  topK: number;
  includeSnippetChars: number;

  /** Marks whether defaults from GET /api/v1/options have been applied yet. */
  hydratedFromServer: boolean;

  setExecutionMode: (mode: ExecutionMode) => void;
  setRetrievalMode: (mode: RetrievalMode) => void;
  setSparseMethod: (method: SparseMethod) => void;
  setDenseMethod: (method: DenseMethod) => void;
  setBm25K1: (value: number) => void;
  setBm25B: (value: number) => void;
  setAlpha: (value: number) => void;
  setCascadeTopN: (value: number) => void;
  setTopK: (value: number) => void;
  setIncludeSnippetChars: (value: number) => void;
  hydrateDefaults: (defaults: RetrievalOptionsDefaults) => void;
  resetToDefaults: (defaults: RetrievalOptionsDefaults) => void;
}

const FALLBACK_DEFAULTS: Omit<
  SearchConfigState,
  | 'hydratedFromServer'
  | 'setExecutionMode'
  | 'setRetrievalMode'
  | 'setSparseMethod'
  | 'setDenseMethod'
  | 'setBm25K1'
  | 'setBm25B'
  | 'setAlpha'
  | 'setCascadeTopN'
  | 'setTopK'
  | 'setIncludeSnippetChars'
  | 'hydrateDefaults'
  | 'resetToDefaults'
> = {
  executionMode: 'basic',
  retrievalMode: 'bm25',
  sparseMethod: 'bm25',
  denseMethod: 'sbert',
  bm25K1: 1.5,
  bm25B: 0.75,
  alpha: 0.5,
  cascadeTopN: 200,
  topK: 10,
  includeSnippetChars: 300,
};

export const useSearchConfigStore = create<SearchConfigState>()(
  persist(
    (set) => ({
      ...FALLBACK_DEFAULTS,
      hydratedFromServer: false,

      setExecutionMode: (executionMode) => {
        set({ executionMode });
      },
      setRetrievalMode: (retrievalMode) => {
        set({ retrievalMode });
      },
      setSparseMethod: (sparseMethod) => {
        set({ sparseMethod });
      },
      setDenseMethod: (denseMethod) => {
        set({ denseMethod });
      },
      setBm25K1: (bm25K1) => {
        set({ bm25K1 });
      },
      setBm25B: (bm25B) => {
        set({ bm25B });
      },
      setAlpha: (alpha) => {
        set({ alpha });
      },
      setCascadeTopN: (cascadeTopN) => {
        set({ cascadeTopN });
      },
      setTopK: (topK) => {
        set({ topK });
      },
      setIncludeSnippetChars: (includeSnippetChars) => {
        set({ includeSnippetChars });
      },
      hydrateDefaults: (defaults) => {
        set((state) =>
          // Only apply server defaults once, and never clobber a value the
          // user already changed in a previous session (persisted).
          state.hydratedFromServer
            ? state
            : { ...state, ...defaults, hydratedFromServer: true },
        );
      },
      resetToDefaults: (defaults) => {
        set({ ...defaults, hydratedFromServer: true });
      },
    }),
    {
      name: 'ir-search-config',
      partialize: (state) => ({
        hydratedFromServer: state.hydratedFromServer,
        executionMode: state.executionMode,
        retrievalMode: state.retrievalMode,
        sparseMethod: state.sparseMethod,
        denseMethod: state.denseMethod,
        bm25K1: state.bm25K1,
        bm25B: state.bm25B,
        alpha: state.alpha,
        cascadeTopN: state.cascadeTopN,
        topK: state.topK,
        includeSnippetChars: state.includeSnippetChars,
      }),
    },
  ),
);
