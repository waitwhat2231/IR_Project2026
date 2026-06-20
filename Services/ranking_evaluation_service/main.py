# Services/ranking_evaluation_service/main.py
"""
Main — entry point of the Evaluation Service (Requirement 8).

What this file does, step by step:
  1) Loads all models once via HybridRetriever (BM25 + TF-IDF + SBERT + Word2Vec).
     We reuse the same loaded objects to evaluate the single models too -> saves memory.
  2) Builds a lightweight BM25Retriever that reuses the already-loaded index
     (without reloading it from disk).
  3) Evaluates each model: TF-IDF, BM25, SBERT, Word2Vec, Hybrid-Parallel, Hybrid-Serial.
  4) Prints a brief comparison table and saves all results in an organized way
     under data/evaluation/.

Run:
    conda activate ir_project
    python Services/ranking_evaluation_service/main.py
Options:
    --phase  baseline|enhanced   (default baseline = before extra features)
    --top_k  retrieval depth      (default 1000)
    --models comma-separated list (default: all)
             values: tfidf,bm25,sbert,word2vec,hybrid_parallel,hybrid_serial

What baseline vs enhanced actually means:
    baseline  — queries are used exactly as stored in processed_queries.json.
                No spell correction, no synonym expansion.
    enhanced  — each query is first passed through QueryRefiner:
                  1. spell-correct the original text
                  2. expand with WordNet synonyms
                  3. re-preprocess the expanded text to get new tokens
                The retrievers then see a richer, corrected version of each query.
                Comparing baseline vs enhanced in the API shows whether
                query refinement genuinely improves retrieval quality.

Note: `build_bm25_retriever()` and `apply_enhanced_refinement()` below are
imported directly by Services/gateway/main.py too, so that an evaluation
triggered on-demand through the API (when a cached result is missing) runs
through the exact same logic as running this file from the CLI.
"""

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from shared.config import DATA_DIR, MODEL_DIR
from Services.retrieval_service.hybrid_retriever import HybridRetriever
from Services.retrieval_service.bm25_retriever import BM25Retriever
from Services.retrieval_service.query_refiner import QueryRefiner
from Services.PreprocessingService.preprocessor import TextPreprocessor
from Services.ranking_evaluation_service.evaluator import RankingEvaluator


def build_bm25_retriever(hybrid, k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    """
    Builds a BM25Retriever that reuses the InvertedIndexManager already loaded
    inside HybridRetriever, instead of loading it from disk again (~246 MB).
    """
    bm25 = BM25Retriever(k1=k1, b=b)
    bm25.idx = hybrid.bm25
    bm25.dataset_name = hybrid.bm25.dataset_name if hybrid.bm25 else None
    return bm25


def apply_enhanced_refinement(
    queries: dict,
    refiner: QueryRefiner,
    preprocessor: TextPreprocessor,
) -> dict:
    """
    Takes the evaluator's query dict and returns a NEW dict where every query
    has been spell-corrected and synonym-expanded.

    The original dict is NOT modified — we deep-copy so baseline results
    are unaffected if both phases are run in the same process.

    Input  queries[qid]:
        {
          "original":         "shoud teachers get tenur?",
          "processed_str":    "should teacher get tenure",
          "processed_tokens": ["should", "teacher", "get", "tenure"],
        }

    Output queries[qid] for enhanced:
        {
          "original":         "should teachers get tenure?"     # spell-corrected
          "processed_str":    "teacher get tenure employ job"   # re-preprocessed after expansion
          "processed_tokens": ["teacher", "get", "tenure", "employ", "job"],
        }

    Why each step matters:
      - spell_correct: fixes typos in the original so SBERT (which gets the
        raw text) doesn't embed a misspelled word that has no representation.
        `refiner` should already have had `load_corpus_vocabulary()` called on
        it (see main(), below) so real proper nouns / domain terms that exist
        in this corpus (e.g. "Hitler") aren't mistaken for typos and rewritten
        into an unrelated dictionary word (e.g. "hitter").
      - expand_with_synonyms: adds related terms (e.g. "ban" -> "prohibit",
        "forbid") so lexical models (BM25/TF-IDF) can match documents that
        use different wording than the query.
      - re-preprocess: normalises and stems the expanded text the same way
        documents were processed, so the expanded tokens are guaranteed to
        match what's in the index.
    """
    enhanced = copy.deepcopy(queries)

    for qid, data in enhanced.items():
        original_text = data.get("original", "")

        # Step 1: spell-correct the raw query text
        corrected = refiner.correct_spelling(original_text)

        # Step 2: get synonym-expanded token list from the corrected text
        expanded_tokens = refiner.expand_with_synonyms(corrected)

        # expanded_tokens is a flat list of individual terms (strings),
        # already lowercased and stripped of punctuation by expand_with_synonyms.
        # Join them into a single string so TextPreprocessor can normalise and
        # stem them the same way documents were processed in step2_preprocess.
        expanded_text = " ".join(expanded_tokens)

        # Step 3: re-preprocess so expanded terms go through the full pipeline
        # (normalisation, tokenisation, stopword removal, stemming/lemmatisation).
        # This guarantees the tokens match the index vocabulary.
        processed = preprocessor.process(expanded_text)

        # SBERT gets the spell-corrected ORIGINAL text (not stemmed, not expanded)
        # because it uses WordPiece tokenisation internally; stemmed/expanded
        # text degrades its pretrained representations.
        data["original"]         = corrected
        data["processed_str"]    = processed["processed_str"]
        data["processed_tokens"] = processed["processed_tokens"]

    return enhanced


def main():
    parser = argparse.ArgumentParser(description="IR System Evaluation (Requirement 8)")
    parser.add_argument("--dataset", default="webis-touche2020")
    parser.add_argument(
        "--phase", default="baseline",
        choices=["baseline", "enhanced"],
        help=(
            "baseline: evaluate with raw processed queries (no refinement).\n"
            "enhanced: apply spell correction + synonym expansion before evaluating.\n"
            "Run BOTH phases and use GET /api/v1/evaluation/compare to see the diff."
        ),
    )
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--models", default="all",
                        help="comma-separated list or 'all'")
    args = parser.parse_args()

    dataset_name = args.dataset
    base_models_dir = MODEL_DIR

    all_models = ["tfidf", "bm25", "sbert", "word2vec",
                  "hybrid_parallel", "hybrid_serial"]
    selected = all_models if args.models == "all" else \
        [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 70)
    print(" IR System Evaluation Service — Requirement 8")
    print(f" dataset={dataset_name} | phase={args.phase} | top_k={args.top_k}")
    print(f" models = {selected}")
    print("=" * 70)

    # -- Step 1: load all models ---------------------------------------------
    print("\n[1/3] Loading all retrievers into memory...")
    hybrid = HybridRetriever()
    hybrid.load_all_retrievers(
        bm25_dir=base_models_dir / f"bm25_{dataset_name}",
        tfidf_prefix=base_models_dir / f"tfidf_{dataset_name}",
        sbert_dir=base_models_dir / f"sbert_{dataset_name}",
        w2v_dir=base_models_dir / f"word2vec_{dataset_name}",
    )
    bm25 = build_bm25_retriever(hybrid) if hybrid.bm25 else None

    # -- Step 2: set up evaluator --------------------------------------------
    print("\n[2/3] Running evaluation...")
    evaluator = RankingEvaluator(dataset_name, top_k=args.top_k)

    # -- Key change: for enhanced, swap in refined queries -------------------
    if args.phase == "enhanced":
        print("\n  [enhanced] Applying query refinement to all test queries...")

        history_path = PROJECT_ROOT / "data" / "search_history.json"
        refiner      = QueryRefiner(history_path=history_path)
        preprocessor = TextPreprocessor(use_stemming=True)

        # Protect real corpus vocabulary (proper nouns, domain terms, e.g.
        # "Hitler") from being mistaken for typos and rewritten into an
        # unrelated dictionary word (e.g. "hitter") -- mirrors what the
        # gateway does automatically for live "enhanced" search requests,
        # so a CLI-run evaluation and an API-triggered one behave identically.
        if hybrid.bm25:
            refiner.load_corpus_vocabulary(hybrid.bm25.index.keys())

        original_count = len(evaluator.queries)
        evaluator.queries = apply_enhanced_refinement(
            evaluator.queries, refiner, preprocessor
        )
        print(f"  [enhanced] Refined {original_count} queries.")
        print(f"  [enhanced] Example:")

        # Print one before/after to make the refinement visible in the console
        sample_qid = next(iter(evaluator.queries))
        print(f"    qid={sample_qid}")
        print(f"    original tokens (baseline) : would come from processed_queries.json")
        print(f"    tokens after refinement    : {evaluator.queries[sample_qid]['processed_tokens'][:10]}")
        print()
    else:
        print("\n  [baseline] Using raw processed queries — no refinement applied.")

    # -- Step 3: evaluate each model -----------------------------------------
    model_specs = {
        "tfidf":           (evaluator.build_run_tfidf,           hybrid.tfidf),
        "bm25":            (evaluator.build_run_bm25,            bm25),
        "sbert":           (evaluator.build_run_sbert,           hybrid.sbert),
        "word2vec":        (evaluator.build_run_word2vec,        hybrid.w2v),
        "hybrid_parallel": (evaluator.build_run_hybrid_parallel, hybrid),
        "hybrid_serial":   (evaluator.build_run_hybrid_serial,   hybrid),
    }

    results_by_model = {}
    for name in selected:
        if name not in model_specs:
            print(f"  [skip] unknown model '{name}'")
            continue
        run_builder, retriever = model_specs[name]
        if retriever is None:
            print(f"  [skip] '{name}' — model not loaded (missing files).")
            continue
        results_by_model[name] = evaluator.evaluate_model(name, run_builder, retriever)

    if not results_by_model:
        print("\n[ERROR] No models were evaluated. Check that model files exist.")
        return

    # -- Step 4: save and print ----------------------------------------------
    print("\n[3/3] Saving results...")
    out_dir = evaluator.save_results(args.phase, results_by_model)

    metric_order = ["MAP", "Recall@1000", "P@10", "nDCG@10"]
    print("\n" + "=" * 70)
    print(f" Comparison (primary metrics) — phase={args.phase}")
    print("=" * 70)
    header = f"{'model':<18}" + "".join(f"{m:>13}" for m in metric_order)
    print(header)
    print("-" * len(header))
    for name, res in results_by_model.items():
        agg = res["aggregate"]
        print(f"{name:<18}" + "".join(f"{agg.get(m, 0):>13.4f}" for m in metric_order))
    print("=" * 70)
    print(f"\nDetailed report + CSV + per-query JSON saved under:\n  {out_dir}")
    print(
        f"\nTo compare baseline vs enhanced in the API:\n"
        f"  GET /api/v1/evaluation/compare?dataset={dataset_name}"
    )


if __name__ == "__main__":
    main()