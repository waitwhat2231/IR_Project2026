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
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from shared.config import MODEL_DIR
from Services.retrieval_service.hybrid_retriever import HybridRetriever
from Services.retrieval_service.bm25_retriever import BM25Retriever
from Services.ranking_evaluation_service.evaluator import RankingEvaluator


def build_bm25_retriever(hybrid, k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    """
    Builds a BM25Retriever and reuses the InvertedIndexManager already loaded
    inside HybridRetriever, instead of loading it from disk again (~246MB).
    """
    bm25 = BM25Retriever(k1=k1, b=b)
    bm25.idx = hybrid.bm25            # inject the pre-loaded index
    bm25.dataset_name = hybrid.bm25.dataset_name if hybrid.bm25 else None
    return bm25


def main():
    # -- Step 0: parse command-line arguments --------------------------------
    parser = argparse.ArgumentParser(description="IR System Evaluation (Requirement 8)")
    parser.add_argument("--dataset", default="webis-touche2020")
    parser.add_argument("--phase", default="baseline",
                        help="baseline (before extra features) or enhanced (after them)")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--models", default="all",
                        help="comma-separated list or 'all'")
    args = parser.parse_args()

    dataset_name = args.dataset
    base_models_dir = PROJECT_ROOT / "data" / "models"

    # Determine which models to evaluate
    all_models = ["tfidf", "bm25", "sbert", "word2vec",
                  "hybrid_parallel", "hybrid_serial"]
    selected = all_models if args.models == "all" else \
        [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 70)
    print(" IR System Evaluation Service — Requirement 8")
    print(f" dataset={dataset_name} | phase={args.phase} | top_k={args.top_k}")
    print(f" models = {selected}")
    print("=" * 70)

    # -- Step 1: load all models once via HybridRetriever --------------------
    print("\n[1/3] Loading all retrievers into memory...")
    hybrid = HybridRetriever()
    hybrid.load_all_retrievers(
        bm25_dir=base_models_dir / f"bm25_{dataset_name}",
        tfidf_prefix=base_models_dir / f"tfidf_{dataset_name}",
        sbert_dir=base_models_dir / f"sbert_{dataset_name}",
        w2v_dir=base_models_dir / f"word2vec_{dataset_name}",
    )

    # Build a BM25Retriever that reuses the same loaded index
    bm25 = build_bm25_retriever(hybrid) if hybrid.bm25 else None

    # -- Step 2: set up the evaluator and run evaluation per model -----------
    print("\n[2/3] Running evaluation...")
    evaluator = RankingEvaluator(dataset_name, top_k=args.top_k)

    # Map: model name -> (run-builder function, the model object it needs)
    # We check each model is available before evaluating so one missing model
    # does not break the whole run.
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
        print("\n[ERROR] No models were evaluated. Check that the model files exist.")
        return

    # -- Step 3: save results and print the comparison table -----------------
    print("\n[3/3] Saving results...")
    out_dir = evaluator.save_results(args.phase, results_by_model)

    # Print the final comparison table on screen
    metric_order = ["MAP", "Recall@1000", "P@10", "nDCG@10"]
    print("\n" + "=" * 70)
    print(" Comparison (primary metrics)")
    print("=" * 70)
    header = f"{'model':<18}" + "".join(f"{m:>13}" for m in metric_order)
    print(header)
    print("-" * len(header))
    for name, res in results_by_model.items():
        agg = res["aggregate"]
        print(f"{name:<18}" + "".join(f"{agg.get(m, 0):>13.4f}" for m in metric_order))
    print("=" * 70)
    print(f"\nDetailed report + CSV + per-query JSON saved under:\n  {out_dir}")


if __name__ == "__main__":
    main()
