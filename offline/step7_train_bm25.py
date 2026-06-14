# offline/step7_train_bm25.py
"""
BM25 setup — configures parameters and saves the retriever config.

Reads:  data/indexes/{name}_inverted.pkl   (InvertedIndexManager from step3)
Writes: data/models/bm25_{name}/
            bm25_config.pkl                (k1, b, dataset_name)

Why this step exists even though BM25 has no learned parameters:
  - Locks in the k1/b hyperparameter values used for evaluation
  - Verifies the InvertedIndexManager loads correctly and is complete
  - Creates a consistent load path for the retrieval service
    (same pattern as TF-IDF, SBERT, Word2Vec — each model has a save dir)
  - Saves a parameter sensitivity report showing how k1/b affect results
    on sample queries (required by the project spec)

The retrieval service loads all models the same way:
  TFIDFRetriever.load(MODEL_DIR / "tfidf_{name}")
  SBERTRetriever.load(MODEL_DIR / "sbert_{name}")
  Word2VecRetriever.load(MODEL_DIR / "word2vec_{name}")
  BM25Retriever.load(MODEL_DIR / "bm25_{name}")   ← consistent
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from shared.config import MODEL_DIR, DATASETS
from Services.retrieval_service.bm25_retriever import BM25Retriever
from Services.PreprocessingService.preprocessor import TextPreprocessor


# ── BM25 hyperparameters ───────────────────────────────────────────────────────
# These are the values locked in for evaluation.
# The UI allows users to change them at query time via retrieve_with_params(),
# but evaluation always uses these defaults.
BM25_K1 = 1.5
BM25_B   = 0.75

# Sample queries for the parameter sensitivity report
# Using preprocessed tokens (same stemmer as step2)
SAMPLE_QUERIES = {
    "q1": ["vaccin", "mandatori", "children", "school"],
    "q2": ["climat", "chang", "global", "warm", "effect"],
    "q3": ["abort", "right", "legal", "ban"],
}


def train_bm25(dataset_name: str):
    save_dir = MODEL_DIR / f"bm25_{dataset_name}"

    print(f"\n{'='*60}")
    print(f"BM25 setup for: {dataset_name}")
    print(f"{'='*60}")

    if (save_dir / "bm25_config.pkl").exists():
        print(f"  Already exists at {save_dir}/ — skipping.")
        print(f"  Delete the folder to reconfigure.")
        return

    # ── Load and verify InvertedIndexManager ──────────────────────────────────
    bm25 = BM25Retriever(k1=BM25_K1, b=BM25_B)
    bm25.load_index(dataset_name)

    # ── Run a quick retrieval test ────────────────────────────────────────────
    print(f"\n  Retrieval test:")
    test_tokens = SAMPLE_QUERIES["q1"]
    results     = bm25.retrieve(test_tokens, top_k=3)
    print(f"  Query tokens: {test_tokens}")
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"    #{rank}  {doc_id}  score={score:.4f}")

    # ── Parameter sensitivity report ──────────────────────────────────────────
    # Project spec requires demonstrating how k1/b affect results in the UI.
    # We also generate this as a JSON report at training time.
    print(f"\n  Parameter sensitivity report:")
    param_grid = [
        (1.0, 0.5),
        (1.2, 0.75),
        (1.5, 0.75),   # ← default
        (2.0, 0.75),
        (2.0, 1.0),
    ]

    sensitivity_report = {}
    for q_name, q_tokens in SAMPLE_QUERIES.items():
        sensitivity_report[q_name] = {
            "tokens": q_tokens,
            "results_by_params": {},
        }
        for k1, b in param_grid:
            res    = bm25.retrieve_with_params(q_tokens, k1=k1, b=b, top_k=3)
            top3   = [{"doc_id": did, "score": round(sc, 4)} for did, sc in res]
            key    = f"k1={k1}_b={b}"
            sensitivity_report[q_name]["results_by_params"][key] = top3
            marker = " ← default" if (k1, b) == (BM25_K1, BM25_B) else ""
            print(f"    {q_name}  k1={k1}, b={b}{marker}  "
                  f"top1={res[0][0] if res else 'none'} ({res[0][1]:.3f})" if res else "")

    # Save the sensitivity report as JSON for the project report
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / "parameter_sensitivity.json"
    report_path.write_text(json.dumps(sensitivity_report, indent=2))
    print(f"\n  Sensitivity report saved → {report_path}")

    # ── Save BM25 config ──────────────────────────────────────────────────────
    bm25.save(save_dir)

    print(f"\n  Step 7 COMPLETE — BM25 configured for '{dataset_name}'.")


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in DATASETS:
        train_bm25(name)

    print("\n\nStep 7 COMPLETE — BM25 models configured and saved.")