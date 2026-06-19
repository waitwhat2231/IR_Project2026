# Services/ranking_evaluation_service/main.py
"""
Main — entry point of the Evaluation Service (Requirement 8).

Steps:
  1) Load all models once via HybridRetriever.
  2) Build a BM25Retriever reusing the already-loaded index.
  3) Evaluate each model: TF-IDF, BM25, SBERT, Word2Vec, Hybrid-Parallel, Hybrid-Serial.
  4) Save results under data/evaluation/{dataset}/{phase}/.

Run:
    conda activate ir_project
    python Services/ranking_evaluation_service/main.py --phase baseline
    python Services/ranking_evaluation_service/main.py --phase enhanced

What baseline vs enhanced actually does:
    baseline  → queries used exactly as stored in processed_queries.json.
                No modification. Establishes the performance floor.

    enhanced  → each query's content words (nouns/adjectives only) are expanded
                with WordNet synonyms before retrieval. No spell correction is
                applied because TREC test queries are professionally written —
                spell-correcting them either does nothing or damages domain terms
                ("vaping" → "raping" via edit-distance correction).
                Verbs are explicitly excluded from expansion because their WordNet
                senses are too ambiguous ("get" → "perplex", "amaze", "baffle"…).

                BM25 and TF-IDF benefit most (extra tokens reach documents that
                use different vocabulary). SBERT still receives the original text
                (its WordPiece tokenizer was not trained on stemmed/expanded input).

After both phases, compare via the API:
    GET /api/v1/evaluation/compare?dataset=webis-touche2020
"""

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from shared.config import MODEL_DIR
from Services.retrieval_service.hybrid_retriever import HybridRetriever
from Services.retrieval_service.bm25_retriever import BM25Retriever
from Services.PreprocessingService.preprocessor import TextPreprocessor
from Services.ranking_evaluation_service.evaluator import RankingEvaluator


def build_bm25_retriever(hybrid, k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    """Reuse the index already loaded inside HybridRetriever — avoids a 246 MB reload."""
    bm25 = BM25Retriever(k1=k1, b=b)
    bm25.idx = hybrid.bm25
    bm25.dataset_name = hybrid.bm25.dataset_name if hybrid.bm25 else None
    return bm25


def _expand_query(text: str, nltk_stops: set, allowed_lexnames: set,
                  content_pos: set) -> set:
    """
    POS-tag the query, then expand content words (nouns + adjectives only)
    with WordNet synonyms.

    Why nouns and adjectives only, not verbs:
      Verbs in WordNet are massively polysemous. "get" has 27 senses and
      most are unrelated to its function-word use in debate queries
      ("get tenure" = obtain, but WordNet expands it as "get = perplex",
      giving synonyms like "baffle", "stupefy", "nonplus"). That is noise,
      not signal. Nouns and adjectives are far more stable across senses.

    Why NLTK stopwords, not the custom protected_stopwords list:
      The custom list has only 45 words. NLTK's English stopwords list has
      ~180 words and correctly classifies function words ("get", "use",
      "give", "take") that would otherwise slip through to verb expansion.
    """
    from nltk.corpus import wordnet
    from nltk import pos_tag, word_tokenize

    tagged = pos_tag(word_tokenize(text))
    expanded: set = set()

    for word, pos in tagged:
        cw = word.lower().strip("?!.,;:")
        if cw in nltk_stops or len(cw) <= 2:
            continue
        expanded.add(cw)

        if pos not in content_pos:
            continue  # skip verbs and function words entirely

        wn_pos = wordnet.NOUN if pos.startswith("N") else wordnet.ADJ
        for syn in wordnet.synsets(cw, pos=wn_pos):
            if syn.lexname() not in allowed_lexnames:
                continue
            for lemma in syn.lemmas():
                lname = lemma.name().lower()
                if (
                    "_" not in lname
                    and "-" not in lname
                    and len(lname) > 2
                    and lname not in nltk_stops
                ):
                    expanded.add(lname)

    return expanded


def apply_enhanced_refinement(
    queries: dict,
    preprocessor: TextPreprocessor,
) -> dict:
    """
    Returns a NEW query dict where each query's lexical representation
    (processed_str, processed_tokens) is enriched with synonym-expanded terms.

    data["original"] is intentionally NOT changed:
      - SBERT reads "original" and embeds the raw sentence.
      - The original text is already correct; changing it would not improve
        SBERT's embedding (it would hurt it if the text became an incoherent
        bag of synonyms).
      - BM25 and TF-IDF read processed_tokens / processed_str and DO benefit
        from the expanded token set.

    The original queries dict is deep-copied — the baseline dict is unaffected
    so both phases can be run in the same process without interference.
    """
    import nltk
    nltk.download("wordnet",                    quiet=True)
    nltk.download("omw-1.4",                    quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt",                      quiet=True)
    nltk.download("punkt_tab",                  quiet=True)
    nltk.download("stopwords",                  quiet=True)

    from nltk.corpus import stopwords as nltk_stopwords_corpus
    nltk_stops = set(nltk_stopwords_corpus.words("english"))

    # Nouns and adjectives only — see _expand_query docstring for reasoning
    content_pos = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}

    # WordNet lexicographer file names that tend to yield on-topic synonyms.
    # "noun.act" is included because it covers abstract process-nouns like
    # "penalty" (→ punishment, penalization) and "tenure" (→ incumbency).
    allowed_lexnames = {
        "noun.communication",
        "noun.cognition",
        "noun.state",
        "noun.phenomenon",
        "noun.act",
        "noun.attribute",
        "noun.group",
        "noun.event",
        "adj.all",
    }

    enhanced = copy.deepcopy(queries)

    for qid, data in enhanced.items():
        original_text = data.get("original", "")

        expanded_words = _expand_query(
            original_text, nltk_stops, allowed_lexnames, content_pos
        )

        # Re-preprocess expanded words through the full pipeline
        # (normalise, tokenise, remove stopwords, stem) so tokens match the index
        expanded_text = " ".join(expanded_words)
        processed = preprocessor.process(expanded_text)

        # Only the lexical representations change
        data["processed_str"]    = processed["processed_str"]
        data["processed_tokens"] = processed["processed_tokens"]
        # data["original"] is LEFT UNCHANGED — SBERT reads this

    return enhanced


def main():
    parser = argparse.ArgumentParser(description="IR System Evaluation (Requirement 8)")
    parser.add_argument("--dataset", default="webis-touche2020")
    parser.add_argument(
        "--phase",
        default="baseline",
        choices=["baseline", "enhanced"],
        help=(
            "baseline: raw processed queries, no modification.\n"
            "enhanced: noun/adjective synonym expansion before retrieval."
        ),
    )
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument(
        "--models",
        default="all",
        help="comma-separated list or 'all': tfidf,bm25,sbert,word2vec,hybrid_parallel,hybrid_serial",
    )
    args = parser.parse_args()

    dataset_name   = args.dataset
    all_models     = ["tfidf", "bm25", "sbert", "word2vec", "hybrid_parallel", "hybrid_serial"]
    selected       = all_models if args.models == "all" else \
                     [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 70)
    print(" IR System Evaluation Service — Requirement 8")
    print(f" dataset={dataset_name} | phase={args.phase} | top_k={args.top_k}")
    print(f" models = {selected}")
    print("=" * 70)

    # ── 1. Load all models ────────────────────────────────────────────────────
    print("\n[1/3] Loading all retrievers...")
    hybrid = HybridRetriever()
    hybrid.load_all_retrievers(
        bm25_dir    = MODEL_DIR / f"bm25_{dataset_name}",
        tfidf_prefix= MODEL_DIR / f"tfidf_{dataset_name}",
        sbert_dir   = MODEL_DIR / f"sbert_{dataset_name}",
        w2v_dir     = MODEL_DIR / f"word2vec_{dataset_name}",
    )
    bm25 = build_bm25_retriever(hybrid) if hybrid.bm25 else None

    # ── 2. Set up evaluator ───────────────────────────────────────────────────
    print("\n[2/3] Running evaluation...")
    evaluator    = RankingEvaluator(dataset_name, top_k=args.top_k)
    preprocessor = TextPreprocessor(use_stemming=True)

    if args.phase == "enhanced":
        print("\n  [enhanced] Applying synonym expansion to test queries...")
        print("  (no spell correction — TREC queries are already correctly spelled)")

        original_queries = evaluator.queries
        evaluator.queries = apply_enhanced_refinement(original_queries, preprocessor)

        # Show one before/after example so the console confirms real changes happened
        sample_qid = next(iter(evaluator.queries))
        before = original_queries[sample_qid]["processed_tokens"]
        after  = evaluator.queries[sample_qid]["processed_tokens"]
        new    = sorted(set(after) - set(before))
        print(f"\n  Sample qid={sample_qid}:")
        print(f"    baseline tokens  : {before}")
        print(f"    enhanced tokens  : {after}")
        print(f"    net new terms    : {new}")
        print()
    else:
        print("\n  [baseline] Using raw processed queries — no refinement applied.")

    # ── 3. Evaluate each model ────────────────────────────────────────────────
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
            print(f"  [skip] '{name}' — model not loaded.")
            continue
        results_by_model[name] = evaluator.evaluate_model(name, run_builder, retriever)

    if not results_by_model:
        print("\n[ERROR] No models evaluated. Check that model files exist.")
        return

    # ── 4. Save + print comparison table ────────────────────────────────────
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
    print(f"\nSaved under: {out_dir}")
    if args.phase == "enhanced":
        print("\nTo compare baseline vs enhanced:")
        print(f"  GET /api/v1/evaluation/compare?dataset={dataset_name}")


if __name__ == "__main__":
    main()