# Services/ranking_evaluation_service/scorer.py
"""
Scorer — pure IR effectiveness metrics (no file I/O, no model loading).

This module is the "math core" of the Ranking & Evaluation Service (Requirement 8).
It takes:
  - a ranked result list per query:  ranked = [(doc_id, score), ...]  (descending)
  - the relevance judgments (qrels) for that query:  relevant = {doc_id: grade}
and returns the standard IR metrics.

Metrics required by the project spec (Requirement 8):
  - Precision@k
  - Recall@k
  - Average Precision (AP)  -> averaged across queries yields MAP
  - nDCG@k  (uses graded relevance)

Conventions:
  - For binary metrics (P, R, MAP) a document is "relevant" when grade >= REL_THRESHOLD.
  - nDCG uses the graded relevance values directly (0, 1, 2, ...).
  - Any document not present in qrels is treated as non-relevant (grade = 0).
"""

import math
from typing import Dict, List, Tuple

# Short alias for a ranked result list
RankedList = List[Tuple[str, float]]
# Short alias for the relevance judgments of a single query
Qrels = Dict[str, int]

# Threshold: a grade >= 1 means "relevant" (suits Touche 2020 with grades 0/1/2)
REL_THRESHOLD = 1


# -----------------------------------------------------------------------------
# 1) Precision@k
# -----------------------------------------------------------------------------
def precision_at_k(ranked: RankedList, relevant: Qrels, k: int,
                   rel_threshold: int = REL_THRESHOLD) -> float:
    """
    Fraction of relevant documents within the top k results.
    P@k = (relevant docs in the top k) / k
    """
    if k <= 0:
        return 0.0
    # Count how many of the first k docs have a grade at or above the threshold
    hits = sum(
        1 for doc_id, _ in ranked[:k]
        if relevant.get(doc_id, 0) >= rel_threshold
    )
    return hits / k


# -----------------------------------------------------------------------------
# 2) Recall@k
# -----------------------------------------------------------------------------
def recall_at_k(ranked: RankedList, relevant: Qrels, k: int,
                rel_threshold: int = REL_THRESHOLD) -> float:
    """
    Fraction of all relevant documents that the system retrieved within the
    top k results, relative to the total number of relevant docs in qrels.
    Recall@k = (retrieved & relevant in the top k) / (total relevant)
    """
    # Total number of known relevant docs for this query (the denominator)
    total_relevant = sum(1 for g in relevant.values() if g >= rel_threshold)
    if total_relevant == 0:
        return 0.0
    hits = sum(
        1 for doc_id, _ in ranked[:k]
        if relevant.get(doc_id, 0) >= rel_threshold
    )
    return hits / total_relevant


# -----------------------------------------------------------------------------
# 3) Average Precision (AP)  ->  basis for MAP
# -----------------------------------------------------------------------------
def average_precision(ranked: RankedList, relevant: Qrels,
                      rel_threshold: int = REL_THRESHOLD) -> float:
    """
    Average precision over the full ranked list.
    At every relevant document we compute the precision up to its position,
    then average over the total number of relevant docs. This metric rewards
    placing relevant documents at earlier ranks.
    """
    total_relevant = sum(1 for g in relevant.values() if g >= rel_threshold)
    if total_relevant == 0:
        return 0.0

    hits = 0          # number of relevant docs seen so far
    ap_sum = 0.0      # accumulator of partial precisions
    for i, (doc_id, _) in enumerate(ranked):
        if relevant.get(doc_id, 0) >= rel_threshold:
            hits += 1
            # precision at position (i+1) = relevant-so-far / position
            ap_sum += hits / (i + 1)

    return ap_sum / total_relevant


# -----------------------------------------------------------------------------
# 4) nDCG@k (Normalized Discounted Cumulative Gain) — uses graded relevance
# -----------------------------------------------------------------------------
def dcg_at_k(ranked: RankedList, relevant: Qrels, k: int) -> float:
    """
    DCG: sum of gains (relevance grades) discounted logarithmically by rank.
    A highly relevant document at an early position contributes more.
    """
    dcg = 0.0
    for i, (doc_id, _) in enumerate(ranked[:k]):
        grade = relevant.get(doc_id, 0)
        # discount factor log2(rank+1); rank starts at 1, hence (i+2)
        dcg += grade / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked: RankedList, relevant: Qrels, k: int) -> float:
    """
    nDCG = actual DCG / ideal DCG (IDCG).
    IDCG is computed by sorting all known relevance grades in descending order
    (the ideal ranking). The result lies in [0, 1] where 1 is a perfect ranking.
    """
    # Ideal DCG: sort known grades descending and take the top k
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


# -----------------------------------------------------------------------------
# 5) Aggregate all metrics for a single query
# -----------------------------------------------------------------------------
def evaluate_query(ranked: RankedList, relevant: Qrels) -> Dict[str, float]:
    """
    Computes all metrics for a single query and returns them as a dict.
    We provide several cut-offs to enable deeper analysis in the report,
    while highlighting the four required primary metrics (MAP, Recall, P@10, nDCG).
    """
    return {
        # The four primary metrics required by the spec
        "MAP":          average_precision(ranked, relevant),  # query AP (averaged later)
        "Recall@1000":  recall_at_k(ranked, relevant, 1000),
        "P@10":         precision_at_k(ranked, relevant, 10),
        "nDCG@10":      ndcg_at_k(ranked, relevant, 10),
        # Extra cut-offs for analysis and comparison
        "P@5":          precision_at_k(ranked, relevant, 5),
        "Recall@100":   recall_at_k(ranked, relevant, 100),
        "nDCG@100":     ndcg_at_k(ranked, relevant, 100),
    }


# -----------------------------------------------------------------------------
# 6) Score a full run (all queries) then compute the averages
# -----------------------------------------------------------------------------
def score_run(run: Dict[str, RankedList],
              qrels: Dict[str, Qrels]) -> Dict[str, object]:
    """
    run:   {query_id: [(doc_id, score), ...]}   the system's results per query
    qrels: {query_id: {doc_id: grade}}          the reference relevance judgments

    Steps:
      1) For every query that has relevance judgments, compute its metrics
         via evaluate_query.
      2) Average each metric across all queries (this turns AP into MAP).

    Returns a dict with:
      - "aggregate":   the mean of each metric across queries.
      - "per_query":   metrics for each individual query (for report analysis).
      - "num_queries": the number of queries that were evaluated.
    """
    per_query: Dict[str, Dict[str, float]] = {}

    # (1) Compute metrics for every query present in qrels
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])          # if the system returned nothing, use an empty list
        per_query[qid] = evaluate_query(ranked, relevant)

    # (2) Average each metric across all queries
    aggregate: Dict[str, float] = {}
    if per_query:
        metric_names = next(iter(per_query.values())).keys()
        n = len(per_query)
        for m in metric_names:
            aggregate[m] = sum(q[m] for q in per_query.values()) / n

    return {
        "aggregate":   aggregate,
        "per_query":   per_query,
        "num_queries": len(per_query),
    }
