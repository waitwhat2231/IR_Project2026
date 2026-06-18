# Services/ranking_evaluation_service/evaluator.py
"""
Evaluator — orchestrator of the Ranking & Evaluation Service (Requirement 8).

Responsibilities of this file:
  1) Load the reference data once:
       - qrels (relevance judgments): data/raw/{dataset}/qrels.json
       - processed_queries:           data/processed/{dataset}/processed_queries.json
  2) Turn each retrieval model into a uniform "run":
       run = {query_id: [(doc_id, score), ...]}
     while respecting that each model expects a different input:
       - TF-IDF   -> processed query text (processed_str)
       - BM25     -> processed token list
       - Word2Vec -> processed token list
       - SBERT    -> raw query text, because it uses internal WordPiece tokenization
       - Hybrid   -> combines both (lexical + semantic)
  3) Compute metrics via scorer.score_run.
  4) Save results in an organized way (JSON + CSV + text report) under:
       data/evaluation/{dataset}/{phase}/
     where phase distinguishes "baseline" (before extra features) from
     "enhanced" (after them).
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, cast

# Add the project root to the path so imports work when run directly
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.config import DATA_DIR, RAW_DIR, PROCESSED_DIR
from Services.ranking_evaluation_service.scorer import score_run

# Short alias for a ranked result list
RankedList = List[Tuple[str, float]]


class RankingEvaluator:
    """
    Loads the reference data once, then builds a run for each model and scores it.
    The class separates run building from metric computation from saving,
    following the separation-of-concerns principle (SOA / Clean Architecture).
    """

    def __init__(self, dataset_name: str, top_k: int = 1000):
        self.dataset_name = dataset_name
        self.top_k = top_k  # retrieval depth used per model (affects Recall and MAP)

        # -- Step 1: load the relevance judgments (qrels) --------------------
        qrels_path = RAW_DIR / dataset_name / "qrels.json"
        if not qrels_path.exists():
            raise FileNotFoundError(f"qrels not found: {qrels_path}")
        with open(qrels_path, encoding="utf-8") as f:
            # qrels = {qid: {doc_id: relevance_grade}}
            self.qrels: Dict[str, Dict[str, int]] = json.load(f)

        # -- Step 2: load the processed queries -----------------------------
        queries_path = PROCESSED_DIR / dataset_name / "processed_queries.json"
        if not queries_path.exists():
            raise FileNotFoundError(f"processed_queries not found: {queries_path}")
        with open(queries_path, encoding="utf-8") as f:
            raw_queries = json.load(f)

        # Keep only queries that have relevance judgments (qrels) — the evaluable ones
        self.queries: Dict[str, dict] = {
            qid: data for qid, data in raw_queries.items() if qid in self.qrels
        }

        print(f"[Evaluator] dataset={dataset_name}")
        print(f"[Evaluator] queries with qrels = {len(self.queries)}")
        print(f"[Evaluator] retrieval depth top_k = {self.top_k}")

    # -------------------------------------------------------------------------
    # Build the runs per model (each returns {qid: [(doc_id, score), ...]})
    # -------------------------------------------------------------------------

    def build_run_tfidf(self, tfidf) -> Dict[str, RankedList]:
        """TF-IDF: uses the processed text (processed_str) and has a fast batch method."""
        # Build a {qid: processed_str} dict, then call batch retrieval once
        batch = {qid: data["processed_str"] for qid, data in self.queries.items()}
        return tfidf.retrieve_batch(batch, top_k=self.top_k)

    def build_run_bm25(self, bm25) -> Dict[str, RankedList]:
        """BM25: uses the processed token list and has a batch method."""
        batch = {qid: data["processed_tokens"] for qid, data in self.queries.items()}
        return bm25.retrieve_batch(batch, top_k=self.top_k)

    def build_run_sbert(self, sbert) -> Dict[str, RankedList]:
        """SBERT: has no batch method, so we iterate the queries one by one.
        Important: we pass the raw (original) text, not the processed one."""
        run: Dict[str, RankedList] = {}
        for qid, data in self.queries.items():
            run[qid] = sbert.retrieve(data["original"], top_k=self.top_k)
        return run

    def build_run_word2vec(self, w2v) -> Dict[str, RankedList]:
        """Word2Vec: retrieve per query using the processed tokens."""
        run: Dict[str, RankedList] = {}
        for qid, data in self.queries.items():
            run[qid] = w2v.retrieve(data["processed_tokens"], top_k=self.top_k)
        return run

    def build_run_hybrid_parallel(self, hybrid) -> Dict[str, RankedList]:
        """Parallel hybrid: fuse BM25 (lexical) with SBERT (semantic) via alpha."""
        run: Dict[str, RankedList] = {}
        for qid, data in self.queries.items():
            run[qid] = hybrid.retrieve_hybrid(
                query_raw=data["original"],
                query_tokens=data["processed_tokens"],
                sparse_method="bm25",
                dense_method="sbert",
                alpha=0.5,
                top_k=self.top_k,
            )
        return run

    def build_run_hybrid_serial(self, hybrid) -> Dict[str, RankedList]:
        """Serial hybrid: first-stage BM25 filtering, then SBERT re-ranking."""
        run: Dict[str, RankedList] = {}
        for qid, data in self.queries.items():
            run[qid] = hybrid.retrieve_serial(
                query_raw=data["original"],
                query_tokens=data["processed_tokens"],
                sparse_method="bm25",
                dense_method="sbert",
                cascade_top_n=self.top_k,  # pass enough depth for the first stage
                top_k=self.top_k,
            )
        return run

    # -------------------------------------------------------------------------
    # Evaluate a single model: build the run then compute metrics with timing
    # -------------------------------------------------------------------------

    def evaluate_model(self, model_name: str, run_builder, retriever) -> dict:
        """
        model_name : the model's name for display and saving.
        run_builder: one of the build_run_* methods above.
        retriever  : the loaded model object passed to the builder.

        Returns the results dict (aggregate + per_query + timing info).
        """
        print(f"\n[Evaluator] > Evaluating: {model_name} ...")
        start = time.perf_counter()

        # (1) Build the ranked results per query
        run = run_builder(retriever)

        # (2) Compute metrics (AP becomes MAP after averaging)
        scored = score_run(run, self.qrels)

        elapsed = time.perf_counter() - start
        scored["elapsed_sec"] = round(elapsed, 2)

        # FIX: Force the strict linter to accept the type using 'cast'
        agg = cast(Dict[str, float], scored["aggregate"])

        print(
            f"[Evaluator] done {model_name} in {elapsed:.1f}s | "
            f"MAP={agg.get('MAP', 0):.4f}  "
            f"Recall@1000={agg.get('Recall@1000', 0):.4f}  "
            f"P@10={agg.get('P@10', 0):.4f}  "
            f"nDCG@10={agg.get('nDCG@10', 0):.4f}"
        )

        return scored

    # -------------------------------------------------------------------------
    # Save results in an organized way
    # -------------------------------------------------------------------------

    def save_results(self, phase: str, results_by_model: Dict[str, dict]) -> Path:
        """
        Saves results under: data/evaluation/{dataset}/{phase}/
          - metrics_summary.json : full summary (metadata + averages per model)
          - comparison.csv       : comparison table (models x metrics) for the report
          - report.txt           : readable text report + per-query nDCG@10 breakdown
          - per_query/{model}.json : per-query metrics for each model

        phase: "baseline" before extra features, or "enhanced" after them.
        """
        # Prepare the output folder
        out_dir = DATA_DIR / "evaluation" / self.dataset_name / phase
        per_query_dir = out_dir / "per_query"
        per_query_dir.mkdir(parents=True, exist_ok=True)

        # Metric column order: the four primary metrics first, then the extras
        metric_order = [
            "MAP",
            "Recall@1000",
            "P@10",
            "nDCG@10",
            "P@5",
            "Recall@100",
            "nDCG@100",
        ]

        # (1) Full JSON summary
        summary = {
            "dataset": self.dataset_name,
            "phase": phase,
            "top_k": self.top_k,
            "num_queries": len(self.queries),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "models": {
                name: {
                    "aggregate": res["aggregate"],
                    "elapsed_sec": res.get("elapsed_sec"),
                }
                for name, res in results_by_model.items()
            },
        }
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # (2) Comparison CSV table (one row per model)
        with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + metric_order + ["elapsed_sec"])
            for name, res in results_by_model.items():
                agg = res["aggregate"]
                row = [name] + [f"{agg.get(m, 0):.4f}" for m in metric_order]
                row.append(res.get("elapsed_sec", ""))
                writer.writerow(row)

        # (3) Per-query metrics for each model (one file per model)
        for name, res in results_by_model.items():
            safe = name.replace(" ", "_")
            with open(per_query_dir / f"{safe}.json", "w", encoding="utf-8") as f:
                json.dump(res["per_query"], f, indent=2, ensure_ascii=False)

        # (4) Readable text report
        self._write_text_report(
            out_dir / "report.txt", phase, results_by_model, metric_order
        )

        print(f"\n[Evaluator] Results saved under: {out_dir}")
        return out_dir

    def _write_text_report(
        self,
        path: Path,
        phase: str,
        results_by_model: Dict[str, dict],
        metric_order: List[str],
    ) -> None:
        """Builds a text report with the comparison table and per-query nDCG@10."""
        lines: List[str] = []
        lines.append("=" * 78)
        lines.append(f"  IR System Evaluation Report  —  Requirement 8")
        lines.append(f"  Dataset : {self.dataset_name}")
        lines.append(f"  Phase   : {phase}   (baseline = before extra features)")
        lines.append(f"  Queries : {len(self.queries)}    top_k = {self.top_k}")
        lines.append("=" * 78)
        lines.append("")

        # Comparison table: a column for the model name + columns for the metrics
        header = f"{'model':<18}" + "".join(f"{m:>13}" for m in metric_order)
        lines.append(header)
        lines.append("-" * len(header))
        for name, res in results_by_model.items():
            agg = res["aggregate"]
            row = f"{name:<18}" + "".join(
                f"{agg.get(m, 0):>13.4f}" for m in metric_order
            )
            lines.append(row)
        lines.append("")

        # Per-query nDCG@10 breakdown (useful to analyze strengths/weaknesses)
        lines.append("=" * 78)
        lines.append("  Per-query nDCG@10 breakdown")
        lines.append("=" * 78)
        # Sort queries numerically when they are numeric
        try:
            qids_sorted = sorted(self.queries, key=lambda x: int(x))
        except ValueError:
            qids_sorted = sorted(self.queries)

        # Header row: query id + each model
        model_names = list(results_by_model.keys())
        head = (
            f"{'qid':<6}" + "".join(f"{n[:12]:>13}" for n in model_names) + "   query"
        )
        lines.append(head)
        lines.append("-" * len(head))
        for qid in qids_sorted:
            cells = "".join(
                f"{results_by_model[n]['per_query'].get(qid, {}).get('nDCG@10', 0):>13.4f}"
                for n in model_names
            )
            original = self.queries[qid].get("original", "")[:45]
            lines.append(f"{qid:<6}{cells}   {original}")

        path.write_text("\n".join(lines), encoding="utf-8")
