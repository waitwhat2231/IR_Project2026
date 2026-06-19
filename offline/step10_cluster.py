# offline/step10_cluster.py
"""
Clusters all documents using their SBERT embeddings.

Why SBERT and not Word2Vec?
  SBERT embeddings are pretrained on large general text → richer semantic
  representation → clusters tend to map to coherent human-readable topics.
  Word2Vec is trained on this corpus alone (382K docs) → more domain-specific
  but noisier at the semantic level for clustering purposes.

Why MiniBatchKMeans and not plain KMeans?
  Plain sklearn KMeans loads the full distance matrix into memory on each
  iteration — at 382K × 384 float32 that's ~587 MB just for the embeddings,
  plus O(N × k) workspace per iteration. MiniBatchKMeans processes the data
  in random batches, converging in far fewer full passes, which is the
  correct choice for N > ~100K.

Algorithm outline:
  1. Load SBERT doc_embeddings.npy (already on disk from step5)
  2. Run MiniBatchKMeans → one cluster label per document
  3. Build doc_cluster_map {doc_id → cluster_id} and save as pickle
  4. For each cluster: find 3 representative docs (closest to centroid)
  5. For each cluster: compute top-10 discriminative terms from the
     preprocessed tokens (term frequency inside cluster, penalised by
     how many other clusters also use the term)
  6. Sample ~10K docs stratified by cluster size, project to 2D with
     UMAP (falls back to PCA if umap-learn is not installed), save as JSON
  7. Save clusters.json (summary) — consumed by ClusterManager at runtime

Reads:
  data/models/sbert_{name}/doc_embeddings.npy
  data/models/sbert_{name}/meta.pkl
  data/processed/{name}/processed_docs.pkl

Writes:
  data/models/clusters_{name}/
      clusters.json        ← cluster summaries (top_terms, sizes, rep docs)
      doc_cluster_map.pkl  ← {doc_id: cluster_id} for all N docs
      scatter_2d.json      ← [{doc_id, x, y, cluster_id}, ...] ~10K sample

Time estimate (CPU):
  KMeans on 382K × 384 : ~10–15 min (MiniBatchKMeans is far faster than KMeans)
  Term counting          : ~5  min  (one streaming pass over processed_docs.pkl)
  UMAP on 10K sample    : ~2  min
  Total                 : ~20 min
"""

import gc
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.append(str(Path(__file__).parent.parent))

from shared.config import DATASETS, MODEL_DIR, PROCESSED_DIR
from shared.pickle_stream import stream_chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _stratified_sample(labels: np.ndarray, n_clusters: int, total: int) -> np.ndarray:
    """
    Sample `total` indices proportionally from each cluster so every cluster
    is represented in the scatter plot regardless of its size.
    """
    rng = np.random.default_rng(42)
    indices: List[int] = []
    n_docs = len(labels)

    for cid in range(n_clusters):
        cluster_idx = np.where(labels == cid)[0]
        # How many samples from this cluster, proportional to its size
        k = max(1, round(total * len(cluster_idx) / n_docs))
        k = min(k, len(cluster_idx))
        chosen = rng.choice(cluster_idx, size=k, replace=False)
        indices.extend(chosen.tolist())

    # Trim or pad to exactly `total` if rounding caused a slight mismatch
    indices = indices[:total]
    return np.array(indices, dtype=np.int64)


def _reduce_2d(vecs: np.ndarray) -> np.ndarray:
    """
    Project high-dimensional vectors to 2D for scatter visualisation.
    UMAP produces much better cluster separation than PCA on semantic embeddings,
    so it is tried first. If umap-learn is not installed, PCA is the fallback.
    """
    try:
        import umap  # pip install umap-learn
        print("  Using UMAP for 2D projection...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            verbose=False,
        )
        return reducer.fit_transform(vecs)
    except ImportError:
        print("  umap-learn not installed — falling back to PCA for 2D projection.")
        print("  Install it with:  pip install umap-learn")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(vecs)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_clusters(
    dataset_name: str,
    n_clusters:   int = 20,
    sample_size:  int = 10_000,
):
    save_dir = MODEL_DIR / f"clusters_{dataset_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if (save_dir / "clusters.json").exists():
        print(f"  Clusters already exist at {save_dir}/ — skipping.")
        print(f"  Delete the folder to re-cluster.")
        return

    # ── 1. Load SBERT embeddings ──────────────────────────────────────────────
    sbert_dir  = MODEL_DIR / f"sbert_{dataset_name}"
    emb_path   = sbert_dir / "doc_embeddings.npy"
    meta_path  = sbert_dir / "meta.pkl"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"SBERT embeddings not found: {emb_path}\n"
            f"Run step5_train_sbert.py first."
        )

    print(f"Loading SBERT embeddings from {emb_path} ...")
    embeddings = np.load(emb_path)            # (N, 384) float32, already L2-normalised
    n_docs, dim = embeddings.shape

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    doc_ids: List[str] = meta["doc_ids"]      # ordered list, same N as embeddings

    assert len(doc_ids) == n_docs, "doc_ids length mismatch — re-run step5"
    print(f"  {n_docs:,} documents, dim={dim}")

    # ── 2. MiniBatchKMeans ────────────────────────────────────────────────────
    print(f"\nRunning MiniBatchKMeans(n_clusters={n_clusters}) ...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=4_096,  # ~16 MB per batch at float32/384-dim — fits in L3 cache
        n_init=5,          # run 5 random initialisations, keep best inertia
        max_iter=150,
        verbose=1,
    )
    labels: np.ndarray = kmeans.fit_predict(embeddings)   # (N,) int
    centroids = kmeans.cluster_centers_.copy()              # (k, 384)

    # Re-normalise centroids (the mean of unit vectors is not a unit vector)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.clip(norms, 1e-10, None)

    print(f"  Clustering done.  Inertia: {kmeans.inertia_:.2f}")
    del kmeans; gc.collect()

    # ── 3. doc_cluster_map: {doc_id → cluster_id} ────────────────────────────
    doc_cluster_map = {
        doc_id: int(label)
        for doc_id, label in zip(doc_ids, labels)
    }
    map_path = save_dir / "doc_cluster_map.pkl"
    with open(map_path, "wb") as f:
        pickle.dump(doc_cluster_map, f, protocol=4)
    print(f"\nSaved doc_cluster_map ({n_docs:,} entries) → {map_path}")

    # ── 4. Per-cluster: size, representative docs ─────────────────────────────
    doc_ids_arr = np.array(doc_ids, dtype=object)
    cluster_info = []

    for cid in range(n_clusters):
        mask          = labels == cid
        cluster_vecs  = embeddings[mask]          # (size, 384)
        cluster_dids  = doc_ids_arr[mask]
        size          = int(mask.sum())
        centroid      = centroids[cid]

        # Representative docs = 3 closest to centroid (highest cosine similarity)
        # cluster_vecs are already L2-normalised, centroid is too → dot product = cosine
        sims    = cluster_vecs @ centroid          # (size,)
        top_idx = np.argsort(sims)[-3:][::-1]
        rep_docs = cluster_dids[top_idx].tolist()

        cluster_info.append({
            "id":                   cid,
            "size":                 size,
            "pct":                  round(100.0 * size / n_docs, 2),
            "top_terms":            [],             # filled in step 5
            "representative_doc_ids": rep_docs,
            # Store centroid for potential future use (nearest-cluster queries)
            # 384 floats × 20 clusters = 7 680 values — negligible in JSON
            "centroid":             centroid.tolist(),
        })

    # ── 5. Top discriminative terms per cluster ───────────────────────────────
    # Strategy: count how often each term appears in cluster C, then divide by
    # how many OTHER clusters also contain the term (like a cluster-level IDF).
    # This surfaces terms that are characteristic of C, not of the whole corpus.
    print("\nExtracting top terms per cluster (streaming processed_docs.pkl) ...")

    corpus_path = PROCESSED_DIR / dataset_name / "processed_docs.pkl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Processed docs not found: {corpus_path}\n"
            f"Run step2_preprocess.py first."
        )

    cluster_term_counts: defaultdict = defaultdict(Counter)

    for chunk in stream_chunks(corpus_path):
        for doc_id, doc_data in chunk.items():
            cid = doc_cluster_map.get(doc_id)
            if cid is None:
                continue
            tokens = doc_data.get("processed_tokens", [])
            # Use a set so each term counts at most once per doc in this cluster
            # (avoids one very long document dominating the cluster's term list)
            cluster_term_counts[cid].update(set(tokens))
        del chunk
        gc.collect()

    # cluster_term_counts[cid][term] = number of docs in cluster cid that contain term

    # corpus_term_cluster_count[term] = number of distinct clusters containing the term
    corpus_term_cluster_count: Counter = Counter()
    for cid, counter in cluster_term_counts.items():
        corpus_term_cluster_count.update(counter.keys())

    for cid in range(n_clusters):
        counter = cluster_term_counts[cid]
        # Score: how prevalent in this cluster vs how spread across clusters
        scored = {
            term: count / corpus_term_cluster_count[term]
            for term, count in counter.items()
        }
        top_terms = sorted(scored, key=scored.__getitem__, reverse=True)[:10]
        cluster_info[cid]["top_terms"] = top_terms

    print("  Top terms extracted.")
    del cluster_term_counts, corpus_term_cluster_count; gc.collect()

    # ── 6. 2D scatter (stratified sample) ─────────────────────────────────────
    actual_sample = min(sample_size, n_docs)
    print(f"\nComputing 2D projection (sample={actual_sample:,}) ...")

    sample_idx     = _stratified_sample(labels, n_clusters, total=actual_sample)
    sample_vecs    = embeddings[sample_idx].astype(np.float32)
    sample_labels  = labels[sample_idx]
    sample_doc_ids = doc_ids_arr[sample_idx]

    del embeddings; gc.collect()   # free the big matrix before UMAP

    coords_2d = _reduce_2d(sample_vecs)     # (sample, 2)

    scatter = [
        {
            "doc_id":     str(did),
            "x":          round(float(x), 4),
            "y":          round(float(y), 4),
            "cluster_id": int(cid),
        }
        for did, (x, y), cid in zip(sample_doc_ids, coords_2d, sample_labels)
    ]

    scatter_path = save_dir / "scatter_2d.json"
    with open(scatter_path, "w", encoding="utf-8") as f:
        json.dump(scatter, f)
    print(f"  Scatter saved ({len(scatter):,} points) → {scatter_path}")

    # ── 7. clusters.json ──────────────────────────────────────────────────────
    summary = {
        "dataset":          dataset_name,
        "n_clusters":       n_clusters,
        "n_docs":           n_docs,
        "embedding_source": "sbert",
        "dim":              dim,
        "generated_at":     datetime.now().isoformat(timespec="seconds"),
        "clusters":         cluster_info,
    }
    clusters_path = save_dir / "clusters.json"
    with open(clusters_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Clustering COMPLETE for '{dataset_name}'")
    print(f"  {save_dir}/")
    print(f"    clusters.json       — {n_clusters} cluster summaries")
    print(f"    doc_cluster_map.pkl — {n_docs:,} doc→cluster assignments")
    print(f"    scatter_2d.json     — {len(scatter):,} sample points for 2D chart")
    print(f"{'='*60}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build document clusters (step 10)")
    parser.add_argument("--k",      type=int, default=20,     help="Number of clusters (default 20)")
    parser.add_argument("--sample", type=int, default=10_000, help="2D scatter sample size (default 10000)")
    args = parser.parse_args()

    for name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        build_clusters(name, n_clusters=args.k, sample_size=args.sample)

    print("\nStep 10 COMPLETE — document clusters built.")