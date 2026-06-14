import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Direct imports matching your shared/config.py layout
from shared.config import MODEL_DIR, DATASETS, SBERT_MODEL, TOP_K

def test_sbert_retrieval(query: str, dataset_name: str, top_k: int = TOP_K):
    print(f"--- Testing SBERT Retrieval for Dataset: {dataset_name} ---")
    print(f"Query: '{query}'\n")

    # 1. Resolve paths using your pathlib config setup
    model_dir = MODEL_DIR / f"sbert_{dataset_name}"
    faiss_path = model_dir / "faiss.index"
    meta_path = model_dir / "meta.pkl"

    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Trained SBERT assets missing in {model_dir}.\n"
            f"Please ensure step5_train_sbert.py has executed successfully."
        )

    # 2. Load the binary FAISS index matrix
    print("[1/3] Loading FAISS index and metadata configurations...")
    index = faiss.read_index(str(faiss_path))
    
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Safely fallback to config defaults if keys are missing from pickle metadata
    model_name = metadata.get("model_name", SBERT_MODEL)
    doc_ids = metadata.get("doc_ids") 
    doc_text_dict = metadata.get("doc_texts", {}) # Fallback if you cached raw texts for viewing

    # 3. Compute dense spatial array representation of the query
    print(f"[2/3] Transforming query string via model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # FAISS strict typing rule: vectors must be flattened 2D float32 numpy arrays
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    
    # 4. Search the vector space index 
    print(f"[3/3] Querying index for top-{top_k} nearest semantic neighbors...")
    similarities, indices = index.search(query_vector, top_k)

    # 5. Output Ranked Matches
    print("\n======= SYSTEM MATCHES =======")
    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
        if idx == -1:  # Sentinel padding check for empty index values
            continue
            
        target_doc_id = doc_ids[idx]
        raw_text = doc_text_dict.get(target_doc_id, "[Raw context text not available in meta.pkl]")
        text_preview = raw_text[:130] + "..." if len(raw_text) > 130 else raw_text

        print(f"Rank {rank}: [Doc ID: {target_doc_id}] (Score/Similarity: {sim:.4f})")
        print(f"   Excerpt: {text_preview}\n")


if __name__ == "__main__":
    # Verifies against the explicit dataset key string configured inside your dict
    DATASET_KEY = "webis-touche2020" 
    
    # Test query checking structural thematic retrieval 
    SAMPLE_QUERY = "Should cellular devices be banned from school grounds?"
    
    if DATASET_KEY not in DATASETS:
        print(f"Warning: '{DATASET_KEY}' is missing from config.DATASETS definitions.")
    
    try:
        test_sbert_retrieval(query=SAMPLE_QUERY, dataset_name=DATASET_KEY)
    except Exception as e:
        print(f"\nExecution Failed: {e}")