import pickle
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent)) #Current file path
from shared.config import PROCESSED_DIR, INDEX_DIR, DATASETS

def build_and_save_inverted_index(dataset_name: str):
    proc_path = PROCESSED_DIR / dataset_name
    index_path = INDEX_DIR / dataset_name
    index_path.mkdir(parents=True, exist_ok=True)

    docs_file = proc_path / "processed_docs.pkl"
    out_file = index_path / "inverted_index.pkl"

    print(f"\n{'='*60}")
    print(f"Building Inverted Index for: {dataset_name}")
    print(f"{'='*60}")

    if not docs_file.exists():
        print(f"  [Error] {docs_file} not found. Please run step2 first!")
        return

    print("  Loading preprocessed documents...")
    with open(docs_file, "rb") as f:
        processed_docs = pickle.load(f)

# ── Building the Inverted Index ─────────────────────────────────────────────

    inverted_index = {}

    for doc_id, doc_data in tqdm(processed_docs.items(), desc="  Indexing tokens"):
        # processed_tokens تم إنشاؤها في الخطوة الثانية كمصفوفة كلمات
        tokens = doc_data.get("processed_tokens", [])
        
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {}
            
            # حساب التكرار (Term Frequency) للكلمة داخل الوثيقة الحالية
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = 1
            else:
                inverted_index[token][doc_id] += 1

    print(f"  Saving Inverted Index ({len(inverted_index):,} unique tokens)...")
    with open(out_file, "wb") as f:
        pickle.dump(inverted_index, f)
        
    print(f"  Success! Inverted Index saved to → {out_file}")


if __name__ == "__main__":
    for name in DATASETS:
        build_and_save_inverted_index(name)
    
    print("\n--- Quick Example of what the Inverted Index looks like in memory ---")
    sample_inverted_index = {
        "vaccin": {"doc_1": 3, "doc_5": 1},
        "coronaviru": {"doc_1": 2, "doc_12": 4}
    }
    print(sample_inverted_index)
    print("Step 3 COMPLETE — Inverted Index built successfully.")