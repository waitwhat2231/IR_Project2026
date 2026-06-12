import pickle
from pathlib import Path

# الرجوع خطوة للأعلى لأن الملف موجود داخل مجلد offline
BASE_DIR = Path(__file__).resolve().parent.parent
index_file = BASE_DIR / "data" / "indexes" / "webis-touche2020" / "inverted_index.pkl"

print(f"[*] Opening your real Inverted Index at:\n    {index_file}")

# باقي الكود كما هو تماماً دون تغيير...
print("[*] Reading data...")
with open(index_file, "rb") as f:
    inverted_index = pickle.load(f)

print(f"[+] Total unique words found: {len(inverted_index):,}")

test_words = ["health", "education", "internet"]

print("\n--- Searching the index for real words ---")
for word in test_words:
    if word in inverted_index:
        postings = inverted_index[word]
        print(f"\nWord: '{word}'")
        print(f" -> Appeared in {len(postings):,} different documents.")
        sample_docs = list(postings.items())[:3]
        print(f" -> Sample documents and frequencies (TF): {sample_docs}")
    else:
        print(f"\nWord: '{word}' not found in the index.")