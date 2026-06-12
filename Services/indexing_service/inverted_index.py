import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.config import INDEX_DIR

class InvertedIndexManager:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.index_file = INDEX_DIR / dataset_name / "inverted_index.pkl"
        self.index = {}
        self.load_index()

    def load_index(self):
        """Download the index of hearts from the offline saved file"""
        if self.index_file.exists():
            print(f"[*] Loading Inverted Index for {self.dataset_name} into memory...")
            with open(self.index_file, "rb") as f:
                self.index = pickle.load(f)
            print(f"[+] Loaded {len(self.index):,} terms successfully.")
        else:
            print(f"[-] Warning: Inverted Index file not found at {self.index_file}")

    def get_postings(self, term: str) -> dict:
        """Returning documents and repeating the word in them (Postings List)"""
        return self.index.get(term.lower(), {})

    def get_doc_frequency(self, term: str) -> int:
        """Counting the number of documents in which the word appeared (DF - very useful for TF-IDF)"""
        return len(self.get_postings(term))