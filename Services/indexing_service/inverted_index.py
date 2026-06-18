import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.config import INDEX_DIR


class InvertedIndexManager:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        # Note: I'm using the file path used in your step3 script
        self.index_file = INDEX_DIR / f"{dataset_name}_inverted.pkl"

        # Core BM25 Data
        self.doc_ids: list = []
        self.index = {}  # {term: {doc_id: tf}}
        self.doc_lengths = {}  # {doc_id: length}
        self.N = 0  # Total docs
        self.avg_dl = 0  # Average document length
        self.df = {}  # {term: document_frequency}

        self.load_index()

    def load_index(self):
        if self.index_file.exists():
            print(f"[*] Loading Inverted Index for {self.dataset_name}...")
            with open(self.index_file, "rb") as f:
                # We load the whole object that we saved in step 3
                data = pickle.load(f)

                # Check if it's the whole object or just the dict
                if hasattr(data, "index"):
                    self.index = data.index
                    self.df = data.df
                    self.doc_lengths = data.doc_lengths
                    self.N = data.N
                    self.avg_dl = data.avg_dl
                else:
                    # Fallback if you just saved the dict
                    self.index = data
            print(f"[+] Loaded {len(self.index):,} terms.")
        else:
            print(f"[-] Warning: {self.index_file} not found.")

    def get_postings(self, term: str) -> dict:
        return self.index.get(term.lower(), {})

    def idf(self, term: str) -> float:
        """Required for BM25: Log-based IDF calculation."""
        import math

        df = len(self.get_postings(term))
        # Standard BM25 IDF formula
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)
