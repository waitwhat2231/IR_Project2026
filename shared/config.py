# shared/config.py
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR     = DATA_DIR / "indexes"
MODEL_DIR     = DATA_DIR / "models"

# Create all directories
for d in [RAW_DIR, PROCESSED_DIR, INDEX_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Datasets
DATASETS = {
    "webis-touche2020": "beir/webis-touche2020/v2"
}

# MongoDB
MONGO_URI  = "mongodb://localhost:27017"    # local dev
MONGO_URI_DOCKER = "mongodb://mongodb:27017"  # inside Docker
MONGO_DB   = "ir_system"
MONGO_COLL = "documents"

# BM25 defaults
BM25_K1 = 1.5
BM25_B  = 0.75

# Embedding model
SBERT_MODEL = "all-MiniLM-L6-v2"

# Service ports
PORTS = {
    "gateway":          8000,
    "preprocessing":    8001,
    "indexing":         8002,
    "retrieval":        8003,
    "query_refinement": 8004,
}

TOP_K          = 10
CANDIDATE_SIZE = 100   # hybrid serial first stage