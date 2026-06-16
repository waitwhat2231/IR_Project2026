# offline/step9_load_to_mongodb.py
"""
Step 9 — Load the original documents into MongoDB.

WHY THIS STEP EXISTS
--------------------
Steps 2-7 only needed the *processed* text (stems/tokens) to build the index
and train the models. But when we finally show search results to a user, we
must display the ORIGINAL title and text of each document.

Storing all 382K original documents inside every model would be wasteful.
Instead we keep them in one place — MongoDB — and look them up by id when needed.

WHAT THIS SCRIPT DOES
---------------------
1. Reads the original documents produced by step 1:
       data/raw/{dataset}/docs.pkl
   Shape:  { doc_id: {"title": "...", "text": "..."} }

2. Inserts every document into MongoDB, using the document's own id as the
   MongoDB primary key (_id). This is the important part of the task:
   the IDs are seeded AND actually used as the lookup key, so later the
   retrieval result (which is just a list of doc_ids + scores) can fetch the
   full text in one fast query.

3. The load is "idempotent": running it twice will not create duplicates,
   it simply overwrites (upserts) the same _id.

CONNECTION SETTINGS (from shared/config.py, lines 21-24)
--------------------------------------------------------
    MONGO_URI         -> mongodb://localhost:27017   (running the script on your PC)
    MONGO_URI_DOCKER  -> mongodb://mongodb:27017     (running inside the docker network)
    MONGO_DB          -> ir_system                   (database name)
    MONGO_COLL        -> documents                   (collection name)

You can override the URI with an environment variable MONGO_URI if needed,
otherwise it falls back to the value in config.
"""

import os
import sys
import pickle
from pathlib import Path

from tqdm import tqdm
from pymongo import MongoClient, ASCENDING, ReplaceOne
from pymongo.errors import ServerSelectionTimeoutError

# Make the project root importable so "shared.config" works when run directly
sys.path.append(str(Path(__file__).parent.parent))

from shared.config import (
    RAW_DIR,
    DATASETS,
    MONGO_URI,
    MONGO_URI_DOCKER,   # kept available for running inside docker (see note below)
    MONGO_DB,
    MONGO_COLL,
)

# How many documents we send to MongoDB per network round-trip.
# Bigger batches = faster, but use more memory. 1000 is a safe default.
BATCH_SIZE = 1000


def get_collection():
    """
    Open a connection to MongoDB and return the target collection.

    URI resolution order:
      1. environment variable MONGO_URI  (lets you point anywhere without code changes)
      2. config.MONGO_URI                (default: localhost, what you use normally)

    If you run THIS script from inside another docker container that sits on the
    same docker network as the database, set MONGO_URI=mongodb://mongodb:27017
    (that value is exposed as MONGO_URI_DOCKER in the config).
    """
    uri = os.getenv("MONGO_URI", MONGO_URI)
    print(f"[*] Connecting to MongoDB at: {uri}")
    print(f"    Docker-network URI (if needed): {MONGO_URI_DOCKER}")

    # serverSelectionTimeoutMS: fail fast (5s) instead of hanging if Mongo is down
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)

    # "ping" forces an actual connection so we can show a clear error early
    client.admin.command("ping")

    db = client[MONGO_DB]            # database name from config  -> "ir_system"
    coll = db[MONGO_COLL]           # collection name from config -> "documents"
    print(f"[+] Connected. database='{MONGO_DB}'  collection='{MONGO_COLL}'")
    return client, coll


def load_docs(dataset_name: str) -> dict:
    """
    Load the original documents saved by step 1.

    docs.pkl is a SINGLE pickle block (not the chunked stream used in step 2),
    so a plain pickle.load() is correct here.

    Returns: { doc_id: {"title": "...", "text": "..."} }
    """
    docs_file = RAW_DIR / dataset_name / "docs.pkl"
    if not docs_file.exists():
        raise FileNotFoundError(
            f"{docs_file} not found. Run offline/step1_download.py first."
        )

    print(f"[*] Loading original documents from: {docs_file}")
    with open(docs_file, "rb") as f:
        docs = pickle.load(f)
    print(f"[+] Loaded {len(docs):,} documents.")
    return docs


def seed_dataset(coll, dataset_name: str, docs: dict):
    """
    Insert every document into MongoDB.

    Schema written per document (matches shared/database.py so the online
    read methods keep working):
        {
            "_id":     <doc_id>,     # the document's own id IS the primary key
            "dataset": <name>,       # which dataset this doc belongs to
            "title":   <title>,      # full original title
            "text":    <text>,       # full original text
        }

    We use ReplaceOne(..., upsert=True) so the operation is idempotent:
      - first run  -> inserts the document
      - later runs -> overwrites the same _id (no duplicates, no errors)
    """
    total = len(docs)
    written = 0

    # We build operations in batches and send each batch with bulk_write.
    operations = []
    # docs.items() gives us (doc_id, {"title", "text"}) pairs
    for doc_id, doc in tqdm(docs.items(), total=total,
                            desc=f"Seeding '{dataset_name}'", unit="doc"):
        operations.append(
            ReplaceOne(
                {"_id": doc_id},                       # match by the document id
                {
                    "_id":     doc_id,                 # seed the ID and USE it as key
                    "dataset": dataset_name,
                    "title":   doc.get("title", "") or "",
                    "text":    doc.get("text", "") or "",
                },
                upsert=True,                           # insert if missing, replace if present
            )
        )

        # Once we have a full batch, flush it to MongoDB and clear the buffer
        if len(operations) >= BATCH_SIZE:
            coll.bulk_write(operations, ordered=False)
            written += len(operations)
            operations.clear()

    # Flush whatever is left in the final (partial) batch
    if operations:
        coll.bulk_write(operations, ordered=False)
        written += len(operations)

    print(f"[+] Seeded {written:,} documents for '{dataset_name}'.")


def verify(coll, dataset_name: str, docs: dict):
    """
    Prove that the data — and especially the IDs — were stored and are usable.

    We pick one real doc_id from the source file and fetch it back from MongoDB
    by its _id. If it comes back, the IDs are genuinely being used as keys.
    """
    count = coll.count_documents({"dataset": dataset_name})
    print(f"\n[Verification] documents in DB for '{dataset_name}': {count:,}")

    # Take the first id from the source and look it up by _id in MongoDB
    sample_id = next(iter(docs.keys()))
    found = coll.find_one({"_id": sample_id})
    print(f"[Verification] sample lookup by _id = '{sample_id}'")
    if found:
        title = (found.get("title") or "")[:70]
        print(f"               -> FOUND. title: {title}")
    else:
        print("               -> NOT FOUND (something went wrong).")


def main():
    # Connect once, reuse the connection for all datasets
    try:
        client, coll = get_collection()
    except ServerSelectionTimeoutError:
        print("\n[ERROR] Could not reach MongoDB.")
        print("        Start it first with:  docker compose up -d")
        print("        (or make sure MongoDB is running on the URI in config).")
        sys.exit(1)

    # Create an index on "dataset" so filtered counts/queries stay fast.
    # (_id already has a unique index automatically.)
    coll.create_index([("dataset", ASCENDING)])

    # DATASETS comes from config; here it is just webis-touche2020
    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        docs = load_docs(dataset_name)
        seed_dataset(coll, dataset_name, docs)
        verify(coll, dataset_name, docs)

    client.close()
    print("\nStep 9 COMPLETE — original documents are now in MongoDB.")


if __name__ == "__main__":
    main()
