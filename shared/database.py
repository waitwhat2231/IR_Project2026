import os
from typing import List, Dict, Optional
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError
from tqdm import tqdm

from shared.config import MONGO_URI, MONGO_DB, MONGO_COLL


class DocumentDatabase:
    """
    Wraps MongoDB. Stores and retrieves ORIGINAL (raw) documents.

    Schema per document:
    {
        "_id":     "doc_001",          # doc_id IS the MongoDB _id → free index
        "dataset": "trec-covid",
        "title":   "Original title",
        "text":    "Original full text..."
    }

    Using doc_id as _id gives us:
    - Automatic unique index (fast O(log n) lookup)
    - No separate index needed for the most common query pattern
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection: Optional[str] = None,
    ):
        self.uri: str = uri or os.getenv("MONGO_URI", MONGO_URI)
        self.db_name: str = db_name or MONGO_DB
        self.coll_name: str = collection or MONGO_COLL
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.coll: Optional[Collection] = None

    @property
    def _coll(self) -> Collection:
        """Type-narrowed accessor — asserts connect() has been called."""
        assert (
            self.coll is not None
        ), "DocumentDatabase: call connect() before using the database"
        return self.coll

    def connect(self):
        self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        # Verify connection
        self.client.admin.command("ping")
        self.db = self.client[self.db_name]
        self.coll = self.db[self.coll_name]
        # Ensure index on dataset field for filtered queries
        self.coll.create_index([("dataset", ASCENDING)])
        print(f"MongoDB connected: {self.uri} / {self.db_name}")
        print(f"Total documents in DB: {self.coll.count_documents({})}")

    def disconnect(self):
        if self.client:
            self.client.close()

    # ─── WRITE (offline only) ────────────────────────────────────

    def insert_dataset(
        self,
        docs: Dict[str, Dict],
        dataset: str,
        batch_size: int = 1000,
    ):
        """
        Bulk insert original documents.
        Called ONCE during the offline phase (step8_load_to_mongodb.py).

        Uses _id = doc_id so we get a free unique index.
        ON CONFLICT → skip (idempotent, safe to re-run).
        """
        items = list(docs.items())
        total = len(items)
        inserted = 0

        for i in tqdm(
            range(0, total, batch_size), desc=f"Inserting {dataset} into MongoDB"
        ):
            batch = items[i : i + batch_size]
            mongo_docs = [
                {
                    "_id": doc_id,  # doc_id AS the primary key
                    "dataset": dataset,
                    "title": doc.get("title", "")[:2000],  # cap at 2KB
                    "text": doc.get("text", "")[:100000],  # cap at 100KB
                }
                for doc_id, doc in batch
            ]
            try:
                result = self._coll.insert_many(
                    mongo_docs,
                    ordered=False,  # continue on duplicate key errors
                )
                inserted += len(result.inserted_ids)
            except BulkWriteError as e:
                # Some docs may already exist — count only new ones
                inserted += e.details.get("nInserted", 0)

        print(
            f"Dataset '{dataset}': {inserted:,} new docs inserted "
            f"(total in collection: {self.count(dataset):,})"
        )

    # ─── READ (online — called after every retrieval) ────────────

    def get_by_ids(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """
        THE KEY ONLINE METHOD.

        After the retrieval model returns top-10 doc_ids,
        we call this ONCE to get their original content.

        MongoDB query:
            db.documents.find({ _id: { $in: [id1, id2, ..., id10] } })

        Uses the _id index → essentially O(1) per document.
        One round-trip to MongoDB for all 10 documents.

        Returns: {doc_id: {title, text}}
        """
        cursor = self._coll.find(
            {"_id": {"$in": doc_ids}},
            {"_id": 1, "title": 1, "text": 1},  # projection: only these fields
        )
        return {
            doc["_id"]: {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
            for doc in cursor
        }

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """Fetch a single document."""
        doc = self._coll.find_one({"_id": doc_id}, {"_id": 1, "title": 1, "text": 1})
        if doc:
            return {"title": doc.get("title", ""), "text": doc.get("text", "")}
        return None

    def count(self, dataset: Optional[str] = None) -> int:
        query = {"dataset": dataset} if dataset else {}
        return self._coll.count_documents(query)

    def dataset_exists(self, dataset: str) -> bool:
        return self._coll.count_documents({"dataset": dataset}, limit=1) > 0


# Module-level singleton — import this everywhere
db = DocumentDatabase()
