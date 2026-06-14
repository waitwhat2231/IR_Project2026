import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from Services.retrieval_service.word2vec_retriever import Word2VecRetriever
from Services.retrieval_service.tfidf_retriever import TFIDFRetriever


def test_tfidf_retriever_returns_matching_document_first():
	corpus = {
		"doc-1": "apple banana apple",
		"doc-2": "car engine road",
	}

	retriever = TFIDFRetriever().fit(corpus)
	results = retriever.retrieve("apple", top_k=2)

	assert results[0][0] == "doc-1"
	assert results[0][1] >= results[1][1]


def test_word2vec_retriever_roundtrip(tmp_path: Path):
	corpus = {
		"doc-1": ["apple", "banana", "apple"],
		"doc-2": ["car", "engine", "road"],
		"doc-3": ["apple", "fruit"],
	}

	retriever = Word2VecRetriever(vector_size=32, min_count=1, epochs=10, workers=1, seed=13)
	retriever.fit(corpus)

	prefix = tmp_path / "word2vec_test"
	retriever.save(prefix)

	loaded = Word2VecRetriever.load(prefix)
	results = loaded.retrieve("apple", top_k=3)

	assert loaded.doc_embeddings is not None
	assert loaded.doc_embeddings.shape[0] == 3
	assert results[0][0] in {"doc-1", "doc-3"}
