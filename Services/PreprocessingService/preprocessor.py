# services/preprocessing_service/preprocessor.py
import re
import string
import unicodedata
from typing import List, Dict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class TextPreprocessor:
    """
    Full NLP preprocessing pipeline.

    Pipeline order:
      1. Unicode normalize → ASCII
      2. Lowercase
      3. Remove URLs, HTML, numbers, punctuation
      4. Tokenize (NLTK word_tokenize)
      5. Remove stopwords
      6. Stem (Porter) OR Lemmatize (WordNet)

    Returns two formats:
      - processed_str:    "stemmed token1 token2"  (for TF-IDF)
      - processed_tokens: ["stemmed", "token1", "token2"]  (for BM25/W2V)

    IMPORTANT: Use the SAME instance for documents AND queries.
    """

    def __init__(
        self,
        language:          str  = "english",
        use_stemming:      bool = True,
        use_lemmatization: bool = False,
        remove_stopwords:  bool = True,
        stemmer_type:      str  = "porter",   # "porter" or "snowball"
        min_token_length:  int  = 2,
    ):
        self.language          = language
        self.use_stemming      = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords  = remove_stopwords
        self.min_token_length  = min_token_length

        self.stop_words = set(stopwords.words(language))

        if stemmer_type == "porter":
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = SnowballStemmer(language)

        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()

    def normalize(self, text: str) -> str:
        """Step 1-4: Unicode, lowercase, remove noise, punctuation."""
        # Unicode NFKD → ASCII
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove numbers
        text = re.sub(r"\d+", " ", text)
        # Remove punctuation
        text = text.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
        # Collapse whitespace
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> List[str]:
        """Step 5: NLTK word_tokenize."""
        return word_tokenize(text)

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Step 6: Remove stopwords and short tokens."""
        return [
            t for t in tokens
            if len(t) >= self.min_token_length
            and (not self.remove_stopwords or t not in self.stop_words)
        ]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Step 7a: Porter stemming."""
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Step 7b: WordNet lemmatization."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def process(self, text: str) -> Dict:
        """
        Full pipeline. Returns dict with two formats.
        """
        normalized = self.normalize(text)
        tokens     = self.tokenize(normalized)
        tokens     = self.filter_tokens(tokens)

        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        elif self.use_stemming:
            tokens = self.stem_tokens(tokens)

        return {
            "processed_str":    " ".join(tokens),
            "processed_tokens": tokens,
        }

    def process_batch(self, texts: List[str]) -> List[Dict]:
        return [self.process(t) for t in texts]