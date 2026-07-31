"""
Week 2 - Feature Engineering
-----------------------------
Bag-of-Words, TF-IDF and simple word embeddings implemented from scratch
with NumPy only (no sklearn.feature_extraction.text, no gensim/word2vec).
"""

import math
import numpy as np
from collections import Counter


def add_bigrams(tokens: list) -> list:
    """Appends 'word1_word2' bigram tokens to a unigram token list. This is
    plain Python string-joining, not an external n-gram library."""
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


class Vocabulary:
    """Builds a word -> index mapping from a list of tokenized documents."""

    def __init__(self, max_features: int = 5000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.word2idx = {}
        self.idx2word = []

    def fit(self, tokenized_docs: list):
        df = Counter()
        for tokens in tokenized_docs:
            for w in set(tokens):
                df[w] += 1
        # keep words that appear in at least min_df docs, most frequent first
        candidates = [(w, c) for w, c in df.items() if c >= self.min_df]
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[: self.max_features]
        self.idx2word = [w for w, _ in candidates]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.doc_freq = dict(candidates)
        return self

    def __len__(self):
        return len(self.idx2word)


class BagOfWords:
    """Raw term-count feature matrix."""

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def transform(self, tokenized_docs: list) -> np.ndarray:
        n, d = len(tokenized_docs), len(self.vocab)
        X = np.zeros((n, d), dtype=np.float32)
        for i, tokens in enumerate(tokenized_docs):
            counts = Counter(tokens)
            for w, c in counts.items():
                j = self.vocab.word2idx.get(w)
                if j is not None:
                    X[i, j] = c
        return X


class TfidfVectorizerScratch:
    """TF-IDF built from scratch:
        tf(t,d)  = count(t,d) / len(d)
        idf(t)   = log(N / (1 + df(t))) + 1
        tfidf    = tf * idf, then L2-normalized per document.
    """

    def __init__(self, max_features: int = 5000, min_df: int = 2, use_bigrams: bool = False):
        self.vocab = Vocabulary(max_features=max_features, min_df=min_df)
        self.idf_ = None
        self.use_bigrams = use_bigrams

    def _prep(self, tokenized_docs: list) -> list:
        if self.use_bigrams:
            return [add_bigrams(t) for t in tokenized_docs]
        return tokenized_docs

    def fit(self, tokenized_docs: list):
        tokenized_docs = self._prep(tokenized_docs)
        self.vocab.fit(tokenized_docs)
        n_docs = len(tokenized_docs)
        self.idf_ = np.zeros(len(self.vocab), dtype=np.float32)
        for w, j in self.vocab.word2idx.items():
            df = self.vocab.doc_freq[w]
            self.idf_[j] = math.log(n_docs / (1 + df)) + 1
        return self

    def transform(self, tokenized_docs: list) -> np.ndarray:
        tokenized_docs = self._prep(tokenized_docs)
        n, d = len(tokenized_docs), len(self.vocab)
        X = np.zeros((n, d), dtype=np.float32)
        for i, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            counts = Counter(tokens)
            length = len(tokens)
            for w, c in counts.items():
                j = self.vocab.word2idx.get(w)
                if j is not None:
                    tf = c / length
                    X[i, j] = tf * self.idf_[j]
        # L2 normalize each row
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms
        return X

    def fit_transform(self, tokenized_docs: list) -> np.ndarray:
        self.fit(tokenized_docs)
        return self.transform(tokenized_docs)


class CooccurrenceEmbedding:
    """A tiny from-scratch word-embedding: build a word-word co-occurrence
    matrix (window based), apply PPMI weighting, then reduce dimensions
    with truncated SVD (numpy.linalg.svd). Document vectors are the mean
    of their words' embeddings. This mirrors the idea behind GloVe/word2vec
    without depending on an external embedding library.
    """

    def __init__(self, vocab: Vocabulary, window: int = 4, dim: int = 50):
        self.vocab = vocab
        self.window = window
        self.dim = dim
        self.embeddings = None

    def fit(self, tokenized_docs: list):
        V = len(self.vocab)
        cooc = np.zeros((V, V), dtype=np.float32)
        for tokens in tokenized_docs:
            idxs = [self.vocab.word2idx[t] for t in tokens if t in self.vocab.word2idx]
            for center_pos, center in enumerate(idxs):
                start = max(0, center_pos - self.window)
                end = min(len(idxs), center_pos + self.window + 1)
                for ctx_pos in range(start, end):
                    if ctx_pos != center_pos:
                        cooc[center, idxs[ctx_pos]] += 1

        total = cooc.sum()
        row_sums = cooc.sum(axis=1, keepdims=True)
        col_sums = cooc.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = (row_sums @ col_sums) / max(total, 1)
            pmi = np.log((cooc + 1e-9) / (expected + 1e-9))
        ppmi = np.maximum(pmi, 0)

        # Truncated SVD via numpy for dimensionality reduction
        U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
        k = min(self.dim, U.shape[1])
        self.embeddings = U[:, :k] * S[:k]
        return self

    def transform(self, tokenized_docs: list) -> np.ndarray:
        n = len(tokenized_docs)
        dim = self.embeddings.shape[1]
        X = np.zeros((n, dim), dtype=np.float32)
        for i, tokens in enumerate(tokenized_docs):
            idxs = [self.vocab.word2idx[t] for t in tokens if t in self.vocab.word2idx]
            if idxs:
                X[i] = self.embeddings[idxs].mean(axis=0)
        return X
