"""
Candidate generation and blocking engine.
"""
import math
from array import array
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Set, Tuple
import numpy as np
import pandas as pd

def get_char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    if not text:
        return []
    words = text.split()
    ngrams = []
    for w in words:
        padded = f" {w} "
        w_len = len(padded)
        for n in range(min_n, min(max_n + 1, w_len + 1)):
            for i in range(w_len - n + 1):
                ngrams.append(padded[i : i + n])
    return ngrams

def get_word_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [w for w in text.split() if len(w) > 1]

class InvertedIndexEngine:
    def __init__(self, token_fn, max_document_frequency: int = 50_000):
        self.token_fn = token_fn
        self.max_document_frequency = max_document_frequency
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, Tuple[array, array]] = {}
        self.doc_norms: np.ndarray = np.array([])
        self.num_docs = 0

    def fit_index(self, corpus: Sequence[str]):
        self.num_docs = len(corpus)
        if self.num_docs == 0:
            return
        doc_freq = Counter()
        for text in corpus:
            doc_freq.update(set(self.token_fn(text)))
        for t, df in doc_freq.items():
            if df <= self.max_document_frequency:
                self.idf[t] = math.log((1.0 + self.num_docs) / (1.0 + df)) + 1.0
        norms = np.zeros(self.num_docs, dtype=np.float32)
        for doc_id, text in enumerate(corpus):
            counts = Counter(self.token_fn(text))
            sq_sum = 0.0
            for t, count in counts.items():
                if t not in self.idf:
                    continue
                w = (1.0 + math.log(count)) * self.idf[t]
                sq_sum += w * w
                posting = self.postings.get(t)
                if posting is None:
                    posting = (array("I"), array("f"))
                    self.postings[t] = posting
                posting[0].append(doc_id)
                posting[1].append(w)
            norms[doc_id] = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
        self.doc_norms = norms

    def query_top_k(
        self, query_text: str, top_k: int = 15, min_score: float = 0.12
    ) -> List[Tuple[int, float]]:
        if self.num_docs == 0:
            return []
        tokens = self.token_fn(query_text)
        if not tokens:
            return []
        q_counts = Counter(tokens)
        q_weights = {}
        q_sq_sum = 0.0
        for t, count in q_counts.items():
            if t in self.idf:
                w = (1.0 + math.log(count)) * self.idf[t]
                q_weights[t] = w
                q_sq_sum += w * w
        if q_sq_sum == 0.0:
            return []
        q_norm = math.sqrt(q_sq_sum)
        scores = defaultdict(float)
        for t, qw in q_weights.items():
            doc_ids, doc_weights = self.postings[t]
            for pos in range(len(doc_ids)):
                scores[doc_ids[pos]] += qw * doc_weights[pos]
        results = []
        for doc_id, dot in scores.items():
            sim = dot / (q_norm * self.doc_norms[doc_id])
            if sim >= min_score:
                results.append((doc_id, sim))
        if not results:
            return []
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]
