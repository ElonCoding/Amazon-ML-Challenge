"""
Candidate generation and blocking engine.

Uses per-source, per-country TF-IDF inverted indexes over name character
n-grams and address word tokens, plus a bounded numeric-token lookup. The
candidate cap is a configurable project setting, not a numeric challenge rule.
"""
import math
from array import array
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Set, Tuple
import numpy as np
import pandas as pd


def get_char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    """
    Extracts character n-grams with whitespace boundary tokens.
    """
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
    """
    Tokenizes text into words.
    """
    if not text:
        return []
    return [w for w in text.split() if len(w) > 1]


class InvertedIndexEngine:
    """
    TF-IDF Inverted Index with cosine scoring. Terms appearing in too many
    documents are omitted from postings: these terms are weak retrieval
    signals and otherwise create very large posting traversals on this corpus.
    """
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

        # Pass one stores only corpus-level document frequencies. Retaining a
        # Counter for every document multiplies memory use across millions of
        # target records.
        doc_freq = Counter()
        for text in corpus:
            doc_freq.update(set(self.token_fn(text)))

        # Compute IDF weights
        for t, df in doc_freq.items():
            if df <= self.max_document_frequency:
                self.idf[t] = math.log((1.0 + self.num_docs) / (1.0 + df)) + 1.0

        # Pass two regenerates per-document term counts and retains postings
        # only for informative terms. This trades a second tokenization pass
        # for much lower peak memory and less work per query.
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

        # Accumulate dot product scores
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

        # Sort by similarity score descending
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]


class MultiIndexBlocker:
    def __init__(
        self,
        max_candidates_per_entity: int = 20,
        name_top_k: int = 12,
        address_top_k: int = 6,
        min_name_sim: float = 0.12,
        min_address_sim: float = 0.15,
    ):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.name_top_k = name_top_k
        self.address_top_k = address_top_k
        self.min_name_sim = min_name_sim
        self.min_address_sim = min_address_sim

    def generate_candidates(
        self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Generates candidate target IDs for each entity in S1.
        Returns:
            candidate_map: Dict[s1_entity_id -> List[target_entity_ids]]
        """
        candidate_map: Dict[str, List[str]] = {
            s1_id: [] for s1_id in df_s1["entity_id"]
        }

        # Dynamic country partitioning (open set)
        unique_countries = df_s1["clean_country"].unique()

        for country in unique_countries:
            s1_sub = df_s1[df_s1["clean_country"] == country].reset_index(drop=True)
            if s1_sub.empty:
                continue

            # Keep the two large target frames separate. Besides avoiding a
            # full concatenated copy, independent retrieval prevents a strong
            # Source 2 neighborhood from consuming every Source 3 top-K slot.
            candidates_by_source = []
            for target_frame in (df_s2, df_s3):
                target_sub = target_frame[
                    target_frame["clean_country"] == country
                ].reset_index(drop=True)
                if target_sub.empty:
                    candidates_by_source.append({})
                else:
                    candidates_by_source.append(
                        self._block_country_partition(s1_sub, target_sub)
                    )
                del target_sub

            for s1_id in s1_sub["entity_id"]:
                source_lists = [
                    source_map.get(s1_id, []) for source_map in candidates_by_source
                ]
                merged = []
                seen = set()
                max_rank = max((len(items) for items in source_lists), default=0)
                for rank in range(max_rank):
                    for items in source_lists:
                        if rank < len(items) and items[rank] not in seen:
                            seen.add(items[rank])
                            merged.append(items[rank])
                            if len(merged) >= self.max_candidates_per_entity:
                                break
                    if len(merged) >= self.max_candidates_per_entity:
                        break
                candidate_map[s1_id] = merged

        return candidate_map

    def _block_country_partition(
        self, s1_sub: pd.DataFrame, target_sub: pd.DataFrame
    ) -> Dict[str, List[str]]:
        s1_ids = s1_sub["entity_id"].array
        # Keep IDs in the underlying Arrow-backed array; converting millions
        # of IDs to a second Python list adds substantial duplicate memory.
        target_ids = target_sub["entity_id"].array
        n_s1 = len(s1_ids)

        # 1. Build Index A: Name Character N-Gram Inverted Index
        name_index = InvertedIndexEngine(token_fn=get_char_ngrams)
        name_index.fit_index(target_sub["clean_name"].array)

        # 2. Build Index B: Address Word Token Inverted Index
        addr_index = InvertedIndexEngine(token_fn=get_word_tokens)
        addr_index.fit_index(target_sub["clean_address"].array)

        # 3. Build Index C: Postal/PIN Code Inverted Index
        pin_index = defaultdict(list)
        for tidx, digits in enumerate(target_sub["digits"]):
            for d in digits:
                # Retrieval below only ever inspects the first ten records for
                # a numeric token, so do not retain arbitrarily long postings.
                if len(d) >= 4 and len(pin_index[d]) < 10:
                    pin_index[d].append(tidx)

        result: Dict[str, List[str]] = {}

        for i in range(n_s1):
            s1_id = s1_ids[i]
            s1_name = s1_sub.at[i, "clean_name"]
            s1_addr = s1_sub.at[i, "clean_address"]
            s1_digits = s1_sub.at[i, "digits"]

            cand_scores: Dict[int, float] = defaultdict(float)

            # Query Name Index
            name_hits = name_index.query_top_k(
                s1_name, top_k=self.name_top_k, min_score=self.min_name_sim
            )
            for tidx, score in name_hits:
                cand_scores[tidx] += score * 1.5

            # Query Address Index
            addr_hits = addr_index.query_top_k(
                s1_addr, top_k=self.address_top_k, min_score=self.min_address_sim
            )
            for tidx, score in addr_hits:
                cand_scores[tidx] += score * 1.0

            # Query PIN Index
            for d in s1_digits:
                if len(d) >= 4 and d in pin_index:
                    for tidx in pin_index[d][:10]:
                        cand_scores[tidx] += 0.35

            if not cand_scores:
                result[s1_id] = []
            else:
                # Rank by accumulated score descending
                sorted_cands = sorted(
                    cand_scores.items(), key=lambda x: x[1], reverse=True
                )
                capped_cands = [
                    str(target_ids[tidx])
                    for tidx, _ in sorted_cands[: self.max_candidates_per_entity]
                ]
                result[s1_id] = capped_cands

        return result

