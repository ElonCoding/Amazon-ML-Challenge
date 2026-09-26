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
