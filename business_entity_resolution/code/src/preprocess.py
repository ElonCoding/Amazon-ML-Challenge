"""
Preprocessing and Text Canonicalization Engine.
"""
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple

def clean_text_basic(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    s = str(text).lower()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = s.replace(".", " ")
    s = s.replace(",", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
