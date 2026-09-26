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

LEGAL_SUFFIX_MAP = {
    r"\bcorp\.?\b": "corporation",
    r"\bco\.?\b": "company",
    r"\binc\.?\b": "incorporated",
    r"\bltd\.?\b": "limited",
    r"\bpvt\.?\b": "private",
    r"\bllc\.?\b": "llc",
    r"\bsarl\.?\b": "sarl",
    r"\bsas\.?\b": "sas",
    r"\bsa\.?\b": "sa",
    r"\bgmbh\.?\b": "gmbh",
    r"\bplc\.?\b": "plc",
    r"\bpvt\s+ltd\.?\b": "private limited",
    r"\bprivate\s+limited\b": "private limited",
}

def normalize_business_name(name: Optional[str]) -> str:
    s = clean_text_basic(name)
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

ADDRESS_ABBREV_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bln\.?\b": "lane",
    r"\bfl\.?\b": "floor",
    r"\bste\.?\b": "suite",
    r"\bbldg\.?\b": "building",
    r"\bmarg\b": "road",
    r"\bopp\.?\b": "opposite",
    r"\bnr\.?\b": "near",
    r"\bsec\.?\b": "sector",
    r"\bpl\.?\b": "place",
    r"\bpkwy\.?\b": "parkway",
}

def normalize_address(address: Optional[str]) -> str:
    s = clean_text_basic(address)
    for pattern, replacement in ADDRESS_ABBREV_MAP.items():
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
