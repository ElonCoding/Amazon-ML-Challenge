"""
Preprocessing and Text Canonicalization Engine.
Adheres to:
- REQ-IN-1: Strict TSV parsing with sep="\t".
- REQ-IN-2: Open geography handling (supports US, India, France, and arbitrary strings).
- REQ-IN-3: Text normalization (legal suffixes, address abbreviations, punctuation).
"""

import re
import pandas as pd
from typing import Dict, List, Optional, Tuple


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


def clean_text_basic(text: Optional[str]) -> str:
    """
    Base string cleaning:
    - Null safe
    - Lowercase
    - Replace '&' with 'and'
    - Normalize punctuation and whitespace
    """
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


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalizes business name by expanding/stripping legal suffixes.
    """
    s = clean_text_basic(name)
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_address(address: Optional[str]) -> str:
    """
    Normalizes business address components and abbreviations.
    """
    s = clean_text_basic(address)
    for pattern, replacement in ADDRESS_ABBREV_MAP.items():
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


import pyarrow as pa
from typing import Dict, List, Optional, Tuple, Set


def extract_address_digits(address: Optional[str]) -> List[str]:
    """
    Extracts numerical sequences (PIN codes, ZIP codes, plot/door numbers).
    """
    if address is None or pd.isna(address):
        return []
    s = str(address)
    # Match sequences of 2 or more digits
    digits = re.findall(r"\b\d{2,8}\b", s)
    return digits


def _apply_in_chunks(series: pd.Series, function, chunk_size: int = 100_000, arrow_strings: bool = False):
    """Apply a text transform in bounded chunks to limit temporary Python objects."""
    chunks = []
    py_list = series.tolist()
    for start in range(0, len(py_list), chunk_size):
        sub = py_list[start : start + chunk_size]
        res = [function(x) for x in sub]
        if arrow_strings:
            res_series = pd.Series(pa.array(res, type=pa.string()), dtype="string[pyarrow]")
        else:
            res_series = pd.Series(res, dtype=object)
        chunks.append(res_series)
    del py_list
    if not chunks:
        dtype = "string[pyarrow]" if arrow_strings else object
        return pd.Series([], dtype=dtype)
    return pd.concat(chunks, ignore_index=True)


def load_and_preprocess_tsv(
    file_path: str,
    filter_ids: Optional[Set[str]] = None,
    max_extra_rows: int = 0,
) -> pd.DataFrame:
    """
    Loads TSV file with strict sep="\t", performs schema checks and adds cleaned columns.
    When filter_ids is provided, streams the file in chunks to keep memory usage low.
    """
    # Arrow-backed input strings use substantially less memory than Python
    # object strings for the multi-million-row challenge tables.
    if filter_ids is not None:
        chunks = []
        extra_collected = 0
        for chunk in pd.read_csv(
            file_path,
            sep="\t",
            chunksize=100_000,
            keep_default_na=False,
            dtype_backend="pyarrow",
        ):
            in_filter = chunk["entity_id"].isin(filter_ids)
            selected = chunk[in_filter]
            if max_extra_rows > 0 and extra_collected < max_extra_rows:
                needed = max_extra_rows - extra_collected
                extra = chunk[~in_filter].head(needed)
                extra_collected += len(extra)
                selected = pd.concat([selected, extra], ignore_index=True)
            if len(selected) > 0:
                chunks.append(selected)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path, sep="\t", nrows=0, keep_default_na=False, dtype_backend="pyarrow")
    else:
        df = pd.read_csv(
            file_path,
            sep="\t",
            keep_default_na=False,
            dtype_backend="pyarrow",
        )

    required_cols = ["entity_id", "business_name", "business_address", "country"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {file_path}")

    # Canonicalize text columns
    df["clean_name"] = _apply_in_chunks(
        df["business_name"], normalize_business_name, arrow_strings=False
    )
    df["clean_address"] = _apply_in_chunks(
        df["business_address"], normalize_address, arrow_strings=False
    )
    # Open-set country normalization (just strip & lowercase, do NOT filter or map to fixed set)
    df["clean_country"] = (
        df["country"].fillna("").astype(str).str.strip().str.lower().astype("category")
    )
    df["digits"] = _apply_in_chunks(df["business_address"], extract_address_digits)

    # Downstream retrieval and scoring use only these canonical fields and the
    # ID. Drop the duplicate raw text columns after canonicalization to keep
    # the resident tables smaller.
    df.drop(columns=["business_name", "business_address", "country"], inplace=True)

    return df
