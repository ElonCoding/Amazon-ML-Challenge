"""Low-memory, disk-backed exact-key inference fallback for constrained hosts.

The script writes complete challenge TSVs but retrieves only candidates whose
normalized country/name or country/address keys match exactly. It does not
replace the project's fuzzy multi-index blocker or establish full-data recall.
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "business_entity_resolution" / "code" / "src"
sys.path.insert(0, str(SRC))

from feature_extraction import FEATURE_NAMES, extract_pair_features
from preprocess import (
    extract_address_digits,
    normalize_address,
    normalize_business_name,
)
from train import EntityResolutionClassifier

LOOKUP_SQL = """
WITH found AS (
    SELECT * FROM (
        SELECT entity_id, source, country, clean_name, clean_address, digits, 2 AS priority
        FROM targets INDEXED BY idx_name
        WHERE country=? AND clean_name=? AND source='S2' AND clean_name<>''
        ORDER BY entity_id LIMIT ?
    )
    UNION ALL
    SELECT * FROM (
        SELECT entity_id, source, country, clean_name, clean_address, digits, 2 AS priority
        FROM targets INDEXED BY idx_name
        WHERE country=? AND clean_name=? AND source='S3' AND clean_name<>''
        ORDER BY entity_id LIMIT ?
    )
    UNION ALL
    SELECT * FROM (
        SELECT entity_id, source, country, clean_name, clean_address, digits, 1 AS priority
        FROM targets INDEXED BY idx_address
        WHERE country=? AND clean_address=? AND source='S2' AND clean_address<>''
        ORDER BY entity_id LIMIT ?
    )
    UNION ALL
    SELECT * FROM (
        SELECT entity_id, source, country, clean_name, clean_address, digits, 1 AS priority
        FROM targets INDEXED BY idx_address
        WHERE country=? AND clean_address=? AND source='S3' AND clean_address<>''
        ORDER BY entity_id LIMIT ?
    )
)
SELECT entity_id, source, country, clean_name, clean_address, digits, priority
FROM found
"""


def normalized_chunk(frame):
    names = [normalize_business_name(value) for value in frame["business_name"].tolist()]
    addresses = [normalize_address(value) for value in frame["business_address"].tolist()]
    countries = [str(value).strip().lower() for value in frame["country"].tolist()]
    digits = ["|".join(extract_address_digits(value)) for value in frame["business_address"].tolist()]
    return names, addresses, countries, digits


def build_target_index(test_dir: Path, database_path: Path, chunk_size: int = 50_000):
    building_path = database_path.with_suffix(database_path.suffix + ".building")
    if building_path.exists():
        building_path.unlink()

    connection = sqlite3.connect(building_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("PRAGMA page_size=4096")
    connection.execute(
        "CREATE TABLE targets ("
        "source TEXT NOT NULL, entity_id TEXT NOT NULL, country TEXT NOT NULL, "
        "clean_name TEXT NOT NULL, clean_address TEXT NOT NULL, digits TEXT NOT NULL, "
        "PRIMARY KEY (source, entity_id)) WITHOUT ROWID"
    )

    total = 0
    start = time.monotonic()
    for source, filename in (("S2", "test_source2.tsv"), ("S3", "test_source3.tsv")):
        path = test_dir / filename
        print(f"Indexing {source}: {path}", flush=True)
        for frame in pd.read_csv(
            path,
            sep="\t",
            chunksize=chunk_size,
            keep_default_na=False,
            dtype_backend="pyarrow",
        ):
            required = {"entity_id", "business_name", "business_address", "country"}
            if not required.issubset(frame.columns):
                raise ValueError(f"Missing required columns in {path}")
            names, addresses, countries, digits = normalized_chunk(frame)
            rows = zip(
                [source] * len(frame),
                frame["entity_id"].astype(str).tolist(),
                countries,
                names,
                addresses,
                digits,
            )
            connection.executemany(
                "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?)", rows
            )
            total += len(frame)
            if total % 500_000 < len(frame):
                connection.commit()
                print(f"Indexed {total:,} targets in {time.monotonic() - start:.0f}s", flush=True)
        connection.commit()

    print("Building SQLite lookup indexes...", flush=True)
    connection.execute(
        "CREATE INDEX idx_name ON targets(country, clean_name, source, entity_id)"
    )
    connection.execute(
        "CREATE INDEX idx_address ON targets(country, clean_address, source, entity_id)"
    )
    connection.commit()
    connection.close()
    os.replace(building_path, database_path)
    print(
        f"Indexed {total:,} targets; database {database_path.stat().st_size / 1024**3:.2f} GB; "
        f"elapsed {time.monotonic() - start:.0f}s",
        flush=True,
    )


def retrieve_candidates(connection, country, name, address, cap):
    by_source = {"S2": {}, "S3": {}}
    limit = max(cap, 1)
    params = (
        country, name, limit, country, name, limit,
        country, address, limit, country, address, limit,
    )
    for row in connection.execute(LOOKUP_SQL, params):
        source = row[1]
        old = by_source[source].get(row[0])
        priority = 3 if old else row[6]
        by_source[source][row[0]] = [row[:6], priority]

    ranked = {}
    for source, matches in by_source.items():
        ranked[source] = sorted(
            matches.values(), key=lambda item: (-item[1], item[0][0])
        )[:cap]

    chosen = []
    seen = set()
    max_rank = max((len(items) for items in ranked.values()), default=0)
    for rank in range(max_rank):
        for source in ("S2", "S3"):
            items = ranked[source]
            if rank < len(items):
                row = items[rank][0]
                if row[0] not in seen:
                    seen.add(row[0])
                    chosen.append(row)
                    if len(chosen) >= cap:
                        return chosen
    return chosen


def process_inference(test_dir: Path, model_path: Path, output_dir: Path, database_path: Path,
                      cap: int, batch_anchors: int):
    if database_path.exists():
        print(f"Reusing target index: {database_path}", flush=True)
    else:
        build_target_index(test_dir, database_path)

    classifier = EntityResolutionClassifier.load(str(model_path))
    booster = classifier.booster
    if booster is None:
        raise RuntimeError("Model artifact has no LightGBM booster")
    threshold = classifier.optimal_threshold
    connection = sqlite3.connect(database_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_partial = output_dir / "candidate_pairs.tsv.partial"
    matching_partial = output_dir / "matching_results.tsv.partial"
    source1_path = test_dir / "test_source1.tsv"
    candidate_total = 0
    matched_total = 0
    processed = 0
    start = time.monotonic()

    with candidate_partial.open("w", encoding="utf-8", newline="") as candidate_file, \
            matching_partial.open("w", encoding="utf-8", newline="") as matching_file:
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        matching_file.write("source1_entity_id\tmatched_entity_ids\n")

        for frame in pd.read_csv(
            source1_path,
            sep="\t",
            chunksize=batch_anchors,
            keep_default_na=False,
            dtype_backend="pyarrow",
        ):
            names, addresses, countries, digits = normalized_chunk(frame)
            ids = frame["entity_id"].astype(str).tolist()
            pair_features = []
            candidate_lists = []
            for anchor_index, (entity_id, country, name, address, anchor_digits) in enumerate(
                zip(ids, countries, names, addresses, digits)
            ):
                candidates = retrieve_candidates(connection, country, name, address, cap)
                candidate_lists.append([row[0] for row in candidates])
                s1_row = {
                    "entity_id": entity_id,
                    "clean_country": country,
                    "clean_name": name,
                    "clean_address": address,
                    "digits": anchor_digits.split("|") if anchor_digits else [],
                }
                for row in candidates:
                    target_row = {
                        "entity_id": row[0],
                        "clean_country": row[2],
                        "clean_name": row[3],
                        "clean_address": row[4],
                        "digits": row[5].split("|") if row[5] else [],
                    }
                    features = extract_pair_features(s1_row, target_row)
                    pair_features.append([features[name] for name in FEATURE_NAMES])

            predictions = [[] for _ in ids]
            # Candidate and feature lists share order, so scores map directly to IDs.
            if pair_features:
                pair_target_ids = []
                for anchor_index, candidate_ids in enumerate(candidate_lists):
                    for target_id in candidate_ids:
                        pair_target_ids.append((anchor_index, target_id))
                probabilities = booster.predict(np.asarray(pair_features, dtype=np.float32))
                for (anchor_index, target_id), probability in zip(pair_target_ids, probabilities):
                    if probability >= threshold:
                        predictions[anchor_index].append(target_id)

            for entity_id, candidates, matches in zip(ids, candidate_lists, predictions):
                candidate_file.write(f"{entity_id}\t{','.join(candidates)}\n")
                matching_file.write(f"{entity_id}\t{','.join(matches)}\n")
                candidate_total += len(candidates)
                matched_total += len(matches)
            processed += len(ids)
            if processed % 10_000 < len(ids):
                candidate_file.flush()
                matching_file.flush()
                print(
                    f"Scored {processed:,}/{1_732_544:,} Source 1 rows; "
                    f"candidates={candidate_total:,}, matches={matched_total:,}; "
                    f"elapsed={time.monotonic() - start:.0f}s",
                    flush=True,
                )

    connection.close()
    os.replace(candidate_partial, output_dir / "candidate_pairs.tsv")
    os.replace(matching_partial, output_dir / "matching_results.tsv")
    print(
        f"Complete: rows={processed:,}, candidates={candidate_total:,}, matches={matched_total:,}, "
        f"threshold={threshold:.3f}, elapsed={time.monotonic() - start:.0f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=ROOT / "dataset" / "student_resource" / "dataset" / "test")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "validation_5000_cap40_model.joblib")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--database-path", type=Path, default=ROOT / "models" / "streaming_test_candidate_index.sqlite")
    parser.add_argument("--candidate-cap", type=int, default=40)
    parser.add_argument("--batch-anchors", type=int, default=2000)
    args = parser.parse_args()
    process_inference(
        args.test_dir,
        args.model_path,
        args.output_dir,
        args.database_path,
        args.candidate_cap,
        args.batch_anchors,
    )


if __name__ == "__main__":
    main()
