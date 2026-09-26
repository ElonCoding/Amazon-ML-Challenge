"""
Unit and Integration Tests for Business Entity Resolution Pipeline.
Tests:
- TSV delimiter and header schema compliance.
- Singleton empty string representation.
- Prefix validity (S2/S3 only, zero S1 self-links).
- Candidate subset integrity (matched <= candidate).
- Uniqueness of entity rows.
- Open-set geography resilience (France support).
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "business_entity_resolution", "code", "src")))
from preprocess import normalize_business_name, normalize_address, extract_address_digits
from blocking import MultiIndexBlocker
from evaluate import compute_entity_f_beta, compute_macro_f05


class TestEntityResolutionPipeline(unittest.TestCase):
    def test_singleton_metric_scoring(self):
        self.assertEqual(compute_entity_f_beta(set(), set()), 1.0)
        self.assertEqual(compute_entity_f_beta(set(), {"S2-00001"}), 0.0)
        self.assertEqual(compute_entity_f_beta({"S2-00001"}, set()), 0.0)
        self.assertEqual(compute_entity_f_beta({"S2-00001"}, {"S2-00001"}), 1.0)

    def test_text_normalization(self):
        self.assertEqual(normalize_business_name("Amazon Corp."), "amazon corporation")
        self.assertEqual(normalize_business_name("Tata Pvt Ltd"), "tata private limited")
        self.assertEqual(normalize_address("123 Main St, Suite 400"), "123 main street suite 400")
        self.assertIn("60607", extract_address_digits("Chicago, IL 60607"))

    def test_france_open_set_blocking(self):
        df_s1 = pd.DataFrame([{
            "entity_id": "S1-FR01",
            "business_name": "Lumiere Technologies",
            "business_address": "12 Rue de la Paix, Paris",
            "country": "France",
            "clean_name": "lumiere technologies",
            "clean_address": "12 rue de la paix paris",
            "clean_country": "france",
            "digits": [],
        }])
        df_s2 = pd.DataFrame([{
            "entity_id": "S2-FR01",
            "business_name": "Lumiere Tech SARL",
            "business_address": "12 Rue de la Paix, 75002 Paris",
            "country": "France",
            "clean_name": "lumiere tech sarl",
            "clean_address": "12 rue de la paix 75002 paris",
            "clean_country": "france",
            "digits": ["75002"],
        }])
        df_s3 = pd.DataFrame(columns=df_s2.columns)

        blocker = MultiIndexBlocker(max_candidates_per_entity=5)
        cands = blocker.generate_candidates(df_s1, df_s2, df_s3)
        self.assertIn("S1-FR01", cands)
        self.assertIn("S2-FR01", cands["S1-FR01"])

    def test_output_file_schema_and_integrity(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        match_path = os.path.join(base_dir, "output", "matching_results.tsv")
        cand_path = os.path.join(base_dir, "output", "candidate_pairs.tsv")

        if not os.path.exists(match_path) or not os.path.exists(cand_path):
            self.skipTest("Output TSVs not generated yet.")

        df_match = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
        self.assertEqual(list(df_match.columns), ["source1_entity_id", "matched_entity_ids"])
        self.assertEqual(len(df_match), len(df_match["source1_entity_id"].unique()), "Duplicate S1 IDs found.")

        df_cand = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
        self.assertEqual(list(df_cand.columns), ["source1_entity_id", "candidate_entity_ids"])
        self.assertEqual(len(df_cand), len(df_cand["source1_entity_id"].unique()), "Duplicate S1 IDs found.")

        cand_map = {
            row["source1_entity_id"]: set(row["candidate_entity_ids"].split(",")) if row["candidate_entity_ids"] else set()
            for _, row in df_cand.iterrows()
        }
        for _, row in df_match.iterrows():
            s1_id = row["source1_entity_id"]
            matched = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
            cands = cand_map.get(s1_id, set())
            for m in matched:
                self.assertIn(m, cands, f"Matched ID {m} for {s1_id} was not in candidates.")
                self.assertTrue(m.startswith(("S2-", "S3-")), f"Invalid target prefix: {m}")
                self.assertFalse(m.startswith("S1-"), f"Self match detected: {m}")

if __name__ == "__main__":
    unittest.main()
