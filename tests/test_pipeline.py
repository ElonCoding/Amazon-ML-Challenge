"""
Unit and Integration Tests for Business Entity Resolution Pipeline.
"""
import os, sys, unittest, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "business_entity_resolution", "code", "src")))
from evaluate import compute_entity_f_beta

class TestEntityResolutionPipeline(unittest.TestCase):
    def test_singleton_metric_scoring(self):
        self.assertEqual(compute_entity_f_beta(set(), set()), 1.0)
        self.assertEqual(compute_entity_f_beta(set(), {"S2-00001"}), 0.0)
        self.assertEqual(compute_entity_f_beta({"S2-00001"}, set()), 0.0)
        self.assertEqual(compute_entity_f_beta({"S2-00001"}, {"S2-00001"}), 1.0)

    def test_text_normalization(self):
        from preprocess import normalize_business_name, normalize_address, extract_address_digits
        self.assertEqual(normalize_business_name("Amazon Corp."), "amazon corporation")
        self.assertEqual(normalize_business_name("Tata Pvt Ltd"), "tata private limited")
        self.assertEqual(normalize_address("123 Main St, Suite 400"), "123 main street suite 400")
        self.assertIn("60607", extract_address_digits("Chicago, IL 60607"))
