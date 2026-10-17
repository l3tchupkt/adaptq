import os
import unittest
from unittest.mock import patch

from adaptq.cpu_detection import detect_cpu_features, get_cpu_report


class TestCPUDetection(unittest.TestCase):
    def test_default_detection(self):
        report = detect_cpu_features()
        self.assertIn("architecture", report)
        self.assertIn("avx2", report)
        self.assertIn("fma", report)
        self.assertIn("status", report)
        text = get_cpu_report()
        self.assertIn("AdapTQ CPU capability check", text)
        self.assertIn("Architecture:", text)
        self.assertIn("Status:", text)

    def test_force_avx2_override(self):
        with patch.dict(os.environ, {"ADAPTQ_FORCE_AVX2": "1"}):
            report = detect_cpu_features()
            self.assertTrue(report["avx2"])
            self.assertTrue(report["fma"])
            self.assertEqual(report["status"], "AVX2 backend available")

    def test_force_scalar_override(self):
        with patch.dict(os.environ, {"ADAPTQ_FORCE_SCALAR": "1"}):
            report = detect_cpu_features()
            self.assertFalse(report["avx2"])
            self.assertFalse(report["fma"])
            self.assertEqual(report["status"], "AVX2 backend unavailable")
            self.assertIn("reason", report)


if __name__ == "__main__":
    unittest.main()
