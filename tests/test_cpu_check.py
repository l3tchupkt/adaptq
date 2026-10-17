import unittest
from unittest.mock import patch

from adaptq.cpu_check import check_cpu_support


class TestCpuCheck(unittest.TestCase):
    def test_supported_architectures(self):
        for arch in ["x86_64", "AMD64", "x64"]:
            with patch("platform.machine", return_value=arch):
                # Should not raise
                check_cpu_support()

    def test_unsupported_arm_architecture(self):
        for arch in ["aarch64", "arm64", "armv7l"]:
            with patch("platform.machine", return_value=arch):
                with self.assertRaises(RuntimeError) as ctx:
                    check_cpu_support()
                self.assertIn("ARM NEON support is not available yet", str(ctx.exception))
                self.assertIn(arch, str(ctx.exception))

    def test_unsupported_other_architecture(self):
        with patch("platform.machine", return_value="riscv64"):
            with self.assertRaises(RuntimeError) as ctx:
                check_cpu_support()
            self.assertIn("AdapTQ requires an x86_64 CPU", str(ctx.exception))
            self.assertIn("riscv64", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
