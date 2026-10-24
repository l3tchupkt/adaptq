import subprocess
import sys
import textwrap
from pathlib import Path


def test_transformers_backend_imports_without_torch():
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys

        class BlockOptionalBackends:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch blocked for test")
                if name == "transformers" or name.startswith("transformers."):
                    raise ImportError("transformers blocked for test")
                return None

        sys.meta_path.insert(0, BlockOptionalBackends())

        import adaptq.runtime_py.backends.transformers_hf as backend
        assert backend._HF_AVAILABLE is False
        print("import-ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout
