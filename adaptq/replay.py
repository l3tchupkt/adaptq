"""
adaptq.replay — Python interface to AdapTQ V2 Replay + Compare

Provides Python-friendly wrappers around the C++ ReplayEngine,
SessionSnapshot, and RuntimeContext for session capture, deterministic
replay, branching, and cross-strategy comparison.

These wrappers use the subprocess CLI (adaptq binary) since the pybind11
extension currently exposes V1 MHAContext only. A native pybind11 binding
for ReplayEngine is planned for V3.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

__all__ = [
    "ReplayEngine",
    "ReplayResult",
    "CompareResult",
    "snapshot_info",
    "snapshot_to_json",
]


# ---------------------------------------------------------------------------
# Locate the adaptq CLI binary
# ---------------------------------------------------------------------------

def _find_binary() -> Optional[str]:
    """Find the adaptq CLI binary (build_v2/adapTQ_demo or system path)."""
    # 1. Check alongside this package (editable installs)
    pkg_dir = Path(__file__).parent.parent
    import sys
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    binary_name = f"adapTQ_demo{exe_suffix}"
    
    candidates = [
        pkg_dir / "build_release" / binary_name,
        pkg_dir / "build" / binary_name,
        pkg_dir / "build_v2" / binary_name,   # legacy fallback
        pkg_dir / binary_name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # 2. Fall back to PATH
    import shutil
    found = shutil.which("adaptq") or shutil.which("adapTQ_demo")
    return found


_BINARY: Optional[str] = None


def _get_binary() -> str:
    global _BINARY
    if _BINARY is None:
        _BINARY = _find_binary()
    if _BINARY is None:
        raise RuntimeError(
            "adaptq CLI binary not found. "
            "Build the project first:\n"
            "  cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build_release --parallel\n"
            "Or install from source:\n"
            "  pip install ."
        )
    return _BINARY


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------

_VALID_OUTPUT_FORMATS = {"json", "csv", "md", "tex"}

# AQSS V2 header written by replay/session_snapshot.cpp::save().
# <IIiiiiiiQ = magic, version, 6 signed 32-bit fields, flags.
_SNAPSHOT_HEADER = struct.Struct("<IIiiiiiiQ")
_SNAPSHOT_MAGIC = 0x41515353
_SNAPSHOT_VERSION = 2


def _validate_output_format(output_format: str) -> None:
    if output_format not in _VALID_OUTPUT_FORMATS:
        allowed = ", ".join(sorted(_VALID_OUTPUT_FORMATS))
        raise ValueError(
            f"unsupported output_format {output_format!r}; expected one of: {allowed}"
        )


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read JSON replay result from {path}: {exc}") from exc


def _run_cli(command: List[str], error_prefix: str) -> dict:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{error_prefix}:\n{result.stderr}")
    if not result.stdout.strip():
        raise RuntimeError(f"{error_prefix}: CLI returned no JSON summary")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{error_prefix}: invalid JSON summary from CLI: {exc}"
        ) from exc


def _run_json_or_file(
    command: List[str],
    *,
    output_format: str,
    output_path: Optional[Union[str, Path]],
    temp_suffix: str,
    error_prefix: str,
) -> dict:
    """Run a CLI report once and return its structured JSON summary.

    When the caller supplies output_path, the requested format is written by
    the CLI and a JSON copy of the same report is emitted on stdout. Without
    output_path, JSON is captured in a temporary file for the Python result.
    """
    if output_path is not None:
        command += [
            "--format", output_format,
            "--output", str(output_path),
            "--summary-json",
        ]
        return _run_cli(command, error_prefix)

    with tempfile.NamedTemporaryFile(
        suffix=temp_suffix, delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        command += ["--format", "json", "--output", tmp_path]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{error_prefix}:\n{result.stderr}")
        return _load_json(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# ReplayResult — Python-side view of ReplayReport
# ---------------------------------------------------------------------------

class ReplayResult:
    """
    Result of a replay or compare run.

    Attributes
    ----------
    strategy : str
        Name of the strategy used.
    n_tokens_replayed : int
        Number of tokens replayed.
    wall_time_ms : float
        Total wall time in milliseconds.
    metrics : list[dict]
        Per-token metrics (only populated if collect_metrics=True).
    raw : dict
        Full parsed JSON from the CLI.
    """

    def __init__(self, raw: dict):
        self.raw = raw
        self.strategy: str = raw.get("strategy", "")
        self.n_tokens_replayed: int = raw.get("n_tokens_replayed", 0)
        self.wall_time_ms: float = raw.get("wall_time_ms", 0.0)
        self.metrics: List[dict] = raw.get("metrics", [])

    def __repr__(self) -> str:
        return (
            f"ReplayResult(strategy={self.strategy!r}, "
            f"n_tokens={self.n_tokens_replayed}, "
            f"wall_time_ms={self.wall_time_ms:.2f})"
        )


class CompareResult:
    """
    Result of a multi-strategy compare run.

    Attributes
    ----------
    rows : list[dict]
        One dict per strategy with keys: strategy, n_tokens, wall_time_ms,
        avg_latency_us, avg_quality, avg_bits_per_dim.
    """

    def __init__(self, rows: List[dict]):
        self.rows = rows

    def best_quality(self) -> Optional[dict]:
        """Return the row with the highest avg_quality."""
        valid = [r for r in self.rows if r.get("avg_quality", -1) >= 0]
        if not valid:
            return None
        return max(valid, key=lambda r: r["avg_quality"])

    def fastest(self) -> Optional[dict]:
        """Return the row with the lowest wall_time_ms."""
        if not self.rows:
            return None
        return min(self.rows, key=lambda r: r.get("wall_time_ms", float("inf")))

    def to_markdown(self) -> str:
        """Render the comparison as a Markdown table."""
        if not self.rows:
            return "*(no results)*"
        header = "| Strategy | Tokens | Wall (ms) | Latency µs | Quality | Bits/Dim |"
        sep    = "|---|---|---|---|---|---|"
        lines  = [header, sep]
        for r in self.rows:
            lines.append(
                f"| `{r.get('strategy', '?')}` "
                f"| {r.get('n_tokens', 0)} "
                f"| {r.get('wall_time_ms', 0):.2f} "
                f"| {r.get('avg_latency_us', -1):.2f} "
                f"| {r.get('avg_quality', -1):.4f} "
                f"| {r.get('avg_bits_per_dim', -1):.2f} |"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CompareResult({len(self.rows)} strategies)"


# ---------------------------------------------------------------------------
# ReplayEngine — Python interface
# ---------------------------------------------------------------------------

class ReplayEngine:
    """
    Python interface to the AdapTQ V2 ReplayEngine.

    Uses the ``adaptq replay`` and ``adaptq compare`` CLI subcommands
    internally. Snapshot files stay on disk; only the structured JSON result
    is captured through Python when no output_path is requested.

    Example
    -------
    >>> engine = ReplayEngine()
    >>> result = engine.replay("session.aqss", collect_metrics=True)
    >>> print(result)
    ReplayResult(strategy='har_fixed', n_tokens=128, wall_time_ms=3.14)

    >>> compare = engine.compare("session.aqss",
    ...                          strategies=["har_fixed", "fp_passthrough"])
    >>> print(compare.to_markdown())
    """

    def replay(
        self,
        snapshot_path: Union[str, Path],
        *,
        strategy: Optional[str] = None,
        from_token: Optional[int] = None,
        collect_metrics: bool = False,
        output_format: str = "json",
        output_path: Optional[Union[str, Path]] = None,
    ) -> ReplayResult:
        """
        Replay a session snapshot.

        Parameters
        ----------
        snapshot_path : str or Path
            Path to the ``.aqss`` snapshot file.
        strategy : str, optional
            Override strategy (``"har_fixed"`` or ``"fp_passthrough"``).
            If omitted, uses the strategy embedded in the snapshot config.
        from_token : int, optional
            Branch mode: warm-up to this token index then return.
        collect_metrics : bool
            Collect per-token ComputeMetrics (slower but richer output).
        output_format : str
            Output format for ``output_path``: ``"json"`` (default), ``"csv"``,
            ``"md"``, or ``"tex"``. Ignored when ``output_path`` is omitted,
            because the Python API returns a structured ``ReplayResult``.
        output_path : str or Path, optional
            Write the replay report to this file using ``output_format``.

        Returns
        -------
        ReplayResult
        """
        if from_token is not None and from_token < 0:
            raise ValueError("from_token must be non-negative or None")
        _validate_output_format(output_format)
        binary = _get_binary()
        cmd = [binary, "replay", str(snapshot_path)]
        if strategy:
            cmd += ["--strategy", strategy]
        if from_token is not None:
            cmd += ["--from-token", str(from_token)]
        if collect_metrics:
            cmd.append("--metrics")

        raw = _run_json_or_file(
            cmd,
            output_format=output_format,
            output_path=output_path,
            temp_suffix=".json",
            error_prefix="adaptq replay failed",
        )
        return ReplayResult(raw)

    def compare(
        self,
        snapshot_path: Union[str, Path],
        *,
        strategies: List[str],
        output_format: str = "json",
        output_path: Optional[Union[str, Path]] = None,
    ) -> CompareResult:
        """
        Compare multiple strategies on the same session snapshot.

        Parameters
        ----------
        snapshot_path : str or Path
            Path to the ``.aqss`` snapshot file.
        strategies : list[str]
            Strategy names to compare (e.g. ``["har_fixed", "fp_passthrough"]``).
        output_format : str
            Output format for ``output_path``: ``"json"``, ``"csv"``, ``"md"``,
            or ``"tex"``. Ignored when ``output_path`` is omitted because the
            Python API returns a structured ``CompareResult``.
        output_path : str or Path, optional
            Write comparison output to this file using ``output_format``.

        Returns
        -------
        CompareResult
        """
        if not strategies:
            raise ValueError("strategies list must be non-empty")
        _validate_output_format(output_format)

        binary = _get_binary()
        strats_csv = ",".join(strategies)
        cmd = [
            binary, "compare", str(snapshot_path),
            "--strategies", strats_csv,
        ]

        rows = _run_json_or_file(
            cmd,
            output_format=output_format,
            output_path=output_path,
            temp_suffix=".json",
            error_prefix="adaptq compare failed",
        )
        if not isinstance(rows, list):
            raise RuntimeError("adaptq compare failed: expected a JSON array")
        return CompareResult(rows)


# ---------------------------------------------------------------------------
# Snapshot utilities
# ---------------------------------------------------------------------------

def snapshot_info(path: Union[str, Path]) -> dict:
    """
    Return metadata from the header of a ``.aqss`` snapshot file.

    Only the fixed-size snapshot header is read; no replay is performed and
    no per-head storage or token log data is loaded.

    Returns
    -------
    dict with keys: n_tokens, n_layers, n_heads, dim, bits, version,
    file_size_bytes, has_token_log, has_strategy_state.
    """
    snapshot_path = Path(path)
    try:
        with snapshot_path.open("rb") as f:
            header = f.read(_SNAPSHOT_HEADER.size)
    except OSError as exc:
        raise RuntimeError(
            f"snapshot_info: cannot read {snapshot_path}: {exc}"
        ) from exc

    if len(header) != _SNAPSHOT_HEADER.size:
        raise RuntimeError(
            "snapshot_info: truncated snapshot header "
            f"(expected {_SNAPSHOT_HEADER.size} bytes, got {len(header)})"
        )

    (
        magic,
        version,
        n_layers,
        n_heads,
        dim,
        bits,
        n_tokens,
        n_heads_total,
        flags,
    ) = _SNAPSHOT_HEADER.unpack(header)

    if magic != _SNAPSHOT_MAGIC:
        raise RuntimeError("snapshot_info: invalid magic (not an AQSS file)")
    if version > _SNAPSHOT_VERSION:
        raise RuntimeError(
            f"snapshot_info: snapshot version {version} > current version "
            f"{_SNAPSHOT_VERSION}"
        )
    if any(value < 0 for value in (n_layers, n_heads, dim, n_tokens, n_heads_total)):
        raise RuntimeError("snapshot_info: invalid negative value in snapshot header")

    file_size = snapshot_path.stat().st_size

    return {
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dim": dim,
        "bits": bits,
        "version": version,
        "file_size_bytes": file_size,
        "has_token_log": bool(flags & 1),
        "has_strategy_state": bool(flags & 2),
    }


def snapshot_to_json(
    path: Union[str, Path],
    json_path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> str:
    """
    Export snapshot header metadata as a formatted JSON string.

    Optionally writes the JSON content to ``json_path`` if specified.

    Parameters
    ----------
    path : str or Path
        Path to the ``.aqss`` binary snapshot file.
    json_path : str or Path, optional
        Optional path where the formatted JSON string will be saved.
    indent : int, default 2
        JSON indentation level for formatting.

    Returns
    -------
    str
        Formatted JSON string containing snapshot metadata.
    """
    info = snapshot_info(path)
    json_str = json.dumps(info, indent=indent)
    if json_path is not None:
        out_path = Path(json_path)
        out_path.write_text(json_str, encoding="utf-8")
    return json_str

