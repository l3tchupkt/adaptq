"""
CPU architecture and capability validation for AdapTQ.
Ensures early, clear error reporting on unsupported platforms.
"""
from __future__ import annotations

import platform
import warnings


def check_cpu_support() -> None:
    """
    Check if the host CPU is supported by AdapTQ.
    AdapTQ requires an x86_64 architecture with AVX2 and FMA support for high performance.
    Raises RuntimeError on unsupported CPU architectures (e.g., ARM / aarch64).
    """
    machine = platform.machine().lower()

    # Supported x86_64 architecture aliases
    if machine in ("x86_64", "amd64", "x64"):
        return

    # Clear diagnostics for ARM platforms
    if "arm" in machine or "aarch" in machine:
        raise RuntimeError(
            f"AdapTQ requires an x86_64 CPU with AVX2 and FMA support. "
            f"Detected: {machine}. ARM NEON support is not available yet."
        )

    # General unsupported architecture fallback
    raise RuntimeError(
        f"AdapTQ requires an x86_64 CPU with AVX2 and FMA support. "
        f"Detected: {machine}."
    )
