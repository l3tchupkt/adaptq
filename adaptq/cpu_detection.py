"""
adaptq.cpu_detection
====================
Runtime CPU feature detection utility for AdapTQ.

Detects AVX2 and FMA CPU instruction capabilities across Linux, macOS,
and Windows, providing diagnostics and status reporting for backend selection.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Dict, Union


def detect_cpu_features() -> Dict[str, Union[str, bool]]:
    """
    Detects host architecture and AVX2/FMA instruction support.
    Returns a dictionary of features and status.
    """
    arch = platform.machine().lower()
    is_x86 = arch in ("x86_64", "amd64", "x86", "i386", "i686")

    has_avx2 = False
    has_fma = False

    # Check environment variable overrides
    if os.environ.get("ADAPTQ_FORCE_AVX2", "0") in ("1", "true", "TRUE"):
        has_avx2 = True
        has_fma = True
    elif os.environ.get("ADAPTQ_DISABLE_AVX2", "0") in ("1", "true", "TRUE") or \
         os.environ.get("ADAPTQ_FORCE_SCALAR", "0") in ("1", "true", "TRUE"):
        has_avx2 = False
        has_fma = False
    elif is_x86:
        system = platform.system()
        if system == "Linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    flags_line = next((line for line in content.splitlines() if line.startswith("flags")), "")
                    flags = set(flags_line.split())
                    has_avx2 = "avx2" in flags
                    has_fma = "fma" in flags
            except Exception:
                pass
        elif system == "Darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"], text=True)
                flags = set(out.lower().split())
                has_avx2 = "avx2" in flags
                has_fma = "fma" in flags
            except Exception:
                pass
        elif system == "Windows":
            try:
                # Use ctypes to invoke __cpuid or kernel IsProcessorFeaturePresent
                import ctypes
                # Check AVX2 (PF_AVX2_INSTRUCTIONS_AVAILABLE = 40)
                kernel32 = ctypes.windll.kernel32
                if hasattr(kernel32, "IsProcessorFeaturePresent"):
                    has_avx2 = bool(kernel32.IsProcessorFeaturePresent(40))
                    # FMA is available on all modern AVX2 x86_64 processors
                    has_fma = has_avx2
            except Exception:
                has_avx2 = True  # Standard fallback on modern 64-bit Windows
                has_fma = True

    status = "AVX2 backend available" if (has_avx2 and has_fma) else "AVX2 backend unavailable"
    reason = None
    if not (has_avx2 and has_fma):
        if not is_x86:
            reason = f"Non-x86 architecture ({arch}) defaults to scalar backend"
        elif not has_avx2 and not has_fma:
            reason = "CPU does not support AVX2 or FMA instructions"
        elif not has_avx2:
            reason = "CPU does not support AVX2 instructions"
        else:
            reason = "CPU does not support FMA instructions"

    report = {
        "architecture": arch,
        "is_x86": is_x86,
        "avx2": has_avx2,
        "fma": has_fma,
        "status": status,
    }
    if reason:
        report["reason"] = reason

    return report


def get_cpu_report() -> str:
    """Formats CPU feature report into a diagnostic text summary."""
    features = detect_cpu_features()
    lines = [
        "AdapTQ CPU capability check",
        "",
        f"Architecture: {features['architecture']}",
        f"AVX2: {'supported' if features['avx2'] else 'unavailable'}",
        f"FMA: {'supported' if features['fma'] else 'unavailable'}",
        f"Status: {features['status']}",
    ]
    if "reason" in features:
        lines.append(f"Reason: {features['reason']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_cpu_report())
