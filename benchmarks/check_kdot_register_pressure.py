#!/usr/bin/env python3
"""Check optimized AVX2 K-dot register usage for accidental vector spills."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kernels" / "avx2" / "kdot_avx2.cpp"
FUNCTION_RE = re.compile(r"\bkdot4_quad_avx2<([234])>")
YMM_RE = re.compile(r"\bymm([0-9]+)\b")
STACK_VECTOR_RE = re.compile(
    r"\bvmov[a-z0-9]*\b[^\n]*\(%(?:rsp|rbp)\)",
    re.IGNORECASE,
)


def find_compiler() -> str:
    configured = os.environ.get("CXX")
    if configured:
        return configured
    for candidate in ("c++", "g++", "clang++"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("No C++ compiler found; set CXX to g++ or clang++.")


def demangle(compiler: str, assembly: str) -> str:
    cxxfilt = shutil.which("c++filt")
    if cxxfilt is None:
        return assembly
    result = subprocess.run(
        [cxxfilt],
        input=assembly,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def extract_function(assembly: str, bits: str) -> Optional[str]:
    lines = assembly.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = FUNCTION_RE.search(line)
        if match and match.group(1) == bits:
            start = index
            break
    if start is None:
        return None

    for index in range(start + 1, len(lines)):
        if ".cfi_endproc" in lines[index] or lines[index].startswith(".size"):
            return "\n".join(lines[start : index + 1])
    return "\n".join(lines[start:])


def main() -> int:
    if not SOURCE.is_file():
        print(f"missing source: {SOURCE}", file=sys.stderr)
        return 2

    compiler = find_compiler()

    with tempfile.TemporaryDirectory(prefix="adaptq-kdot-regs-") as tmp:
        assembly_path = Path(tmp) / "kdot_avx2.s"
        command = [
            compiler,
            "-std=c++17",
            "-O3",
            "-mavx2",
            "-mfma",
            "-fno-inline",
            "-fno-omit-frame-pointer",
            "-S",
            str(SOURCE),
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "include" / "adaptq"),
            "-I",
            str(ROOT / "kernels"),
            "-o",
            str(assembly_path),
        ]

        try:
            subprocess.run(command, check=True)
        except FileNotFoundError:
            print(f"compiler not found: {compiler}", file=sys.stderr)
            return 2
        except subprocess.CalledProcessError as exc:
            print(f"compiler failed with exit code {exc.returncode}", file=sys.stderr)
            return 2

        assembly = assembly_path.read_text(encoding="utf-8")
        demangled = demangle(compiler, assembly)

        found = 0
        failed = False
        for bits in ("2", "3", "4"):
            body = extract_function(demangled, bits)
            if body is None:
                continue

            found += 1
            registers = sorted({int(match) for match in YMM_RE.findall(body)})
            spills = [
                line.strip()
                for line in body.splitlines()
                if STACK_VECTOR_RE.search(line)
            ]

            print(f"bits={bits} ymm_registers={len(registers)}")
            if registers:
                print(
                    f"bits={bits} ymm_names="
                    + ",".join(f"ymm{reg}" for reg in registers)
                )
            print(f"bits={bits} vector_stack_accesses={len(spills)}")
            if spills:
                failed = True
                for line in spills:
                    print(f"bits={bits} spill={line}")

        if found != 3:
            print(
                f"kdot4_quad_avx2 specializations emitted: {found}/3",
                file=sys.stderr,
            )
            return 2

        if failed:
            print(
                "register-pressure check failed: vector values spill to the stack",
                file=sys.stderr,
            )
            return 1

        print("register-pressure check passed: no vector register spills detected")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
