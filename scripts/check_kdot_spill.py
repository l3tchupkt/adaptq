#!/usr/bin/env python3
"""
scripts/check_kdot_spill.py — Vector Register Spill Detector for 4-way AVX2 K-dot
================================================================================
Inspects compiled assembly of `kernels/avx2/kdot_avx2.cpp` for:
  - kdot4_quad_avx2<2>
  - kdot4_quad_avx2<3>
  - kdot4_quad_avx2<4>

Verifies that:
  1. All three specializations (2, 3, 4 bits) are present in the assembly.
     (Fails if any expected specialization is missing to prevent false passes).
  2. Measures active YMM register utilization per specialization.
  3. Detects vector stack accesses (vmovups/vmovaps/vmovdqu to/from stack frames).
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

SPECIALIZATIONS = [2, 3, 4]

# Patterns for GCC / Clang / MSVC mangled or symbol names for kdot4_quad_avx2
FUNC_PATTERNS = {
    2: re.compile(r'(?:_Z16kdot4_quad_avx2ILi2EE|kdot4_quad_avx2<2>)'),
    3: re.compile(r'(?:_Z16kdot4_quad_avx2ILi3EE|kdot4_quad_avx2<3>)'),
    4: re.compile(r'(?:_Z16kdot4_quad_avx2ILi4EE|kdot4_quad_avx2<4>)'),
}

YMM_REGEX = re.compile(r'%?ymm([0-9]|1[0-5])\b', re.IGNORECASE)
VECTOR_SPILL_REGEX = re.compile(
    r'\b(?:vmovaps|vmovups|vmovdqa|vmovdqu|vmovdqa32|vmovdqu32)\s+.*'
    r'(?:[0-9]*\([%a-z0-9]+\)|\[[%a-z0-9\s\+\-\*]+\])',
    re.IGNORECASE
)
STACK_REG_REGEX = re.compile(r'(?:%?rsp|%?rbp|%?esp|%?ebp)', re.IGNORECASE)


def compile_to_asm(source_file, compiler="g++", extra_flags=None):
    """Compiles source_file to assembly using compiler with -O3 -mavx2 -mfma."""
    if extra_flags is None:
        extra_flags = []
    
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    include_dir = os.path.join(repo_dir, "include")
    kernels_dir = os.path.join(repo_dir, "kernels")
    
    tmp_asm = tempfile.NamedTemporaryFile(suffix=".s", delete=False)
    tmp_asm.close()

    cmd = [
        compiler,
        "-S",
        "-O3",
        "-mavx2",
        "-mfma",
        "-std=c++17",
        f"-I{include_dir}",
        f"-I{kernels_dir}",
        "-o", tmp_asm.name,
        source_file
    ] + extra_flags

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if os.path.exists(tmp_asm.name):
            os.remove(tmp_asm.name)
        raise RuntimeError(f"Failed to compile {source_file} to assembly with {compiler}: {e}")

    with open(tmp_asm.name, "r", encoding="utf-8", errors="replace") as f:
        asm_content = f.read()

    os.remove(tmp_asm.name)
    return asm_content


def extract_function_blocks(asm_text):
    """Parses assembly text and partitions into functions."""
    lines = asm_text.splitlines()
    functions = {}
    current_func = None
    current_lines = []

    label_regex = re.compile(r'^([a-zA-Z0-9_$.]+):')
    globl_regex = re.compile(r'^\s*\.(?:globl|global)\s+([a-zA-Z0-9_$.]+)')

    for line in lines:
        match_lbl = label_regex.match(line.strip())
        if match_lbl:
            label = match_lbl.group(1)
            # Check if this label matches any of our target functions
            matched_bits = None
            for bits, pat in FUNC_PATTERNS.items():
                if pat.search(label):
                    matched_bits = bits
                    break
            
            if matched_bits is not None:
                if current_func is not None:
                    functions[current_func] = current_lines
                current_func = matched_bits
                current_lines = [line]
                continue

        if current_func is not None:
            # Check for end of function directives (.cfi_endproc, .size, or next global label)
            if line.strip().startswith('.cfi_endproc') or line.strip().startswith('.size'):
                current_lines.append(line)
                functions[current_func] = current_lines
                current_func = None
                current_lines = []
            else:
                current_lines.append(line)

    if current_func is not None and current_lines:
        functions[current_func] = current_lines

    return functions


def analyze_specialization(bits, lines):
    """Analyzes a single specialization function for YMM count and vector spills."""
    ymm_used = set()
    spills = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('.') or stripped.startswith('#'):
            continue

        for m in YMM_REGEX.finditer(stripped):
            ymm_used.add(int(m.group(1)))

        if VECTOR_SPILL_REGEX.search(stripped) and STACK_REG_REGEX.search(stripped):
            spills.append(stripped)

    return {
        "bits": bits,
        "ymm_count": len(ymm_used),
        "ymm_registers": sorted(list(ymm_used)),
        "spill_count": len(spills),
        "spills": spills
    }


def main():
    parser = argparse.ArgumentParser(description="Check 4-way AVX2 K-dot for vector register spills")
    parser.add_argument("--asm", type=str, default=None, help="Path to pre-generated assembly file")
    parser.add_argument("--compiler", type=str, default="g++", help="C++ compiler to generate assembly")
    parser.add_argument("--max-spills", type=int, default=0, help="Maximum allowed vector stack spills (default: 0)")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kdot_src = os.path.join(repo_dir, "kernels", "avx2", "kdot_avx2.cpp")

    if args.asm:
        with open(args.asm, "r", encoding="utf-8", errors="replace") as f:
            asm_content = f.read()
    else:
        if not os.path.exists(kdot_src):
            print(f"Error: Could not find {kdot_src}")
            return 1
        try:
            asm_content = compile_to_asm(kdot_src, compiler=args.compiler)
        except Exception as e:
            print(f"Compilation error: {e}")
            print("Note: To analyze precompiled assembly, pass --asm <file>.")
            return 2

    func_blocks = extract_function_blocks(asm_content)

    print("=" * 65)
    print("  4-way AVX2 K-dot Vector Register Spill Checker")
    print("=" * 65)

    all_passed = True
    missing_specs = []

    for bits in SPECIALIZATIONS:
        if bits not in func_blocks:
            missing_specs.append(bits)
            print(f"  [ FAIL ] kdot4_quad_avx2<{bits}>: NOT FOUND in emitted assembly!")
            all_passed = False
            continue

        result = analyze_specialization(bits, func_blocks[bits])
        ymm_str = f"YMM regs used: {result['ymm_count']}/16 ({result['ymm_registers']})"
        spill_status = f"Vector stack spills: {result['spill_count']}"

        passed = (result['spill_count'] <= args.max_spills)
        status_tag = "[ PASS ]" if passed else "[ FAIL ]"
        print(f"  {status_tag} kdot4_quad_avx2<{bits}>: {ymm_str} | {spill_status}")

        if not passed:
            all_passed = False
            for s in result['spills']:
                print(f"           Spill instruction: {s}")

    print("=" * 65)
    if missing_specs:
        print(f"FAILED: Expected specializations missing: {missing_specs}")
        return 1

    if not all_passed:
        print("FAILED: Vector register spills detected exceeding threshold.")
        return 1

    print("SUCCESS: All 3 specializations emitted and within spill budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
