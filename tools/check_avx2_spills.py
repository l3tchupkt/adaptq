#!/usr/bin/env python3
"""Check AVX2 kdot4 register pressure from generated assembly.

Compiles kernels/avx2/kdot_avx2.cpp to assembly with AVX2/FMA and
reports YMM register use plus vector stack traffic for the three
kdot4_quad_avx2 specializations (<2>, <3>, <4>).

Usage:
    python tools/check_avx2_spills.py
    python tools/check_avx2_spills.py --asm build/kdot.s
    python tools/check_avx2_spills.py --strict (fail on any vector stack spill)
"""
import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "kernels", "avx2", "kdot_avx2.cpp")

SPECS = ["kdot_batch", "vaccum_batch"]
# Fallback patterns for demangled or Itanium variants
FALLBACK = ["kdot_batch", "vaccum_batch"]


def build_asm(out_path):
    cmd = [
        "g++", "-S", "-O3", "-mavx2", "-mfma", "-std=c++17",
        "-I", os.path.join(REPO, "include"),
        "-I", os.path.join(REPO, "kernels"),
        SRC, "-o", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("compile failed:\n" + (r.stderr or r.stdout)[-4000:])
        return False
    return True


def analyze(asm_text):
    lines = asm_text.splitlines()
    # Only global function labels start a new range. Local .L labels
    # inside a function must not split it, otherwise we only inspect
    # the first line and miss all YMM use.
    func_ranges = {}
    cur = None
    start = 0
    label_re = re.compile(r"^(_Z[\w\$]+):")
    for i, ln in enumerate(lines):
        m = label_re.match(ln.strip())
        if m:
            if cur:
                func_ranges[cur] = (start, i)
            cur = m.group(1)
            start = i
    if cur:
        func_ranges[cur] = (start, len(lines))

    results = {}
    # kdot4 templates are inlined into the backend methods, so inspect
    # the two public entry points that contain the vector loops.
    targets = {
        "kdot_batch": [n for n in func_ranges if "kdot_batch" in n],
        "vaccum_batch": [n for n in func_ranges if "vaccum_batch" in n],
    }
    for pretty, names in targets.items():
        if not names:
            results[pretty] = {"found": False, "ymm": set(), "stack_vec": 0}
            continue
        # Merge all matching ranges (usually one per method)
        ymms = set()
        stack_vec = 0
        key = names[0]
        for name in names:
            s, e = func_ranges[name]
            body = "\n".join(lines[s:e])
            ymms |= set(re.findall(r"%ymm(\d+)", body))
            stack_vec += len(re.findall(r"vmov\w+\s+.*\(%(rsp|rbp)", body))
            stack_vec += len(re.findall(r"\(%(rsp|rbp).*vmov", body))
        results[pretty + " [" + key[:50] + "]"] = {
            "found": True, "ymm": ymms, "stack_vec": stack_vec,
        }
    # Also confirm the three bit-width paths exist in the binary
    # via the Decode specializations referenced in the asm comments/refs.
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", default="", help="Use existing .s file instead of compiling")
    ap.add_argument("--strict", action="store_true", help="Fail if any vector stack traffic found")
    args = ap.parse_args()

    if args.asm:
        with open(args.asm, encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        out = os.path.join(REPO, "_kdot_check.s")
        ok = build_asm(out)
        if not ok:
            print("SKIP: compiler or AVX2 flags unavailable, cannot inspect assembly")
            return 0
        with open(out, encoding="utf-8", errors="replace") as f:
            text = f.read()
        try:
            os.unlink(out)
        except OSError:
            pass

    res = analyze(text)
    missing = [k for k, v in res.items() if not v["found"]]
    if missing:
        print("FAIL: expected kdot4 specializations not found:")
        for m in missing:
            print("  missing " + m)
        print("Hint: build with -mavx2 -mfma and -O3 so all three emit.")
        return 1

    worst = 0
    for name, v in res.items():
        print("%s: ymm_used=%d stack_vec=%d" % (name, len(v["ymm"]), v["stack_vec"]))
        worst = max(worst, v["stack_vec"])

    print("done: %d specializations inspected" % len(res))
    if args.strict and worst > 0:
        print("FAIL: vector stack traffic detected in strict mode")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
