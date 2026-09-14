# GitHub Security Advisory Draft

**Title:** Unvalidated Snapshot Metadata and Quantization Bit-Widths Leading to OOM and Stack Buffer Overflow  
**Affected versions:** AdapTQ <= 0.2.1  
**Fixed version:** AdapTQ 0.2.2  
**CVE ID:** No CVE assigned.
**CWE:** CWE-121 (Stack-based Buffer Overflow), CWE-128 (Wrap-around Error)
**CVSS Score:** 5.5 (Medium) - CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:N/A:H

## Impact
A locally crafted or maliciously modified `SessionSnapshot` (`.aqss` file) could specify out-of-bounds constraints for `dim`, `n_heads`, `bits`, or `n_tokens`. During the loading phase in `SessionSnapshot::load()`, these parameters were utilized to allocate `std::vector` structures prior to boundary validation. Supplying negative capacities or extreme numerical values resulted in uncontrolled allocation failure (`std::bad_alloc`) or potential buffer over-reads. 

Additionally, providing quantization bit-widths greater than 4 via the public API bypassed internal validation, triggering a stack-based buffer overflow (`STATUS_STACK_BUFFER_OVERRUN`) during the scalar fallback execution path.

## Root Cause
1. **Snapshot Loading:** The binary decoding logic inside `session_snapshot.cpp` trusted the integers pulled from the file header, invoking `resize()` on the storage buffers using these parsed sizes without verifying if they fell within structurally safe or mathematically positive limits.
2. **Buffer Overrun:** The internal scalar kernel assumed `bits` would never exceed 4, hardcoding a stack buffer of size 16 (`float ecb[16]`). The public API previously did not enforce this limit, allowing `bits >= 5` to write out of bounds.

## Attack Prerequisites
An attacker must be able to supply a manipulated `.aqss` session file to the host system and trigger the `ReplayEngine` or `SessionSnapshot` loader, or invoke the C API directly with invalid `bits` configurations.

## Mitigation
Upgrade to `AdapTQ >= 0.2.2`. 
- The `SessionSnapshot::load()` code now implements strict bounds validation ensuring dimensions are strictly positive and capacity ranges are validated prior to any memory allocation.
- The C API (`adaptq_create`, `adaptq_mha_create`) now enforces that `bits` strictly equals 2, 3, or 4.

## Credits
- **Reporters:** @ThunderKhan, @badshahaditya57-sketch
- **Remediation Creator:** @l3tchupkt (letchu)
