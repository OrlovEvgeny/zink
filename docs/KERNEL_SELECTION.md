# Kernel Selection Guide

Zink supports multiple ternary matrix-vector multiplication kernels, each optimized for different hardware targets. Kernel selection is resolved at compile time via `kernels/dispatch.zig` — there is no runtime branching on kernel type.

## Available Kernels

### Reference (scalar)

**File:** `kernels/reference.zig`

Pure scalar implementation. Unpacks each 2-bit ternary weight and multiplies against the i8 activation. No SIMD, no lookup tables.

- **Use case:** Correctness oracle. Every optimized kernel must produce identical output to the reference for all inputs.
- **Constraints:** None (works on any target).
- **Performance:** Baseline. Not intended for production.

### I2_S Generic

**File:** `kernels/i2s_generic.zig`

Portable SIMD implementation using Zig's `@Vector` type. Processes multiple elements per iteration via the compiler's auto-vectorization. Uses the I2_S (Integer 2-bit SIMD) approach from Bitnet.cpp: MAD-based (multiply-accumulate with decomposition), lossless at 2 bits per weight.

- **Use case:** Default kernel for targets without specialized SIMD intrinsics.
- **Constraints:** Input dimension must be a comptime multiple of 128.
- **Performance:** Significantly faster than reference on any target with vector hardware. Compiler maps `@Vector` to available SIMD (NEON, SSE, etc.).

Ref: Bitnet.cpp ACL 2025, Section 3.2.2 (I2_S design).

### I2_S Helium

**File:** `kernels/i2s_helium.zig`

ARM Helium MVE-specific implementation. Uses 128-bit MVE vectors (Q-registers), VLDRB/VSTRB for loads/stores, VMLADAV for multiply-accumulate-dual-across-vector, and VCTP for tail predication.

- **Use case:** Cortex-M55 targets (STM32N6570-DK, etc.).
- **Constraints:** Requires ARM Helium MVE. Dimension must be a multiple of 128.
- **Performance:** Exploits the M55's 2-beat-per-tick MVE pipeline. VMLADAV computes a full dot product reduction in a single instruction.

### I2_S RISC-V Vector

**File:** `kernels/i2s_rvv.zig`

RISC-V Vector 1.0 implementation. VLEN-agnostic using `vsetvli` for dynamic vector length configuration, `vle8`/`vse8` for loads/stores, and `vmacc`/`vredsum` for accumulate and reduction.

- **Use case:** RISC-V targets with Vector extension (e.g., C908 core).
- **Constraints:** Requires RVV 1.0. Dimension must be a multiple of 128.
- **Performance:** Scales with VLEN. Tested against muRISCV-NN benchmarks.

### TL1 (Ternary Lookup Table)

**File:** `kernels/tl1.zig`

Lookup-table-based kernel. Instead of multiplying each activation by a ternary weight, TL1 precomputes a 9-entry lookup table from pairs of activation values. Each table entry holds the sum for one of the 9 possible (w0, w1) combinations where w0, w1 in {-1, 0, 1}.

- **Use case:** When LUT construction cost is amortized across multiple projections sharing the same input vector. In a transformer layer, Q/K/V projections all use the same `norm_out` input, so one LUT serves three matmuls.
- **Constraints:** LUT memory overhead: `9 * (input_dim / 2) * sizeof(i16)` bytes. For hidden_size=1536, that's 13,824 bytes.
- **Performance:** Eliminates all multiplications in the matmul. Trades compute for memory. Faster than I2_S when the LUT is reused 3+ times.
- **Rebuild requirement:** The LUT must be rebuilt when the input vector changes. In `transformer.zig`, the LUT is rebuilt for the O projection (input is `attn_out`, not `norm_out`) and again for the FFN down projection (input is `ffn_gate` after SwiGLU).

Ref: Bitnet.cpp ACL 2025, Section 3.3 (TL1 design).

### FATNN (Binary Decomposition)

**File:** `kernels/fatnn.zig`

Decomposes each ternary weight into two binary vectors (alpha, beta) such that the original ternary value can be recovered. Replaces multiply-accumulate with bitwise AND + popcount operations, reducing computational complexity.

- **Use case:** Targets with fast popcount but no efficient multiply-accumulate (some embedded RISC cores).
- **Constraints:** Requires pre-decomposed weight storage. Doubles weight memory (two binary vectors instead of one ternary vector).
- **Performance:** O(2N) bitwise ops vs O(4N) for naive ternary. Competitive when popcount is hardware-accelerated.

Ref: FATNN (ICCV 2021).

## Comptime Dispatch

Kernel selection is a comptime parameter of `InferenceEngine`:

```zig
const Engine = InferenceEngine(model_config, .{
    .name = "stm32n6",
    .sram_bytes = 4 * 1024 * 1024 + 200 * 1024,
    .kernel_type = .tl1,  // or .i2s_generic, .i2s_helium, .i2s_rvv, .fatnn
    .dma_chunk_size = 4096,
    .dma_alignment = 16,
});
```

`dispatch.zig` provides:
- `ternaryMatVec()` — direct matmul using the selected kernel
- `ternaryMatVecDispatch()` — dispatches to the selected kernel with optional TL1 LUT
- `buildTL1Lut()` — constructs the TL1 lookup table from an input vector

All dispatch is resolved at comptime. The compiled binary contains only the code for the selected kernel.

## Selection Guidelines

| Target | Recommended Kernel | Reason |
|--------|--------------------|--------|
| Cortex-M55 (Helium) | `i2s_helium` or `tl1` | Helium MVE gives best I2_S throughput; TL1 wins when LUT reuse is high |
| Cortex-M55 (no Helium) | `i2s_generic` | Falls back to compiler auto-vectorization |
| RISC-V with RVV | `i2s_rvv` | Native vector instructions |
| RISC-V without RVV | `i2s_generic` or `fatnn` | Generic SIMD or popcount-based |
| Host testing | `i2s_generic` | Portable, maps to SSE/AVX on x86 |

## Testing Strategy

Every optimized kernel is tested against the scalar reference:

```zig
// In tests/test_kernels.zig
test "i2s_generic matches reference" {
    // Generate random packed weights and activations
    // Run both reference and i2s_generic kernels
    // Assert identical output for every element
}
```

The reference kernel is the correctness oracle. No optimized kernel is considered correct unless it matches the reference for all inputs, including edge cases (all zeros, all -1, alternating patterns).
