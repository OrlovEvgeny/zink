# Zink Architecture

Zink is a ternary neural network inference engine for microcontrollers, written in Zig. It runs BitNet b1.58 models on Cortex-M55 and RISC-V Vector hardware without an operating system, dynamic memory allocation, or floating-point at runtime (except for weight requantization scales).

## Layer Model

Zink is organized into five layers, each depending only on layers below it.

```
  API         main.zig            Public entry point (InferenceEngine)
    |
  Model       model/              TransformerLayer, KV-cache, .zink loader
    |
  Ops         ops/                RMSNorm, softmax, RoPE, SiLU
    |
  Kernels     kernels/            Ternary matmul primitives (I2_S, TL1, FATNN)
    |
  Core        core/               Arena, ping-pong buffers, tensor descriptor
```

**Kernel layer** (`kernels/`): Raw ternary dot products and matrix-vector multiplications. No model knowledge. Includes a scalar reference kernel for correctness validation, plus optimized variants for generic `@Vector`, ARM Helium MVE, and RISC-V Vector.

**Ops layer** (`ops/`): Fixed-point implementations of RMSNorm, softmax (comptime exp LUT), RoPE (comptime sin/cos + frequency tables), and SiLU/SwiGLU. Uses kernels for any matmul work. No model knowledge.

**Model layer** (`model/`): `TransformerLayer` wires ops and kernels into the standard pre-norm transformer pipeline. `KvCache` implements a Q4-quantized ring buffer. `zink_loader` parses the `.zink` binary format. `config.zig` defines `ModelConfig` and `ModelMemoryLayout`.

**Core layer** (`core/`): `StaticArena` (bump allocator over a fixed buffer, 32-byte aligned), `PingPongBuffers` (two alternating buffers for pipeline processing), and `Tensor` (zero-allocation tensor descriptor).

**API layer** (`main.zig`): `InferenceEngine(cfg, target)` is a comptime-specialized struct that owns all runtime state. Exposes `init()`, `step()`, `reset()`, and `argmaxLogits()`.

## Memory Model

Memory is the binding constraint on MCU inference, not compute. Every byte of SRAM is accounted for at compile time.

**Static allocation only.** No `malloc`/`free` at runtime. All buffers are either embedded in the `InferenceEngine` struct (whose size is a comptime constant) or statically allocated via `StaticArena`.

**Ping-pong activation buffers.** Two buffers of size `hidden_size` bytes. One is read while the other is written. After each transformer layer, they swap roles. This means activation memory is O(hidden_size), not O(num_layers * hidden_size).

**Q4 KV-cache.** Key and value vectors are quantized to 4-bit (Q4) with per-group i16 scales (group_size=64). Stored in a ring buffer with capacity `max_seq_len`. O(1) insert at the write position, O(1) eviction when full. Ref: "Agent Memory Below the Prompt", Shkolnikov, Feb 2026 (0.7-3.0% PPL impact at 4x memory savings).

**Scratch buffers.** `ScratchBuffers(cfg)` holds all intermediate computation buffers: norm output, Q/K/V/O projection outputs, FFN gate/up intermediates, attention scores/probs, i32 accumulators, and per-head key/value extraction buffers. All sizes derived from `ModelConfig` at comptime.

**Weight streaming.** Weights reside in external flash (QSPI). The `streaming.zig` module provides double-buffered DMA transfer. Only one layer's weights need to be in SRAM at a time.

**Compile-time memory fitting.** `ModelConfig.assertFitsIn(sram_bytes)` produces a `@compileError` if the model's total SRAM requirement exceeds the target's capacity, with an exact byte count in the error message.

## Comptime Specialization

Zig's comptime execution is Zink's core advantage over C inference engines.

- **Model dimensions** (`hidden_size`, `num_heads`, `head_dim`, etc.) are comptime parameters. No runtime branching on model shape.
- **Kernel selection** is resolved at comptime via `dispatch.zig`. The `InferenceEngine` is monomorphized for a specific `KernelType` (I2_S generic, I2_S Helium, TL1, etc.). No vtable, no dynamic dispatch.
- **RoPE frequency tables**, exp/sqrt LUTs, and sin/cos tables are computed at compile time and embedded as constants.
- **Buffer sizes** are comptime constants derived from `ModelConfig`. `@compileError` enforces alignment constraints (e.g., hidden_size must be a multiple of 128).
- **Loop unrolling**: inner loops over `head_dim` use `inline for` when the dimension is comptime-known.

## Target Configurations

Each target is defined in `targets/*.zig` as a `TargetConfig` struct:

| Target | MCU | SRAM | Kernel |
|--------|-----|------|--------|
| `stm32n6` | STM32N6570-DK (Cortex-M55 @ 800 MHz) | 4.2 MB | I2_S Helium / TL1 |
| `rp2350` | RP2350 | 520 KB | I2_S generic |
| `generic_cortex_m55` | Generic Cortex-M55 | 2 MB | I2_S Helium |

## Inference Pipeline

For each token, `InferenceEngine.step(token_id)` executes:

1. **Embedding lookup** → copy into ping-pong write buffer, swap
2. For each transformer layer:
   a. **Pre-attention RMSNorm**
   b. **Q/K/V projections** (ternary matmul → i32 accumulator → requantize to i8)
   c. **RoPE** on Q and K per-head
   d. **KV-cache append** (Q4 quantize and store K, V)
   e. **Multi-head attention** (dot product scores → scale → causal mask → softmax → weighted value sum)
   f. **O projection** (ternary matmul)
   g. **Residual add**
   h. **Pre-FFN RMSNorm**
   i. **FFN: SwiGLU** (gate and up projections → silu(gate) * up → down projection)
   j. **Residual add**
   k. Swap ping-pong buffers
3. **Final RMSNorm**
4. **LM head projection** → raw i32 logits over vocabulary
