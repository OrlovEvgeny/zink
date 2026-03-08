[![Zig](https://img.shields.io/badge/Zig-0.15%2B-f7a41d?logo=zig)](https://ziglang.org)
[![Release](https://img.shields.io/github/v/release/OrlovEvgeny/zink)](https://github.com/OrlovEvgeny/zink/releases)
[![Tests](https://github.com/OrlovEvgeny/zink/actions/workflows/test.yml/badge.svg)](https://github.com/OrlovEvgeny/zink/actions/workflows/test.yml)

# Zink

Ternary neural network inference engine for microcontrollers. Runs BitNet b1.58 models ({-1, 0, 1} weights) on Cortex-M55, RISC-V and other embedded targets. Zero runtime allocation, comptime-specialized kernels, Q4 KV-cache. Written in Zig.

## Install

```sh
zig fetch --save git+https://github.com/OrlovEvgeny/zink.git
```

```zig
// build.zig
const zink_dep = b.dependency("zink", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("zink", zink_dep.module("zink"));
```

## Running a Model

### 1. Convert weights to .zink

```sh
pip install numpy safetensors

# From safetensors (auto-infers config from tensor shapes)
python3 tools/zink_pack.py model.safetensors -o model.zink

# With explicit config
python3 tools/zink_pack.py model.safetensors -o model.zink --config bitnet_0_7b

# Generate a small test model for development
python3 tools/zink_pack.py --test
```

### 2. Define model config

```zig
const zink = @import("zink");

const my_model = zink.model.ModelConfig{
    .hidden_size = 1536,
    .num_layers = 24,
    .num_heads = 24,
    .num_kv_heads = 24,    // < num_heads for GQA
    .head_dim = 64,
    .intermediate_size = 4096,
    .vocab_size = 32000,
    .max_seq_len = 512,
    // .sliding_window_size = 256,  // optional, limits attention scope
};

// Compile error if model config is invalid (alignment, head dims, etc.)
comptime { my_model.validate(); }

// Compile error with exact byte counts if model exceeds target SRAM
comptime { my_model.assertFitsIn(4 * 1024 * 1024); }
```

### 3. Create the inference engine

```zig
const target = zink.engine.TargetConfig{
    .name = "stm32n6",
    .sram_bytes = 4 * 1024 * 1024 + 200 * 1024,
    .kernel_type = .i2s_helium,  // .i2s_generic, .tl1, .i2s_rvv, .fatnn
    .dma_chunk_size = 4096,
    .dma_alignment = 16,
};

const Engine = zink.engine.InferenceEngine(my_model, target);
```

### 4. Load weights and run inference

```zig
// Load .zink file (zero-copy from flash or embedded)
const bytes = @embedFile("model.zink");
const zink_file = try zink.model.ZinkFile.parse(bytes);

// Initialize engine (sets weight pointers into .zink buffer)
var engine = try Engine.init(&zink_file);

// Autoregressive generation loop
var token: usize = start_token;
for (0..max_tokens) |_| {
    const logits = engine.step(token);
    token = Engine.argmaxLogits(&logits);
    // emit token...
}

// Reset for a new sequence
engine.reset();
```

## Cross-Compilation

```sh
# ARM Cortex-M55 (Helium MVE)
zig build -Dtarget=thumb-freestanding -Dcpu=cortex_m55 -Doptimize=ReleaseFast

# RISC-V 32-bit with Vector extension
zig build -Dtarget=riscv32-freestanding -Dcpu=generic_rv32+v -Doptimize=ReleaseFast
```

## Memory Planning

Check if a model fits on your target before flashing:

```sh
python3 tools/memory_planner.py --model bitnet_0_7b --target stm32n6

python3 tools/memory_planner.py --hidden 512 --layers 8 --heads 8 \
    --kv-heads 4 --head-dim 64 --intermediate 1408 --vocab 32000 --max-seq 256 \
    --target stm32n6
```

## Tests

```sh
zig build test --summary all
```

## Docs

- [Architecture](docs/ARCHITECTURE.md) — layer model, memory model, comptime specialization
- [.zink Format](docs/ZINK_FORMAT.md) — binary format spec, tensor naming, Python tool
- [Kernel Selection](docs/KERNEL_SELECTION.md) — I2_S, TL1, FATNN, when to use which

## License

[MIT](LICENSE)
