# .zink Binary Format Specification

Version: 1

The `.zink` format stores quantized model weights for zero-copy loading on microcontrollers. All data is little-endian. Tensor payloads are SIMD-aligned for direct use without unpacking.

## File Layout

```
[0]               96-byte header
[96]              Tensor table (32 bytes per entry)
[aligned]         Tensor data (16-byte aligned payloads)
```

## Header (96 bytes)

| Offset | Size | Type   | Field                  | Description |
|--------|------|--------|------------------------|-------------|
| 0      | 4    | u32    | magic                  | `0x5A494E4B` ("ZINK") |
| 4      | 2    | u16    | version                | Format version (currently 1) |
| 6      | 2    | u16    | flags                  | Reserved, must be 0 |
| 8      | 4    | u32    | hidden_size            | Model hidden dimension |
| 12     | 2    | u16    | num_layers             | Number of transformer layers |
| 14     | 2    | u16    | num_heads              | Number of attention heads |
| 16     | 2    | u16    | num_kv_heads           | Number of KV heads (< num_heads for GQA) |
| 18     | 2    | u16    | head_dim               | Dimension per head |
| 20     | 4    | u32    | intermediate_size      | FFN intermediate dimension |
| 24     | 4    | u32    | vocab_size             | Vocabulary size |
| 28     | 4    | u32    | max_seq_len            | Maximum sequence length |
| 32     | 4    | u32    | rope_theta_fixed       | RoPE theta in Q16.16 fixed-point |
| 36     | 2    | u16    | kv_group_size          | Q4 KV-cache quantization group size |
| 38     | 1    | u8     | quant_type             | Default quantization type |
| 39     | 1    | -      | (padding)              | Alignment padding |
| 40     | 4    | u32    | num_tensors            | Number of tensor table entries |
| 44     | 4    | u32    | tensor_table_offset    | Byte offset of tensor table |
| 48     | 4    | u32    | data_offset            | Byte offset of tensor data section |
| 52     | 4    | -      | (padding)              | Alignment padding for u64 |
| 56     | 8    | u64    | total_size             | Total file size in bytes |
| 64     | 32   | -      | reserved               | Reserved, must be zero |

## Tensor Table Entry (32 bytes)

Each tensor is described by a 32-byte entry in the tensor table.

| Offset | Size | Type   | Field              | Description |
|--------|------|--------|--------------------|-------------|
| 0      | 8    | u64    | name_hash          | FNV-1a 64-bit hash of the tensor name |
| 8      | 4    | u32    | offset             | Byte offset from data section start |
| 12     | 4    | u32    | rows               | Number of rows |
| 16     | 4    | u32    | cols               | Number of columns |
| 20     | 4    | u32    | packed_size_bytes  | Size of packed data in bytes |
| 24     | 1    | u8     | quant_type         | Quantization type enum |
| 25     | 7    | -      | reserved           | Padding, must be zero |

Tensor data is located at `data_offset + entry.offset` in the file.

## Quantization Types

| Value | Name         | Description |
|-------|--------------|-------------|
| 0     | `i2_s`       | I2_S ternary packing: 4 values per byte, 2 bits each, LSB first. Two's complement: -1=0b11, 0=0b00, 1=0b01. |
| 1     | `tl1_packed` | TL1-specific packing (reserved) |
| 2     | `raw_int8`   | Raw int8 values, one byte per element |
| 3     | `f16`        | IEEE 754 half-precision float (reserved) |
| 4     | `q4`         | 4-bit quantized with per-group scales |

## Tensor Naming Convention

Tensors are identified by FNV-1a 64-bit hash of their string name.

**Global tensors:**
- `embedding.weight` — Token embedding table (raw_int8, vocab_size x hidden_size)
- `final_rmsnorm.weight` — Final layer norm weights (raw_int8, hidden_size)
- `lm_head.weight` — Language model head (i2_s, vocab_size x hidden_size)
- `lm_head.scale` — LM head requantization scale (raw_int8, 4 bytes = f32)

**Per-layer tensors** (prefix: `layers.{N}.`):
- `attention.q_proj.weight` / `.scale` — Query projection
- `attention.k_proj.weight` / `.scale` — Key projection
- `attention.v_proj.weight` / `.scale` — Value projection
- `attention.o_proj.weight` / `.scale` — Output projection
- `ffn.gate_proj.weight` / `.scale` — FFN gate projection (SwiGLU)
- `ffn.up_proj.weight` / `.scale` — FFN up projection (SwiGLU)
- `ffn.down_proj.weight` / `.scale` — FFN down projection
- `attention_norm.weight` — Pre-attention RMSNorm (raw_int8, Q0.7)
- `ffn_norm.weight` — Pre-FFN RMSNorm (raw_int8, Q0.7)

## Alignment

- Tensor data payloads are aligned to 16 bytes (`DATA_ALIGNMENT`).
- The data section starts at a 16-byte aligned offset.
- Padding between tensors uses zero bytes (valid ternary values that contribute nothing to dot products).

## Zero-Copy Design

The format is designed for memory-mapped access on MCU. The Zig loader (`zink_loader.zig`) casts pointers directly into the file buffer without copying or unpacking. This requires:
- All tensor data to be at aligned offsets
- The file to remain mapped/accessible for the lifetime of inference
- The packing format to match the kernel's expected input layout

## FNV-1a Hash

```
h = 0xCBF29CE484222325
for each byte in name:
    h ^= byte
    h *= 0x100000001B3
    h &= 0xFFFFFFFFFFFFFFFF
```

Implemented identically in `zink_loader.zig` (Zig) and `zink_pack.py` (Python).

## Python Packing Tool

```bash
# Generate test file
python3 tools/zink_pack.py --test

# Convert from safetensors
python3 tools/zink_pack.py model.safetensors -o model.zink --config bitnet_0_7b

# Convert with auto-inferred config
python3 tools/zink_pack.py model.safetensors -o model.zink

# Verify integrity
python3 tools/zink_pack.py model.safetensors -o model.zink --verify
```
