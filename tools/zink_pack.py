#!/usr/bin/env python3
"""
zink_pack.py — Convert model weights to .zink format.

Usage:
    python3 tools/zink_pack.py --test                        # Generate test .zink file
    python3 tools/zink_pack.py model.safetensors -o model.zink
    python3 tools/zink_pack.py model.safetensors -o model.zink --verify
    python3 tools/zink_pack.py model.safetensors -o model.zink --config config.json

The .zink format:
  - 96-byte header (magic "ZINK", model params, tensor table offset)
  - Tensor table: 32 bytes per entry (name_hash, offset, rows, cols, size, quant_type)
  - Tensor data: SIMD-aligned (16-byte), zero-copy design
"""

import struct
import argparse
import json
import sys
import os

try:
    import numpy as np
except ImportError:
    np = None

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

MAGIC = 0x5A494E4B  # "ZINK"
HEADER_SIZE = 96
TENSOR_ENTRY_SIZE = 32
DATA_ALIGNMENT = 16

# Quant types matching Zig enum.
QUANT_I2_S = 0
QUANT_TL1_PACKED = 1
QUANT_RAW_INT8 = 2
QUANT_F16 = 3
QUANT_Q4 = 4

# Predefined model configs matching src/model/config.zig.
PREDEFINED_CONFIGS = {
    "bitnet_test": {
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 2,
        "num_kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 384,
        "vocab_size": 256,
        "max_seq_len": 64,
        "rope_theta": 10000.0,
        "kv_group_size": 64,
    },
    "bitnet_0_7b": {
        "hidden_size": 1536,
        "num_layers": 24,
        "num_heads": 24,
        "num_kv_heads": 24,
        "head_dim": 64,
        "intermediate_size": 4096,
        "vocab_size": 32000,
        "max_seq_len": 512,
        "rope_theta": 10000.0,
        "kv_group_size": 64,
    },
    "bitnet_3b": {
        "hidden_size": 3200,
        "num_layers": 26,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 100,
        "intermediate_size": 8704,
        "vocab_size": 32000,
        "max_seq_len": 512,
        "rope_theta": 10000.0,
        "kv_group_size": 100,
    },
}

# HuggingFace tensor name → Zink tensor name mapping.
HF_TO_ZINK_MAP = {
    "model.embed_tokens.weight": "embedding.weight",
    "model.norm.weight": "final_rmsnorm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Per-layer HuggingFace patterns → Zink suffixes.
HF_LAYER_MAP = {
    "self_attn.q_proj.weight": "attention.q_proj.weight",
    "self_attn.k_proj.weight": "attention.k_proj.weight",
    "self_attn.v_proj.weight": "attention.v_proj.weight",
    "self_attn.o_proj.weight": "attention.o_proj.weight",
    "mlp.gate_proj.weight": "ffn.gate_proj.weight",
    "mlp.up_proj.weight": "ffn.up_proj.weight",
    "mlp.down_proj.weight": "ffn.down_proj.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}


def fnv1a_hash(name: str) -> int:
    """FNV-1a 64-bit hash. Must match the Zig implementation."""
    h = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def pack_i2s(ternary: "np.ndarray") -> bytes:
    """Pack ternary values {-1, 0, 1} into 2-bit two's complement.

    Encoding: -1 = 0b11, 0 = 0b00, 1 = 0b01.
    4 values per byte, LSB first.
    """
    flat = ternary.flatten().astype(np.int8)
    assert len(flat) % 4 == 0, f"Length {len(flat)} not divisible by 4"

    result = bytearray()
    for i in range(0, len(flat), 4):
        byte = 0
        for j in range(4):
            w = int(flat[i + j])
            assert w in (-1, 0, 1), f"Non-ternary value {w} at index {i+j}"
            # Two's complement for 2-bit: -1 → 0b11, 0 → 0b00, 1 → 0b01
            bits = w & 0x3
            byte |= bits << (j * 2)
        result.append(byte)
    return bytes(result)


def quantize_bitnet_b158(weights: "np.ndarray"):
    """Per-tensor AbsMean quantization for BitNet b1.58.

    gamma = mean(|W|)
    W_ternary = round(clamp(W / gamma, -1, 1))

    Ref: BitNet b1.58 paper, Section 3.1.
    """
    gamma = np.mean(np.abs(weights))
    if gamma == 0:
        return np.zeros_like(weights, dtype=np.int8), 0.0
    scaled = weights / gamma
    clamped = np.clip(scaled, -1, 1)
    ternary = np.round(clamped).astype(np.int8)
    return ternary, float(gamma)


def quantize_activations_int8(activations: "np.ndarray"):
    """Per-tensor int8 activation quantization (matching BitNet b1.58 scheme).

    Not per-block, not per-channel.
    """
    absmax = np.max(np.abs(activations))
    if absmax == 0:
        return np.zeros_like(activations, dtype=np.int8), 0.0
    scale = 127.0 / absmax
    quantized = np.clip(np.round(activations * scale), -128, 127).astype(np.int8)
    return quantized, float(absmax / 127.0)


def align_offset(offset: int, alignment: int = DATA_ALIGNMENT) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def map_hf_tensor_name(hf_name: str) -> str | None:
    """Map a HuggingFace tensor name to Zink convention.

    Examples:
        model.layers.0.self_attn.q_proj.weight → layers.0.attention.q_proj.weight
        model.embed_tokens.weight → embedding.weight
    """
    if hf_name in HF_TO_ZINK_MAP:
        return HF_TO_ZINK_MAP[hf_name]

    # Layer tensors: model.layers.{N}.{suffix}
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        # model.layers.N.rest...
        if len(parts) >= 4:
            layer_idx = parts[2]
            remainder = ".".join(parts[3:])
            if remainder in HF_LAYER_MAP:
                return f"layers.{layer_idx}.{HF_LAYER_MAP[remainder]}"

    return None


def write_header(
    f,
    hidden_size,
    num_layers,
    num_heads,
    num_kv_heads,
    head_dim,
    intermediate_size,
    vocab_size,
    max_seq_len,
    rope_theta,
    kv_group_size,
    quant_type,
    num_tensors,
    tensor_table_offset,
    data_offset,
    total_size,
):
    """Write 96-byte .zink header."""
    rope_theta_fixed = int(rope_theta * 65536) & 0xFFFFFFFF

    header = struct.pack(
        "<IHH"  # magic, version, flags (8)
        "IHHHHIII"  # hidden_size..max_seq_len (24)
        "IHBx"  # rope_theta_fixed, kv_group_size, quant_type, pad (8)
        "III"  # num_tensors, tensor_table_offset, data_offset (12)
        "xxxx"  # alignment padding for u64 (4)
        "Q"  # total_size (8)
        "32s",  # reserved (32) = 96 total
        MAGIC,
        1,  # version
        0,  # flags
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_seq_len,
        rope_theta_fixed,
        kv_group_size,
        quant_type,
        num_tensors,
        tensor_table_offset,
        data_offset,
        total_size,
        b"\x00" * 32,
    )
    assert len(header) == HEADER_SIZE, f"Header size {len(header)} != {HEADER_SIZE}"
    f.write(header)


def write_tensor_entry(f, name_hash, offset, rows, cols, packed_size, quant_type):
    """Write 32-byte tensor table entry."""
    entry = struct.pack(
        "<QIIIIB7s",
        name_hash,
        offset,
        rows,
        cols,
        packed_size,
        quant_type,
        b"\x00" * 7,
    )
    assert len(entry) == TENSOR_ENTRY_SIZE
    f.write(entry)


def write_zink_file(output_path: str, config: dict, packed_tensors: list):
    """Write a complete .zink file from config and packed tensor list."""
    num_tensors = len(packed_tensors)
    tensor_table_offset = HEADER_SIZE
    data_offset = align_offset(tensor_table_offset + num_tensors * TENSOR_ENTRY_SIZE)

    current_data_offset = 0
    for t in packed_tensors:
        t["offset"] = current_data_offset
        current_data_offset = align_offset(current_data_offset + len(t["data"]))

    total_size = data_offset + current_data_offset

    with open(output_path, "wb") as f:
        write_header(
            f,
            config["hidden_size"],
            config["num_layers"],
            config["num_heads"],
            config["num_kv_heads"],
            config["head_dim"],
            config["intermediate_size"],
            config["vocab_size"],
            config["max_seq_len"],
            config.get("rope_theta", 10000.0),
            config.get("kv_group_size", 64),
            QUANT_I2_S,
            num_tensors,
            tensor_table_offset,
            data_offset,
            total_size,
        )

        for t in packed_tensors:
            write_tensor_entry(
                f,
                t["hash"],
                t["offset"],
                t["rows"],
                t["cols"],
                len(t["data"]),
                t["quant_type"],
            )

        current_pos = f.tell()
        if current_pos < data_offset:
            f.write(b"\x00" * (data_offset - current_pos))

        for t in packed_tensors:
            f.write(t["data"])
            remainder = len(t["data"]) % DATA_ALIGNMENT
            if remainder:
                f.write(b"\x00" * (DATA_ALIGNMENT - remainder))

    return total_size, num_tensors


def generate_test_zink(output_path: str):
    """Generate a small test .zink file for round-trip testing."""
    if np is None:
        print("numpy required for test generation. Install: pip install numpy")
        sys.exit(1)

    rng = np.random.default_rng(42)
    config = PREDEFINED_CONFIGS["bitnet_test"]

    tensors = {}
    scale_tensors = {}
    for layer in range(config["num_layers"]):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            name = f"layers.{layer}.attention.{proj}.weight"
            weights_f32 = rng.standard_normal(
                size=(config["hidden_size"], config["hidden_size"])
            ).astype(np.float32)
            ternary, gamma = quantize_bitnet_b158(weights_f32)
            tensors[name] = ternary
            scale_tensors[f"layers.{layer}.attention.{proj}.scale"] = gamma

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            name = f"layers.{layer}.ffn.{proj}.weight"
            if proj == "down_proj":
                shape = (config["hidden_size"], config["intermediate_size"])
            else:
                shape = (config["intermediate_size"], config["hidden_size"])
            weights_f32 = rng.standard_normal(size=shape).astype(np.float32)
            ternary, gamma = quantize_bitnet_b158(weights_f32)
            tensors[name] = ternary
            scale_tensors[f"layers.{layer}.ffn.{proj}.scale"] = gamma

        for norm in ["attention_norm", "ffn_norm"]:
            name = f"layers.{layer}.{norm}.weight"
            tensors[name] = np.full(config["hidden_size"], 127, dtype=np.int8)

    embed = rng.integers(
        -128, 127,
        size=(config["vocab_size"], config["hidden_size"]),
        dtype=np.int8,
    )
    tensors["embedding.weight"] = embed
    tensors["final_rmsnorm.weight"] = np.full(config["hidden_size"], 127, dtype=np.int8)

    lm_head_f32 = rng.standard_normal(
        size=(config["vocab_size"], config["hidden_size"])
    ).astype(np.float32)
    lm_ternary, lm_gamma = quantize_bitnet_b158(lm_head_f32)
    tensors["lm_head.weight"] = lm_ternary
    scale_tensors["lm_head.scale"] = lm_gamma

    packed_tensors = pack_tensor_dict(tensors, scale_tensors)
    total_size, num_tensors = write_zink_file(output_path, config, packed_tensors)

    file_size = os.path.getsize(output_path)
    print(f"Wrote {output_path}: {file_size} bytes, {num_tensors} tensors")
    print(f"  hidden_size={config['hidden_size']}, layers={config['num_layers']}")

    verify_zink_file(output_path)
    print("  Verification: PASSED")


def pack_tensor_dict(tensors: dict, scale_tensors: dict) -> list:
    """Pack a dict of named tensors into the list format for write_zink_file."""
    packed = []
    for name, weights in tensors.items():
        is_raw = (
            name.endswith("_norm.weight")
            or name in ("embedding.weight", "final_rmsnorm.weight")
        )
        if is_raw:
            data = weights.flatten().tobytes()
            rows = weights.shape[0] if weights.ndim > 1 else weights.shape[0]
            cols = weights.shape[1] if weights.ndim > 1 else 1
            packed.append({
                "name": name,
                "hash": fnv1a_hash(name),
                "rows": rows,
                "cols": cols,
                "data": data,
                "quant_type": QUANT_RAW_INT8,
            })
        else:
            data = pack_i2s(weights)
            packed.append({
                "name": name,
                "hash": fnv1a_hash(name),
                "rows": weights.shape[0],
                "cols": weights.shape[1],
                "data": data,
                "quant_type": QUANT_I2_S,
            })

    for name, gamma in scale_tensors.items():
        data = struct.pack("<f", gamma)
        packed.append({
            "name": name,
            "hash": fnv1a_hash(name),
            "rows": 1,
            "cols": 1,
            "data": data,
            "quant_type": QUANT_RAW_INT8,
        })

    return packed


def convert_safetensors(input_path: str, output_path: str, config: dict):
    """Convert a .safetensors file to .zink format.

    Loads all tensors, maps HuggingFace names to Zink convention,
    applies BitNet b1.58 quantization to weight tensors, and writes
    the .zink file.
    """
    if np is None:
        print("ERROR: numpy required. Install: pip install numpy", file=sys.stderr)
        sys.exit(1)
    if safe_open is None:
        print("ERROR: safetensors required. Install: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path}...")
    model = safe_open(input_path, framework="numpy")
    tensor_names = model.keys()
    print(f"  Found {len(list(tensor_names))} tensors")

    tensors = {}
    scale_tensors = {}
    skipped = []

    for hf_name in model.keys():
        zink_name = map_hf_tensor_name(hf_name)
        if zink_name is None:
            skipped.append(hf_name)
            continue

        weight = model.get_tensor(hf_name).astype(np.float32)

        # Determine how to handle the tensor.
        if zink_name.endswith("_norm.weight"):
            # RMSNorm weights: quantize to Q0.7 (scale to [-128, 127]).
            absmax = np.max(np.abs(weight))
            if absmax > 0:
                scaled = np.clip(np.round(weight / absmax * 127), -128, 127).astype(np.int8)
            else:
                scaled = np.zeros_like(weight, dtype=np.int8)
            tensors[zink_name] = scaled

        elif zink_name == "embedding.weight":
            # Embedding: quantize to int8 per-tensor.
            quantized, _ = quantize_activations_int8(weight)
            tensors[zink_name] = quantized

        elif zink_name == "final_rmsnorm.weight":
            absmax = np.max(np.abs(weight))
            if absmax > 0:
                scaled = np.clip(np.round(weight / absmax * 127), -128, 127).astype(np.int8)
            else:
                scaled = np.zeros_like(weight, dtype=np.int8)
            tensors[zink_name] = scaled

        else:
            # Projection weights: apply BitNet b1.58 ternary quantization.
            ternary, gamma = quantize_bitnet_b158(weight)

            # Pad columns to a multiple of 4 for I2_S packing (4 ternary values
            # per byte). Padding uses zeros, which are valid ternary values and
            # contribute nothing to dot products.
            rows, cols = ternary.shape
            pad_cols = (4 - cols % 4) % 4
            if pad_cols > 0:
                ternary = np.pad(ternary, ((0, 0), (0, pad_cols)), constant_values=0)

            tensors[zink_name] = ternary

            # Store scale for requantization.
            scale_name = zink_name.replace(".weight", ".scale")
            scale_tensors[scale_name] = gamma

    if skipped:
        print(f"  Skipped {len(skipped)} unmapped tensors:")
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    # Validate expected tensors.
    num_layers = config["num_layers"]
    expected_count = 0
    missing = []

    # Check global tensors.
    for gname in ["embedding.weight", "final_rmsnorm.weight", "lm_head.weight"]:
        if gname in tensors:
            expected_count += 1
        else:
            missing.append(gname)

    # Check layer tensors.
    layer_suffixes = [
        "attention.q_proj.weight", "attention.k_proj.weight",
        "attention.v_proj.weight", "attention.o_proj.weight",
        "ffn.gate_proj.weight", "ffn.up_proj.weight", "ffn.down_proj.weight",
        "attention_norm.weight", "ffn_norm.weight",
    ]
    for layer_idx in range(num_layers):
        for suffix in layer_suffixes:
            name = f"layers.{layer_idx}.{suffix}"
            if name in tensors:
                expected_count += 1
            else:
                missing.append(name)

    if missing:
        print(f"  WARNING: {len(missing)} expected tensors missing:")
        for m in missing[:10]:
            print(f"    {m}")

    print(f"  Mapped {expected_count} tensors, {len(scale_tensors)} scales")

    packed_tensors = pack_tensor_dict(tensors, scale_tensors)
    total_size, num_tensors = write_zink_file(output_path, config, packed_tensors)

    file_size = os.path.getsize(output_path)
    print(f"Wrote {output_path}: {file_size} bytes ({file_size / (1024*1024):.1f} MB), {num_tensors} tensors")


def verify_zink_file(path: str):
    """Read back a .zink file and verify structural integrity."""
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small: {len(data)} < {HEADER_SIZE}")

    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {hex(magic)} (expected {hex(MAGIC)})")

    version = struct.unpack_from("<H", data, 4)[0]
    if version > 1:
        raise ValueError(f"Unsupported version: {version}")

    num_tensors = struct.unpack_from("<I", data, 40)[0]
    tensor_table_offset = struct.unpack_from("<I", data, 44)[0]
    data_offset = struct.unpack_from("<I", data, 48)[0]

    # Verify tensor table bounds.
    table_end = tensor_table_offset + num_tensors * TENSOR_ENTRY_SIZE
    if table_end > len(data):
        raise ValueError(f"Tensor table exceeds file: {table_end} > {len(data)}")

    # Verify each tensor entry.
    for i in range(num_tensors):
        entry_offset = tensor_table_offset + i * TENSOR_ENTRY_SIZE
        name_hash, offset, rows, cols, packed_size, quant_type = struct.unpack_from(
            "<QIIIIB", data, entry_offset
        )
        tensor_end = data_offset + offset + packed_size
        if tensor_end > len(data):
            raise ValueError(
                f"Tensor {i} (hash={hex(name_hash)}) data exceeds file: "
                f"{tensor_end} > {len(data)}"
            )

    print(f"  Verified: {num_tensors} tensors, all offsets valid")


def infer_config_from_safetensors(path: str) -> dict:
    """Attempt to infer model config from safetensors metadata and tensor shapes."""
    if safe_open is None:
        print("ERROR: safetensors required. Install: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    model = safe_open(path, framework="numpy")

    # Count layers by looking for layer tensor names.
    max_layer = -1
    for name in model.keys():
        if name.startswith("model.layers."):
            parts = name.split(".")
            if len(parts) >= 3:
                try:
                    layer_idx = int(parts[2])
                    max_layer = max(max_layer, layer_idx)
                except ValueError:
                    pass
    num_layers = max_layer + 1

    # Get hidden_size from embed_tokens or q_proj.
    hidden_size = None
    for name in model.keys():
        if "embed_tokens" in name:
            shape = model.get_tensor(name).shape
            hidden_size = shape[-1]
            vocab_size = shape[0]
            break

    if hidden_size is None:
        for name in model.keys():
            if "q_proj.weight" in name:
                shape = model.get_tensor(name).shape
                hidden_size = shape[0]
                break

    if hidden_size is None:
        raise ValueError("Cannot infer hidden_size from model tensors")

    # Get intermediate_size from gate_proj.
    intermediate_size = None
    for name in model.keys():
        if "gate_proj.weight" in name:
            shape = model.get_tensor(name).shape
            intermediate_size = shape[0]
            break

    # Get num_heads from q_proj vs k_proj shapes (for GQA detection).
    num_heads = None
    num_kv_heads = None
    for name in model.keys():
        if "self_attn.q_proj.weight" in name:
            q_shape = model.get_tensor(name).shape
            break
    for name in model.keys():
        if "self_attn.k_proj.weight" in name:
            k_shape = model.get_tensor(name).shape
            break

    # Infer head count. For standard MHA, q and k have same shape.
    # For GQA, k is smaller by the GQA ratio.
    head_dim = 64  # Default, may need override via --config.
    if hidden_size:
        num_heads = hidden_size // head_dim
        if 'k_shape' in dir() and 'q_shape' in dir():
            num_kv_heads = k_shape[0] // head_dim
        else:
            num_kv_heads = num_heads

    config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads or hidden_size // head_dim,
        "num_kv_heads": num_kv_heads or num_heads or hidden_size // head_dim,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size or hidden_size * 4,
        "vocab_size": vocab_size if 'vocab_size' in dir() else 32000,
        "max_seq_len": 512,
        "rope_theta": 10000.0,
        "kv_group_size": head_dim,
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Convert models to .zink format")
    parser.add_argument("input", nargs="?", help="Input model file (.safetensors)")
    parser.add_argument("-o", "--output", default="model.zink", help="Output .zink file")
    parser.add_argument("--test", action="store_true", help="Generate test .zink file")
    parser.add_argument("--verify", action="store_true", help="Verify output .zink file after writing")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Model config JSON file or predefined name (bitnet_0_7b, bitnet_3b)",
    )
    args = parser.parse_args()

    if args.test:
        output = args.output if args.output != "model.zink" else "test_model.zink"
        generate_test_zink(output)
    elif args.input:
        # Determine config.
        if args.config:
            if args.config in PREDEFINED_CONFIGS:
                config = PREDEFINED_CONFIGS[args.config]
                print(f"Using predefined config: {args.config}")
            elif os.path.isfile(args.config):
                with open(args.config) as f:
                    config = json.load(f)
                print(f"Using config from {args.config}")
            else:
                print(f"ERROR: Unknown config '{args.config}'", file=sys.stderr)
                print(f"  Predefined: {', '.join(PREDEFINED_CONFIGS.keys())}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Inferring config from model tensors...")
            config = infer_config_from_safetensors(args.input)
            print(f"  Inferred: hidden={config['hidden_size']}, "
                  f"layers={config['num_layers']}, "
                  f"heads={config['num_heads']}/{config['num_kv_heads']}")

        convert_safetensors(args.input, args.output, config)

        if args.verify:
            print("Verifying output...")
            verify_zink_file(args.output)
            print("  Verification: PASSED")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
