#!/usr/bin/env python3
"""
memory_planner.py — Compute exact SRAM memory breakdown for a Zink model config.

Matches the comptime ModelMemoryLayout computation in src/model/config.zig.

Usage:
    python3 tools/memory_planner.py --model bitnet_0_7b --target stm32n6

    python3 tools/memory_planner.py --hidden 128 --layers 2 --heads 2 \\
        --kv-heads 2 --head-dim 64 --intermediate 384 --vocab 256 --max-seq 64

    python3 tools/memory_planner.py --model bitnet_0_7b --target stm32n6 \\
        --sliding-window 256
"""

import argparse
import sys

TARGETS = {
    "stm32n6": {"name": "STM32N6570-DK", "sram_bytes": 4 * 1024 * 1024 + 200 * 1024},
    "rp2350": {"name": "RP2350", "sram_bytes": 520 * 1024},
    "generic_m55": {"name": "Generic Cortex-M55", "sram_bytes": 2 * 1024 * 1024},
}

# Must match PREDEFINED_CONFIGS in zink_pack.py and src/model/config.zig.
MODELS = {
    "bitnet_test": {
        "hidden_size": 128, "num_layers": 2, "num_heads": 2,
        "num_kv_heads": 2, "head_dim": 64, "intermediate_size": 384,
        "vocab_size": 256, "max_seq_len": 64, "kv_group_size": 64,
    },
    "bitnet_0_7b": {
        "hidden_size": 1536, "num_layers": 24, "num_heads": 24,
        "num_kv_heads": 24, "head_dim": 64, "intermediate_size": 4096,
        "vocab_size": 32000, "max_seq_len": 512, "kv_group_size": 64,
    },
    "bitnet_3b": {
        "hidden_size": 3200, "num_layers": 26, "num_heads": 32,
        "num_kv_heads": 32, "head_dim": 100, "intermediate_size": 8704,
        "vocab_size": 32000, "max_seq_len": 512, "kv_group_size": 100,
    },
}


def compute_memory_layout(
    hidden_size,
    num_layers,
    num_heads,
    num_kv_heads,
    head_dim,
    intermediate_size,
    vocab_size,
    max_seq_len,
    kv_group_size=64,
    sliding_window_size=None,
):
    """Compute memory breakdown matching ModelMemoryLayout.compute() in Zig."""
    # Ping-pong activation buffers: 2 × hidden_size × sizeof(i8).
    ping_pong = 2 * hidden_size

    # Q4 KV-cache.
    # The ring buffer always allocates max_seq_len positions, regardless of
    # sliding window size. Sliding window only limits which positions
    # participate in attention, not the storage capacity of the ring buffer.
    # This matches src/model/config.zig ModelMemoryLayout.compute().
    effective_seq_len = max_seq_len

    packed_dim = head_dim // 2
    num_groups_per_head = head_dim // kv_group_size
    scales_per_head = num_groups_per_head * 2  # sizeof(i16)
    kv_per_head_per_pos = packed_dim + scales_per_head
    kv_total = num_layers * num_kv_heads * effective_seq_len * kv_per_head_per_pos * 2

    # Scratch buffers from ScratchBuffers struct (matches config.zig exactly).
    scratch_buffers = (
        hidden_size * 1  # norm_out
        + hidden_size * 1  # q_out
        + hidden_size * 1  # k_out
        + hidden_size * 1  # v_out
        + hidden_size * 1  # attn_out
        + intermediate_size * 1  # ffn_gate
        + intermediate_size * 1  # ffn_up
        + hidden_size * 1  # proj_out
        + hidden_size * 1  # residual
        + max_seq_len * 4  # scores (i32)
        + max_seq_len * 2  # probs (u16)
        + hidden_size * 4  # accum (i32)
        + intermediate_size * 4  # accum_large (i32)
        + head_dim * 1  # key_buf
        + head_dim * 1  # val_buf
    )

    # TL1 LUT: 9 entries per activation pair, i16 each.
    tl1_lut = 9 * (hidden_size // 2) * 2

    # Weight memory: ternary weights at 2 bits per element (I2_S packed).
    # Per layer: Q, K, V, O projections (hidden² each) + gate, up (inter×hidden)
    # + down (hidden×inter).
    per_layer_weight_bits = (
        4 * hidden_size * hidden_size  # Q, K, V, O
        + 2 * intermediate_size * hidden_size  # gate, up
        + hidden_size * intermediate_size  # down
    ) * 2  # 2 bits per weight
    per_layer_weight_bytes = per_layer_weight_bits // 8
    # RMSNorm weights: 2 per layer × hidden_size × 1 byte.
    per_layer_norm_bytes = 2 * hidden_size
    total_layer_weights = num_layers * (per_layer_weight_bytes + per_layer_norm_bytes)

    # Global weights: embedding (vocab × hidden × 1 byte),
    # lm_head (vocab × hidden × 2 bits / 8), final_rmsnorm (hidden × 1 byte).
    embedding_bytes = vocab_size * hidden_size
    lm_head_bytes = vocab_size * hidden_size * 2 // 8
    final_rms_bytes = hidden_size
    total_global_weights = embedding_bytes + lm_head_bytes + final_rms_bytes

    total_weights = total_layer_weights + total_global_weights

    # SRAM total (runtime buffers only — weights stream from flash).
    sram_total = ping_pong + kv_total + scratch_buffers + tl1_lut

    return {
        "ping_pong": ping_pong,
        "kv_cache": kv_total,
        "kv_effective_seq_len": effective_seq_len,
        "scratch_buffers": scratch_buffers,
        "tl1_lut": tl1_lut,
        "sram_total": sram_total,
        "weight_layer_total": total_layer_weights,
        "weight_global_total": total_global_weights,
        "weight_embedding": embedding_bytes,
        "weight_total": total_weights,
    }


def format_bytes(n):
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n} B"


def main():
    parser = argparse.ArgumentParser(description="Zink memory budget planner")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Predefined model ({', '.join(MODELS.keys())})")
    parser.add_argument("--hidden", type=int, default=None, help="hidden_size")
    parser.add_argument("--layers", type=int, default=None, help="num_layers")
    parser.add_argument("--heads", type=int, default=None, help="num_heads")
    parser.add_argument("--kv-heads", type=int, default=None, help="num_kv_heads")
    parser.add_argument("--head-dim", type=int, default=None, help="head_dim")
    parser.add_argument("--intermediate", type=int, default=None, help="intermediate_size")
    parser.add_argument("--vocab", type=int, default=None, help="vocab_size")
    parser.add_argument("--max-seq", type=int, default=None, help="max_seq_len")
    parser.add_argument("--kv-group-size", type=int, default=None, help="KV Q4 group size")
    parser.add_argument("--sliding-window", type=int, default=None,
                        help="Sliding window attention size")
    parser.add_argument("--target", type=str, default=None,
                        help=f"Target ({', '.join(TARGETS.keys())})")
    args = parser.parse_args()

    # Build config from --model or individual flags.
    if args.model:
        if args.model not in MODELS:
            print(f"ERROR: Unknown model '{args.model}'. Available: {', '.join(MODELS.keys())}",
                  file=sys.stderr)
            sys.exit(1)
        config = dict(MODELS[args.model])  # copy
    elif args.hidden is not None:
        # All individual flags required.
        required = ["hidden", "layers", "heads", "kv_heads", "head_dim",
                     "intermediate", "vocab", "max_seq"]
        missing = [f for f in required if getattr(args, f) is None]
        if missing:
            print(f"ERROR: Missing required flags: {', '.join('--' + f.replace('_', '-') for f in missing)}",
                  file=sys.stderr)
            sys.exit(1)
        config = {
            "hidden_size": args.hidden,
            "num_layers": args.layers,
            "num_heads": args.heads,
            "num_kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "intermediate_size": args.intermediate,
            "vocab_size": args.vocab,
            "max_seq_len": args.max_seq,
            "kv_group_size": args.kv_group_size or args.head_dim,
        }
    else:
        parser.print_help()
        sys.exit(1)

    # Override with individual flags if both --model and flags given.
    if args.hidden is not None:
        config["hidden_size"] = args.hidden
    if args.layers is not None:
        config["num_layers"] = args.layers
    if args.heads is not None:
        config["num_heads"] = args.heads
    if args.kv_heads is not None:
        config["num_kv_heads"] = args.kv_heads
    if args.head_dim is not None:
        config["head_dim"] = args.head_dim
    if args.intermediate is not None:
        config["intermediate_size"] = args.intermediate
    if args.vocab is not None:
        config["vocab_size"] = args.vocab
    if args.max_seq is not None:
        config["max_seq_len"] = args.max_seq
    if args.kv_group_size is not None:
        config["kv_group_size"] = args.kv_group_size

    # Validate.
    if config["hidden_size"] % 128 != 0:
        print(f"ERROR: hidden_size ({config['hidden_size']}) must be a multiple of 128",
              file=sys.stderr)
        sys.exit(1)
    if config["intermediate_size"] % 128 != 0:
        print(f"ERROR: intermediate_size ({config['intermediate_size']}) must be a multiple of 128",
              file=sys.stderr)
        sys.exit(1)
    if config["hidden_size"] != config["num_heads"] * config["head_dim"]:
        print(f"ERROR: hidden_size ({config['hidden_size']}) != "
              f"num_heads ({config['num_heads']}) * head_dim ({config['head_dim']})",
              file=sys.stderr)
        sys.exit(1)

    layout = compute_memory_layout(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_kv_heads=config["num_kv_heads"],
        head_dim=config["head_dim"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        kv_group_size=config.get("kv_group_size", config["head_dim"]),
        sliding_window_size=args.sliding_window,
    )

    print("=" * 60)
    print("Zink Memory Budget Planner")
    print("=" * 60)
    print()
    print("Model Configuration:")
    print(f"  hidden_size      = {config['hidden_size']}")
    print(f"  num_layers       = {config['num_layers']}")
    print(f"  num_heads        = {config['num_heads']}")
    print(f"  num_kv_heads     = {config['num_kv_heads']}")
    print(f"  head_dim         = {config['head_dim']}")
    print(f"  intermediate     = {config['intermediate_size']}")
    print(f"  vocab_size       = {config['vocab_size']}")
    print(f"  max_seq_len      = {config['max_seq_len']}")
    print(f"  kv_group_size    = {config.get('kv_group_size', config['head_dim'])}")
    if args.sliding_window:
        print(f"  sliding_window   = {args.sliding_window}")
    print()

    print("SRAM Breakdown (runtime buffers):")
    print(f"  Ping-pong buffers   : {layout['ping_pong']:>10} ({format_bytes(layout['ping_pong'])})")
    kv_note = ""
    if args.sliding_window:
        kv_note = f" [ring buffer stores full max_seq_len={config['max_seq_len']}]"
    print(f"  Q4 KV-cache         : {layout['kv_cache']:>10} ({format_bytes(layout['kv_cache'])}){kv_note}")
    print(f"  Scratch buffers     : {layout['scratch_buffers']:>10} ({format_bytes(layout['scratch_buffers'])})")
    print(f"  TL1 LUT             : {layout['tl1_lut']:>10} ({format_bytes(layout['tl1_lut'])})")
    print(f"  ---")
    print(f"  SRAM Total          : {layout['sram_total']:>10} ({format_bytes(layout['sram_total'])})")

    print()
    print("Weight Storage (flash):")
    print(f"  Layer weights       : {layout['weight_layer_total']:>10} ({format_bytes(layout['weight_layer_total'])})")
    print(f"  Embedding table     : {layout['weight_embedding']:>10} ({format_bytes(layout['weight_embedding'])})")
    print(f"  Global weights      : {layout['weight_global_total']:>10} ({format_bytes(layout['weight_global_total'])})")
    print(f"  ---")
    print(f"  Total model size    : {layout['weight_total']:>10} ({format_bytes(layout['weight_total'])})")

    if args.target:
        if args.target not in TARGETS:
            print(f"\nERROR: Unknown target '{args.target}'. Available: {', '.join(TARGETS.keys())}",
                  file=sys.stderr)
            sys.exit(1)

        target = TARGETS[args.target]
        sram = target["sram_bytes"]
        print()
        print(f"Target: {target['name']} ({format_bytes(sram)} SRAM)")
        used_pct = layout["sram_total"] / sram * 100

        if layout["sram_total"] <= sram:
            remaining = sram - layout["sram_total"]
            print(f"  Status: FITS ({used_pct:.1f}% used, {format_bytes(remaining)} remaining)")
        else:
            over = layout["sram_total"] - sram
            print(f"  Status: DOES NOT FIT ({used_pct:.1f}% of SRAM, {format_bytes(over)} over budget)")
            sys.exit(1)

    print()


if __name__ == "__main__":
    main()
