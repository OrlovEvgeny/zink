const std = @import("std");

pub const ModelConfig = struct {
    hidden_size: comptime_int,
    num_layers: comptime_int,
    num_heads: comptime_int,
    num_kv_heads: comptime_int,
    head_dim: comptime_int,
    intermediate_size: comptime_int,
    vocab_size: comptime_int,
    max_seq_len: comptime_int,
    rope_theta: comptime_float = 10000.0,
    kv_group_size: comptime_int = 64,

    // Sliding window attention: limits how many past tokens participate in
    // attention. Orthogonal to the ring buffer (which limits total KV storage).
    // null = full context (all validLen() positions).
    // Gemma/Mistral use sliding_window_size = 4096 or similar.
    sliding_window_size: ?comptime_int = null,

    pub fn validate(comptime self: ModelConfig) void {
        if (self.hidden_size % 128 != 0) {
            @compileError(std.fmt.comptimePrint(
                "hidden_size ({}) must be a multiple of 128 for I2_S kernel alignment",
                .{self.hidden_size},
            ));
        }
        if (self.hidden_size != self.num_heads * self.head_dim) {
            @compileError(std.fmt.comptimePrint(
                "hidden_size ({}) != num_heads ({}) * head_dim ({})",
                .{ self.hidden_size, self.num_heads, self.head_dim },
            ));
        }
        if (self.num_heads % self.num_kv_heads != 0) {
            @compileError(std.fmt.comptimePrint(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                .{ self.num_heads, self.num_kv_heads },
            ));
        }
        if (self.head_dim % self.kv_group_size != 0) {
            @compileError(std.fmt.comptimePrint(
                "head_dim ({}) must be divisible by kv_group_size ({})",
                .{ self.head_dim, self.kv_group_size },
            ));
        }
        if (self.intermediate_size % 128 != 0) {
            @compileError(std.fmt.comptimePrint(
                "intermediate_size ({}) must be a multiple of 128 for I2_S kernel alignment",
                .{self.intermediate_size},
            ));
        }
        if (self.max_seq_len == 0) {
            @compileError("max_seq_len must be > 0");
        }
        if (self.vocab_size == 0) {
            @compileError("vocab_size must be > 0");
        }
        if (self.sliding_window_size) |sw| {
            if (sw == 0) {
                @compileError("sliding_window_size must be > 0 when set");
            }
        }
    }

    /// Byte counts for SRAM layout, computed at comptime.
    pub fn memoryLayout(comptime self: ModelConfig) ModelMemoryLayout {
        self.validate();
        return ModelMemoryLayout.compute(self);
    }

    /// Compile-time check that the model fits in the target's SRAM.
    pub fn assertFitsIn(comptime self: ModelConfig, comptime sram_bytes: usize) void {
        const layout = self.memoryLayout();
        if (layout.total > sram_bytes) {
            @compileError(std.fmt.comptimePrint(
                "Model requires {} bytes but target has {} bytes SRAM ({} bytes over budget)",
                .{ layout.total, sram_bytes, layout.total - sram_bytes },
            ));
        }
    }
};

pub const ModelMemoryLayout = struct {
    // Ping-pong activation buffers: 2 × hidden_size × sizeof(i8).
    ping_pong_bytes: usize,
    // Q4 KV-cache: num_kv_heads × head_dim/2 × max_seq_len × 2 (K+V)
    // plus scale storage.
    kv_cache_bytes: usize,
    // Scratch space for intermediate computations.
    scratch_bytes: usize,
    // TL1 LUT: 9 entries × (hidden_size/2) × sizeof(i16) if TL1 is used.
    tl1_lut_bytes: usize,
    total: usize,

    fn compute(comptime cfg: ModelConfig) ModelMemoryLayout {
        const ping_pong = 2 * cfg.hidden_size;

        // Q4 packs two values per byte. Per head per position: head_dim/2 bytes.
        // Plus one i16 scale per group.
        const packed_dim = cfg.head_dim / 2;
        const num_groups_per_head = cfg.head_dim / cfg.kv_group_size;
        const scales_per_head = num_groups_per_head * @sizeOf(i16);
        const kv_per_head_per_pos = packed_dim + scales_per_head;
        // Each layer has its own KV-cache (InferenceEngine creates [num_layers]KvCache).
        const kv_total = cfg.num_layers * cfg.num_kv_heads * cfg.max_seq_len * kv_per_head_per_pos * 2;

        // Full ScratchBuffers struct from transformer.zig:
        //   i8 buffers: norm_out + q_out + k_out + v_out + attn_out + proj_out + residual
        //   i8 buffers: ffn_gate + ffn_up
        //   i32 buffers: scores, accum, accum_large
        //   u16 buffer: probs
        //   i8 buffers: key_buf + val_buf
        const scratch = 7 * cfg.hidden_size // norm_out, q_out, k_out, v_out, attn_out, proj_out, residual
        + 2 * cfg.intermediate_size // ffn_gate, ffn_up
        + cfg.max_seq_len * @sizeOf(i32) // scores
        + cfg.max_seq_len * @sizeOf(u16) // probs
        + cfg.hidden_size * @sizeOf(i32) // accum
        + cfg.intermediate_size * @sizeOf(i32) // accum_large
        + 2 * cfg.head_dim; // key_buf, val_buf

        // TL1 LUT: 9 entries per activation pair, i16 each.
        const tl1_lut = 9 * (cfg.hidden_size / 2) * @sizeOf(i16);

        const total = ping_pong + kv_total + scratch + tl1_lut;

        return .{
            .ping_pong_bytes = ping_pong,
            .kv_cache_bytes = kv_total,
            .scratch_bytes = scratch,
            .tl1_lut_bytes = tl1_lut,
            .total = total,
        };
    }
};

// Minimal test config for unit testing.
pub const bitnet_test = ModelConfig{
    .hidden_size = 128,
    .num_layers = 2,
    .num_heads = 2,
    .num_kv_heads = 2,
    .head_dim = 64,
    .intermediate_size = 384,
    .vocab_size = 256,
    .max_seq_len = 64,
    .kv_group_size = 64,
};

// Sliding window test config: same as bitnet_test but with window_size=16.
pub const bitnet_test_windowed = ModelConfig{
    .hidden_size = 128,
    .num_layers = 2,
    .num_heads = 2,
    .num_kv_heads = 2,
    .head_dim = 64,
    .intermediate_size = 384,
    .vocab_size = 256,
    .max_seq_len = 64,
    .kv_group_size = 64,
    .sliding_window_size = 16,
};

// GQA test config: 8 query heads sharing 2 KV heads (4:1 ratio).
pub const bitnet_test_gqa = ModelConfig{
    .hidden_size = 512,
    .num_layers = 2,
    .num_heads = 8,
    .num_kv_heads = 2,
    .head_dim = 64,
    .intermediate_size = 1408,
    .vocab_size = 256,
    .max_seq_len = 64,
    .kv_group_size = 64,
};

// BitNet b1.58 0.7B — 24 layers, 1536 hidden.
// Ref: BitNet b1.58 paper, Table 1.
pub const bitnet_b158_0_7b = ModelConfig{
    .hidden_size = 1536,
    .num_layers = 24,
    .num_heads = 24,
    .num_kv_heads = 24,
    .head_dim = 64,
    .intermediate_size = 4096,
    .vocab_size = 32000,
    .max_seq_len = 512,
    .rope_theta = 10000.0,
    .kv_group_size = 64,
};

// BitNet b1.58 3B — 26 layers, 3200 hidden.
// head_dim = 3200/32 = 100; kv_group_size must divide head_dim.
// intermediate_size rounded to 8704 (from 8640) for I2_S 128-element alignment.
// Ref: BitNet b1.58 paper, Table 1.
pub const bitnet_b158_3b = ModelConfig{
    .hidden_size = 3200,
    .num_layers = 26,
    .num_heads = 32,
    .num_kv_heads = 32,
    .head_dim = 100,
    .intermediate_size = 8704,
    .vocab_size = 32000,
    .max_seq_len = 512,
    .rope_theta = 10000.0,
    .kv_group_size = 100,
};

test "bitnet_test config validates" {
    comptime bitnet_test.validate();
}

test "bitnet_test_windowed config validates" {
    comptime bitnet_test_windowed.validate();
}

test "bitnet_test_gqa config validates" {
    comptime bitnet_test_gqa.validate();
}

test "bitnet_b158_0_7b config validates" {
    comptime bitnet_b158_0_7b.validate();
}

test "bitnet_b158_3b config validates" {
    comptime bitnet_b158_3b.validate();
}

test "production config memory layouts compute" {
    const layout_07b = comptime bitnet_b158_0_7b.memoryLayout();
    try std.testing.expect(layout_07b.total > 0);
    try std.testing.expect(layout_07b.kv_cache_bytes > layout_07b.ping_pong_bytes);

    const layout_3b = comptime bitnet_b158_3b.memoryLayout();
    try std.testing.expect(layout_3b.total > layout_07b.total);
}

test "memory layout computes without overflow" {
    const layout = comptime bitnet_test.memoryLayout();
    try std.testing.expect(layout.total > 0);
    try std.testing.expect(layout.ping_pong_bytes == 256);
    try std.testing.expect(layout.total == layout.ping_pong_bytes + layout.kv_cache_bytes + layout.scratch_bytes + layout.tl1_lut_bytes);

    // KV-cache must include num_layers factor.
    // Per-layer: num_kv_heads(2) * max_seq(64) * (head_dim/2 + scales) * 2(K+V)
    //   = 2 * 64 * (32 + 2) * 2 = 8704
    // Total = 2 layers * 8704 = 17408
    const per_layer_kv = bitnet_test.num_kv_heads * bitnet_test.max_seq_len * (bitnet_test.head_dim / 2 + (bitnet_test.head_dim / bitnet_test.kv_group_size) * @sizeOf(i16)) * 2;
    try std.testing.expectEqual(per_layer_kv * bitnet_test.num_layers, layout.kv_cache_bytes);
}

test "memory layout scratch matches ScratchBuffers" {
    const layout = comptime bitnet_test.memoryLayout();
    // Verify scratch accounts for all ScratchBuffers fields.
    const expected_scratch = 7 * bitnet_test.hidden_size // i8 buffers
    + 2 * bitnet_test.intermediate_size // ffn_gate, ffn_up
    + bitnet_test.max_seq_len * @sizeOf(i32) // scores
    + bitnet_test.max_seq_len * @sizeOf(u16) // probs
    + bitnet_test.hidden_size * @sizeOf(i32) // accum
    + bitnet_test.intermediate_size * @sizeOf(i32) // accum_large
    + 2 * bitnet_test.head_dim; // key_buf, val_buf
    try std.testing.expectEqual(expected_scratch, layout.scratch_bytes);
}

test "assertFitsIn passes for large SRAM" {
    comptime bitnet_test.assertFitsIn(1024 * 1024);
}
