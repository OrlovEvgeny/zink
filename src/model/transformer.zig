const std = @import("std");
const config_mod = @import("config.zig");
const ModelConfig = config_mod.ModelConfig;
const dispatch = @import("../kernels/dispatch.zig");
const KernelType = dispatch.KernelType;
const tl1 = @import("../kernels/tl1.zig");
const rmsnorm = @import("../ops/rmsnorm.zig");
const softmax_mod = @import("../ops/softmax.zig");
const silu_mod = @import("../ops/silu.zig");
const rope_mod = @import("../ops/rope.zig");
const attention_mod = @import("../ops/attention.zig");
const kv_cache_mod = @import("../core/kv_cache.zig");

/// Comptime-specialized scratch buffers for one transformer layer.
/// All sizes derived from ModelConfig. No runtime allocation.
pub fn ScratchBuffers(comptime cfg: ModelConfig) type {
    comptime cfg.validate();

    return struct {
        norm_out: [cfg.hidden_size]i8 = std.mem.zeroes([cfg.hidden_size]i8),
        q_out: [cfg.hidden_size]i8 = std.mem.zeroes([cfg.hidden_size]i8),
        k_out: [cfg.hidden_size]i8 = std.mem.zeroes([cfg.hidden_size]i8),
        v_out: [cfg.hidden_size]i8 = std.mem.zeroes([cfg.hidden_size]i8),
        attn_out: [cfg.hidden_size]i8 = std.mem.zeroes([cfg.hidden_size]i8),
        ffn_gate: [cfg.intermediate_size]i8 = std.mem.zeroes([cfg.intermediate_size]i8),
        ffn_up: [cfg.intermediate_size]i8 = std.mem.zeroes([cfg.intermediate_size]i8),
        proj_out: [cfg.hidden_size]i8 = std.mem.zeroes([cfg.hidden_size]i8),
        residual: [cfg.hidden_size]i8 = std.mem.zeroes([cfg.hidden_size]i8),

        // Attention working buffers.
        scores: [cfg.max_seq_len]i32 = std.mem.zeroes([cfg.max_seq_len]i32),
        probs: [cfg.max_seq_len]u16 = std.mem.zeroes([cfg.max_seq_len]u16),

        // i32 accumulators for matmul output before quantization.
        accum: [cfg.hidden_size]i32 = std.mem.zeroes([cfg.hidden_size]i32),
        accum_large: [cfg.intermediate_size]i32 = std.mem.zeroes([cfg.intermediate_size]i32),

        // Per-head key/value buffers for attention.
        key_buf: [cfg.head_dim]i8 = std.mem.zeroes([cfg.head_dim]i8),
        val_buf: [cfg.head_dim]i8 = std.mem.zeroes([cfg.head_dim]i8),
    };
}

/// Comptime-specialized transformer layer.
/// Wires ternary matmul kernels, RMSNorm, RoPE, attention, and SwiGLU FFN
/// into the standard pre-norm transformer pipeline.
///
/// Weight pointers are zero-copy references into .zink file data.
pub fn TransformerLayer(comptime cfg: ModelConfig, comptime kernel: KernelType) type {
    comptime cfg.validate();

    const hidden = cfg.hidden_size;
    const inter = cfg.intermediate_size;
    const max_seq_len = cfg.max_seq_len;
    const heads_per_kv = cfg.num_heads / cfg.num_kv_heads;

    // TL1 LUT buffer size: 9 entries per pair, hidden_size/2 pairs, i16.
    // Ref: Bitnet.cpp ACL 2025, Section 3.3 (TL1 design).
    const tl1_lut_size = 9 * (hidden / 2);
    // Same for intermediate_size (used for down projection LUT).
    const tl1_lut_size_inter = 9 * (inter / 2);

    return struct {
        const Self = @This();

        // Ternary projection weights [out_dim][in_dim/4] packed.
        wq: *const [hidden][hidden / 4]u8,
        wk: *const [hidden][hidden / 4]u8,
        wv: *const [hidden][hidden / 4]u8,
        wo: *const [hidden][hidden / 4]u8,
        w_gate: *const [inter][hidden / 4]u8,
        w_up: *const [inter][hidden / 4]u8,
        w_down: *const [hidden][inter / 4]u8,

        // Per-tensor f32 scales for requantizing i32 accumulators to i8.
        sq: f32,
        sk: f32,
        sv: f32,
        so: f32,
        s_gate: f32,
        s_up: f32,
        s_down: f32,

        // RMSNorm weight arrays (Q0.7 stored as i8).
        rms_attn_weight: *const [hidden]i8,
        rms_ffn_weight: *const [hidden]i8,

        /// Run one transformer layer step.
        ///
        /// Pipeline:
        /// 1. RMSNorm → QKV projections → RoPE → KV-cache → attention
        /// 2. Residual add
        /// 3. RMSNorm → FFN (SwiGLU) → residual add
        pub fn forward(
            self: *const Self,
            input: *const [hidden]i8,
            output: *[hidden]i8,
            scratch: *ScratchBuffers(cfg),
            kv_cache: *kv_cache_mod.KvCache(cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len, cfg.kv_group_size),
            position: usize,
        ) void {
            // Pre-attention RMSNorm.
            rmsnorm.rmsnorm(input, self.rms_attn_weight, &scratch.norm_out, 4);

            // TL1 path: build LUT once from norm_out, reuse for Q/K/V/O projections.
            // LUT construction is O(hidden_size) but eliminates all multiplications
            // in the subsequent 4 matmuls, which are O(hidden_size²).
            // Ref: Bitnet.cpp ACL 2025, Section 3.3.
            var tl1_lut_hidden: [tl1_lut_size]i16 = undefined;
            if (kernel == .tl1) {
                dispatch.buildTL1Lut(hidden, &scratch.norm_out, &tl1_lut_hidden);
            }

            const lut_hidden: ?[]const i16 = if (kernel == .tl1) &tl1_lut_hidden else null;

            // QKV projections: norm_out × W_q/k/v → i32 accumulators → quantize to i8.
            dispatch.ternaryMatVecDispatch(kernel, hidden, hidden, self.wq, &scratch.norm_out, &scratch.accum, lut_hidden);
            quantizeI32ToI8(&scratch.accum, self.sq, &scratch.q_out);

            dispatch.ternaryMatVecDispatch(kernel, hidden, hidden, self.wk, &scratch.norm_out, &scratch.accum, lut_hidden);
            quantizeI32ToI8(&scratch.accum, self.sk, &scratch.k_out);

            dispatch.ternaryMatVecDispatch(kernel, hidden, hidden, self.wv, &scratch.norm_out, &scratch.accum, lut_hidden);
            quantizeI32ToI8(&scratch.accum, self.sv, &scratch.v_out);

            // RoPE on Q and K per-head.
            for (0..cfg.num_heads) |h| {
                const q_head = scratch.q_out[h * cfg.head_dim ..][0..cfg.head_dim];
                rope_mod.applyRoPE(cfg.head_dim, cfg.rope_theta, q_head, position);
            }
            for (0..cfg.num_kv_heads) |h| {
                const k_head = scratch.k_out[h * cfg.head_dim ..][0..cfg.head_dim];
                rope_mod.applyRoPE(cfg.head_dim, cfg.rope_theta, k_head, position);
            }

            // Append K, V to cache.
            const k_heads: *const [cfg.num_kv_heads][cfg.head_dim]i8 = @ptrCast(&scratch.k_out);
            const v_heads: *const [cfg.num_kv_heads][cfg.head_dim]i8 = @ptrCast(&scratch.v_out);
            kv_cache.append(k_heads, v_heads);

            // Multi-head attention.
            // Sliding window limits which positions participate. The ring buffer
            // stores all validLen() positions, but we only attend to the most
            // recent window_len entries (or all of them if no window is set).
            const seq_len = kv_cache.windowLen(cfg.sliding_window_size);
            @memset(&scratch.attn_out, 0);

            // When using a ring buffer with sliding window, we need the start
            // position within the ring buffer. The most recent `seq_len` entries
            // are at indices [(write_pos - seq_len) % max_seq_len .. write_pos).
            const ring_start = if (seq_len <= kv_cache.write_pos)
                kv_cache.write_pos - seq_len
            else
                max_seq_len + kv_cache.write_pos - seq_len;

            for (0..cfg.num_heads) |h| {
                const kv_head = h / heads_per_kv;
                const q_head: []const i8 = scratch.q_out[h * cfg.head_dim ..][0..cfg.head_dim];

                // Compute scores against windowed cached keys.
                for (0..seq_len) |i| {
                    const ring_pos = (ring_start + i) % max_seq_len;
                    kv_cache.getKey(kv_head, ring_pos, &scratch.key_buf);
                    scratch.scores[i] = attention_mod.dotProduct(q_head, &scratch.key_buf);
                }

                // Scale, mask, softmax.
                const recip = comptime attention_mod.headDimSqrtRecip(cfg.head_dim);
                attention_mod.scaleScores(&scratch.scores, seq_len, recip);
                // Causal mask: within the window, the current token is at
                // position (seq_len - 1). Positions beyond that are masked.
                attention_mod.applyCausalMask(&scratch.scores, seq_len, seq_len - 1);

                // Convert i32 scores to i8 for softmax LUT.
                var scores_i8: [cfg.max_seq_len]i8 = undefined;
                for (0..seq_len) |i| {
                    scores_i8[i] = @intCast(std.math.clamp(scratch.scores[i], -128, 127));
                }
                softmax_mod.softmax(scores_i8[0..seq_len], scratch.probs[0..seq_len], &softmax_mod.default_exp_lut);

                // Weighted sum of values.
                const attn_head: *[cfg.head_dim]i8 = scratch.attn_out[h * cfg.head_dim ..][0..cfg.head_dim];
                var head_acc: [cfg.head_dim]i32 = std.mem.zeroes([cfg.head_dim]i32);
                for (0..seq_len) |i| {
                    const ring_pos = (ring_start + i) % max_seq_len;
                    kv_cache.getValue(kv_head, ring_pos, &scratch.val_buf);
                    const prob: i32 = scratch.probs[i];
                    for (0..cfg.head_dim) |d| {
                        head_acc[d] += prob * @as(i32, scratch.val_buf[d]);
                    }
                }
                for (0..cfg.head_dim) |d| {
                    attn_head[d] = @intCast(std.math.clamp(@divTrunc(head_acc[d], 65536), -128, 127));
                }
            }

            // Rebuild LUT from attn_out for the O projection. The spec's "reuse LUT
            // across Q/K/V/O" applies within the Q/K/V group (same norm_out input),
            // not here: the O projection's input is attn_out, a different vector.
            if (kernel == .tl1) {
                dispatch.buildTL1Lut(hidden, &scratch.attn_out, &tl1_lut_hidden);
            }
            const lut_attn: ?[]const i16 = if (kernel == .tl1) &tl1_lut_hidden else null;
            dispatch.ternaryMatVecDispatch(kernel, hidden, hidden, self.wo, &scratch.attn_out, &scratch.accum, lut_attn);
            quantizeI32ToI8(&scratch.accum, self.so, &scratch.proj_out);

            // Residual connection: input + proj_out.
            addResidual(input, &scratch.proj_out, &scratch.residual);

            // Pre-FFN RMSNorm.
            rmsnorm.rmsnorm(&scratch.residual, self.rms_ffn_weight, &scratch.norm_out, 4);

            // FFN: SwiGLU.
            // Rebuild LUT from norm_out for FFN projections (gate, up share same input).
            if (kernel == .tl1) {
                dispatch.buildTL1Lut(hidden, &scratch.norm_out, &tl1_lut_hidden);
            }
            const lut_ffn: ?[]const i16 = if (kernel == .tl1) &tl1_lut_hidden else null;

            // gate = matmul(norm_out, w_gate)
            dispatch.ternaryMatVecDispatch(kernel, inter, hidden, self.w_gate, &scratch.norm_out, &scratch.accum_large, lut_ffn);
            quantizeI32ToI8Large(inter, &scratch.accum_large, self.s_gate, &scratch.ffn_gate);

            // up = matmul(norm_out, w_up)
            dispatch.ternaryMatVecDispatch(kernel, inter, hidden, self.w_up, &scratch.norm_out, &scratch.accum_large, lut_ffn);
            quantizeI32ToI8Large(inter, &scratch.accum_large, self.s_up, &scratch.ffn_up);

            // SwiGLU: silu(gate) * up
            silu_mod.swiglu(&scratch.ffn_gate, &scratch.ffn_up, &scratch.ffn_gate);

            // down = matmul(ffn_intermediate, w_down)
            // Rebuild LUT from ffn_gate since input to down projection is different.
            var tl1_lut_inter: [tl1_lut_size_inter]i16 = undefined;
            if (kernel == .tl1) {
                dispatch.buildTL1Lut(inter, &scratch.ffn_gate, &tl1_lut_inter);
            }
            const lut_down: ?[]const i16 = if (kernel == .tl1) &tl1_lut_inter else null;
            dispatch.ternaryMatVecDispatch(kernel, hidden, inter, self.w_down, &scratch.ffn_gate, &scratch.accum, lut_down);
            quantizeI32ToI8(&scratch.accum, self.s_down, &scratch.proj_out);

            // Final residual: residual + proj_out → output.
            addResidual(&scratch.residual, &scratch.proj_out, output);
        }

    };
}

/// Requantize i32 accumulators to i8 using per-tensor scale.
/// Per-tensor int8 activation quantization matches BitNet b1.58 training scheme.
fn quantizeI32ToI8(accum: anytype, scale: f32, output: anytype) void {
    const len = accum.len;
    if (scale == 0) {
        @memset(output, 0);
        return;
    }
    const inv_scale = 1.0 / scale;
    for (0..len) |i| {
        const val: f32 = @floatFromInt(accum[i]);
        const scaled = val * inv_scale;
        output[i] = @intCast(std.math.clamp(@as(i32, @intFromFloat(scaled)), -128, 127));
    }
}

fn quantizeI32ToI8Large(comptime size: usize, accum: *const [size]i32, scale: f32, output: *[size]i8) void {
    if (scale == 0) {
        @memset(output, 0);
        return;
    }
    const inv_scale = 1.0 / scale;
    for (0..size) |i| {
        const val: f32 = @floatFromInt(accum[i]);
        const scaled = val * inv_scale;
        output[i] = @intCast(std.math.clamp(@as(i32, @intFromFloat(scaled)), -128, 127));
    }
}

/// Element-wise residual addition: out[i] = clamp(a[i] + b[i], -128, 127).
fn addResidual(a: anytype, b: anytype, output: anytype) void {
    for (0..a.len) |i| {
        const sum: i16 = @as(i16, a[i]) + @as(i16, b[i]);
        output[i] = @intCast(std.math.clamp(sum, -128, 127));
    }
}

// Re-export dotProduct from attention for use in forward().
// The attention module's dotProduct is not pub, so we reference it via
// the module-level computeScores indirectly. Instead, inline the dot here.
fn dotProductInline(a: []const i8, b: []const i8) i32 {
    std.debug.assert(a.len == b.len);
    var acc: i32 = 0;
    for (0..a.len) |i| {
        acc += @as(i32, a[i]) * @as(i32, b[i]);
    }
    return acc;
}

test "ScratchBuffers size is comptime-known" {
    const cfg = config_mod.bitnet_test;
    const Scratch = ScratchBuffers(cfg);
    // Verify the struct can be instantiated.
    const size = @sizeOf(Scratch);
    try std.testing.expect(size > 0);
}

test "quantizeI32ToI8 basic" {
    var accum = [_]i32{ 1000, -500, 250, 0 };
    var output: [4]i8 = undefined;
    quantizeI32ToI8(&accum, 10.0, &output);

    // 1000 / 10 = 100, -500/10 = -50, 250/10 = 25, 0
    try std.testing.expectEqual(@as(i8, 100), output[0]);
    try std.testing.expectEqual(@as(i8, -50), output[1]);
    try std.testing.expectEqual(@as(i8, 25), output[2]);
    try std.testing.expectEqual(@as(i8, 0), output[3]);
}

test "quantizeI32ToI8 clamps to range" {
    var accum = [_]i32{ 5000, -5000 };
    var output: [2]i8 = undefined;
    quantizeI32ToI8(&accum, 10.0, &output);
    try std.testing.expectEqual(@as(i8, 127), output[0]);
    try std.testing.expectEqual(@as(i8, -128), output[1]);
}

test "addResidual basic" {
    const a = [_]i8{ 50, -50, 100, -100 };
    const b = [_]i8{ 20, -20, 30, -30 };
    var out: [4]i8 = undefined;
    addResidual(&a, &b, &out);
    try std.testing.expectEqual(@as(i8, 70), out[0]);
    try std.testing.expectEqual(@as(i8, -70), out[1]);
    try std.testing.expectEqual(@as(i8, 127), out[2]); // clamped
    try std.testing.expectEqual(@as(i8, -128), out[3]); // clamped
}

test "TransformerLayer type instantiation" {
    const cfg = config_mod.bitnet_test;
    const Layer = TransformerLayer(cfg, .i2s_generic);
    // Verify the type exists and has the expected fields.
    try std.testing.expect(@sizeOf(Layer) > 0);
}
