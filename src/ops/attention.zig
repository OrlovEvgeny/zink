const std = @import("std");

/// Attention computation for a single head.
/// Computes: attn_output = softmax(Q @ K^T / sqrt(d)) @ V
///
/// All arithmetic is in fixed-point (i32 accumulators, i8 values).
/// Uses the KV-cache for K and V vectors.

/// Compute dot product between two i8 vectors.
pub fn dotProduct(a: []const i8, b: []const i8) i32 {
    std.debug.assert(a.len == b.len);
    var acc: i32 = 0;
    for (0..a.len) |i| {
        acc += @as(i32, a[i]) * @as(i32, b[i]);
    }
    return acc;
}

/// Compute attention scores for a single query against all cached keys.
/// scores[i] = Q · K[i] for i in 0..seq_len.
/// Caller provides the dequantized key vectors.
pub fn computeScores(
    query: []const i8,
    keys: []const []const i8,
    scores: []i32,
    seq_len: usize,
) void {
    for (0..seq_len) |pos| {
        scores[pos] = dotProduct(query, keys[pos]);
    }
}

/// Scale scores by 1/sqrt(head_dim) in fixed-point.
/// head_dim_sqrt_recip is 1/sqrt(d) in Q0.16 format.
pub fn scaleScores(
    scores: []i32,
    seq_len: usize,
    comptime head_dim_sqrt_recip_q16: u32,
) void {
    for (0..seq_len) |i| {
        // score * (1/sqrt(d)) in Q0.16, shift back.
        const scaled: i64 = @as(i64, scores[i]) * @as(i64, head_dim_sqrt_recip_q16);
        scores[i] = @intCast(@divTrunc(scaled, 65536));
    }
}

/// Apply causal mask: set scores beyond the current position to minimum.
pub fn applyCausalMask(
    scores: []i32,
    seq_len: usize,
    current_pos: usize,
) void {
    for (current_pos + 1..seq_len) |i| {
        scores[i] = std.math.minInt(i32);
    }
}

/// Weighted sum of value vectors using attention probabilities.
/// probs: Q0.16 probabilities from softmax.
/// values: dequantized V vectors per position.
/// output: resulting attention vector.
pub fn weightedSum(
    probs: []const u16,
    values: []const []const i8,
    output: []i8,
    head_dim: usize,
    seq_len: usize,
) void {
    for (0..head_dim) |d| {
        var acc: i32 = 0;
        for (0..seq_len) |pos| {
            acc += @as(i32, probs[pos]) * @as(i32, values[pos][d]);
        }
        // probs are Q0.16, so divide by 65536 to get back to i8 range.
        const result = @divTrunc(acc, 65536);
        output[d] = @intCast(std.math.clamp(result, -128, 127));
    }
}

/// Comptime: compute 1/sqrt(head_dim) in Q0.16 format.
pub fn headDimSqrtRecip(comptime head_dim: usize) u32 {
    const val: f64 = 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));
    return @intFromFloat(val * 65536.0);
}

test "dot product known values" {
    const a = [_]i8{ 10, 20, 30 };
    const b = [_]i8{ 1, 2, 3 };
    const result = dotProduct(&a, &b);
    // 10*1 + 20*2 + 30*3 = 10 + 40 + 90 = 140
    try std.testing.expectEqual(@as(i32, 140), result);
}

test "headDimSqrtRecip computation" {
    const recip64 = comptime headDimSqrtRecip(64);
    // 1/sqrt(64) = 0.125, in Q0.16 = 8192
    try std.testing.expectEqual(@as(u32, 8192), recip64);
}

test "causal mask zeroes future positions" {
    var scores = [_]i32{ 100, 200, 300, 400 };
    applyCausalMask(&scores, 4, 1);
    try std.testing.expectEqual(@as(i32, 100), scores[0]);
    try std.testing.expectEqual(@as(i32, 200), scores[1]);
    try std.testing.expectEqual(std.math.minInt(i32), scores[2]);
    try std.testing.expectEqual(std.math.minInt(i32), scores[3]);
}
