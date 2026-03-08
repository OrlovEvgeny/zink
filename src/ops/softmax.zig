const std = @import("std");

/// Fixed-point softmax with comptime-generated exp lookup table.
/// Uses Q0.16 format for probabilities (0 to 65535 ≈ 0.0 to ~1.0).

/// Comptime-generated exp LUT for softmax.
/// Maps i8 input [-128, 127] → Q0.16 exp value.
/// exp(x/scale) where scale normalizes the input range.
fn generateExpLut(comptime scale: f32) [256]u16 {
    var lut: [256]u16 = undefined;
    for (0..256) |i| {
        const x: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(i)))));
        const exp_val = @exp(x / scale);
        // Clamp to u16 range. The relative magnitudes are what matter.
        const scaled = exp_val * 256.0;
        lut[i] = @intFromFloat(std.math.clamp(scaled, 0.0, 65535.0));
    }
    return lut;
}

/// Default scale: maps i8 range to a reasonable exp range.
/// scale=32 means exp(-4) to exp(~4), avoiding extreme saturation.
pub const default_exp_lut = generateExpLut(32.0);

/// Compute softmax over int8 logits, producing Q0.16 probabilities.
/// Subtracts max for numerical stability before LUT lookup.
pub fn softmax(
    logits: []const i8,
    output: []u16,
    comptime exp_lut: *const [256]u16,
) void {
    std.debug.assert(logits.len == output.len);
    const n = logits.len;
    if (n == 0) return;

    // Find max for stability.
    var max_val: i8 = logits[0];
    for (logits[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    // Accumulate exp(x - max) via LUT.
    var sum: u32 = 0;
    for (0..n) |i| {
        const shifted: i16 = @as(i16, logits[i]) - @as(i16, max_val);
        // Clamp to i8 range for LUT index.
        const idx: u8 = @bitCast(@as(i8, @intCast(std.math.clamp(shifted, -128, 127))));
        const exp_val = exp_lut[idx];
        output[i] = exp_val;
        sum += exp_val;
    }

    if (sum == 0) {
        // Uniform distribution fallback.
        const uniform: u16 = @intCast(65535 / n);
        for (output) |*v| v.* = uniform;
        return;
    }

    // Normalize: output[i] = (output[i] << 16) / sum, capped at u16.
    for (output) |*v| {
        const normalized: u32 = (@as(u32, v.*) << 16) / sum;
        v.* = @intCast(@min(normalized, 65535));
    }
}

/// Convenience: find argmax of softmax output.
pub fn argmax(probs: []const u16) usize {
    var best_idx: usize = 0;
    var best_val: u16 = 0;
    for (probs, 0..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    return best_idx;
}

test "softmax max element gets highest probability" {
    const logits = [_]i8{ 10, 50, 20, -30 };
    var output: [4]u16 = undefined;
    softmax(&logits, &output, &default_exp_lut);

    // Element at index 1 (value 50) should have highest probability.
    try std.testing.expectEqual(@as(usize, 1), argmax(&output));
}

test "softmax probabilities sum to ~1.0 (Q0.16)" {
    const logits = [_]i8{ 10, 20, 30, 40 };
    var output: [4]u16 = undefined;
    softmax(&logits, &output, &default_exp_lut);

    var sum: u32 = 0;
    for (output) |v| sum += v;

    // In Q0.16, 1.0 = 65536. Sum should be close.
    const diff: i32 = @as(i32, @intCast(sum)) - 65536;
    const abs_diff: u32 = @intCast(if (diff < 0) -diff else diff);
    try std.testing.expect(abs_diff < 1000);
}

test "softmax uniform input" {
    const logits = [_]i8{ 42, 42, 42, 42 };
    var output: [4]u16 = undefined;
    softmax(&logits, &output, &default_exp_lut);

    // All equal input → approximately uniform output.
    for (output) |v| {
        const expected: i32 = 65536 / 4;
        const diff = @as(i32, v) - expected;
        const abs_diff: u32 = @intCast(if (diff < 0) -diff else diff);
        try std.testing.expect(abs_diff < 2000);
    }
}
