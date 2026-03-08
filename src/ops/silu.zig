const std = @import("std");

/// Fixed-point SiLU (Swish) activation: silu(x) = x * sigmoid(x)
/// Uses a comptime-generated LUT for sigmoid, then multiplies.
///
/// SiLU is used in the SwiGLU FFN variant of BitNet/LLaMA architectures.

/// Comptime-generated sigmoid LUT.
/// Maps i8 input [-128, 127] → Q0.8 sigmoid value [0, 255].
fn generateSigmoidLut() [256]u8 {
    var lut: [256]u8 = undefined;
    for (0..256) |i| {
        const x: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(i)))));
        // Scale x to a useful range. x/32 maps [-128,127] to [-4, 4].
        const sigmoid = 1.0 / (1.0 + @exp(-x / 32.0));
        lut[i] = @intFromFloat(sigmoid * 255.0);
    }
    return lut;
}

pub const sigmoid_lut = generateSigmoidLut();

/// Apply SiLU activation in-place on int8 values.
/// silu(x) = x * sigmoid(x)
/// Result is clamped back to i8 range.
pub fn silu(input: []const i8, output: []i8) void {
    std.debug.assert(input.len == output.len);
    for (0..input.len) |i| {
        const x: i16 = input[i];
        const idx: u8 = @bitCast(input[i]);
        const sig: i16 = sigmoid_lut[idx]; // [0, 255] ≈ [0, 1) in Q0.8

        // x * sigmoid(x) / 256 to scale back from Q0.8.
        const result = @divTrunc(x * sig, 256);
        output[i] = @intCast(std.math.clamp(result, -128, 127));
    }
}

/// SwiGLU gate: output = silu(gate) * up
/// Both gate and up are int8 vectors of the same length.
pub fn swiglu(gate: []const i8, up: []const i8, output: []i8) void {
    std.debug.assert(gate.len == up.len);
    std.debug.assert(gate.len == output.len);

    for (0..gate.len) |i| {
        const idx: u8 = @bitCast(gate[i]);
        const sig: i16 = sigmoid_lut[idx];

        // silu(gate) * up / 256
        const gate_val: i16 = gate[i];
        const up_val: i16 = up[i];
        const activated = @divTrunc(gate_val * sig, 256);
        // Divide by 128 (not 256): activated is already in i8 range after the
        // /256 step. This second division scales the i8×i8 product back to i8:
        // max |activated| × max |up_val| = 127×127 = 16129, /128 = 126 → fits i8.
        const result = @divTrunc(activated * up_val, 128);
        output[i] = @intCast(std.math.clamp(result, -128, 127));
    }
}

test "silu positive values" {
    const input = [_]i8{ 64, 127, 32 };
    var output: [3]i8 = undefined;
    silu(&input, &output);

    // SiLU(x) ≈ x for large positive x, but scaled down by sigmoid.
    for (output) |v| {
        try std.testing.expect(v > 0);
    }
}

test "silu zero" {
    const input = [_]i8{0};
    var output: [1]i8 = undefined;
    silu(&input, &output);
    // silu(0) = 0 * 0.5 = 0
    try std.testing.expectEqual(@as(i8, 0), output[0]);
}

test "silu negative values are attenuated" {
    const input = [_]i8{ -64, -127 };
    var output: [2]i8 = undefined;
    silu(&input, &output);

    // SiLU(-x) → small negative value (sigmoid suppresses magnitude).
    for (0..2) |i| {
        const abs_out: u8 = @intCast(if (output[i] < 0) -@as(i16, output[i]) else output[i]);
        const abs_in: u8 = @intCast(if (input[i] < 0) -@as(i16, input[i]) else input[i]);
        try std.testing.expect(abs_out <= abs_in);
    }
}

test "silu vs f32 reference bounded error" {
    const input = [_]i8{ 0, 32, 64, 96, 127, -32, -64, -96 };
    var output: [8]i8 = undefined;
    silu(&input, &output);

    for (0..8) |i| {
        const x: f32 = @floatFromInt(input[i]);
        const sig = 1.0 / (1.0 + @exp(-x / 32.0));
        const expected_f = x * sig / 256.0 * 256.0; // scale matches LUT
        const expected: i8 = @intFromFloat(std.math.clamp(expected_f, -128.0, 127.0));
        const err = @as(i32, output[i]) - @as(i32, expected);
        const abs_err: u32 = @intCast(if (err < 0) -err else err);
        try std.testing.expect(abs_err <= 3);
    }
}
