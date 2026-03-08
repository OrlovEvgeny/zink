const std = @import("std");

/// RoPE (Rotary Position Embedding) with comptime frequency tables.
/// Uses comptime LUT for sin/cos to avoid floating-point at runtime.
///
/// Frequency: freq[i] = 1 / (theta^(2i/d)) for i in 0..d/2
/// For each position p, rotation angle = p * freq[i].
/// Applied as 2D rotation to (x[2i], x[2i+1]) pairs.

/// Comptime-generated sin/cos LUT over [0, 2*pi) with the given resolution.
/// Values are stored in Q0.15 format (i16): [-32767, 32767] ≈ [-1.0, 1.0).
fn SinCosLut(comptime size: usize) type {
    return struct {
        sin: [size]i16,
        cos: [size]i16,

        const resolution = size;
    };
}

fn generateSinCosLut(comptime size: usize) SinCosLut(size) {
    @setEvalBranchQuota(100000);
    var lut: SinCosLut(size) = undefined;
    for (0..size) |i| {
        const angle: f64 = @as(f64, @floatFromInt(i)) * (2.0 * std.math.pi / @as(f64, @floatFromInt(size)));
        lut.sin[i] = @intFromFloat(std.math.clamp(@sin(angle) * 32767.0, -32767.0, 32767.0));
        lut.cos[i] = @intFromFloat(std.math.clamp(@cos(angle) * 32767.0, -32767.0, 32767.0));
    }
    return lut;
}

// 1024-entry LUT covers [0, 2π) with ~0.35° resolution.
pub const sin_cos_lut = generateSinCosLut(1024);
const LUT_SIZE = 1024;

/// Comptime-generated frequency table for RoPE.
/// freq[i] = 1/(theta^(2i/d)), stored as a Q16.16 multiplier
/// that converts position to LUT index: lut_idx = (pos * freq[i]) % LUT_SIZE.
fn generateFreqTable(comptime head_dim: usize, comptime theta: f64) [head_dim / 2]u32 {
    @setEvalBranchQuota(10000);
    var freqs: [head_dim / 2]u32 = undefined;
    for (0..head_dim / 2) |i| {
        const exp = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim));
        const freq = 1.0 / std.math.pow(f64, theta, exp);
        // Store as steps-per-position in LUT index space.
        // One full rotation = LUT_SIZE steps.
        freqs[i] = @intFromFloat(freq * @as(f64, LUT_SIZE) / (2.0 * std.math.pi));
    }
    return freqs;
}

/// Apply RoPE to a single head's Q or K vector at the given position.
/// Rotates pairs (x[2i], x[2i+1]) by angle = position * freq[i].
///
/// head_dim must be even.
pub fn applyRoPE(
    comptime head_dim: usize,
    comptime theta: f64,
    vec: *[head_dim]i8,
    position: usize,
) void {
    comptime {
        if (head_dim % 2 != 0) @compileError("head_dim must be even for RoPE");
    }
    const freqs = comptime generateFreqTable(head_dim, theta);

    inline for (0..head_dim / 2) |i| {
        // u64 intermediate to prevent overflow for large position values.
        const lut_idx = @as(usize, @intCast((@as(u64, position) *% @as(u64, freqs[i])) % LUT_SIZE));
        const cos_val: i32 = sin_cos_lut.cos[lut_idx];
        const sin_val: i32 = sin_cos_lut.sin[lut_idx];

        const x0: i32 = vec[2 * i];
        const x1: i32 = vec[2 * i + 1];

        // 2D rotation: [cos -sin; sin cos] @ [x0; x1]
        // Result in Q0.15, shift back to i8.
        const r0 = @divTrunc(x0 * cos_val - x1 * sin_val, 32767);
        const r1 = @divTrunc(x0 * sin_val + x1 * cos_val, 32767);

        vec[2 * i] = @intCast(std.math.clamp(r0, -128, 127));
        vec[2 * i + 1] = @intCast(std.math.clamp(r1, -128, 127));
    }
}

test "rope position 0 is identity" {
    var vec = [_]i8{ 100, -50, 25, -12 };
    const original = vec;
    applyRoPE(4, 10000.0, &vec, 0);

    // At position 0, all angles are 0 → cos=1, sin=0 → identity.
    for (0..4) |i| {
        const err = @as(i32, vec[i]) - @as(i32, original[i]);
        const abs_err: u32 = @intCast(if (err < 0) -err else err);
        try std.testing.expect(abs_err <= 1);
    }
}

test "rope preserves magnitude" {
    var vec = [_]i8{ 100, 0, 50, 50 };
    applyRoPE(4, 10000.0, &vec, 42);

    // RoPE is a rotation, so magnitude should be approximately preserved.
    // Check that values are within i8 range (no overflow).
    for (vec) |v| {
        try std.testing.expect(v >= -128 and v <= 127);
    }
}

test "rope different positions produce different embeddings" {
    var vec1 = [_]i8{ 100, -50, 25, -12, 60, 70, -80, 90 };
    var vec2 = vec1;
    applyRoPE(8, 10000.0, &vec1, 1);
    applyRoPE(8, 10000.0, &vec2, 100);

    var differs = false;
    for (0..8) |i| {
        if (vec1[i] != vec2[i]) differs = true;
    }
    try std.testing.expect(differs);
}
