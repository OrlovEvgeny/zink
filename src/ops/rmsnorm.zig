const std = @import("std");

/// Fixed-point RMSNorm.
/// y[i] = x[i] * weight[i] / rms(x)
/// where rms(x) = sqrt(mean(x²))
///
/// Uses integer arithmetic throughout. The inverse sqrt is computed
/// via Newton-Raphson in Q16.16 fixed-point.

/// Q16.16 fixed-point type for intermediate precision.
const Q16_16 = i32;
const FRAC_BITS = 16;
const ONE: Q16_16 = 1 << FRAC_BITS;

/// Integer Newton-Raphson for 1/sqrt(x) in Q16.16.
/// x_fp is in Q16.16 format. Returns 1/sqrt(x) in Q16.16.
/// 6 iterations suffice for 16-bit fractional precision.
fn invSqrtNewtonRaphson(x_fp: Q16_16) Q16_16 {
    if (x_fp <= 0) return ONE;

    // Initial estimate: find leading bit, approximate 1/sqrt.
    const x_u: u32 = @intCast(x_fp);
    const leading = @clz(x_u);
    // Rough initial: 1/sqrt(2^k) ≈ 2^(-k/2).
    const half_shift = (31 - leading) / 2;
    var guess: i64 = @as(i64, ONE) >> @intCast(half_shift);
    if (guess == 0) guess = 1;

    // Newton-Raphson: g' = g * (3 - x * g²) / 2
    const x_64: i64 = x_fp;
    for (0..6) |_| {
        const g2 = @divTrunc(guess * guess, ONE);
        const xg2 = @divTrunc(x_64 * g2, ONE);
        const three_minus = 3 * ONE - xg2;
        guess = @divTrunc(guess * three_minus, 2 * ONE);
    }

    return @intCast(std.math.clamp(guess, 0, std.math.maxInt(i32)));
}

/// Compute RMSNorm over int8 input, producing int8 output.
/// weights are per-element scaling factors in Q0.7 (i8 representing [-1, 1)).
/// epsilon_shift: log2(epsilon * N), added to prevent division by zero.
pub fn rmsnorm(
    input: []const i8,
    weights: []const i8,
    output: []i8,
    comptime epsilon_shift: u5,
) void {
    std.debug.assert(input.len == weights.len);
    std.debug.assert(input.len == output.len);
    const n = input.len;

    // Compute sum of squares in i32 to avoid overflow.
    var sum_sq: i64 = 0;
    for (input) |x| {
        const x32: i64 = x;
        sum_sq += x32 * x32;
    }

    // mean_sq in Q16.16: (sum_sq << 16) / N + epsilon.
    const n_i64: i64 = @intCast(n);
    const mean_sq_fp: Q16_16 = @intCast(@divTrunc(sum_sq << FRAC_BITS, n_i64) + (@as(i64, 1) << epsilon_shift));

    // 1/sqrt(mean_sq) in Q16.16.
    const inv_rms = invSqrtNewtonRaphson(mean_sq_fp);

    // y[i] = clamp(x[i] * weight[i] * inv_rms >> 16, -128, 127)
    for (0..n) |i| {
        const x: i32 = input[i];
        const w: i32 = weights[i];
        // x * inv_rms is in Q16.16, then multiply by weight and shift back.
        const normalized = @divTrunc(x * inv_rms, ONE);
        const scaled = @divTrunc(normalized * w, 128); // weights are Q0.7
        output[i] = @intCast(std.math.clamp(scaled, -128, 127));
    }
}

test "rmsnorm basic" {
    const input = [_]i8{ 64, 64, 64, 64 };
    const weights = [_]i8{ 127, 127, 127, 127 }; // ≈1.0 in Q0.7
    var output: [4]i8 = undefined;
    rmsnorm(&input, &weights, &output, 4);

    // All inputs are equal, so RMSNorm should produce values close to the weights.
    // rms = 64, normalized = 1.0, scaled ≈ 127/128 ≈ 0.99 → ~63 (x * w / 128)
    // Exact value depends on fixed-point precision.
    for (output) |v| {
        try std.testing.expect(v > 50 and v < 127);
    }
}

test "rmsnorm vs f32 reference bounded error" {
    const input = [_]i8{ 100, -50, 25, -12 };
    const weights = [_]i8{ 127, 127, 127, 127 };
    var output: [4]i8 = undefined;
    rmsnorm(&input, &weights, &output, 4);

    // f32 reference.
    const eps: f32 = @as(f32, @floatFromInt(@as(i32, 1) << 4)) / @as(f32, 65536.0);
    var sum_sq: f32 = 0;
    for (input) |x| {
        const xf: f32 = @floatFromInt(x);
        sum_sq += xf * xf;
    }
    const rms = @sqrt(sum_sq / 4.0 + eps);

    for (0..4) |i| {
        const expected: f32 = @as(f32, @floatFromInt(input[i])) / rms * (@as(f32, @floatFromInt(weights[i])) / 128.0);
        const expected_i8: i8 = @intFromFloat(std.math.clamp(expected, -128.0, 127.0));
        const err = @as(i32, output[i]) - @as(i32, expected_i8);
        const abs_err: u32 = @intCast(if (err < 0) -err else err);
        // Allow some fixed-point quantization error.
        try std.testing.expect(abs_err <= 5);
    }
}
