const std = @import("std");
const reference = @import("reference.zig");

/// FATNN ternary decomposition kernel.
/// Decomposes ternary weights into two binary vectors (alpha, beta) such that:
///   w[i] = alpha[i] - beta[i]  where alpha, beta ∈ {0, 1}
///
/// This reduces ternary dot product to:
///   dot(w, x) = dot(alpha, x) - dot(beta, x)
///
/// Each binary dot product is computed via bitwise AND + popcount,
/// reducing O(4N) multiply-accumulate to O(2N) bitwise ops.
/// Ref: FATNN, ICCV 2021.

/// Decompose ternary weights {-1, 0, 1} into binary vectors alpha and beta.
/// alpha[i] = (w[i] == 1)  → bit 1 for positive weights
/// beta[i]  = (w[i] == -1) → bit 1 for negative weights
/// Bit-packed: 8 weights per byte, MSB first.
pub fn packFATNN(
    ternary_weights: []const i8,
    alpha: []u8,
    beta: []u8,
) void {
    std.debug.assert(ternary_weights.len % 8 == 0);
    const n_bytes = ternary_weights.len / 8;
    std.debug.assert(alpha.len >= n_bytes);
    std.debug.assert(beta.len >= n_bytes);

    for (0..n_bytes) |byte_idx| {
        var a_byte: u8 = 0;
        var b_byte: u8 = 0;
        inline for (0..8) |bit| {
            const w = ternary_weights[byte_idx * 8 + bit];
            const shift: u3 = 7 - @as(u3, @intCast(bit));
            if (w == 1) a_byte |= @as(u8, 1) << shift;
            if (w == -1) b_byte |= @as(u8, 1) << shift;
        }
        alpha[byte_idx] = a_byte;
        beta[byte_idx] = b_byte;
    }
}

/// Binary dot product via AND + popcount.
/// mask: bit-packed binary vector (1 = selected, 0 = not).
/// activations: int8 values, same length as unpacked mask.
/// n: number of elements (must be multiple of 8).
fn binaryDot(mask: []const u8, activations: []const i8, n: usize) i32 {
    var acc: i32 = 0;
    for (0..n / 8) |byte_idx| {
        const bits = mask[byte_idx];
        const base = byte_idx * 8;
        // Iterate set bits.
        inline for (0..8) |bit| {
            const shift: u3 = 7 - @as(u3, @intCast(bit));
            if (bits & (@as(u8, 1) << shift) != 0) {
                acc += activations[base + bit];
            }
        }
    }
    return acc;
}

/// FATNN ternary dot product.
/// dot(w, x) = dot(alpha, x) - dot(beta, x)
pub fn ternaryDotFATNN(
    alpha: []const u8,
    beta: []const u8,
    activations: []const i8,
    n: usize,
) i32 {
    return binaryDot(alpha, activations, n) - binaryDot(beta, activations, n);
}

const testing = std.testing;

test "FATNN matches reference N=128" {
    var prng = std.Random.DefaultPrng.init(0xFA700);
    const rand = prng.random();

    const N = 128;
    var weights_i8: [N]i8 = undefined;
    for (&weights_i8) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    // Reference.
    var packed_ref: [N / 4]u8 = undefined;
    reference.packTernary(&weights_i8, &packed_ref);
    const ref_result = reference.ternaryDotProductScalar(N, &packed_ref, &activations);

    // FATNN.
    var alpha: [N / 8]u8 = undefined;
    var beta: [N / 8]u8 = undefined;
    packFATNN(&weights_i8, &alpha, &beta);
    const fatnn_result = ternaryDotFATNN(&alpha, &beta, &activations, N);

    try testing.expectEqual(ref_result, fatnn_result);
}

test "FATNN decomposition correctness" {
    const weights = [_]i8{ 1, -1, 0, 1, -1, -1, 0, 0 };
    var alpha: [1]u8 = undefined;
    var beta: [1]u8 = undefined;
    packFATNN(&weights, &alpha, &beta);

    // alpha: 1,0,0,1,0,0,0,0 → 0b10010000 = 0x90
    try testing.expectEqual(@as(u8, 0x90), alpha[0]);
    // beta: 0,1,0,0,1,1,0,0 → 0b01001100 = 0x4C
    try testing.expectEqual(@as(u8, 0x4C), beta[0]);
}

test "FATNN all zeros" {
    const weights = [_]i8{0} ** 8;
    var alpha: [1]u8 = undefined;
    var beta: [1]u8 = undefined;
    packFATNN(&weights, &alpha, &beta);

    const activations = [_]i8{ 10, 20, 30, 40, 50, 60, 70, 80 };
    const result = ternaryDotFATNN(&alpha, &beta, &activations, 8);
    try testing.expectEqual(@as(i32, 0), result);
}

test "FATNN all ones" {
    const weights = [_]i8{1} ** 8;
    var alpha: [1]u8 = undefined;
    var beta: [1]u8 = undefined;
    packFATNN(&weights, &alpha, &beta);

    const activations = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = ternaryDotFATNN(&alpha, &beta, &activations, 8);
    try testing.expectEqual(@as(i32, 36), result);
}
