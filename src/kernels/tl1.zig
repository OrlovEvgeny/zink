const std = @import("std");
const reference = @import("reference.zig");

/// TL1 lookup table kernel for ternary dot product.
/// Builds a LUT from activation pairs, then uses ternary weight pair indices
/// to look up precomputed partial sums.
///
/// LUT is rebuilt once per input vector, reused across Q/K/V/O projections.
/// LUT construction is O(N) but eliminates all multiplications in the
/// subsequent matmuls, which are O(N²).
/// Ref: Bitnet.cpp ACL 2025, Section 3.3 (TL1 design).

/// Number of LUT entries per activation pair. Each pair of ternary weights
/// can be one of 9 combinations: {-1,0,1} × {-1,0,1}.
pub const LUT_ENTRIES_PER_PAIR = 9;

/// Build the TL1 lookup table from activation pairs.
/// For each consecutive pair (a[2i], a[2i+1]), precomputes all 9 possible
/// dot products with weight pairs {-1,0,1}×{-1,0,1}.
///
/// lut layout: [n_pairs][9] entries, stored flat.
/// Index mapping for weight pair (w0, w1) where w0,w1 ∈ {-1,0,1}:
///   idx = (w0+1)*3 + (w1+1)
pub fn buildTL1Lut(
    activations: []const i8,
    lut: []i16,
) void {
    std.debug.assert(activations.len % 2 == 0);
    const n_pairs = activations.len / 2;
    std.debug.assert(lut.len >= n_pairs * LUT_ENTRIES_PER_PAIR);

    for (0..n_pairs) |p| {
        const a0: i16 = activations[p * 2];
        const a1: i16 = activations[p * 2 + 1];
        const base = p * LUT_ENTRIES_PER_PAIR;

        // w0, w1 ∈ {-1, 0, 1} → 9 combinations.
        var idx: usize = 0;
        inline for ([_]i16{ -1, 0, 1 }) |w0| {
            inline for ([_]i16{ -1, 0, 1 }) |w1| {
                lut[base + idx] = w0 * a0 + w1 * a1;
                idx += 1;
            }
        }
    }
}

/// Map a pair of ternary weights to a LUT index.
/// w0, w1 ∈ {-1, 0, 1}. Index = (w0+1)*3 + (w1+1).
fn weightPairToIndex(w0: i2, w1: i2) u4 {
    const idx0: u4 = @intCast(@as(i4, w0) + 1);
    const idx1: u4 = @intCast(@as(i4, w1) + 1);
    return idx0 * 3 + idx1;
}

/// Pack ternary weight pairs into 4-bit indices for TL1 lookup.
/// Each byte holds two pair indices (lo and hi nibble).
pub fn packTL1Weights(
    ternary_weights: []const i8,
    output: []u8,
) void {
    std.debug.assert(ternary_weights.len % 4 == 0);
    const n_pairs = ternary_weights.len / 2;
    std.debug.assert(output.len >= n_pairs / 2);

    var pair_idx: usize = 0;
    while (pair_idx < n_pairs) : (pair_idx += 2) {
        const w0: i2 = @intCast(ternary_weights[pair_idx * 2]);
        const w1: i2 = @intCast(ternary_weights[pair_idx * 2 + 1]);
        const w2: i2 = @intCast(ternary_weights[(pair_idx + 1) * 2]);
        const w3: i2 = @intCast(ternary_weights[(pair_idx + 1) * 2 + 1]);

        const lo = weightPairToIndex(w0, w1);
        const hi = weightPairToIndex(w2, w3);
        output[pair_idx / 2] = (@as(u8, hi) << 4) | @as(u8, lo);
    }
}

/// TL1 ternary dot product using lookup table.
/// weight_indices: packed 4-bit pair indices (2 pairs per byte).
/// lut: precomputed from buildTL1Lut.
/// n_pairs: number of activation pairs (= N/2).
pub fn ternaryDotTL1(
    weight_indices: []const u8,
    lut: []const i16,
    n_pairs: usize,
) i32 {
    std.debug.assert(n_pairs % 2 == 0);
    std.debug.assert(weight_indices.len >= n_pairs / 2);

    var acc: i32 = 0;
    var pair: usize = 0;
    while (pair < n_pairs) : (pair += 2) {
        const byte = weight_indices[pair / 2];
        const lo_idx: u4 = @truncate(byte);
        const hi_idx: u4 = @truncate(byte >> 4);

        acc += lut[pair * LUT_ENTRIES_PER_PAIR + lo_idx];
        acc += lut[(pair + 1) * LUT_ENTRIES_PER_PAIR + hi_idx];
    }
    return acc;
}

const testing = std.testing;

test "TL1 matches reference N=128" {
    var prng = std.Random.DefaultPrng.init(0x71_BEEF);
    const rand = prng.random();

    const N = 128;
    var weights_i8: [N]i8 = undefined;
    for (&weights_i8) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    // Reference result.
    var packed_ref: [N / 4]u8 = undefined;
    reference.packTernary(&weights_i8, &packed_ref);
    const ref_result = reference.ternaryDotProductScalar(N, &packed_ref, &activations);

    // TL1 result.
    const n_pairs = N / 2;
    var lut: [n_pairs * LUT_ENTRIES_PER_PAIR]i16 = undefined;
    buildTL1Lut(&activations, &lut);

    var tl1_packed: [n_pairs / 2]u8 = undefined;
    packTL1Weights(&weights_i8, &tl1_packed);

    const tl1_result = ternaryDotTL1(&tl1_packed, &lut, n_pairs);
    try testing.expectEqual(ref_result, tl1_result);
}

test "TL1 weight pair index mapping" {
    // (-1, -1) → (0)*3 + (0) = 0
    try testing.expectEqual(@as(u4, 0), weightPairToIndex(-1, -1));
    // (0, 0) → (1)*3 + (1) = 4
    try testing.expectEqual(@as(u4, 4), weightPairToIndex(0, 0));
    // (1, 1) → (2)*3 + (2) = 8
    try testing.expectEqual(@as(u4, 8), weightPairToIndex(1, 1));
}

test "TL1 LUT values" {
    const activations = [_]i8{ 10, -20 };
    var lut: [LUT_ENTRIES_PER_PAIR]i16 = undefined;
    buildTL1Lut(&activations, &lut);

    // (-1,-1): -10 + 20 = 10
    try testing.expectEqual(@as(i16, 10), lut[0]);
    // (0,0): 0
    try testing.expectEqual(@as(i16, 0), lut[4]);
    // (1,1): 10 + (-20) = -10
    try testing.expectEqual(@as(i16, -10), lut[8]);
    // (1,-1): 10 + 20 = 30
    try testing.expectEqual(@as(i16, 30), lut[6]);
}
