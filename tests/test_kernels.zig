const std = @import("std");
const testing = std.testing;
const zink = @import("zink");

const reference = zink.kernels.reference;
const i2s_generic = zink.kernels.i2s_generic;
const tl1 = zink.kernels.tl1;
const fatnn = zink.kernels.fatnn;
const dispatch = zink.kernels.dispatch;

// Helper: generate random ternary weights and int8 activations.
fn generateTestData(
    comptime N: usize,
    seed: u64,
    weights: *[N]i8,
    activations: *[N]i8,
) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (weights) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
    for (activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);
}

// I2_S generic vs scalar reference across multiple dimensions.

test "i2s vs reference N=128 seed=1" {
    try verifyI2sVsReference(128, 1);
}

test "i2s vs reference N=256 seed=2" {
    try verifyI2sVsReference(256, 2);
}

test "i2s vs reference N=512 seed=3" {
    try verifyI2sVsReference(512, 3);
}

test "i2s vs reference N=1024 seed=4" {
    try verifyI2sVsReference(1024, 4);
}

fn verifyI2sVsReference(comptime N: usize, seed: u64) !void {
    var weights: [N]i8 = undefined;
    var activations: [N]i8 = undefined;
    generateTestData(N, seed, &weights, &activations);

    var buf: [N / 4]u8 = undefined;
    reference.packTernary(&weights, &buf);

    const ref_result = reference.ternaryDotProductScalar(N, &buf, &activations);
    const i2s_result = i2s_generic.ternaryDotI2S(N, &buf, &activations);
    try testing.expectEqual(ref_result, i2s_result);
}

// TL1 vs scalar reference.

test "tl1 vs reference N=128 seed=10" {
    try verifyTL1VsReference(128, 10);
}

test "tl1 vs reference N=256 seed=11" {
    try verifyTL1VsReference(256, 11);
}

test "tl1 vs reference N=512 seed=12" {
    try verifyTL1VsReference(512, 12);
}

fn verifyTL1VsReference(comptime N: usize, seed: u64) !void {
    var weights: [N]i8 = undefined;
    var activations: [N]i8 = undefined;
    generateTestData(N, seed, &weights, &activations);

    var buf_ref: [N / 4]u8 = undefined;
    reference.packTernary(&weights, &buf_ref);
    const ref_result = reference.ternaryDotProductScalar(N, &buf_ref, &activations);

    const n_pairs = N / 2;
    var lut_buf: [n_pairs * tl1.LUT_ENTRIES_PER_PAIR]i16 = undefined;
    tl1.buildTL1Lut(&activations, &lut_buf);

    var tl1_packed: [n_pairs / 2]u8 = undefined;
    tl1.packTL1Weights(&weights, &tl1_packed);

    const tl1_result = tl1.ternaryDotTL1(&tl1_packed, &lut_buf, n_pairs);
    try testing.expectEqual(ref_result, tl1_result);
}

// FATNN vs scalar reference.

test "fatnn vs reference N=128 seed=20" {
    try verifyFATNNVsReference(128, 20);
}

test "fatnn vs reference N=256 seed=21" {
    try verifyFATNNVsReference(256, 21);
}

test "fatnn vs reference N=512 seed=22" {
    try verifyFATNNVsReference(512, 22);
}

test "fatnn vs reference N=1024 seed=23" {
    try verifyFATNNVsReference(1024, 23);
}

fn verifyFATNNVsReference(comptime N: usize, seed: u64) !void {
    var weights: [N]i8 = undefined;
    var activations: [N]i8 = undefined;
    generateTestData(N, seed, &weights, &activations);

    var buf_ref: [N / 4]u8 = undefined;
    reference.packTernary(&weights, &buf_ref);
    const ref_result = reference.ternaryDotProductScalar(N, &buf_ref, &activations);

    var alpha: [N / 8]u8 = undefined;
    var beta: [N / 8]u8 = undefined;
    fatnn.packFATNN(&weights, &alpha, &beta);
    const fatnn_result = fatnn.ternaryDotFATNN(&alpha, &beta, &activations, N);

    try testing.expectEqual(ref_result, fatnn_result);
}

// Dispatch vs scalar reference.

test "dispatch vs reference N=128" {
    var weights: [128]i8 = undefined;
    var activations: [128]i8 = undefined;
    generateTestData(128, 42, &weights, &activations);

    var buf: [32]u8 = undefined;
    reference.packTernary(&weights, &buf);

    const ref_result = reference.ternaryDotProductScalar(128, &buf, &activations);
    const dispatch_result = dispatch.ternaryDotProduct(128, &buf, &activations);
    try testing.expectEqual(ref_result, dispatch_result);
}

// TL1 dispatch (matmul level) vs scalar reference.

test "tl1 dispatch matmul vs reference N=128" {
    try verifyTL1DispatchVsReference(128, 50);
}

test "tl1 dispatch matmul vs reference N=256" {
    try verifyTL1DispatchVsReference(256, 51);
}

fn verifyTL1DispatchVsReference(comptime N: usize, seed: u64) !void {
    const M = 4;
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    var weights_i8: [M][N]i8 = undefined;
    var weights_packed: [M][N / 4]u8 = undefined;
    for (0..M) |row| {
        for (&weights_i8[row]) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
        reference.packTernary(&weights_i8[row], &weights_packed[row]);
    }

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    var ref_output: [M]i32 = undefined;
    reference.ternaryMatVecScalar(M, N, &weights_packed, &activations, &ref_output);

    const n_pairs = N / 2;
    var lut_buf: [n_pairs * tl1.LUT_ENTRIES_PER_PAIR]i16 = undefined;
    tl1.buildTL1Lut(&activations, &lut_buf);

    var tl1_output: [M]i32 = undefined;
    dispatch.ternaryMatVecTL1(M, N, &weights_packed, &lut_buf, &tl1_output);

    for (0..M) |i| {
        try testing.expectEqual(ref_output[i], tl1_output[i]);
    }
}

// FATNN dispatch (matmul level) vs scalar reference.

test "fatnn dispatch matmul vs reference N=128" {
    try verifyFATNNDispatchVsReference(128, 60);
}

test "fatnn dispatch matmul vs reference N=256" {
    try verifyFATNNDispatchVsReference(256, 61);
}

fn verifyFATNNDispatchVsReference(comptime N: usize, seed: u64) !void {
    const M = 4;
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    var weights_i8: [M][N]i8 = undefined;
    var weights_packed: [M][N / 4]u8 = undefined;
    for (0..M) |row| {
        for (&weights_i8[row]) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
        reference.packTernary(&weights_i8[row], &weights_packed[row]);
    }

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    var ref_output: [M]i32 = undefined;
    reference.ternaryMatVecScalar(M, N, &weights_packed, &activations, &ref_output);

    var fatnn_output: [M]i32 = undefined;
    dispatch.ternaryMatVecFATNN(M, N, &weights_packed, &activations, &fatnn_output);

    for (0..M) |i| {
        try testing.expectEqual(ref_output[i], fatnn_output[i]);
    }
}

// Edge cases.

test "all zeros dot product" {
    const weights = [_]i8{0} ** 128;
    var buf: [32]u8 = undefined;
    reference.packTernary(&weights, &buf);
    var activations: [128]i8 = undefined;
    for (&activations) |*a| a.* = 127;

    try testing.expectEqual(@as(i32, 0), reference.ternaryDotProductScalar(128, &buf, &activations));
    try testing.expectEqual(@as(i32, 0), i2s_generic.ternaryDotI2S(128, &buf, &activations));
}

test "all ones dot product" {
    const weights = [_]i8{1} ** 128;
    var buf: [32]u8 = undefined;
    reference.packTernary(&weights, &buf);
    const activations = [_]i8{1} ** 128;

    try testing.expectEqual(@as(i32, 128), reference.ternaryDotProductScalar(128, &buf, &activations));
    try testing.expectEqual(@as(i32, 128), i2s_generic.ternaryDotI2S(128, &buf, &activations));
}

test "alternating weights" {
    var weights: [128]i8 = undefined;
    for (0..128) |i| weights[i] = if (i % 2 == 0) @as(i8, 1) else @as(i8, -1);
    var buf: [32]u8 = undefined;
    reference.packTernary(&weights, &buf);
    const activations = [_]i8{10} ** 128;

    // Sum: 64 * 10 - 64 * 10 = 0
    try testing.expectEqual(@as(i32, 0), reference.ternaryDotProductScalar(128, &buf, &activations));
    try testing.expectEqual(@as(i32, 0), i2s_generic.ternaryDotI2S(128, &buf, &activations));
}

// Pack/unpack round-trip.

test "pack unpack round-trip all values" {
    const test_cases = [_][4]i8{
        .{ 0, 0, 0, 0 },
        .{ 1, 1, 1, 1 },
        .{ -1, -1, -1, -1 },
        .{ 1, -1, 0, 1 },
        .{ -1, 0, 1, -1 },
    };

    for (test_cases) |tc| {
        var buf: [1]u8 = undefined;
        reference.packTernary(&tc, &buf);
        for (0..4) |j| {
            const recovered: i8 = reference.unpackTernary(buf[0], @intCast(j));
            try testing.expectEqual(tc[j], recovered);
        }
    }
}
