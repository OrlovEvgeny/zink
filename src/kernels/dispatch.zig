const std = @import("std");
const builtin = @import("builtin");

const reference = @import("reference.zig");
const i2s_generic = @import("i2s_generic.zig");
const i2s_helium = @import("i2s_helium.zig");
const i2s_rvv = @import("i2s_rvv.zig");
pub const tl1 = @import("tl1.zig");
pub const fatnn = @import("fatnn.zig");

pub const KernelType = enum {
    scalar,
    i2s_generic,
    i2s_helium,
    i2s_rvv,
    tl1,
    fatnn,
};

/// Select the optimal kernel for the compilation target.
pub fn selectKernel(comptime target: std.Target) KernelType {
    if (target.cpu.arch == .thumb and
        std.Target.arm.featureSetHas(target.cpu.features, .mve))
    {
        return .i2s_helium;
    }
    if ((target.cpu.arch == .riscv32 or target.cpu.arch == .riscv64) and
        std.Target.riscv.featureSetHas(target.cpu.features, .v))
    {
        return .i2s_rvv;
    }
    return .i2s_generic;
}

/// Comptime-selected kernel for the current build target.
pub const selected_kernel = selectKernel(builtin.target);

/// Ternary dot product dispatched to the optimal kernel at comptime.
/// No vtable, no runtime branching.
///
/// For TL1/FATNN paths, this falls through to I2_S since dot products
/// use I2_S-packed weights. The TL1/FATNN matmul variants below handle
/// the LUT-based and decomposition-based paths at the matmul level.
pub fn ternaryDotProduct(
    comptime N: usize,
    packed_weights: *const [N / 4]u8,
    activations: *const [N]i8,
) i32 {
    return switch (selected_kernel) {
        .i2s_helium => i2s_helium.ternaryDotI2S_Helium(N, packed_weights, activations),
        .i2s_rvv => i2s_rvv.ternaryDotI2S_RVV(N, packed_weights, activations),
        .i2s_generic => i2s_generic.ternaryDotI2S(N, packed_weights, activations),
        .scalar => reference.ternaryDotProductScalar(N, packed_weights, activations),
        .tl1, .fatnn => i2s_generic.ternaryDotI2S(N, packed_weights, activations),
    };
}

/// Ternary matrix-vector multiply dispatched to the optimal kernel.
/// For I2_S/scalar paths, iterates rows calling ternaryDotProduct.
pub fn ternaryMatVec(
    comptime M: usize,
    comptime N: usize,
    weights: *const [M][N / 4]u8,
    input: *const [N]i8,
    output: *[M]i32,
) void {
    for (0..M) |row| {
        output[row] = ternaryDotProduct(N, &weights[row], input);
    }
}

/// TL1 matrix-vector multiply.
/// Builds the LUT once from input activations, then uses it for all rows.
/// The LUT is rebuilt per input vector and reused across projections
/// by calling this function multiple times with the same lut_buf.
///
/// Caller provides:
/// - weights: I2_S packed ternary weights (used to derive TL1 pair indices per row)
/// - input: activation vector (used to build LUT on first call)
/// - lut_buf: pre-built TL1 LUT (caller builds once via tl1.buildTL1Lut)
/// - output: i32 accumulator results
///
/// Each row's I2_S weights are unpacked to ternary, then re-packed as TL1 pair indices.
/// This is a correctness path; production would store TL1-packed weights in .zink.
pub fn ternaryMatVecTL1(
    comptime M: usize,
    comptime N: usize,
    weights: *const [M][N / 4]u8,
    lut_buf: []const i16,
    output: *[M]i32,
) void {
    const n_pairs = N / 2;
    var tl1_row_packed: [n_pairs / 2]u8 = undefined;

    for (0..M) |row| {
        // Unpack I2_S weights to ternary i8, then repack as TL1 pair indices.
        var weights_i8: [N]i8 = undefined;
        for (0..N / 4) |byte_idx| {
            inline for (0..4) |j| {
                weights_i8[byte_idx * 4 + j] = reference.unpackTernary(weights[row][byte_idx], @intCast(j));
            }
        }
        tl1.packTL1Weights(&weights_i8, &tl1_row_packed);
        output[row] = tl1.ternaryDotTL1(&tl1_row_packed, lut_buf, n_pairs);
    }
}

/// Build TL1 LUT from an activation vector.
/// Wraps tl1.buildTL1Lut for use from transformer layer dispatch.
pub fn buildTL1Lut(comptime N: usize, activations: *const [N]i8, lut_buf: []i16) void {
    tl1.buildTL1Lut(activations, lut_buf);
}

/// FATNN matrix-vector multiply.
/// Decomposes each row's ternary weights into alpha/beta binary vectors
/// and computes dot product via AND + popcount.
///
/// Like the TL1 path, this unpacks I2_S weights per row since .zink
/// doesn't yet store FATNN-format (alpha/beta) weights.
pub fn ternaryMatVecFATNN(
    comptime M: usize,
    comptime N: usize,
    weights: *const [M][N / 4]u8,
    input: *const [N]i8,
    output: *[M]i32,
) void {
    var alpha: [N / 8]u8 = undefined;
    var beta: [N / 8]u8 = undefined;

    for (0..M) |row| {
        var weights_i8: [N]i8 = undefined;
        for (0..N / 4) |byte_idx| {
            inline for (0..4) |j| {
                weights_i8[byte_idx * 4 + j] = reference.unpackTernary(weights[row][byte_idx], @intCast(j));
            }
        }
        fatnn.packFATNN(&weights_i8, &alpha, &beta);
        output[row] = fatnn.ternaryDotFATNN(&alpha, &beta, input, N);
    }
}

/// Dispatched matrix-vector multiply that selects the right path based on KernelType.
/// For TL1: caller must pre-build the LUT and pass it in lut_buf.
/// For other kernels: lut_buf is ignored.
pub fn ternaryMatVecDispatch(
    comptime kernel: KernelType,
    comptime M: usize,
    comptime N: usize,
    weights: *const [M][N / 4]u8,
    input: *const [N]i8,
    output: *[M]i32,
    lut_buf: ?[]const i16,
) void {
    switch (kernel) {
        .tl1 => ternaryMatVecTL1(M, N, weights, lut_buf.?, output),
        .fatnn => ternaryMatVecFATNN(M, N, weights, input, output),
        else => ternaryMatVec(M, N, weights, input, output),
    }
}

test "dispatch selects a kernel" {
    const ref = @import("reference.zig");
    var prng = std.Random.DefaultPrng.init(0xD15A7C8);
    const rand = prng.random();

    const N = 128;
    var weights_i8: [N]i8 = undefined;
    for (&weights_i8) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
    var buf: [N / 4]u8 = undefined;
    ref.packTernary(&weights_i8, &buf);

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    const ref_result = ref.ternaryDotProductScalar(N, &buf, &activations);
    const dispatch_result = ternaryDotProduct(N, &buf, &activations);
    try std.testing.expectEqual(ref_result, dispatch_result);
}

test "TL1 matmul dispatch matches reference" {
    const ref = @import("reference.zig");
    var prng = std.Random.DefaultPrng.init(0xABCD);
    const rand = prng.random();

    const M = 4;
    const N = 128;
    var weights_i8: [M][N]i8 = undefined;
    var weights_packed: [M][N / 4]u8 = undefined;
    for (0..M) |row| {
        for (&weights_i8[row]) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
        ref.packTernary(&weights_i8[row], &weights_packed[row]);
    }

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    // Reference matmul.
    var ref_output: [M]i32 = undefined;
    ref.ternaryMatVecScalar(M, N, &weights_packed, &activations, &ref_output);

    // TL1 matmul via dispatch.
    const n_pairs = N / 2;
    var lut_buf: [n_pairs * tl1.LUT_ENTRIES_PER_PAIR]i16 = undefined;
    tl1.buildTL1Lut(&activations, &lut_buf);

    var tl1_output: [M]i32 = undefined;
    ternaryMatVecTL1(M, N, &weights_packed, &lut_buf, &tl1_output);

    for (0..M) |i| {
        try std.testing.expectEqual(ref_output[i], tl1_output[i]);
    }
}

test "FATNN matmul dispatch matches reference" {
    const ref = @import("reference.zig");
    var prng = std.Random.DefaultPrng.init(0xFA77);
    const rand = prng.random();

    const M = 4;
    const N = 128;
    var weights_i8: [M][N]i8 = undefined;
    var weights_packed: [M][N / 4]u8 = undefined;
    for (0..M) |row| {
        for (&weights_i8[row]) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
        ref.packTernary(&weights_i8[row], &weights_packed[row]);
    }

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    var ref_output: [M]i32 = undefined;
    ref.ternaryMatVecScalar(M, N, &weights_packed, &activations, &ref_output);

    var fatnn_output: [M]i32 = undefined;
    ternaryMatVecFATNN(M, N, &weights_packed, &activations, &fatnn_output);

    for (0..M) |i| {
        try std.testing.expectEqual(ref_output[i], fatnn_output[i]);
    }
}
