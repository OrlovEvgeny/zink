const std = @import("std");
const reference = @import("reference.zig");
const i2s_generic = @import("i2s_generic.zig");

/// I2_S kernel optimized for ARM Cortex-M55 Helium MVE (128-bit vectors).
///
/// Helium MVE provides VMLADAV (multiply-accumulate-dual-across-vector)
/// and VTBL (table lookup, 1 cycle latency) which are ideal for ternary
/// dot products.
/// Ref: ARM Cortex-M55 Software Optimization Guide, Section 4.3.

const is_arm_mve = blk: {
    const target = @import("builtin").target;
    break :blk target.cpu.arch == .thumb and
        target.os.tag == .freestanding and
        std.Target.arm.featureSetHas(target.cpu.features, .mve);
};

/// Ternary dot product using I2_S for Helium MVE.
/// Uses explicit i8→i16 widening to help LLVM emit widening multiply-
/// accumulate instructions (VMLADAV.S16) on Helium instead of i8 mul
/// which lacks a dual-accumulate form on MVE.
///
/// On non-MVE targets, falls back to the generic @Vector path.
pub fn ternaryDotI2S_Helium(
    comptime N: usize,
    packed_weights: *const [N / 4]u8,
    activations: *const [N]i8,
) i32 {
    comptime {
        if (N % 128 != 0) @compileError("N must be a multiple of 128 for I2_S alignment");
    }

    if (!is_arm_mve) {
        return i2s_generic.ternaryDotI2S(N, packed_weights, activations);
    }

    // Process 8 elements per vector iteration. Using 8-wide i16 vectors
    // maps directly to Helium's 128-bit registers holding 8×i16, enabling
    // VMLADAV.S16 (multiply-accumulate-dual-across-vector) which computes
    // the dot product of two i16 vectors in a single instruction.
    // Ref: ARM Helium Programmer's Guide, Section 5.3.
    const VEC_LEN = 8;
    const BYTES_PER_VEC = VEC_LEN / 4; // 2 packed bytes → 8 weights.

    var acc: i32 = 0;
    const num_vecs = N / VEC_LEN;

    for (0..num_vecs) |vec_idx| {
        const byte_offset = vec_idx * BYTES_PER_VEC;
        const elem_offset = vec_idx * VEC_LEN;

        // Unpack 2 bytes into 8 ternary weights.
        var weights_i8: [VEC_LEN]i8 = undefined;
        inline for (0..BYTES_PER_VEC) |b| {
            const pw = packed_weights[byte_offset + b];
            inline for (0..4) |j| {
                weights_i8[b * 4 + j] = reference.unpackTernary(pw, @intCast(j));
            }
        }

        // Widen to i16 explicitly. On Helium, this enables VMOV.S16 (sign-extend)
        // followed by VMLADAV.S16 instead of falling back to i8 multiply + widen.
        const w_i16: @Vector(VEC_LEN, i16) = @as(@Vector(VEC_LEN, i8), weights_i8);
        const a_i16: @Vector(VEC_LEN, i16) = @as(
            @Vector(VEC_LEN, i8),
            activations[elem_offset..][0..VEC_LEN].*,
        );

        const prod = w_i16 * a_i16;
        acc += @reduce(.Add, prod);
    }

    return acc;
}

pub fn ternaryMatVecI2S_Helium(
    comptime M: usize,
    comptime N: usize,
    weights: *const [M][N / 4]u8,
    input: *const [N]i8,
    output: *[M]i32,
) void {
    for (0..M) |row| {
        output[row] = ternaryDotI2S_Helium(N, &weights[row], input);
    }
}

test "helium fallback matches generic N=128" {
    var prng = std.Random.DefaultPrng.init(0xAE1101);
    const rand = prng.random();

    const N = 128;
    var weights_i8: [N]i8 = undefined;
    for (&weights_i8) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
    var buf: [N / 4]u8 = undefined;
    reference.packTernary(&weights_i8, &buf);

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    const ref_result = reference.ternaryDotProductScalar(N, &buf, &activations);
    const helium_result = ternaryDotI2S_Helium(N, &buf, &activations);
    try std.testing.expectEqual(ref_result, helium_result);
}

test "helium matches generic N=256" {
    var prng = std.Random.DefaultPrng.init(0xBE11_0256);
    const rand = prng.random();

    const N = 256;
    var weights_i8: [N]i8 = undefined;
    for (&weights_i8) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
    var buf: [N / 4]u8 = undefined;
    reference.packTernary(&weights_i8, &buf);

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    const ref_result = reference.ternaryDotProductScalar(N, &buf, &activations);
    const helium_result = ternaryDotI2S_Helium(N, &buf, &activations);
    try std.testing.expectEqual(ref_result, helium_result);
}
