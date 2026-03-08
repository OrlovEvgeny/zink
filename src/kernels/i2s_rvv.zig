const std = @import("std");
const reference = @import("reference.zig");
const i2s_generic = @import("i2s_generic.zig");

/// I2_S kernel for RISC-V Vector extension.
/// Uses @Vector with sizes that map well to RVV instructions via LLVM:
/// - LMUL=1 with VLEN≥128: 16-wide i8 vectors for load/unpack
/// - Widening multiply-accumulate maps to vwmacc.vv
///
/// Ref: muRISCV-NN benchmarks on RVV 1.0 (C908 core).

const is_riscv_v = blk: {
    const target = @import("builtin").target;
    break :blk (target.cpu.arch == .riscv32 or target.cpu.arch == .riscv64) and
        std.Target.riscv.featureSetHas(target.cpu.features, .v);
};

/// Ternary dot product using I2_S for RISC-V Vector.
/// Uses 16-wide i16 vectors to exploit RVV widening multiply-accumulate
/// (vwmacc.vv). 16×i16 = 256 bits, fitting LMUL=2 on 128-bit VLEN
/// or LMUL=1 on 256-bit VLEN, both common in RVV 1.0 implementations.
///
/// Falls back to generic @Vector on non-RVV targets.
pub fn ternaryDotI2S_RVV(
    comptime N: usize,
    packed_weights: *const [N / 4]u8,
    activations: *const [N]i8,
) i32 {
    comptime {
        if (N % 128 != 0) @compileError("N must be a multiple of 128 for I2_S alignment");
    }

    if (!is_riscv_v) {
        return i2s_generic.ternaryDotI2S(N, packed_weights, activations);
    }

    // 16-wide i16 vectors: on RVV, LLVM emits vle8 + vsext + vwmul + vredsum.
    // Using i16 width avoids overflow in the per-vector partial sum
    // (max 16 × 127 = 2032, fits i16).
    const VEC_LEN = 16;
    const BYTES_PER_VEC = VEC_LEN / 4;

    var acc: i32 = 0;
    const num_vecs = N / VEC_LEN;

    for (0..num_vecs) |vec_idx| {
        const byte_offset = vec_idx * BYTES_PER_VEC;
        const elem_offset = vec_idx * VEC_LEN;

        var weights_i8: [VEC_LEN]i8 = undefined;
        inline for (0..BYTES_PER_VEC) |b| {
            const pw = packed_weights[byte_offset + b];
            inline for (0..4) |j| {
                weights_i8[b * 4 + j] = reference.unpackTernary(pw, @intCast(j));
            }
        }

        // Widen to i16 for multiply. On RVV, LLVM converts this to
        // vsext.vf2 + vmul.vv (or vwmul.vv if it recognizes the pattern).
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

pub fn ternaryMatVecI2S_RVV(
    comptime M: usize,
    comptime N: usize,
    weights: *const [M][N / 4]u8,
    input: *const [N]i8,
    output: *[M]i32,
) void {
    for (0..M) |row| {
        output[row] = ternaryDotI2S_RVV(N, &weights[row], input);
    }
}

test "rvv fallback matches generic N=128" {
    var prng = std.Random.DefaultPrng.init(0xBEEF_01);
    const rand = prng.random();

    const N = 128;
    var weights_i8: [N]i8 = undefined;
    for (&weights_i8) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
    var buf: [N / 4]u8 = undefined;
    reference.packTernary(&weights_i8, &buf);

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    const ref_result = reference.ternaryDotProductScalar(N, &buf, &activations);
    const rvv_result = ternaryDotI2S_RVV(N, &buf, &activations);
    try std.testing.expectEqual(ref_result, rvv_result);
}

test "rvv fallback matches generic N=256" {
    var prng = std.Random.DefaultPrng.init(0xBEEF_0256);
    const rand = prng.random();

    const N = 256;
    var weights_i8: [N]i8 = undefined;
    for (&weights_i8) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
    var buf: [N / 4]u8 = undefined;
    reference.packTernary(&weights_i8, &buf);

    var activations: [N]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    const ref_result = reference.ternaryDotProductScalar(N, &buf, &activations);
    const rvv_result = ternaryDotI2S_RVV(N, &buf, &activations);
    try std.testing.expectEqual(ref_result, rvv_result);
}
