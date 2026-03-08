const std = @import("std");
const reference = @import("reference.zig");

/// I2_S ternary dot product using Zig @Vector for portable SIMD.
/// Processes 16 elements per iteration via 128-bit vectors.
///
/// I2_S block alignment = 128 elements.
/// Ref: Bitnet.cpp ACL 2025, Section 3.2.2.
pub fn ternaryDotI2S(
    comptime N: usize,
    packed_weights: *const [N / 4]u8,
    activations: *const [N]i8,
) i32 {
    comptime {
        if (N % 128 != 0) @compileError("N must be a multiple of 128 for I2_S alignment");
    }

    const VEC_LEN = 16;
    const BYTES_PER_VEC = VEC_LEN / 4; // 4 packed bytes → 16 weights.

    var acc: i32 = 0;
    const num_vecs = N / VEC_LEN;

    for (0..num_vecs) |vec_idx| {
        const byte_offset = vec_idx * BYTES_PER_VEC;
        const elem_offset = vec_idx * VEC_LEN;

        // Unpack 4 bytes into 16 ternary weights.
        var weights_arr: [VEC_LEN]i8 = undefined;
        inline for (0..BYTES_PER_VEC) |b| {
            const pw = packed_weights[byte_offset + b];
            inline for (0..4) |j| {
                const w: i8 = reference.unpackTernary(pw, @intCast(j));
                weights_arr[b * 4 + j] = w;
            }
        }

        const w_vec: @Vector(VEC_LEN, i16) = @as(
            @Vector(VEC_LEN, i8),
            weights_arr,
        );
        const a_vec: @Vector(VEC_LEN, i16) = @as(
            @Vector(VEC_LEN, i8),
            activations[elem_offset..][0..VEC_LEN].*,
        );

        // Multiply and horizontal sum via widened multiply + reduce.
        const prod = w_vec * a_vec;
        acc += @reduce(.Add, prod);
    }

    return acc;
}

/// I2_S matrix-vector multiply. Row-by-row using the dot kernel.
pub fn ternaryMatVecI2S(
    comptime M: usize,
    comptime N: usize,
    weights: *const [M][N / 4]u8,
    input: *const [N]i8,
    output: *[M]i32,
) void {
    for (0..M) |row| {
        output[row] = ternaryDotI2S(N, &weights[row], input);
    }
}

const testing = std.testing;

test "i2s_generic matches reference N=128" {
    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const rand = prng.random();

    var weights_i8: [128]i8 = undefined;
    for (&weights_i8) |*w| {
        const r = rand.intRangeAtMost(i8, -1, 1);
        w.* = r;
    }
    var buf: [32]u8 = undefined;
    reference.packTernary(&weights_i8, &buf);

    var activations: [128]i8 = undefined;
    for (&activations) |*a| {
        a.* = rand.intRangeAtMost(i8, -128, 127);
    }

    const ref_result = reference.ternaryDotProductScalar(128, &buf, &activations);
    const i2s_result = ternaryDotI2S(128, &buf, &activations);
    try testing.expectEqual(ref_result, i2s_result);
}

test "i2s_generic matches reference N=256" {
    var prng = std.Random.DefaultPrng.init(0xcafebabe);
    const rand = prng.random();

    var weights_i8: [256]i8 = undefined;
    for (&weights_i8) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
    var buf: [64]u8 = undefined;
    reference.packTernary(&weights_i8, &buf);

    var activations: [256]i8 = undefined;
    for (&activations) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    const ref_result = reference.ternaryDotProductScalar(256, &buf, &activations);
    const i2s_result = ternaryDotI2S(256, &buf, &activations);
    try testing.expectEqual(ref_result, i2s_result);
}

test "i2s_generic matvec matches reference" {
    var prng = std.Random.DefaultPrng.init(0x12345678);
    const rand = prng.random();

    const M = 4;
    const N = 128;
    var weights_i8: [M][N]i8 = undefined;
    var buf: [M][N / 4]u8 = undefined;

    for (0..M) |row| {
        for (&weights_i8[row]) |*w| w.* = rand.intRangeAtMost(i8, -1, 1);
        reference.packTernary(&weights_i8[row], &buf[row]);
    }

    var input: [N]i8 = undefined;
    for (&input) |*a| a.* = rand.intRangeAtMost(i8, -128, 127);

    var ref_output: [M]i32 = undefined;
    reference.ternaryMatVecScalar(M, N, &buf, &input, &ref_output);

    var i2s_output: [M]i32 = undefined;
    ternaryMatVecI2S(M, N, &buf, &input, &i2s_output);

    for (0..M) |i| {
        try testing.expectEqual(ref_output[i], i2s_output[i]);
    }
}
