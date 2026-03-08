const std = @import("std");

/// Extract a single ternary weight from a byte of 2-bit packed values.
/// Two's complement encoding: 0b11 = -1, 0b00 = 0, 0b01 = 1.
/// index 0 is the least significant pair.
pub fn unpackTernary(byte: u8, index: u2) i2 {
    const shift: u3 = @as(u3, index) * 2;
    const bits: u2 = @truncate(byte >> shift);
    return @bitCast(bits);
}

/// Pack i8 ternary values {-1, 0, 1} into 2-bit two's complement encoding.
/// 4 values per byte, LSB first.
pub fn packTernary(weights: []const i8, output: []u8) void {
    std.debug.assert(weights.len % 4 == 0);
    std.debug.assert(output.len >= weights.len / 4);

    for (0..weights.len / 4) |i| {
        var byte: u8 = 0;
        for (0..4) |j| {
            const w = weights[i * 4 + j];
            std.debug.assert(w >= -1 and w <= 1);
            const bits: u2 = @bitCast(@as(i2, @intCast(w)));
            byte |= @as(u8, bits) << @intCast(j * 2);
        }
        output[i] = byte;
    }
}

/// Scalar ternary dot product. N must be a comptime multiple of 4.
/// packed_weights: N/4 bytes of 2-bit values.
/// activations: N int8 values.
/// Returns i32 accumulator.
pub fn ternaryDotProductScalar(
    comptime N: usize,
    packed_weights: *const [N / 4]u8,
    activations: *const [N]i8,
) i32 {
    comptime {
        if (N % 4 != 0) @compileError("N must be a multiple of 4");
    }

    var acc: i32 = 0;
    for (0..N / 4) |byte_idx| {
        const pw = packed_weights[byte_idx];
        inline for (0..4) |j| {
            const w: i32 = unpackTernary(pw, @intCast(j));
            const a: i32 = activations[byte_idx * 4 + j];
            acc += w * a;
        }
    }
    return acc;
}

/// Scalar ternary matrix-vector multiply.
/// weights: M rows x N/4 bytes each. input: N activations. output: M results.
pub fn ternaryMatVecScalar(
    comptime M: usize,
    comptime N: usize,
    weights: *const [M][N / 4]u8,
    input: *const [N]i8,
    output: *[M]i32,
) void {
    for (0..M) |row| {
        output[row] = ternaryDotProductScalar(N, &weights[row], input);
    }
}

test "unpackTernary encoding" {
    // Byte 0b01_00_11_01 = weights [1, -1, 0, 1] (LSB first)
    const byte: u8 = 0b01_00_11_01;
    try std.testing.expectEqual(@as(i2, 1), unpackTernary(byte, 0));
    try std.testing.expectEqual(@as(i2, -1), unpackTernary(byte, 1));
    try std.testing.expectEqual(@as(i2, 0), unpackTernary(byte, 2));
    try std.testing.expectEqual(@as(i2, 1), unpackTernary(byte, 3));
}

test "packTernary round-trip" {
    const weights = [_]i8{ 1, -1, 0, 1, 0, 0, -1, -1 };
    var buf: [2]u8 = undefined;
    packTernary(&weights, &buf);

    for (0..8) |i| {
        const byte_idx = i / 4;
        const bit_idx: u2 = @intCast(i % 4);
        const recovered: i8 = unpackTernary(buf[byte_idx], bit_idx);
        try std.testing.expectEqual(weights[i], recovered);
    }
}

test "dot product known values" {
    // weights = [1, -1, 0, 1], activations = [10, 20, 30, 40]
    // expected = 10 - 20 + 0 + 40 = 30
    const weights = [_]i8{ 1, -1, 0, 1 };
    var buf: [1]u8 = undefined;
    packTernary(&weights, &buf);

    const activations = [_]i8{ 10, 20, 30, 40 };
    const result = ternaryDotProductScalar(4, &buf, &activations);
    try std.testing.expectEqual(@as(i32, 30), result);
}

test "dot product all zeros" {
    const weights = [_]i8{ 0, 0, 0, 0 };
    var buf: [1]u8 = undefined;
    packTernary(&weights, &buf);

    const activations = [_]i8{ 127, -128, 50, -50 };
    const result = ternaryDotProductScalar(4, &buf, &activations);
    try std.testing.expectEqual(@as(i32, 0), result);
}

test "dot product all negative one" {
    const weights = [_]i8{ -1, -1, -1, -1 };
    var buf: [1]u8 = undefined;
    packTernary(&weights, &buf);

    const activations = [_]i8{ 10, 20, 30, 40 };
    const result = ternaryDotProductScalar(4, &buf, &activations);
    try std.testing.expectEqual(@as(i32, -100), result);
}

test "ternaryMatVecScalar" {
    const w0 = [_]i8{ 1, -1, 0, 1 };
    const w1 = [_]i8{ -1, -1, -1, -1 };

    var buf: [2][1]u8 = undefined;
    packTernary(&w0, &buf[0]);
    packTernary(&w1, &buf[1]);

    const input = [_]i8{ 10, 20, 30, 40 };
    var output: [2]i32 = undefined;
    ternaryMatVecScalar(2, 4, &buf, &input, &output);
    try std.testing.expectEqual(@as(i32, 30), output[0]);
    try std.testing.expectEqual(@as(i32, -100), output[1]);
}
