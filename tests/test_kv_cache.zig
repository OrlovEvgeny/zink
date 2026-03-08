const std = @import("std");
const testing = std.testing;
const zink = @import("zink");

const Q4Quantizer = zink.core.Q4Quantizer;
const KvCache = zink.core.KvCache;

test "Q4 round-trip bounded error group_size=8" {
    const Q4 = Q4Quantizer(8);
    const input = [_]i8{ 100, -50, 25, -12, 6, -3, 1, 0 };
    var q4buf: [4]u8 = undefined;
    var scales: [1]i16 = undefined;
    Q4.quantize(&input, &q4buf, &scales);

    var output: [8]i8 = undefined;
    Q4.dequantize(&q4buf, &scales, &output);

    for (0..8) |i| {
        const err = @as(i32, input[i]) - @as(i32, output[i]);
        const abs_err: u32 = @intCast(if (err < 0) -err else err);
        try testing.expect(abs_err <= 20);
    }
}

test "Q4 round-trip group_size=64" {
    const Q4 = Q4Quantizer(64);
    var input: [64]i8 = undefined;
    var prng = std.Random.DefaultPrng.init(0xCA_C4E0);
    const rand = prng.random();
    for (&input) |*v| v.* = rand.intRangeAtMost(i8, -100, 100);

    var q4buf: [32]u8 = undefined;
    var scales: [1]i16 = undefined;
    Q4.quantize(&input, &q4buf, &scales);

    var output: [64]i8 = undefined;
    Q4.dequantize(&q4buf, &scales, &output);

    for (0..64) |i| {
        const err = @as(i32, input[i]) - @as(i32, output[i]);
        const abs_err: u32 = @intCast(if (err < 0) -err else err);
        try testing.expect(abs_err <= 20);
    }
}

test "Q4 all zeros" {
    const Q4 = Q4Quantizer(8);
    const input = [_]i8{0} ** 8;
    var q4buf: [4]u8 = undefined;
    var scales: [1]i16 = undefined;
    Q4.quantize(&input, &q4buf, &scales);

    var output: [8]i8 = undefined;
    Q4.dequantize(&q4buf, &scales, &output);

    for (output) |v| try testing.expectEqual(@as(i8, 0), v);
}

test "KvCache append read round-trip" {
    var cache = KvCache(2, 8, 4, 8){};

    var k = [2][8]i8{
        .{ 100, -50, 25, -12, 6, -3, 1, 0 },
        .{ -100, 50, -25, 12, -6, 3, -1, 0 },
    };
    var v = [2][8]i8{
        .{ 50, 50, 50, 50, -50, -50, -50, -50 },
        .{ 10, 20, 30, 40, -10, -20, -30, -40 },
    };

    cache.append(&k, &v);
    try testing.expectEqual(@as(usize, 1), cache.validLen());

    // Read back and verify bounded error for both heads.
    for (0..2) |head| {
        var k_out: [8]i8 = undefined;
        cache.getKey(head, 0, &k_out);
        for (0..8) |i| {
            const err = @as(i32, k[head][i]) - @as(i32, k_out[i]);
            const abs_err: u32 = @intCast(if (err < 0) -err else err);
            try testing.expect(abs_err <= 20);
        }
    }
}

test "KvCache ring buffer wraps correctly" {
    var cache = KvCache(1, 8, 3, 8){};
    const zeros = [1][8]i8{.{0} ** 8};

    for (0..3) |_| cache.append(&zeros, &zeros);
    try testing.expectEqual(@as(usize, 3), cache.validLen());

    // Fourth append wraps.
    cache.append(&zeros, &zeros);
    try testing.expectEqual(@as(usize, 3), cache.validLen());
    try testing.expectEqual(@as(usize, 1), cache.write_pos);
}

test "KvCache multi-head independence" {
    var cache = KvCache(2, 8, 4, 8){};

    // Only set head 0 to non-zero.
    var k = [2][8]i8{
        .{ 127, 127, 127, 127, 127, 127, 127, 127 },
        .{ 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    var v = k;
    cache.append(&k, &v);

    // Head 1 should read back as zeros.
    var v_out: [8]i8 = undefined;
    cache.getValue(1, 0, &v_out);
    for (v_out) |val| try testing.expectEqual(@as(i8, 0), val);
}

test "KvCache clear resets state" {
    var cache = KvCache(1, 8, 4, 8){};
    const zeros = [1][8]i8{.{0} ** 8};
    cache.append(&zeros, &zeros);
    cache.append(&zeros, &zeros);
    cache.clear();
    try testing.expectEqual(@as(usize, 0), cache.validLen());
    try testing.expectEqual(@as(usize, 0), cache.write_pos);
}

test "KvCache windowLen with no window returns validLen" {
    var cache = KvCache(1, 8, 4, 8){};
    const zeros = [1][8]i8{.{0} ** 8};
    cache.append(&zeros, &zeros);
    cache.append(&zeros, &zeros);
    cache.append(&zeros, &zeros);
    const no_window: ?comptime_int = null;
    try testing.expectEqual(@as(usize, 3), cache.windowLen(no_window));
}

test "KvCache windowLen limits to window_size" {
    var cache = KvCache(1, 8, 8, 8){};
    const zeros = [1][8]i8{.{0} ** 8};

    // Append 6 entries.
    for (0..6) |_| cache.append(&zeros, &zeros);
    try testing.expectEqual(@as(usize, 6), cache.validLen());

    // Window of 3 should return 3.
    const window_3: ?comptime_int = 3;
    try testing.expectEqual(@as(usize, 3), cache.windowLen(window_3));

    // Window larger than validLen returns validLen.
    const window_10: ?comptime_int = 10;
    try testing.expectEqual(@as(usize, 6), cache.windowLen(window_10));
}

test "KvCache windowLen after ring wrap" {
    var cache = KvCache(1, 8, 4, 8){};
    const zeros = [1][8]i8{.{0} ** 8};

    // Fill past capacity (wrap).
    for (0..6) |_| cache.append(&zeros, &zeros);
    try testing.expectEqual(@as(usize, 4), cache.validLen());

    // Window of 2 should return 2.
    const window_2: ?comptime_int = 2;
    try testing.expectEqual(@as(usize, 2), cache.windowLen(window_2));
}
