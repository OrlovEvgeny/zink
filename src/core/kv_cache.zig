const std = @import("std");

/// Q4 symmetric quantizer with per-group scales.
/// Packs two 4-bit values per byte, using signed nibbles.
/// group_size must be even.
pub fn Q4Quantizer(comptime group_size: usize) type {
    comptime {
        if (group_size % 2 != 0) @compileError("group_size must be even");
        if (group_size == 0) @compileError("group_size must be > 0");
    }

    return struct {
        /// Quantize int8 values to Q4. Output is packed (2 values per byte).
        /// scale is set to the max absolute value in the group (used for dequant).
        pub fn quantize(
            input: []const i8,
            output: []u8,
            scales: []i16,
        ) void {
            std.debug.assert(input.len % group_size == 0);
            const num_groups = input.len / group_size;
            std.debug.assert(output.len >= input.len / 2);
            std.debug.assert(scales.len >= num_groups);

            for (0..num_groups) |g| {
                const group_start = g * group_size;
                const group = input[group_start .. group_start + group_size];

                // Per-group absmax for scale.
                var absmax: i16 = 0;
                for (group) |v| {
                    const abs: i16 = if (v < 0) -@as(i16, v) else @as(i16, v);
                    if (abs > absmax) absmax = abs;
                }
                scales[g] = absmax;

                // Quantize to 4-bit range [-8, 7].
                const out_start = group_start / 2;
                for (0..group_size / 2) |pair| {
                    const lo = quantizeValue(group[pair * 2], absmax);
                    const hi = quantizeValue(group[pair * 2 + 1], absmax);
                    const lo_u4: u4 = @bitCast(lo);
                    const hi_u4: u4 = @bitCast(hi);
                    output[out_start + pair] = (@as(u8, hi_u4) << 4) | @as(u8, lo_u4);
                }
            }
        }

        /// Dequantize Q4 packed values back to int8.
        pub fn dequantize(
            input: []const u8,
            scales: []const i16,
            output: []i8,
        ) void {
            const num_elements = output.len;
            std.debug.assert(num_elements % group_size == 0);
            std.debug.assert(input.len >= num_elements / 2);

            const num_groups = num_elements / group_size;
            for (0..num_groups) |g| {
                const group_start = g * group_size;
                const scale = scales[g];
                const in_start = group_start / 2;

                for (0..group_size / 2) |pair| {
                    const byte = input[in_start + pair];
                    const lo: i4 = @bitCast(@as(u4, @truncate(byte)));
                    const hi: i4 = @bitCast(@as(u4, @truncate(byte >> 4)));

                    output[group_start + pair * 2] = dequantizeValue(lo, scale);
                    output[group_start + pair * 2 + 1] = dequantizeValue(hi, scale);
                }
            }
        }

        fn quantizeValue(val: i8, absmax: i16) i4 {
            if (absmax == 0) return 0;
            // Scale to [-8, 7] range. Use i32 to avoid overflow.
            const scaled: i32 = @divTrunc(@as(i32, val) * 7, @as(i32, absmax));
            const clamped = std.math.clamp(scaled, -8, 7);
            return @intCast(clamped);
        }

        fn dequantizeValue(qval: i4, scale: i16) i8 {
            // Reverse: val ≈ qval * absmax / 7
            const result: i32 = @divTrunc(@as(i32, qval) * @as(i32, scale), 7);
            return @intCast(std.math.clamp(result, -128, 127));
        }
    };
}

/// Ring-buffer KV-cache with Q4 quantization.
/// Fixed capacity = max_seq_len. O(1) insert, O(1) eviction of oldest token.
///
/// Q4 KV-cache with group_size=64.
/// 0.7–3.0% PPL impact at 4× memory savings.
/// Ref: "Agent Memory Below the Prompt", Shkolnikov, Feb 2026, Table 9.
pub fn KvCache(
    comptime num_kv_heads: usize,
    comptime head_dim: usize,
    comptime max_seq_len: usize,
    comptime group_size: usize,
) type {
    const packed_dim = head_dim / 2;
    const num_groups = head_dim / group_size;

    comptime {
        if (head_dim % group_size != 0) @compileError("head_dim must be divisible by group_size");
        if (head_dim % 2 != 0) @compileError("head_dim must be even for Q4 packing");
    }

    const Q4 = Q4Quantizer(group_size);

    return struct {
        const Self = @This();

        // Q4-packed KV storage. [position][head][packed_bytes].
        k_cache: [max_seq_len][num_kv_heads][packed_dim]u8 =
            std.mem.zeroes([max_seq_len][num_kv_heads][packed_dim]u8),
        v_cache: [max_seq_len][num_kv_heads][packed_dim]u8 =
            std.mem.zeroes([max_seq_len][num_kv_heads][packed_dim]u8),

        // Per-group scales for dequantization.
        k_scales: [max_seq_len][num_kv_heads][num_groups]i16 =
            std.mem.zeroes([max_seq_len][num_kv_heads][num_groups]i16),
        v_scales: [max_seq_len][num_kv_heads][num_groups]i16 =
            std.mem.zeroes([max_seq_len][num_kv_heads][num_groups]i16),

        write_pos: usize = 0,
        len: usize = 0,

        /// Append a KV pair for all heads. Quantizes to Q4 in place.
        pub fn append(
            self: *Self,
            k_vecs: *const [num_kv_heads][head_dim]i8,
            v_vecs: *const [num_kv_heads][head_dim]i8,
        ) void {
            const pos = self.write_pos;

            for (0..num_kv_heads) |h| {
                Q4.quantize(&k_vecs[h], &self.k_cache[pos][h], &self.k_scales[pos][h]);
                Q4.quantize(&v_vecs[h], &self.v_cache[pos][h], &self.v_scales[pos][h]);
            }

            self.write_pos = (self.write_pos + 1) % max_seq_len;
            if (self.len < max_seq_len) self.len += 1;
        }

        /// Dequantize and return a cached K vector.
        pub fn getKey(self: *const Self, head: usize, pos: usize, output: *[head_dim]i8) void {
            Q4.dequantize(&self.k_cache[pos][head], &self.k_scales[pos][head], output);
        }

        /// Dequantize and return a cached V vector.
        pub fn getValue(self: *const Self, head: usize, pos: usize, output: *[head_dim]i8) void {
            Q4.dequantize(&self.v_cache[pos][head], &self.v_scales[pos][head], output);
        }

        pub fn validLen(self: *const Self) usize {
            return self.len;
        }

        /// Effective attention length when sliding window is applied.
        /// Returns min(validLen(), window_size) if window_size is set,
        /// or validLen() for full-context attention.
        pub fn windowLen(self: *const Self, comptime window_size: ?comptime_int) usize {
            const valid = self.len;
            if (window_size) |w| {
                return @min(valid, w);
            }
            return valid;
        }

        pub fn clear(self: *Self) void {
            self.write_pos = 0;
            self.len = 0;
        }
    };
}

test "Q4 quantize-dequantize round-trip bounded error" {
    const Q4 = Q4Quantizer(8);
    const input = [_]i8{ 100, -50, 25, -12, 6, -3, 1, 0 };
    var q4buf: [4]u8 = undefined;
    var scales: [1]i16 = undefined;
    Q4.quantize(&input, &q4buf, &scales);

    var output: [8]i8 = undefined;
    Q4.dequantize(&q4buf, &scales, &output);

    // Q4 introduces quantization error, but it should be bounded.
    for (0..8) |i| {
        const err = @as(i32, input[i]) - @as(i32, output[i]);
        const abs_err: u32 = @intCast(if (err < 0) -err else err);
        // Max error should be within ~1/7 of absmax ≈ 14 for absmax=100.
        try std.testing.expect(abs_err <= 20);
    }
}

test "KvCache append and read" {
    var cache = KvCache(1, 8, 4, 8){};

    var k = [1][8]i8{.{ 100, -50, 25, -12, 6, -3, 1, 0 }};
    var v = [1][8]i8{.{ -100, 50, -25, 12, -6, 3, -1, 0 }};
    cache.append(&k, &v);

    try std.testing.expectEqual(@as(usize, 1), cache.validLen());

    var k_out: [8]i8 = undefined;
    cache.getKey(0, 0, &k_out);

    // Dequantized values should be close to originals.
    for (0..8) |i| {
        const err = @as(i32, k[0][i]) - @as(i32, k_out[i]);
        const abs_err: u32 = @intCast(if (err < 0) -err else err);
        try std.testing.expect(abs_err <= 20);
    }
}

test "KvCache ring buffer wraps" {
    var cache = KvCache(1, 8, 2, 8){};
    const zeros = [1][8]i8{.{0} ** 8};

    // Fill to capacity.
    cache.append(&zeros, &zeros);
    cache.append(&zeros, &zeros);
    try std.testing.expectEqual(@as(usize, 2), cache.validLen());

    // One more wraps around.
    cache.append(&zeros, &zeros);
    try std.testing.expectEqual(@as(usize, 2), cache.validLen());
    try std.testing.expectEqual(@as(usize, 1), cache.write_pos);
}

test "KvCache clear" {
    var cache = KvCache(1, 8, 4, 8){};
    const zeros = [1][8]i8{.{0} ** 8};
    cache.append(&zeros, &zeros);
    cache.clear();
    try std.testing.expectEqual(@as(usize, 0), cache.validLen());
}
