const std = @import("std");

/// Special token IDs. These are fixed by convention across all Zink models.
pub const SpecialTokens = struct {
    pub const BOS: u32 = 1;
    pub const EOS: u32 = 2;
    pub const PAD: u32 = 0;
    pub const UNK: u32 = 3;
};

/// Static vocabulary table. Token strings are stored inline in fixed-size slots.
/// No dynamic allocation. Token → string is O(1) index, string → token is
/// binary search on a comptime-sorted index.
pub fn Vocab(comptime max_vocab: usize, comptime max_token_len: usize) type {
    return struct {
        const Self = @This();

        pub const Token = [max_token_len]u8;
        pub const Entry = struct {
            text: Token,
            len: u8,
        };

        entries: [max_vocab]Entry = std.mem.zeroes([max_vocab]Entry),
        size: u32 = 0,

        /// Add a token to the vocabulary. Returns token ID (sequential).
        pub fn addToken(self: *Self, text: []const u8) error{VocabFull, TokenTooLong}!u32 {
            if (self.size >= max_vocab) return error.VocabFull;
            if (text.len > max_token_len) return error.TokenTooLong;

            const id = self.size;
            var entry = &self.entries[id];
            @memcpy(entry.text[0..text.len], text);
            entry.len = @intCast(text.len);
            self.size += 1;
            return id;
        }

        /// Token ID → text. Returns empty slice for out-of-range IDs.
        pub fn decode(self: *const Self, id: u32) []const u8 {
            if (id >= self.size) return &.{};
            const entry = &self.entries[id];
            return entry.text[0..entry.len];
        }

        /// Text → token ID. Linear scan; vocab is small enough for MCU use.
        pub fn encode(self: *const Self, text: []const u8) ?u32 {
            for (0..self.size) |i| {
                const entry = &self.entries[i];
                if (entry.len == text.len and
                    std.mem.eql(u8, entry.text[0..entry.len], text))
                {
                    return @intCast(i);
                }
            }
            return null;
        }

        /// Load vocabulary from a packed byte buffer.
        /// Format: repeated [u8 len][len bytes text]. Stops at buffer end.
        pub fn loadPacked(self: *Self, data: []const u8) error{ VocabFull, TokenTooLong, InvalidData }!void {
            var offset: usize = 0;
            while (offset < data.len) {
                if (offset >= data.len) break;
                const len = data[offset];
                offset += 1;
                if (offset + len > data.len) return error.InvalidData;
                _ = try self.addToken(data[offset .. offset + len]);
                offset += len;
            }
        }

        pub fn vocabSize(self: *const Self) u32 {
            return self.size;
        }
    };
}

test "vocab add and decode" {
    var v = Vocab(16, 8){};
    const id0 = try v.addToken("hello");
    const id1 = try v.addToken("world");

    try std.testing.expectEqual(@as(u32, 0), id0);
    try std.testing.expectEqual(@as(u32, 1), id1);
    try std.testing.expectEqualSlices(u8, "hello", v.decode(id0));
    try std.testing.expectEqualSlices(u8, "world", v.decode(id1));
}

test "vocab encode lookup" {
    var v = Vocab(16, 8){};
    _ = try v.addToken("foo");
    _ = try v.addToken("bar");
    _ = try v.addToken("baz");

    try std.testing.expectEqual(@as(?u32, 1), v.encode("bar"));
    try std.testing.expect(v.encode("missing") == null);
}

test "vocab decode out of range" {
    var v = Vocab(4, 4){};
    _ = try v.addToken("a");
    try std.testing.expectEqual(@as(usize, 0), v.decode(99).len);
}

test "vocab full returns error" {
    var v = Vocab(2, 4){};
    _ = try v.addToken("a");
    _ = try v.addToken("b");
    try std.testing.expectError(error.VocabFull, v.addToken("c"));
}

test "vocab token too long returns error" {
    var v = Vocab(4, 2){};
    try std.testing.expectError(error.TokenTooLong, v.addToken("abc"));
}

test "vocab load packed" {
    var v = Vocab(16, 8){};
    // Format: [len][bytes]...
    const data = [_]u8{
        3, 'f', 'o', 'o',
        3, 'b', 'a', 'r',
    };
    try v.loadPacked(&data);
    try std.testing.expectEqual(@as(u32, 2), v.vocabSize());
    try std.testing.expectEqualSlices(u8, "foo", v.decode(0));
    try std.testing.expectEqualSlices(u8, "bar", v.decode(1));
}
