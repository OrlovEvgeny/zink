const std = @import("std");
const vocab_mod = @import("vocab.zig");

/// Byte-level BPE tokenizer for MCU inference.
/// All buffers are statically sized. No dynamic allocation.
///
/// Merge table stores (pair_left, pair_right) → merged_id with priority
/// determined by table order (lower index = higher priority).
pub fn BpeTokenizer(
    comptime max_merges: usize,
    comptime max_vocab: usize,
    comptime max_seq: usize,
    comptime max_token_len: usize,
) type {
    return struct {
        const Self = @This();
        pub const Vocab = vocab_mod.Vocab(max_vocab, max_token_len);

        pub const MergeRule = struct {
            left: u32,
            right: u32,
            merged: u32,
        };

        vocab: Vocab = .{},
        merges: [max_merges]MergeRule = std.mem.zeroes([max_merges]MergeRule),
        num_merges: u32 = 0,

        pub fn addMerge(self: *Self, left: u32, right: u32, merged: u32) error{MergeTableFull}!void {
            if (self.num_merges >= max_merges) return error.MergeTableFull;
            self.merges[self.num_merges] = .{ .left = left, .right = right, .merged = merged };
            self.num_merges += 1;
        }

        /// Encode text to token IDs using BPE merging.
        /// 1. Split input into byte-level tokens.
        /// 2. Iteratively apply the highest-priority merge until no merges apply.
        ///
        /// Returns number of tokens written to output.
        pub fn encode(self: *const Self, text: []const u8, output: []u32) error{SequenceTooLong}!usize {
            if (text.len == 0) return 0;

            // Working buffer: start with byte-level tokens.
            var work: [max_seq]u32 = undefined;
            var work_len: usize = 0;

            for (text) |byte| {
                if (work_len >= max_seq) return error.SequenceTooLong;
                // Look up single-byte token in vocab.
                const single = [1]u8{byte};
                work[work_len] = self.vocab.encode(&single) orelse vocab_mod.SpecialTokens.UNK;
                work_len += 1;
            }

            // Iteratively apply merges. Each pass applies the highest-priority
            // merge found (lowest index in merge table). Repeat until no merge applies.
            var changed = true;
            while (changed) {
                changed = false;
                // Find the highest-priority applicable merge.
                var best_merge_idx: ?usize = null;
                var best_pos: usize = 0;

                for (0..self.num_merges) |m| {
                    const rule = self.merges[m];
                    for (0..work_len -| 1) |pos| {
                        if (work[pos] == rule.left and work[pos + 1] == rule.right) {
                            best_merge_idx = m;
                            best_pos = pos;
                            break;
                        }
                    }
                    // First match in merge table = highest priority.
                    if (best_merge_idx != null) break;
                }

                if (best_merge_idx) |m| {
                    // Apply merge at best_pos: replace pair with merged token.
                    work[best_pos] = self.merges[m].merged;
                    // Shift remaining tokens left by one.
                    var i = best_pos + 1;
                    while (i < work_len - 1) : (i += 1) {
                        work[i] = work[i + 1];
                    }
                    work_len -= 1;
                    changed = true;
                }
            }

            // Copy result to output.
            if (work_len > output.len) return error.SequenceTooLong;
            @memcpy(output[0..work_len], work[0..work_len]);
            return work_len;
        }

        /// Decode token IDs back to text.
        /// Returns number of bytes written to output_buf.
        pub fn decode(self: *const Self, token_ids: []const u32, output_buf: []u8) usize {
            var offset: usize = 0;
            for (token_ids) |id| {
                const text = self.vocab.decode(id);
                if (offset + text.len > output_buf.len) break;
                @memcpy(output_buf[offset .. offset + text.len], text);
                offset += text.len;
            }
            return offset;
        }
    };
}

test "bpe byte-level tokenization" {
    var tok = BpeTokenizer(16, 512, 256, 8){};

    // Add byte-level tokens for ASCII printable range.
    for (0..256) |i| {
        const byte = [1]u8{@intCast(i)};
        _ = try tok.vocab.addToken(&byte);
    }

    var output: [32]u32 = undefined;
    const n = try tok.encode("hi", &output);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(u32, 'h'), output[0]);
    try std.testing.expectEqual(@as(u32, 'i'), output[1]);
}

test "bpe merge" {
    var tok = BpeTokenizer(16, 512, 256, 8){};

    // Byte-level tokens.
    for (0..256) |i| {
        const byte = [1]u8{@intCast(i)};
        _ = try tok.vocab.addToken(&byte);
    }
    // Merged token: "hi" = ID 256.
    _ = try tok.vocab.addToken("hi");
    try tok.addMerge('h', 'i', 256);

    var output: [32]u32 = undefined;
    const n = try tok.encode("hi", &output);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(u32, 256), output[0]);
}

test "bpe decode round-trip" {
    var tok = BpeTokenizer(16, 512, 256, 8){};
    for (0..256) |i| {
        const byte = [1]u8{@intCast(i)};
        _ = try tok.vocab.addToken(&byte);
    }

    const text = "abc";
    var ids: [32]u32 = undefined;
    const n = try tok.encode(text, &ids);

    var buf: [32]u8 = undefined;
    const decoded_len = tok.decode(ids[0..n], &buf);
    try std.testing.expectEqualSlices(u8, text, buf[0..decoded_len]);
}

test "bpe empty input" {
    var tok = BpeTokenizer(16, 512, 256, 8){};
    var output: [32]u32 = undefined;
    const n = try tok.encode("", &output);
    try std.testing.expectEqual(@as(usize, 0), n);
}

test "bpe multi-merge priority" {
    var tok = BpeTokenizer(16, 512, 256, 8){};
    for (0..256) |i| {
        const byte = [1]u8{@intCast(i)};
        _ = try tok.vocab.addToken(&byte);
    }
    _ = try tok.vocab.addToken("ab"); // 256
    _ = try tok.vocab.addToken("bc"); // 257
    // "ab" merge has higher priority (lower index).
    try tok.addMerge('a', 'b', 256);
    try tok.addMerge('b', 'c', 257);

    var output: [32]u32 = undefined;
    const n = try tok.encode("abc", &output);
    // "ab" merges first → [256, 'c'], "bc" can't merge since 'b' is consumed.
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(u32, 256), output[0]);
    try std.testing.expectEqual(@as(u32, 'c'), output[1]);
}
