const std = @import("std");

/// Double-buffered weight loader for external flash (QSPI/OCTOSPI).
/// Two static buffers of chunk_size bytes, 32-byte aligned. DMA fills one
/// while the consumer reads the other, then swap.
///
/// On non-MCU targets (testing): memcpy fallback via the default DmaInterface.
pub fn StreamingWeightLoader(comptime chunk_size: usize, comptime alignment: usize) type {
    return struct {
        const Self = @This();

        bufs: [2][chunk_size]u8 align(alignment) = undefined,
        active: u1 = 0,
        source: []const u8 = &.{},
        source_offset: usize = 0,
        pending_len: usize = 0,
        done: bool = true,

        /// Bind a source buffer (e.g., memory-mapped flash region).
        pub fn bind(self: *Self, data: []const u8) void {
            self.source = data;
            self.source_offset = 0;
            self.done = data.len == 0;
            self.pending_len = 0;
        }

        /// Return the current readable chunk. Only valid after advance() or
        /// the initial prefetchNext() + advance() sequence.
        pub fn currentChunk(self: *const Self) []const u8 {
            return self.bufs[self.active][0..self.pending_len];
        }

        /// Trigger a load of the next chunk into the inactive buffer.
        /// On MCU: this would initiate a DMA transfer.
        /// On host: immediate memcpy.
        pub fn prefetchNext(self: *Self) void {
            if (self.source_offset >= self.source.len) {
                self.done = true;
                return;
            }
            const remaining = self.source.len - self.source_offset;
            const len = @min(remaining, chunk_size);
            const inactive = 1 - self.active;

            @memcpy(self.bufs[inactive][0..len], self.source[self.source_offset .. self.source_offset + len]);
            self.source_offset += len;
            self.pending_len = len;
        }

        /// Swap buffers. The previously prefetched data becomes the current chunk.
        /// Caller must call prefetchNext() before advance().
        pub fn advance(self: *Self) void {
            self.active = 1 - self.active;
        }

        pub fn isDone(self: *const Self) bool {
            return self.done and self.source_offset >= self.source.len;
        }

        /// Iterate through all chunks of the bound source.
        /// Calls handler(chunk) for each chunk. Returns total bytes processed.
        pub fn streamAll(self: *Self, handler: *const fn ([]const u8) void) usize {
            var total: usize = 0;
            self.prefetchNext();
            while (!self.isDone()) {
                self.advance();
                const chunk = self.currentChunk();
                if (chunk.len == 0) break;
                handler(chunk);
                total += chunk.len;
                self.prefetchNext();
            }
            // Handle the final chunk that was prefetched but not yet advanced.
            if (self.pending_len > 0) {
                self.advance();
                const chunk = self.currentChunk();
                if (chunk.len > 0) {
                    handler(chunk);
                    total += chunk.len;
                }
            }
            return total;
        }

        pub fn reset(self: *Self) void {
            self.source_offset = 0;
            self.active = 0;
            self.pending_len = 0;
            self.done = self.source.len == 0;
        }
    };
}

test "streaming basic read" {
    const data = "Hello, streaming world! Extra data padding here.";
    var loader = StreamingWeightLoader(16, 16){};
    loader.bind(data);

    var collected: [128]u8 = undefined;
    var total: usize = 0;

    loader.prefetchNext();
    while (!loader.isDone()) {
        loader.advance();
        const chunk = loader.currentChunk();
        if (chunk.len == 0) break;
        @memcpy(collected[total .. total + chunk.len], chunk);
        total += chunk.len;
        loader.prefetchNext();
    }
    // Drain final pending chunk.
    if (loader.pending_len > 0) {
        loader.advance();
        const chunk = loader.currentChunk();
        @memcpy(collected[total .. total + chunk.len], chunk);
        total += chunk.len;
    }

    try std.testing.expectEqual(data.len, total);
    try std.testing.expectEqualSlices(u8, data, collected[0..total]);
}

test "streaming empty source" {
    var loader = StreamingWeightLoader(16, 16){};
    loader.bind(&.{});
    try std.testing.expect(loader.isDone());
}

test "streaming exact chunk boundary" {
    const data = [_]u8{0xAB} ** 32;
    var loader = StreamingWeightLoader(16, 16){};
    loader.bind(&data);

    var total: usize = 0;
    loader.prefetchNext();
    while (!loader.isDone()) {
        loader.advance();
        const chunk = loader.currentChunk();
        if (chunk.len == 0) break;
        total += chunk.len;
        try std.testing.expectEqual(@as(usize, 16), chunk.len);
        loader.prefetchNext();
    }
    if (loader.pending_len > 0) {
        loader.advance();
        total += loader.currentChunk().len;
    }
    try std.testing.expectEqual(@as(usize, 32), total);
}

test "streaming reset allows re-read" {
    const data = "abcdefghijklmnop"; // exactly 16 bytes
    var loader = StreamingWeightLoader(16, 16){};
    loader.bind(data);

    loader.prefetchNext();
    loader.advance();
    try std.testing.expectEqual(@as(usize, 16), loader.currentChunk().len);

    loader.reset();
    try std.testing.expect(!loader.isDone());
    loader.prefetchNext();
    loader.advance();
    try std.testing.expectEqualSlices(u8, data, loader.currentChunk());
}
