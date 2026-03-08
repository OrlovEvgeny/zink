const std = @import("std");

/// Bump allocator over a fixed-size, statically-allocated buffer.
/// No free — only reset. Suitable for per-inference scratch space.
pub fn StaticArena(comptime size: usize) type {
    return struct {
        const Self = @This();
        const alignment = 32;

        buf: [size]u8 align(alignment) = undefined,
        offset: usize = 0,

        pub const AllocError = error{OutOfMemory};

        /// Bump-allocate a comptime-known number of elements.
        /// Returns a pointer into the arena's buffer.
        pub fn alloc(self: *Self, comptime T: type, comptime count: usize) AllocError!*[count]T {
            const byte_count = count * @sizeOf(T);
            const align_of = @alignOf(T);
            const aligned_offset = std.mem.alignForward(usize, self.offset, align_of);
            const end = aligned_offset + byte_count;

            if (end > size) return error.OutOfMemory;

            const ptr: *[count]T = @ptrCast(@alignCast(self.buf[aligned_offset..end]));
            self.offset = end;
            return ptr;
        }

        /// Reset the arena, invalidating all previous allocations.
        pub fn reset(self: *Self) void {
            self.offset = 0;
        }

        pub fn bytesUsed(self: *const Self) usize {
            return self.offset;
        }

        pub fn bytesRemaining(self: *const Self) usize {
            return size - self.offset;
        }
    };
}

/// Two alternating buffers for pipeline-style processing.
/// Read from one while writing to the other, then swap.
/// Caller must not hold references across swap().
pub fn PingPongBuffers(comptime T: type, comptime size: usize) type {
    return struct {
        const Self = @This();
        const buf_alignment = 32;

        bufs: [2][size]T align(buf_alignment) = std.mem.zeroes([2][size]T),
        active: u1 = 0,

        pub fn readBuf(self: *const Self) []const T {
            return &self.bufs[self.active];
        }

        pub fn writeBuf(self: *Self) []T {
            return &self.bufs[1 - self.active];
        }

        pub fn swap(self: *Self) void {
            self.active = 1 - self.active;
        }
    };
}

test "StaticArena basic allocation" {
    var a = StaticArena(256){};
    const p1 = try a.alloc(i32, 4);
    p1[0] = 42;
    try std.testing.expectEqual(@as(i32, 42), p1[0]);

    const p2 = try a.alloc(u8, 8);
    p2[3] = 0xFF;
    try std.testing.expectEqual(@as(u8, 0xFF), p2[3]);
}

test "StaticArena overflow returns OutOfMemory" {
    var a = StaticArena(16){};
    _ = try a.alloc(u8, 16);
    try std.testing.expectError(error.OutOfMemory, a.alloc(u8, 1));
}

test "StaticArena reset allows reuse" {
    var a = StaticArena(64){};
    _ = try a.alloc(u8, 64);
    a.reset();
    try std.testing.expectEqual(@as(usize, 0), a.bytesUsed());
    _ = try a.alloc(u8, 64);
}

test "PingPongBuffers swap semantics" {
    var pp = PingPongBuffers(i8, 4){};
    const w = pp.writeBuf();
    w[0] = 42;
    pp.swap();
    const r = pp.readBuf();
    try std.testing.expectEqual(@as(i8, 42), r[0]);
}

test "PingPongBuffers read and write are different buffers" {
    var pp = PingPongBuffers(u8, 8){};
    const r_ptr = @intFromPtr(pp.readBuf().ptr);
    const w_ptr = @intFromPtr(pp.writeBuf().ptr);
    try std.testing.expect(r_ptr != w_ptr);
}
