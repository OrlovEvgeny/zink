const std = @import("std");

/// Comptime-dimensioned tensor view over externally-owned storage.
/// No allocations — just a typed pointer with shape metadata.
pub fn Tensor(comptime T: type, comptime shape: anytype) type {
    const ndim = shape.len;
    const total = comptime blk: {
        var product: usize = 1;
        for (shape) |d| product *= d;
        break :blk product;
    };

    return struct {
        const Self = @This();
        pub const element_type = T;
        pub const dimensions = ndim;
        pub const dim_sizes = shape;
        pub const total_elements = total;

        data: *[total]T,

        pub fn fromBuffer(buf: *[total]T) Self {
            return .{ .data = buf };
        }

        pub fn fromSlice(slice: []T) error{BufferTooSmall}!Self {
            if (slice.len < total) return error.BufferTooSmall;
            return .{ .data = slice[0..total] };
        }

        /// Bounds-checked element access via multi-dimensional indices.
        pub fn at(self: Self, indices: [ndim]usize) T {
            return self.data[linearIndex(indices)];
        }

        pub fn setAt(self: Self, indices: [ndim]usize, value: T) void {
            self.data[linearIndex(indices)] = value;
        }

        /// Row slice for 2D tensors.
        pub fn sliceRow(self: Self, row: usize) []T {
            if (ndim != 2) @compileError("sliceRow requires a 2D tensor");
            const cols = shape[1];
            const start = row * cols;
            return self.data[start .. start + cols];
        }

        pub fn asSlice(self: Self) []T {
            return self.data;
        }

        pub fn asConstSlice(self: Self) []const T {
            return self.data;
        }

        fn linearIndex(indices: [ndim]usize) usize {
            var idx: usize = 0;
            var stride: usize = 1;
            comptime var d: usize = ndim;
            inline while (d > 0) {
                d -= 1;
                idx += indices[d] * stride;
                stride *= shape[d];
            }
            return idx;
        }
    };
}

test "1D tensor create and access" {
    var buf = [_]i8{ 10, 20, 30, 40 };
    const t = Tensor(i8, .{4}).fromBuffer(&buf);
    try std.testing.expectEqual(@as(i8, 10), t.at(.{0}));
    try std.testing.expectEqual(@as(i8, 40), t.at(.{3}));
}

test "2D tensor row slicing" {
    var buf = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const Mat = Tensor(i32, .{ 2, 3 });
    const t = Mat.fromBuffer(&buf);
    const row0 = t.sliceRow(0);
    try std.testing.expectEqual(@as(i32, 1), row0[0]);
    try std.testing.expectEqual(@as(i32, 3), row0[2]);
    const row1 = t.sliceRow(1);
    try std.testing.expectEqual(@as(i32, 4), row1[0]);
}

test "2D tensor multi-index access" {
    var buf = [_]u8{ 'a', 'b', 'c', 'd', 'e', 'f' };
    const t = Tensor(u8, .{ 2, 3 }).fromBuffer(&buf);
    try std.testing.expectEqual(@as(u8, 'a'), t.at(.{ 0, 0 }));
    try std.testing.expectEqual(@as(u8, 'd'), t.at(.{ 1, 0 }));
    try std.testing.expectEqual(@as(u8, 'f'), t.at(.{ 1, 2 }));
}

test "3D tensor total elements" {
    const T = Tensor(f32, .{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 24), T.total_elements);
}

test "fromSlice rejects undersized buffer" {
    var buf = [_]i8{ 1, 2 };
    const result = Tensor(i8, .{4}).fromSlice(&buf);
    try std.testing.expectError(error.BufferTooSmall, result);
}
