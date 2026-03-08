const std = @import("std");

pub const MAGIC: u32 = 0x5A494E4B; // "ZINK"
pub const HEADER_SIZE: usize = 96;
pub const TENSOR_ENTRY_SIZE: usize = 32;

pub const QuantType = enum(u8) {
    i2_s = 0,
    tl1_packed = 1,
    raw_int8 = 2,
    f16 = 3,
    q4 = 4,
};

/// .zink file header. 96 bytes, extern layout for zero-copy reads.
pub const ZinkHeader = extern struct {
    magic: u32,
    version: u16,
    flags: u16,

    hidden_size: u32,
    num_layers: u16,
    num_heads: u16,
    num_kv_heads: u16,
    head_dim: u16,
    intermediate_size: u32,
    vocab_size: u32,
    max_seq_len: u32,

    rope_theta_fixed: u32, // Q16.16 fixed-point
    kv_group_size: u16,
    quant_type: u8,
    _pad0: u8 = 0,

    num_tensors: u32,
    tensor_table_offset: u32,
    data_offset: u32,
    total_size: u64,

    _reserved: [32]u8 = [_]u8{0} ** 32,

    comptime {
        if (@sizeOf(ZinkHeader) != HEADER_SIZE) {
            @compileError(std.fmt.comptimePrint(
                "ZinkHeader size is {} but expected {}",
                .{ @sizeOf(ZinkHeader), HEADER_SIZE },
            ));
        }
    }
};

/// Tensor table entry. 32 bytes, extern layout.
pub const TensorEntry = extern struct {
    name_hash: u64,
    offset: u32,
    rows: u32,
    cols: u32,
    packed_size_bytes: u32,
    quant_type: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,

    comptime {
        if (@sizeOf(TensorEntry) != TENSOR_ENTRY_SIZE) {
            @compileError(std.fmt.comptimePrint(
                "TensorEntry size is {} but expected {}",
                .{ @sizeOf(TensorEntry), TENSOR_ENTRY_SIZE },
            ));
        }
    }
};

pub const ParseError = error{
    InvalidMagic,
    UnsupportedVersion,
    TruncatedFile,
    InvalidTensorTable,
};

/// Parsed .zink file handle. All slices point into the original buffer.
pub const ZinkFile = struct {
    header: *const ZinkHeader,
    tensor_table: []const TensorEntry,
    data: []const u8,

    pub fn parse(bytes: []const u8) ParseError!ZinkFile {
        if (bytes.len < HEADER_SIZE) return error.TruncatedFile;

        const header: *const ZinkHeader = @ptrCast(@alignCast(bytes[0..HEADER_SIZE]));

        if (header.magic != MAGIC) return error.InvalidMagic;
        if (header.version > 1) return error.UnsupportedVersion;

        const table_offset = header.tensor_table_offset;
        const num_tensors = header.num_tensors;
        const table_end = table_offset + num_tensors * TENSOR_ENTRY_SIZE;

        if (table_end > bytes.len) return error.TruncatedFile;
        if (header.data_offset > bytes.len) return error.TruncatedFile;

        const table_bytes = bytes[table_offset..table_end];
        const tensor_table: []const TensorEntry = @as(
            [*]const TensorEntry,
            @ptrCast(@alignCast(table_bytes.ptr)),
        )[0..num_tensors];

        // Validate tensor offsets.
        for (tensor_table) |entry| {
            const end = header.data_offset + entry.offset + entry.packed_size_bytes;
            if (end > bytes.len) return error.InvalidTensorTable;
        }

        return .{
            .header = header,
            .tensor_table = tensor_table,
            .data = bytes[header.data_offset..],
        };
    }

    /// Find a tensor by name hash. Linear scan (tensor count is small).
    pub fn findTensor(self: ZinkFile, name_hash: u64) ?*const TensorEntry {
        for (self.tensor_table) |*entry| {
            if (entry.name_hash == name_hash) return entry;
        }
        return null;
    }

    /// Get raw data for a tensor entry.
    pub fn getTensorData(self: ZinkFile, entry: *const TensorEntry) []const u8 {
        return self.data[entry.offset .. entry.offset + entry.packed_size_bytes];
    }

    /// Read a per-tensor scale value stored as a 4-byte raw_int8 tensor
    /// (the bytes are reinterpreted as a little-endian f32).
    /// Returns null if the tensor is not found.
    pub fn getScaleValue(self: ZinkFile, name_hash: u64) ?f32 {
        const entry = self.findTensor(name_hash) orelse return null;
        if (entry.packed_size_bytes < 4) return null;
        const data = self.getTensorData(entry);
        return @bitCast(data[0..4].*);
    }
};

/// FNV-1a 64-bit hash. Same algorithm used by zink_pack.py for tensor name lookup.
pub fn fnv1aHash(name: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (name) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    }
    return hash;
}

test "fnv1a known value" {
    const hash = fnv1aHash("layers.0.attention.q_proj.weight");
    try std.testing.expect(hash != 0);
    // Same input must produce same hash.
    try std.testing.expectEqual(hash, fnv1aHash("layers.0.attention.q_proj.weight"));
}

test "parse valid zink buffer" {
    var buf: [256]u8 align(16) = [_]u8{0} ** 256;
    const header: *ZinkHeader = @ptrCast(@alignCast(&buf));
    header.magic = MAGIC;
    header.version = 1;
    header.num_tensors = 1;
    header.tensor_table_offset = HEADER_SIZE;
    header.data_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;

    // Write a tensor entry.
    const entry: *TensorEntry = @ptrCast(@alignCast(buf[HEADER_SIZE .. HEADER_SIZE + TENSOR_ENTRY_SIZE]));
    entry.name_hash = fnv1aHash("test_tensor");
    entry.offset = 0;
    entry.rows = 4;
    entry.cols = 4;
    entry.packed_size_bytes = 4;
    entry.quant_type = @intFromEnum(QuantType.i2_s);

    const zink = try ZinkFile.parse(&buf);
    try std.testing.expectEqual(@as(u32, MAGIC), zink.header.magic);
    try std.testing.expectEqual(@as(usize, 1), zink.tensor_table.len);

    const found = zink.findTensor(fnv1aHash("test_tensor"));
    try std.testing.expect(found != null);
    try std.testing.expectEqual(@as(u32, 4), found.?.rows);
}

test "reject bad magic" {
    var buf: [256]u8 align(16) = [_]u8{0} ** 256;
    const header: *ZinkHeader = @ptrCast(@alignCast(&buf));
    header.magic = 0xDEADBEEF;
    try std.testing.expectError(error.InvalidMagic, ZinkFile.parse(&buf));
}

test "reject truncated file" {
    var buf: [32]u8 = [_]u8{0} ** 32;
    try std.testing.expectError(error.TruncatedFile, ZinkFile.parse(&buf));
}

test "getScaleValue reads f32 from raw tensor" {
    var buf: [256]u8 align(16) = [_]u8{0} ** 256;
    const header: *ZinkHeader = @ptrCast(@alignCast(&buf));
    header.magic = MAGIC;
    header.version = 1;
    header.num_tensors = 1;
    header.tensor_table_offset = HEADER_SIZE;
    header.data_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;

    const entry: *TensorEntry = @ptrCast(@alignCast(buf[HEADER_SIZE .. HEADER_SIZE + TENSOR_ENTRY_SIZE]));
    entry.name_hash = fnv1aHash("layers.0.attention.q_proj.scale");
    entry.offset = 0;
    entry.rows = 1;
    entry.cols = 1;
    entry.packed_size_bytes = 4;
    entry.quant_type = @intFromEnum(QuantType.raw_int8);

    // Write f32 value 42.5 at data offset.
    const data_start = HEADER_SIZE + TENSOR_ENTRY_SIZE;
    const scale_val: f32 = 42.5;
    const scale_bytes: [4]u8 = @bitCast(scale_val);
    @memcpy(buf[data_start .. data_start + 4], &scale_bytes);

    const zink = try ZinkFile.parse(&buf);
    const result = zink.getScaleValue(fnv1aHash("layers.0.attention.q_proj.scale"));
    try std.testing.expect(result != null);
    try std.testing.expectEqual(scale_val, result.?);

    // Non-existent scale returns null.
    const missing = zink.getScaleValue(fnv1aHash("layers.0.attention.k_proj.scale"));
    try std.testing.expect(missing == null);
}
