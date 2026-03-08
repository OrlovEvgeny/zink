const std = @import("std");
const testing = std.testing;
const zink = @import("zink");

const zink_loader = zink.model.zink_loader;
const ZinkFile = zink.model.ZinkFile;
const ZinkHeader = zink.model.ZinkHeader;

const MAGIC = zink_loader.MAGIC;
const HEADER_SIZE = zink_loader.HEADER_SIZE;
const TENSOR_ENTRY_SIZE = zink_loader.TENSOR_ENTRY_SIZE;

fn buildTestBuffer(buf: *align(16) [512]u8) void {
    @memset(buf, 0);
    const header: *ZinkHeader = @ptrCast(@alignCast(buf));
    header.magic = MAGIC;
    header.version = 1;
    header.hidden_size = 128;
    header.num_layers = 2;
    header.num_heads = 2;
    header.num_kv_heads = 2;
    header.head_dim = 64;
    header.intermediate_size = 384;
    header.vocab_size = 256;
    header.max_seq_len = 64;
    header.num_tensors = 2;
    header.tensor_table_offset = HEADER_SIZE;
    header.data_offset = @intCast(HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE);
    header.total_size = 512;

    // Tensor 0.
    const e0: *zink_loader.TensorEntry = @ptrCast(@alignCast(
        buf[HEADER_SIZE .. HEADER_SIZE + TENSOR_ENTRY_SIZE],
    ));
    e0.name_hash = zink_loader.fnv1aHash("weight_0");
    e0.offset = 0;
    e0.rows = 4;
    e0.cols = 4;
    e0.packed_size_bytes = 4;
    e0.quant_type = @intFromEnum(zink_loader.QuantType.i2_s);

    // Tensor 1.
    const e1: *zink_loader.TensorEntry = @ptrCast(@alignCast(
        buf[HEADER_SIZE + TENSOR_ENTRY_SIZE .. HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE],
    ));
    e1.name_hash = zink_loader.fnv1aHash("weight_1");
    e1.offset = 16;
    e1.rows = 2;
    e1.cols = 8;
    e1.packed_size_bytes = 4;
    e1.quant_type = @intFromEnum(zink_loader.QuantType.raw_int8);

    // Write some tensor data.
    const data_start = HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE;
    buf[data_start] = 0xAB;
    buf[data_start + 16] = 0xCD;
}

test "parse valid zink file" {
    var buf: [512]u8 align(16) = undefined;
    buildTestBuffer(&buf);

    const zf = try ZinkFile.parse(&buf);
    try testing.expectEqual(@as(u32, MAGIC), zf.header.magic);
    try testing.expectEqual(@as(u32, 128), zf.header.hidden_size);
    try testing.expectEqual(@as(usize, 2), zf.tensor_table.len);
}

test "tensor lookup by hash" {
    var buf: [512]u8 align(16) = undefined;
    buildTestBuffer(&buf);

    const zf = try ZinkFile.parse(&buf);
    const found = zf.findTensor(zink_loader.fnv1aHash("weight_0"));
    try testing.expect(found != null);
    try testing.expectEqual(@as(u32, 4), found.?.rows);
    try testing.expectEqual(@as(u32, 4), found.?.cols);

    // Get tensor data.
    const data = zf.getTensorData(found.?);
    try testing.expectEqual(@as(u8, 0xAB), data[0]);
}

test "tensor lookup miss returns null" {
    var buf: [512]u8 align(16) = undefined;
    buildTestBuffer(&buf);

    const zf = try ZinkFile.parse(&buf);
    try testing.expect(zf.findTensor(zink_loader.fnv1aHash("nonexistent")) == null);
}

test "reject bad magic" {
    var buf: [512]u8 align(16) = undefined;
    buildTestBuffer(&buf);
    const header: *ZinkHeader = @ptrCast(@alignCast(&buf));
    header.magic = 0xBADBAD;
    try testing.expectError(error.InvalidMagic, ZinkFile.parse(&buf));
}

test "reject truncated file" {
    var buf: [32]u8 = [_]u8{0} ** 32;
    try testing.expectError(error.TruncatedFile, ZinkFile.parse(&buf));
}

test "reject unsupported version" {
    var buf: [512]u8 align(16) = undefined;
    buildTestBuffer(&buf);
    const header: *ZinkHeader = @ptrCast(@alignCast(&buf));
    header.version = 99;
    try testing.expectError(error.UnsupportedVersion, ZinkFile.parse(&buf));
}

test "fnv1a hash determinism" {
    const h1 = zink_loader.fnv1aHash("layers.0.q_proj.weight");
    const h2 = zink_loader.fnv1aHash("layers.0.q_proj.weight");
    try testing.expectEqual(h1, h2);

    // Different names must produce different hashes (probabilistic, but ~guaranteed).
    const h3 = zink_loader.fnv1aHash("layers.0.k_proj.weight");
    try testing.expect(h1 != h3);
}

test "header struct size" {
    try testing.expectEqual(@as(usize, 96), @sizeOf(ZinkHeader));
}

test "tensor entry struct size" {
    try testing.expectEqual(@as(usize, 32), @sizeOf(zink_loader.TensorEntry));
}
