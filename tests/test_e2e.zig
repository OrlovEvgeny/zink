const std = @import("std");
const testing = std.testing;
const zink = @import("zink");

const config_mod = zink.model.config;
const ModelConfig = config_mod.ModelConfig;
const zink_loader = zink.model.zink_loader;
const ZinkFile = zink.model.ZinkFile;
const ZinkHeader = zink.model.ZinkHeader;
const TensorEntry = zink_loader.TensorEntry;
const InferenceEngine = zink.engine.InferenceEngine;
const TargetConfig = zink.engine.TargetConfig;

const cfg = config_mod.bitnet_test;

const test_target = TargetConfig{
    .name = "test",
    .sram_bytes = 16 * 1024 * 1024,
    .kernel_type = .i2s_generic,
    .dma_chunk_size = 4096,
    .dma_alignment = 16,
};

const Engine = InferenceEngine(cfg, test_target);

// Tensor names needed for a complete model.
const layer_tensor_suffixes = [_][]const u8{
    "attention.q_proj.weight",
    "attention.k_proj.weight",
    "attention.v_proj.weight",
    "attention.o_proj.weight",
    "ffn.gate_proj.weight",
    "ffn.up_proj.weight",
    "ffn.down_proj.weight",
    "attention_norm.weight",
    "ffn_norm.weight",
};

// Per-tensor scale suffixes (f32 stored as 4-byte raw_int8).
const layer_scale_suffixes = [_][]const u8{
    "attention.q_proj.scale",
    "attention.k_proj.scale",
    "attention.v_proj.scale",
    "attention.o_proj.scale",
    "ffn.gate_proj.scale",
    "ffn.up_proj.scale",
    "ffn.down_proj.scale",
};

const global_tensor_names = [_][]const u8{
    "embedding.weight",
    "final_rmsnorm.weight",
    "lm_head.weight",
};

fn countTensors() usize {
    return cfg.num_layers * (layer_tensor_suffixes.len + layer_scale_suffixes.len) + global_tensor_names.len;
}

fn tensorDataSize(comptime name: []const u8) usize {
    // Determine size based on tensor name pattern.
    if (std.mem.indexOf(u8, name, "gate_proj") != null or
        std.mem.indexOf(u8, name, "up_proj") != null)
    {
        return cfg.intermediate_size * cfg.hidden_size / 4;
    }
    if (std.mem.indexOf(u8, name, "down_proj") != null) {
        return cfg.hidden_size * cfg.intermediate_size / 4;
    }
    if (std.mem.indexOf(u8, name, "q_proj") != null or
        std.mem.indexOf(u8, name, "k_proj") != null or
        std.mem.indexOf(u8, name, "v_proj") != null or
        std.mem.indexOf(u8, name, "o_proj") != null)
    {
        return cfg.hidden_size * cfg.hidden_size / 4;
    }
    if (std.mem.indexOf(u8, name, "norm") != null) {
        return cfg.hidden_size;
    }
    if (std.mem.eql(u8, name, "embedding.weight")) {
        return cfg.vocab_size * cfg.hidden_size;
    }
    if (std.mem.eql(u8, name, "final_rmsnorm.weight")) {
        return cfg.hidden_size;
    }
    if (std.mem.eql(u8, name, "lm_head.weight")) {
        return cfg.vocab_size * cfg.hidden_size / 4;
    }
    return 0;
}

fn runtimeTensorDataSize(name: []const u8) usize {
    if (std.mem.indexOf(u8, name, "gate_proj") != null or
        std.mem.indexOf(u8, name, "up_proj") != null)
    {
        return cfg.intermediate_size * cfg.hidden_size / 4;
    }
    if (std.mem.indexOf(u8, name, "down_proj") != null) {
        return cfg.hidden_size * cfg.intermediate_size / 4;
    }
    if (std.mem.indexOf(u8, name, "q_proj") != null or
        std.mem.indexOf(u8, name, "k_proj") != null or
        std.mem.indexOf(u8, name, "v_proj") != null or
        std.mem.indexOf(u8, name, "o_proj") != null)
    {
        return cfg.hidden_size * cfg.hidden_size / 4;
    }
    if (std.mem.indexOf(u8, name, "norm") != null) {
        return cfg.hidden_size;
    }
    if (std.mem.eql(u8, name, "embedding.weight")) {
        return cfg.vocab_size * cfg.hidden_size;
    }
    if (std.mem.eql(u8, name, "final_rmsnorm.weight")) {
        return cfg.hidden_size;
    }
    if (std.mem.eql(u8, name, "lm_head.weight")) {
        return cfg.vocab_size * cfg.hidden_size / 4;
    }
    return 0;
}

fn layerTensorName(buf: []u8, layer_idx: usize, suffix: []const u8) []const u8 {
    const prefix = "layers.";
    var pos: usize = 0;
    @memcpy(buf[pos .. pos + prefix.len], prefix);
    pos += prefix.len;

    buf[pos] = '0' + @as(u8, @intCast(layer_idx % 10));
    pos += 1;

    buf[pos] = '.';
    pos += 1;

    @memcpy(buf[pos .. pos + suffix.len], suffix);
    pos += suffix.len;

    return buf[0..pos];
}

fn runtimeGqaTensorDataSize(name: []const u8) usize {
    const gqa = config_mod.bitnet_test_gqa;
    if (std.mem.indexOf(u8, name, "gate_proj") != null or
        std.mem.indexOf(u8, name, "up_proj") != null)
    {
        return gqa.intermediate_size * gqa.hidden_size / 4;
    }
    if (std.mem.indexOf(u8, name, "down_proj") != null) {
        return gqa.hidden_size * gqa.intermediate_size / 4;
    }
    if (std.mem.indexOf(u8, name, "q_proj") != null or
        std.mem.indexOf(u8, name, "k_proj") != null or
        std.mem.indexOf(u8, name, "v_proj") != null or
        std.mem.indexOf(u8, name, "o_proj") != null)
    {
        return gqa.hidden_size * gqa.hidden_size / 4;
    }
    if (std.mem.indexOf(u8, name, "norm") != null) {
        return gqa.hidden_size;
    }
    if (std.mem.eql(u8, name, "embedding.weight")) {
        return gqa.vocab_size * gqa.hidden_size;
    }
    if (std.mem.eql(u8, name, "final_rmsnorm.weight")) {
        return gqa.hidden_size;
    }
    if (std.mem.eql(u8, name, "lm_head.weight")) {
        return gqa.vocab_size * gqa.hidden_size / 4;
    }
    return 0;
}

// Build a synthetic .zink file in a static buffer.
const ALIGN = 32;
const MAX_BUF = 512 * 1024;

fn buildSyntheticZink(buf: []align(ALIGN) u8) !ZinkFile {
    @memset(buf, 0);

    const num_tensors = countTensors();
    const tensor_table_offset = zink_loader.HEADER_SIZE;
    const data_offset_unaligned = tensor_table_offset + num_tensors * zink_loader.TENSOR_ENTRY_SIZE;
    const data_offset = (data_offset_unaligned + 15) & ~@as(usize, 15);

    // Write header.
    const header: *ZinkHeader = @ptrCast(@alignCast(buf.ptr));
    header.magic = zink_loader.MAGIC;
    header.version = 1;
    header.hidden_size = cfg.hidden_size;
    header.num_layers = cfg.num_layers;
    header.num_heads = cfg.num_heads;
    header.num_kv_heads = cfg.num_kv_heads;
    header.head_dim = cfg.head_dim;
    header.intermediate_size = cfg.intermediate_size;
    header.vocab_size = cfg.vocab_size;
    header.max_seq_len = cfg.max_seq_len;
    header.num_tensors = @intCast(num_tensors);
    header.tensor_table_offset = @intCast(tensor_table_offset);
    header.data_offset = @intCast(data_offset);

    // Write tensor entries and fill data with deterministic patterns.
    var entry_idx: usize = 0;
    var current_data_offset: usize = 0;
    var name_buf: [128]u8 = undefined;

    // Layer tensors (weights + norms).
    for (0..cfg.num_layers) |layer_idx| {
        for (layer_tensor_suffixes) |suffix| {
            const name = layerTensorName(&name_buf, layer_idx, suffix);
            const data_size = runtimeTensorDataSize(name);

            const aligned_data_offset = (current_data_offset + 15) & ~@as(usize, 15);

            const entry: *TensorEntry = @ptrCast(@alignCast(
                buf[tensor_table_offset + entry_idx * zink_loader.TENSOR_ENTRY_SIZE ..][0..zink_loader.TENSOR_ENTRY_SIZE],
            ));
            entry.name_hash = zink_loader.fnv1aHash(name);
            entry.offset = @intCast(aligned_data_offset);
            entry.packed_size_bytes = @intCast(data_size);
            entry.quant_type = @intFromEnum(zink_loader.QuantType.i2_s);

            // For norm weights, use 127 (≈1.0 in Q0.7).
            if (std.mem.indexOf(u8, suffix, "norm") != null) {
                const dst = buf[data_offset + aligned_data_offset .. data_offset + aligned_data_offset + data_size];
                @memset(dst, 127);
            }

            current_data_offset = aligned_data_offset + data_size;
            entry_idx += 1;
        }

        // Per-tensor scales (f32 stored as 4-byte raw_int8).
        for (layer_scale_suffixes) |suffix| {
            const name = layerTensorName(&name_buf, layer_idx, suffix);
            const data_size: usize = 4; // sizeof(f32)
            const aligned_data_offset = (current_data_offset + 15) & ~@as(usize, 15);

            const entry: *TensorEntry = @ptrCast(@alignCast(
                buf[tensor_table_offset + entry_idx * zink_loader.TENSOR_ENTRY_SIZE ..][0..zink_loader.TENSOR_ENTRY_SIZE],
            ));
            entry.name_hash = zink_loader.fnv1aHash(name);
            entry.offset = @intCast(aligned_data_offset);
            entry.packed_size_bytes = @intCast(data_size);
            entry.rows = 1;
            entry.cols = 1;
            entry.quant_type = @intFromEnum(zink_loader.QuantType.raw_int8);

            // Write scale = 10.0 for all projections (non-trivial value).
            const scale_val: f32 = 10.0;
            const scale_bytes: [4]u8 = @bitCast(scale_val);
            @memcpy(buf[data_offset + aligned_data_offset .. data_offset + aligned_data_offset + 4], &scale_bytes);

            current_data_offset = aligned_data_offset + data_size;
            entry_idx += 1;
        }
    }

    // Global tensors.
    for (global_tensor_names) |name| {
        const data_size = runtimeTensorDataSize(name);
        const aligned_data_offset = (current_data_offset + 15) & ~@as(usize, 15);

        const entry: *TensorEntry = @ptrCast(@alignCast(
            buf[tensor_table_offset + entry_idx * zink_loader.TENSOR_ENTRY_SIZE ..][0..zink_loader.TENSOR_ENTRY_SIZE],
        ));
        entry.name_hash = zink_loader.fnv1aHash(name);
        entry.offset = @intCast(aligned_data_offset);
        entry.packed_size_bytes = @intCast(data_size);

        if (std.mem.eql(u8, name, "embedding.weight")) {
            entry.quant_type = @intFromEnum(zink_loader.QuantType.raw_int8);
            // Fill embedding with small values.
            const dst = buf[data_offset + aligned_data_offset .. data_offset + aligned_data_offset + data_size];
            for (dst, 0..) |*b, i| {
                b.* = @bitCast(@as(i8, @intCast(@as(i32, @intCast(i % 256)) - 128)));
            }
        } else if (std.mem.eql(u8, name, "final_rmsnorm.weight")) {
            entry.quant_type = @intFromEnum(zink_loader.QuantType.raw_int8);
            const dst = buf[data_offset + aligned_data_offset .. data_offset + aligned_data_offset + data_size];
            @memset(dst, 127);
        } else {
            entry.quant_type = @intFromEnum(zink_loader.QuantType.i2_s);
        }

        current_data_offset = aligned_data_offset + data_size;
        entry_idx += 1;
    }

    header.total_size = @intCast(data_offset + current_data_offset);

    return ZinkFile.parse(buf[0 .. data_offset + current_data_offset]);
}

test "e2e: engine initialization from synthetic zink" {
    var buf: [MAX_BUF]u8 align(ALIGN) = undefined;
    const zf = try buildSyntheticZink(&buf);
    const engine = try Engine.init(&zf);
    try testing.expectEqual(@as(usize, 0), engine.position);
}

test "e2e: single step produces logits" {
    var buf: [MAX_BUF]u8 align(ALIGN) = undefined;
    const zf = try buildSyntheticZink(&buf);
    var engine = try Engine.init(&zf);

    const logits = engine.step(0);
    // Logits should be an array of vocab_size i32 values.
    try testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);
    // Position should advance.
    try testing.expectEqual(@as(usize, 1), engine.position);
}

test "e2e: deterministic output (same input same output)" {
    var buf: [MAX_BUF]u8 align(ALIGN) = undefined;
    const zf = try buildSyntheticZink(&buf);

    // Run 1.
    var engine1 = try Engine.init(&zf);
    const logits1 = engine1.step(42);

    // Run 2.
    var engine2 = try Engine.init(&zf);
    const logits2 = engine2.step(42);

    for (0..cfg.vocab_size) |i| {
        try testing.expectEqual(logits1[i], logits2[i]);
    }
}

test "e2e: multiple steps advance position" {
    var buf: [MAX_BUF]u8 align(ALIGN) = undefined;
    const zf = try buildSyntheticZink(&buf);
    var engine = try Engine.init(&zf);

    _ = engine.step(0);
    _ = engine.step(1);
    _ = engine.step(2);

    try testing.expectEqual(@as(usize, 3), engine.position);
    // Each layer's KV cache should have 3 entries.
    for (&engine.kv_caches) |*kvc| {
        try testing.expectEqual(@as(usize, 3), kvc.validLen());
    }
}

test "e2e: reset clears state" {
    var buf: [MAX_BUF]u8 align(ALIGN) = undefined;
    const zf = try buildSyntheticZink(&buf);
    var engine = try Engine.init(&zf);

    _ = engine.step(0);
    _ = engine.step(1);
    engine.reset();

    try testing.expectEqual(@as(usize, 0), engine.position);
    for (&engine.kv_caches) |*kvc| {
        try testing.expectEqual(@as(usize, 0), kvc.validLen());
    }
}

test "e2e: argmax selects maximum logit" {
    var logits: [cfg.vocab_size]i32 = std.mem.zeroes([cfg.vocab_size]i32);
    logits[7] = 1000;
    logits[200] = -500;
    try testing.expectEqual(@as(usize, 7), Engine.argmaxLogits(&logits));
}

test "e2e: scale tensors loaded with correct values" {
    var buf: [MAX_BUF]u8 align(ALIGN) = undefined;
    const zf = try buildSyntheticZink(&buf);
    const engine = try Engine.init(&zf);

    // All layers should have scale = 10.0 (set in buildSyntheticZink).
    for (0..cfg.num_layers) |i| {
        try testing.expectEqual(@as(f32, 10.0), engine.layers[i].sq);
        try testing.expectEqual(@as(f32, 10.0), engine.layers[i].sk);
        try testing.expectEqual(@as(f32, 10.0), engine.layers[i].sv);
        try testing.expectEqual(@as(f32, 10.0), engine.layers[i].so);
        try testing.expectEqual(@as(f32, 10.0), engine.layers[i].s_gate);
        try testing.expectEqual(@as(f32, 10.0), engine.layers[i].s_up);
        try testing.expectEqual(@as(f32, 10.0), engine.layers[i].s_down);
    }
}

test "e2e: GQA config validates and instantiates" {
    // GQA: 8 query heads, 2 KV heads (4:1 ratio).
    const gqa_cfg = config_mod.bitnet_test_gqa;
    comptime gqa_cfg.validate();

    const gqa_target = TargetConfig{
        .name = "test_gqa",
        .sram_bytes = 64 * 1024 * 1024,
        .kernel_type = .i2s_generic,
        .dma_chunk_size = 4096,
        .dma_alignment = 16,
    };

    const GqaEngine = InferenceEngine(gqa_cfg, gqa_target);
    // Verify GQA heads_per_kv ratio is computed correctly at comptime.
    // 8 query heads / 2 KV heads = 4.
    try testing.expectEqual(@as(usize, 4), gqa_cfg.num_heads / gqa_cfg.num_kv_heads);
    try testing.expect(@sizeOf(GqaEngine) > 0);
}

test "e2e: sliding window config validates" {
    const sw_cfg = config_mod.bitnet_test_windowed;
    comptime sw_cfg.validate();
    try testing.expectEqual(@as(?comptime_int, 16), sw_cfg.sliding_window_size);
}

test "e2e: GQA forward pass produces logits" {
    const gqa_cfg = config_mod.bitnet_test_gqa;
    const gqa_target = TargetConfig{
        .name = "test_gqa",
        .sram_bytes = 64 * 1024 * 1024,
        .kernel_type = .i2s_generic,
        .dma_chunk_size = 4096,
        .dma_alignment = 16,
    };
    const GqaEngine = InferenceEngine(gqa_cfg, gqa_target);

    const gqa_layer_suffixes = [_][]const u8{
        "attention.q_proj.weight",
        "attention.k_proj.weight",
        "attention.v_proj.weight",
        "attention.o_proj.weight",
        "ffn.gate_proj.weight",
        "ffn.up_proj.weight",
        "ffn.down_proj.weight",
        "attention_norm.weight",
        "ffn_norm.weight",
    };
    const gqa_scale_suffixes = [_][]const u8{
        "attention.q_proj.scale",
        "attention.k_proj.scale",
        "attention.v_proj.scale",
        "attention.o_proj.scale",
        "ffn.gate_proj.scale",
        "ffn.up_proj.scale",
        "ffn.down_proj.scale",
    };

    const gqa_num_tensors = gqa_cfg.num_layers * (gqa_layer_suffixes.len + gqa_scale_suffixes.len) + global_tensor_names.len;
    const gqa_table_offset = zink_loader.HEADER_SIZE;
    const gqa_data_offset = (gqa_table_offset + gqa_num_tensors * zink_loader.TENSOR_ENTRY_SIZE + 15) & ~@as(usize, 15);

    const GQA_BUF = 2 * 1024 * 1024;
    var gqa_buf: [GQA_BUF]u8 align(ALIGN) = undefined;
    @memset(&gqa_buf, 0);

    // Write header.
    const gqa_header: *ZinkHeader = @ptrCast(@alignCast(&gqa_buf));
    gqa_header.magic = zink_loader.MAGIC;
    gqa_header.version = 1;
    gqa_header.hidden_size = gqa_cfg.hidden_size;
    gqa_header.num_layers = gqa_cfg.num_layers;
    gqa_header.num_heads = gqa_cfg.num_heads;
    gqa_header.num_kv_heads = gqa_cfg.num_kv_heads;
    gqa_header.head_dim = gqa_cfg.head_dim;
    gqa_header.intermediate_size = gqa_cfg.intermediate_size;
    gqa_header.vocab_size = gqa_cfg.vocab_size;
    gqa_header.max_seq_len = gqa_cfg.max_seq_len;
    gqa_header.num_tensors = @intCast(gqa_num_tensors);
    gqa_header.tensor_table_offset = @intCast(gqa_table_offset);
    gqa_header.data_offset = @intCast(gqa_data_offset);

    var gqa_entry_idx: usize = 0;
    var gqa_data_pos: usize = 0;
    var gqa_name_buf: [128]u8 = undefined;

    for (0..gqa_cfg.num_layers) |layer_idx| {
        for (gqa_layer_suffixes) |suffix| {
            const name = layerTensorName(&gqa_name_buf, layer_idx, suffix);
            const data_size = runtimeGqaTensorDataSize(name);
            const aligned_pos = (gqa_data_pos + 15) & ~@as(usize, 15);

            const entry: *TensorEntry = @ptrCast(@alignCast(
                gqa_buf[gqa_table_offset + gqa_entry_idx * zink_loader.TENSOR_ENTRY_SIZE ..][0..zink_loader.TENSOR_ENTRY_SIZE],
            ));
            entry.name_hash = zink_loader.fnv1aHash(name);
            entry.offset = @intCast(aligned_pos);
            entry.packed_size_bytes = @intCast(data_size);
            entry.quant_type = @intFromEnum(zink_loader.QuantType.i2_s);

            if (std.mem.indexOf(u8, suffix, "norm") != null) {
                const dst = gqa_buf[gqa_data_offset + aligned_pos .. gqa_data_offset + aligned_pos + data_size];
                @memset(dst, 127);
            }

            gqa_data_pos = aligned_pos + data_size;
            gqa_entry_idx += 1;
        }

        for (gqa_scale_suffixes) |suffix| {
            const name = layerTensorName(&gqa_name_buf, layer_idx, suffix);
            const aligned_pos = (gqa_data_pos + 15) & ~@as(usize, 15);

            const entry: *TensorEntry = @ptrCast(@alignCast(
                gqa_buf[gqa_table_offset + gqa_entry_idx * zink_loader.TENSOR_ENTRY_SIZE ..][0..zink_loader.TENSOR_ENTRY_SIZE],
            ));
            entry.name_hash = zink_loader.fnv1aHash(name);
            entry.offset = @intCast(aligned_pos);
            entry.packed_size_bytes = 4;
            entry.rows = 1;
            entry.cols = 1;
            entry.quant_type = @intFromEnum(zink_loader.QuantType.raw_int8);

            const scale_val: f32 = 10.0;
            const scale_bytes: [4]u8 = @bitCast(scale_val);
            @memcpy(gqa_buf[gqa_data_offset + aligned_pos .. gqa_data_offset + aligned_pos + 4], &scale_bytes);

            gqa_data_pos = aligned_pos + 4;
            gqa_entry_idx += 1;
        }
    }

    for (global_tensor_names) |name| {
        const data_size = runtimeGqaTensorDataSize(name);
        const aligned_pos = (gqa_data_pos + 15) & ~@as(usize, 15);

        const entry: *TensorEntry = @ptrCast(@alignCast(
            gqa_buf[gqa_table_offset + gqa_entry_idx * zink_loader.TENSOR_ENTRY_SIZE ..][0..zink_loader.TENSOR_ENTRY_SIZE],
        ));
        entry.name_hash = zink_loader.fnv1aHash(name);
        entry.offset = @intCast(aligned_pos);
        entry.packed_size_bytes = @intCast(data_size);

        if (std.mem.eql(u8, name, "embedding.weight")) {
            entry.quant_type = @intFromEnum(zink_loader.QuantType.raw_int8);
            const dst = gqa_buf[gqa_data_offset + aligned_pos .. gqa_data_offset + aligned_pos + data_size];
            for (dst, 0..) |*b, i| {
                b.* = @bitCast(@as(i8, @intCast(@as(i32, @intCast(i % 256)) - 128)));
            }
        } else if (std.mem.eql(u8, name, "final_rmsnorm.weight")) {
            entry.quant_type = @intFromEnum(zink_loader.QuantType.raw_int8);
            const dst = gqa_buf[gqa_data_offset + aligned_pos .. gqa_data_offset + aligned_pos + data_size];
            @memset(dst, 127);
        } else {
            entry.quant_type = @intFromEnum(zink_loader.QuantType.i2_s);
        }

        gqa_data_pos = aligned_pos + data_size;
        gqa_entry_idx += 1;
    }

    gqa_header.total_size = @intCast(gqa_data_offset + gqa_data_pos);

    const gqa_zf = try ZinkFile.parse(gqa_buf[0 .. gqa_data_offset + gqa_data_pos]);
    var gqa_engine = try GqaEngine.init(&gqa_zf);

    const logits = gqa_engine.step(0);
    try testing.expectEqual(@as(usize, gqa_cfg.vocab_size), logits.len);
    try testing.expectEqual(@as(usize, 1), gqa_engine.position);

    // Run a second step to exercise multi-head attention with GQA head sharing.
    const logits2 = gqa_engine.step(1);
    try testing.expectEqual(@as(usize, gqa_cfg.vocab_size), logits2.len);
    try testing.expectEqual(@as(usize, 2), gqa_engine.position);
}

test "e2e: sliding window forward pass multi-step" {
    // Use the windowed config (window_size=16) and run 24 steps (past window).
    const sw_cfg = config_mod.bitnet_test_windowed;

    const sw_target = TargetConfig{
        .name = "test_sw",
        .sram_bytes = 16 * 1024 * 1024,
        .kernel_type = .i2s_generic,
        .dma_chunk_size = 4096,
        .dma_alignment = 16,
    };

    const SwEngine = InferenceEngine(sw_cfg, sw_target);

    // Build synthetic .zink for the windowed config (same dims as bitnet_test).
    var sw_buf: [MAX_BUF]u8 align(ALIGN) = undefined;
    const sw_zf = try buildSyntheticZink(&sw_buf);
    var sw_engine = try SwEngine.init(&sw_zf);

    // Run 24 steps, well past the sliding window size of 16.
    for (0..24) |token| {
        _ = sw_engine.step(token % sw_cfg.vocab_size);
    }

    try testing.expectEqual(@as(usize, 24), sw_engine.position);

    // KV-cache should have all 24 entries stored (ring buffer capacity is max_seq_len=64).
    for (&sw_engine.kv_caches) |*kvc| {
        try testing.expectEqual(@as(usize, 24), kvc.validLen());
    }

    // Run one more step and verify it still works (attention window = 16, not 24).
    const logits = sw_engine.step(0);
    try testing.expectEqual(@as(usize, sw_cfg.vocab_size), logits.len);
    try testing.expectEqual(@as(usize, 25), sw_engine.position);
}

test "e2e: missing tensor returns error" {
    // Build a minimal invalid zink buffer (header only, no tensors).
    var buf: [256]u8 align(ALIGN) = std.mem.zeroes([256]u8);
    const header: *ZinkHeader = @ptrCast(@alignCast(&buf));
    header.magic = zink_loader.MAGIC;
    header.version = 1;
    header.num_tensors = 0;
    header.tensor_table_offset = zink_loader.HEADER_SIZE;
    header.data_offset = zink_loader.HEADER_SIZE;
    header.total_size = 256;

    const zf = try ZinkFile.parse(&buf);
    try testing.expectError(error.MissingTensor, Engine.init(&zf));
}
