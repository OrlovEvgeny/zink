const std = @import("std");
const config_mod = @import("model/config.zig");
const ModelConfig = config_mod.ModelConfig;
const zink_loader = @import("model/zink_loader.zig");
const ZinkFile = zink_loader.ZinkFile;
const transformer_mod = @import("model/transformer.zig");
const TransformerLayer = transformer_mod.TransformerLayer;
const ScratchBuffers = transformer_mod.ScratchBuffers;
const kv_cache_mod = @import("core/kv_cache.zig");
const KvCache = kv_cache_mod.KvCache;
const arena_mod = @import("core/arena.zig");
const PingPongBuffers = arena_mod.PingPongBuffers;
const dispatch = @import("kernels/dispatch.zig");
const KernelType = dispatch.KernelType;
const rmsnorm = @import("ops/rmsnorm.zig");
const softmax_mod = @import("ops/softmax.zig");

/// Target hardware configuration. Mirrors the fields in targets/*.zig.
pub const TargetConfig = struct {
    name: []const u8,
    sram_bytes: usize,
    kernel_type: KernelType,
    dma_chunk_size: usize,
    dma_alignment: usize,
};

/// Comptime-specialized inference engine.
/// Owns all runtime state: KV-cache, scratch buffers, ping-pong activations.
/// Weights are zero-copy references into the .zink file.
pub fn InferenceEngine(comptime cfg: ModelConfig, comptime target: TargetConfig) type {
    comptime cfg.validate();

    const Layer = TransformerLayer(cfg, target.kernel_type);

    return struct {
        const Self = @This();

        kv_caches: [cfg.num_layers]KvCache(cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len, cfg.kv_group_size) = [_]KvCache(cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len, cfg.kv_group_size){.{}} ** cfg.num_layers,
        layers: [cfg.num_layers]Layer = undefined,
        scratch: ScratchBuffers(cfg) = .{},
        ping_pong: PingPongBuffers(i8, cfg.hidden_size) = .{},
        position: usize = 0,

        // Final RMSNorm weight and LM head weight (from .zink).
        final_rms_weight: *const [cfg.hidden_size]i8 = undefined,
        lm_head_weight: *const [cfg.vocab_size][cfg.hidden_size / 4]u8 = undefined,
        lm_head_scale: f32 = 1.0,

        // Embedding table pointer (from .zink, raw_int8).
        embedding_table: *const [cfg.vocab_size][cfg.hidden_size]i8 = undefined,

        pub const InitError = error{
            MissingTensor,
        };

        /// Initialize the engine from a parsed .zink file.
        /// All weight pointers are set to reference data within the .zink buffer.
        pub fn init(zink_file: *const ZinkFile) InitError!Self {
            var self = Self{};

            for (0..cfg.num_layers) |layer_idx| {
                self.layers[layer_idx] = try loadLayerWeights(zink_file, layer_idx);
            }

            // Embedding table.
            const embed_entry = zink_file.findTensor(
                zink_loader.fnv1aHash("embedding.weight"),
            ) orelse return error.MissingTensor;
            self.embedding_table = @ptrCast(@alignCast(zink_file.getTensorData(embed_entry).ptr));

            // Final RMSNorm.
            const final_rms_entry = zink_file.findTensor(
                zink_loader.fnv1aHash("final_rmsnorm.weight"),
            ) orelse return error.MissingTensor;
            self.final_rms_weight = @ptrCast(@alignCast(zink_file.getTensorData(final_rms_entry).ptr));

            // LM head.
            const lm_head_entry = zink_file.findTensor(
                zink_loader.fnv1aHash("lm_head.weight"),
            ) orelse return error.MissingTensor;
            self.lm_head_weight = @ptrCast(@alignCast(zink_file.getTensorData(lm_head_entry).ptr));
            self.lm_head_scale = zink_file.getScaleValue(zink_loader.fnv1aHash("lm_head.scale")) orelse 1.0;

            return self;
        }

        /// Run one forward step: token_id → logits.
        /// Returns raw i32 logits over the vocabulary.
        pub fn step(self: *Self, token_id: usize) [cfg.vocab_size]i32 {
            // Embedding lookup.
            const embed: *const [cfg.hidden_size]i8 = &self.embedding_table[token_id % cfg.vocab_size];

            // Copy embedding into ping-pong write buffer.
            const write_buf = self.ping_pong.writeBuf();
            @memcpy(write_buf[0..cfg.hidden_size], embed);
            self.ping_pong.swap();

            // Run each transformer layer.
            for (0..cfg.num_layers) |layer_idx| {
                const input: *const [cfg.hidden_size]i8 = @ptrCast(self.ping_pong.readBuf().ptr);
                const output: *[cfg.hidden_size]i8 = @ptrCast(self.ping_pong.writeBuf().ptr);

                self.layers[layer_idx].forward(
                    input,
                    output,
                    &self.scratch,
                    &self.kv_caches[layer_idx],
                    self.position,
                );
                self.ping_pong.swap();
            }

            // Final RMSNorm.
            const final_input: *const [cfg.hidden_size]i8 = @ptrCast(self.ping_pong.readBuf().ptr);
            var normed: [cfg.hidden_size]i8 = undefined;
            rmsnorm.rmsnorm(final_input, self.final_rms_weight, &normed, 4);

            // LM head projection → logits.
            // lm_head_scale is intentionally not applied here. For greedy decoding
            // (argmax), scale is a monotonic transform that doesn't change token
            // ordering. The field is retained for future use (e.g., temperature
            // sampling where absolute logit magnitudes matter).
            var logits: [cfg.vocab_size]i32 = undefined;
            dispatch.ternaryMatVec(cfg.vocab_size, cfg.hidden_size, self.lm_head_weight, &normed, &logits);

            self.position += 1;
            return logits;
        }

        /// Reset the engine state for a new sequence.
        pub fn reset(self: *Self) void {
            for (&self.kv_caches) |*kvc| kvc.clear();
            self.position = 0;
            self.ping_pong = .{};
        }

        /// Greedy argmax over logits.
        pub fn argmaxLogits(logits: *const [cfg.vocab_size]i32) usize {
            var best_idx: usize = 0;
            var best_val: i32 = std.math.minInt(i32);
            for (0..cfg.vocab_size) |i| {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best_idx = i;
                }
            }
            return best_idx;
        }

        fn getScaleOrDefault(zink_file: *const ZinkFile, layer_idx: usize, comptime suffix: []const u8) f32 {
            const name = layerTensorName(layer_idx, suffix);
            return zink_file.getScaleValue(zink_loader.fnv1aHash(&name)) orelse 1.0;
        }

        fn loadLayerWeights(zink_file: *const ZinkFile, layer_idx: usize) InitError!Layer {
            return Layer{
                .wq = try getTensorPtr(zink_file, layer_idx, "attention.q_proj.weight"),
                .wk = try getTensorPtr(zink_file, layer_idx, "attention.k_proj.weight"),
                .wv = try getTensorPtr(zink_file, layer_idx, "attention.v_proj.weight"),
                .wo = try getTensorPtr(zink_file, layer_idx, "attention.o_proj.weight"),
                .w_gate = try getTensorPtrLarge(zink_file, layer_idx, "ffn.gate_proj.weight"),
                .w_up = try getTensorPtrLarge(zink_file, layer_idx, "ffn.up_proj.weight"),
                .w_down = try getTensorPtrDown(zink_file, layer_idx, "ffn.down_proj.weight"),
                .sq = getScaleOrDefault(zink_file, layer_idx, "attention.q_proj.scale"),
                .sk = getScaleOrDefault(zink_file, layer_idx, "attention.k_proj.scale"),
                .sv = getScaleOrDefault(zink_file, layer_idx, "attention.v_proj.scale"),
                .so = getScaleOrDefault(zink_file, layer_idx, "attention.o_proj.scale"),
                .s_gate = getScaleOrDefault(zink_file, layer_idx, "ffn.gate_proj.scale"),
                .s_up = getScaleOrDefault(zink_file, layer_idx, "ffn.up_proj.scale"),
                .s_down = getScaleOrDefault(zink_file, layer_idx, "ffn.down_proj.scale"),
                .rms_attn_weight = try getRmsPtr(zink_file, layer_idx, "attention_norm.weight"),
                .rms_ffn_weight = try getRmsPtr(zink_file, layer_idx, "ffn_norm.weight"),
            };
        }

        fn getTensorPtr(
            zink_file: *const ZinkFile,
            layer_idx: usize,
            comptime suffix: []const u8,
        ) InitError!*const [cfg.hidden_size][cfg.hidden_size / 4]u8 {
            const name = layerTensorName(layer_idx, suffix);
            const entry = zink_file.findTensor(zink_loader.fnv1aHash(&name)) orelse
                return error.MissingTensor;
            return @ptrCast(@alignCast(zink_file.getTensorData(entry).ptr));
        }

        fn getTensorPtrLarge(
            zink_file: *const ZinkFile,
            layer_idx: usize,
            comptime suffix: []const u8,
        ) InitError!*const [cfg.intermediate_size][cfg.hidden_size / 4]u8 {
            const name = layerTensorName(layer_idx, suffix);
            const entry = zink_file.findTensor(zink_loader.fnv1aHash(&name)) orelse
                return error.MissingTensor;
            return @ptrCast(@alignCast(zink_file.getTensorData(entry).ptr));
        }

        fn getTensorPtrDown(
            zink_file: *const ZinkFile,
            layer_idx: usize,
            comptime suffix: []const u8,
        ) InitError!*const [cfg.hidden_size][cfg.intermediate_size / 4]u8 {
            const name = layerTensorName(layer_idx, suffix);
            const entry = zink_file.findTensor(zink_loader.fnv1aHash(&name)) orelse
                return error.MissingTensor;
            return @ptrCast(@alignCast(zink_file.getTensorData(entry).ptr));
        }

        fn getRmsPtr(
            zink_file: *const ZinkFile,
            layer_idx: usize,
            comptime suffix: []const u8,
        ) InitError!*const [cfg.hidden_size]i8 {
            const name = layerTensorName(layer_idx, suffix);
            const entry = zink_file.findTensor(zink_loader.fnv1aHash(&name)) orelse
                return error.MissingTensor;
            return @ptrCast(@alignCast(zink_file.getTensorData(entry).ptr));
        }

        fn layerTensorName(layer_idx: usize, comptime suffix: []const u8) [layerNameLen(suffix)]u8 {
            const prefix = "layers.";
            // Layer index as single digit for layer_idx < 10, double for < 100.
            var name: [layerNameLen(suffix)]u8 = undefined;
            var pos: usize = 0;

            for (prefix) |c| {
                name[pos] = c;
                pos += 1;
            }

            // Write layer index digits.
            if (layer_idx >= 100) {
                name[pos] = '0' + @as(u8, @intCast(layer_idx / 100));
                pos += 1;
            }
            if (layer_idx >= 10) {
                name[pos] = '0' + @as(u8, @intCast((layer_idx / 10) % 10));
                pos += 1;
            }
            name[pos] = '0' + @as(u8, @intCast(layer_idx % 10));
            pos += 1;

            name[pos] = '.';
            pos += 1;

            for (suffix) |c| {
                name[pos] = c;
                pos += 1;
            }

            return name;
        }

        fn layerNameLen(comptime suffix: []const u8) usize {
            // "layers." + up to 3 digits + "." + suffix
            // For simplicity, assume max 3 digits (up to 999 layers).
            const max_digits = if (cfg.num_layers >= 100) 3 else if (cfg.num_layers >= 10) 2 else 1;
            return "layers.".len + max_digits + 1 + suffix.len;
        }
    };
}

test "InferenceEngine type instantiation" {
    const cfg = config_mod.bitnet_test;
    const target = TargetConfig{
        .name = "test",
        .sram_bytes = 4 * 1024 * 1024,
        .kernel_type = .i2s_generic,
        .dma_chunk_size = 4096,
        .dma_alignment = 16,
    };
    const Engine = InferenceEngine(cfg, target);
    try std.testing.expect(@sizeOf(Engine) > 0);
}

test "layer tensor name generation" {
    const cfg = config_mod.bitnet_test;
    const target = TargetConfig{
        .name = "test",
        .sram_bytes = 4 * 1024 * 1024,
        .kernel_type = .i2s_generic,
        .dma_chunk_size = 4096,
        .dma_alignment = 16,
    };
    const Engine = InferenceEngine(cfg, target);
    const name = Engine.layerTensorName(0, "attention.q_proj.weight");
    try std.testing.expectEqualSlices(u8, "layers.0.attention.q_proj.weight", &name);
}

test "argmax logits" {
    const cfg = config_mod.bitnet_test;
    const target = TargetConfig{
        .name = "test",
        .sram_bytes = 4 * 1024 * 1024,
        .kernel_type = .i2s_generic,
        .dma_chunk_size = 4096,
        .dma_alignment = 16,
    };
    const Engine = InferenceEngine(cfg, target);
    var logits: [cfg.vocab_size]i32 = std.mem.zeroes([cfg.vocab_size]i32);
    logits[42] = 999;
    logits[100] = -500;
    try std.testing.expectEqual(@as(usize, 42), Engine.argmaxLogits(&logits));
}
