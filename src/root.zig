// Zink: ternary neural network inference engine for microcontrollers.

pub const engine = @import("main.zig");
pub const InferenceEngine = engine.InferenceEngine;
pub const TargetConfig = engine.TargetConfig;

pub const model = struct {
    pub const config = @import("model/config.zig");
    pub const ModelConfig = config.ModelConfig;
    pub const zink_loader = @import("model/zink_loader.zig");
    pub const ZinkFile = zink_loader.ZinkFile;
    pub const ZinkHeader = zink_loader.ZinkHeader;
    pub const streaming = @import("model/streaming.zig");
    pub const StreamingWeightLoader = streaming.StreamingWeightLoader;
    pub const transformer = @import("model/transformer.zig");
    pub const TransformerLayer = transformer.TransformerLayer;
    pub const ScratchBuffers = transformer.ScratchBuffers;
};

pub const core = struct {
    pub const tensor = @import("core/tensor.zig");
    pub const Tensor = tensor.Tensor;
    pub const arena = @import("core/arena.zig");
    pub const StaticArena = arena.StaticArena;
    pub const PingPongBuffers = arena.PingPongBuffers;
    pub const kv_cache = @import("core/kv_cache.zig");
    pub const KvCache = kv_cache.KvCache;
    pub const Q4Quantizer = kv_cache.Q4Quantizer;
};

pub const kernels = struct {
    pub const reference = @import("kernels/reference.zig");
    pub const i2s_generic = @import("kernels/i2s_generic.zig");
    pub const tl1 = @import("kernels/tl1.zig");
    pub const fatnn = @import("kernels/fatnn.zig");
    pub const i2s_helium = @import("kernels/i2s_helium.zig");
    pub const i2s_rvv = @import("kernels/i2s_rvv.zig");
    pub const dispatch = @import("kernels/dispatch.zig");
};

pub const tokenizer = struct {
    pub const vocab = @import("tokenizer/vocab.zig");
    pub const Vocab = vocab.Vocab;
    pub const SpecialTokens = vocab.SpecialTokens;
    pub const bpe = @import("tokenizer/bpe.zig");
    pub const BpeTokenizer = bpe.BpeTokenizer;
};

pub const ops = struct {
    pub const rmsnorm = @import("ops/rmsnorm.zig");
    pub const softmax = @import("ops/softmax.zig");
    pub const silu = @import("ops/silu.zig");
    pub const rope = @import("ops/rope.zig");
    pub const attention = @import("ops/attention.zig");
};

test {
    @import("std").testing.refAllDecls(@This());
}
