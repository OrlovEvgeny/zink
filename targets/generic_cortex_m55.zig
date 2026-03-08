// Generic Cortex-M55 target configuration.
// SRAM size is board-specific; caller must verify via assertFitsIn.

const std = @import("std");
const dispatch = @import("../src/kernels/dispatch.zig");

pub const name = "Generic Cortex-M55";

// Default conservative SRAM estimate. Override for specific boards
// by wrapping this config or using ModelConfig.assertFitsIn() directly.
pub const sram_bytes: usize = 2 * 1024 * 1024;
pub const clock_mhz: usize = 400;

pub const has_helium = true;
pub const has_fpu = true;
pub const has_npu = false;
pub const npu_tops: usize = 0;

pub const kernel_type: dispatch.KernelType = .i2s_helium;

pub const flash_bandwidth_mbps: usize = 50;
pub const dma_chunk_size: usize = 8 * 1024;
pub const dma_alignment: usize = 32;

pub const target_query: std.Target.Query = .{
    .cpu_arch = .thumb,
    .os_tag = .freestanding,
    .abi = .eabi,
    .cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_m55 },
};
