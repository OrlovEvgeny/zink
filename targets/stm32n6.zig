// STM32N6570-DK target configuration.
// Cortex-M55 @ 800 MHz, Helium MVE, 4.2 MB contiguous SRAM,
// Neural-ART 600 GOPS NPU, OCTOSPI external flash.

const std = @import("std");
const dispatch = @import("../src/kernels/dispatch.zig");

pub const name = "STM32N6570-DK";

pub const sram_bytes: usize = 4 * 1024 * 1024 + 200 * 1024; // 4.2 MB
pub const clock_mhz: usize = 800;

pub const has_helium = true;
pub const has_fpu = true;
pub const has_npu = true;
pub const npu_tops: usize = 600; // Neural-ART GOPS

pub const kernel_type: dispatch.KernelType = .i2s_helium;

// OCTOSPI flash streaming parameters.
pub const flash_bandwidth_mbps: usize = 100;
pub const dma_chunk_size: usize = 16 * 1024;
pub const dma_alignment: usize = 32;

// Zig cross-compilation target query.
pub const target_query: std.Target.Query = .{
    .cpu_arch = .thumb,
    .os_tag = .freestanding,
    .abi = .eabi,
    .cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_m55 },
};
