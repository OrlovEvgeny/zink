// Raspberry Pi Pico 2 (RP2350) target configuration.
// Cortex-M33 / Hazard3 RISC-V dual core, 520 KB SRAM, no Helium.

const std = @import("std");
const dispatch = @import("../src/kernels/dispatch.zig");

pub const name = "RP2350";

pub const sram_bytes: usize = 520 * 1024;
pub const clock_mhz: usize = 150;

pub const has_helium = false;
pub const has_fpu = true;
pub const has_npu = false;
pub const npu_tops: usize = 0;

pub const kernel_type: dispatch.KernelType = .i2s_generic;

pub const flash_bandwidth_mbps: usize = 24; // QSPI
pub const dma_chunk_size: usize = 4 * 1024;
pub const dma_alignment: usize = 16;

pub const target_query: std.Target.Query = .{
    .cpu_arch = .thumb,
    .os_tag = .freestanding,
    .abi = .eabi,
    .cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_m33 },
};
