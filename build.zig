const std = @import("std");

const BoardConfig = struct {
    name: []const u8,
    target_query: std.Target.Query,
};

const board_stm32n6 = BoardConfig{
    .name = "STM32N6570-DK",
    .target_query = .{
        .cpu_arch = .thumb,
        .os_tag = .freestanding,
        .abi = .eabi,
        .cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_m55 },
    },
};
const board_rp2350 = BoardConfig{
    .name = "RP2350",
    .target_query = .{
        .cpu_arch = .thumb,
        .os_tag = .freestanding,
        .abi = .eabi,
        .cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_m33 },
    },
};
const board_generic_m55 = BoardConfig{
    .name = "Generic Cortex-M55",
    .target_query = .{
        .cpu_arch = .thumb,
        .os_tag = .freestanding,
        .abi = .eabi,
        .cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_m55 },
    },
};

fn lookupBoard(name: []const u8) ?BoardConfig {
    const map = .{
        .{ "stm32n6", board_stm32n6 },
        .{ "rp2350", board_rp2350 },
        .{ "generic_m55", board_generic_m55 },
    };
    inline for (map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) return entry[1];
    }
    return null;
}

pub fn build(b: *std.Build) void {
    const board_option = b.option([]const u8, "board", "Target board (stm32n6, rp2350, generic_m55)");

    // When a board is selected, override target; otherwise use standard options.
    const target = if (board_option) |board_name| blk: {
        const board = lookupBoard(board_name) orelse {
            std.log.err("Unknown board '{s}'. Available: stm32n6, rp2350, generic_m55", .{board_name});
            std.process.exit(1);
        };
        break :blk b.resolveTargetQuery(board.target_query);
    } else b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    // Public module for downstream consumers and test imports.
    const zink_mod = b.addModule("zink", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Static library (cross-compilable to thumb/riscv freestanding).
    const lib = b.addLibrary(.{
        .name = "zink",
        .root_module = zink_mod,
    });
    b.installArtifact(lib);

    // Firmware step: produces a raw binary for the selected board.
    if (board_option != null) {
        const firmware_step = b.step("firmware", "Build firmware binary for target board");
        firmware_step.dependOn(&lib.step);
    }

    // Unit tests (only for native builds — tests can't run on freestanding targets).
    if (board_option == null) {
        const root_tests = b.addTest(.{
            .root_module = zink_mod,
        });
        const run_root_tests = b.addRunArtifact(root_tests);

        const test_step = b.step("test", "Run all tests");
        test_step.dependOn(&run_root_tests.step);

        const test_files = [_][]const u8{
            "tests/test_kernels.zig",
            "tests/test_kv_cache.zig",
            "tests/test_zink.zig",
            "tests/test_e2e.zig",
        };

        for (test_files) |test_file| {
            const test_mod = b.createModule(.{
                .root_source_file = b.path(test_file),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zink", .module = zink_mod },
                },
            });
            const t = b.addTest(.{
                .root_module = test_mod,
            });
            const run_t = b.addRunArtifact(t);
            test_step.dependOn(&run_t.step);
        }
    }
}
