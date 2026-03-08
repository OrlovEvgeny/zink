#!/usr/bin/env python3
"""
benchmark.py — Automated build and benchmark runner for Zink.

Usage:
    python3 tools/benchmark.py                             # Build and run tests
    python3 tools/benchmark.py --build-only                # Just build, measure compile time
    python3 tools/benchmark.py --target thumb-freestanding --cpu cortex_m55
    python3 tools/benchmark.py --all-boards                # Cross-compile all boards
"""

import argparse
import os
import subprocess
import sys
import time


def run_cmd(cmd, description, capture=True):
    """Run a command and return (success, stdout, duration_ms)."""
    print(f"  {description}...", end="", flush=True)
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=300,
        )
        duration_ms = (time.monotonic() - start) * 1000
        if result.returncode == 0:
            print(f" OK ({duration_ms:.0f}ms)")
        else:
            print(f" FAILED ({duration_ms:.0f}ms)")
            if capture and result.stderr:
                print(f"    stderr: {result.stderr[:500]}")
        return result.returncode == 0, result.stdout if capture else "", duration_ms
    except subprocess.TimeoutExpired:
        duration_ms = (time.monotonic() - start) * 1000
        print(f" TIMEOUT ({duration_ms:.0f}ms)")
        return False, "", duration_ms


def build_native(optimize="Debug"):
    """Build the library for the native target."""
    return run_cmd(
        ["zig", "build", f"-Doptimize={optimize}"],
        f"Building native ({optimize})",
    )


def build_board(board):
    """Cross-compile for a board target."""
    return run_cmd(
        ["zig", "build", f"-Dboard={board}"],
        f"Cross-compiling ({board})",
    )


def build_cross(target, cpu=None):
    """Cross-compile for a specific target triple."""
    cmd = ["zig", "build", f"-Dtarget={target}"]
    if cpu:
        cmd.append(f"-Dcpu={cpu}")
    return run_cmd(cmd, f"Cross-compiling ({target})")


def run_tests(summary=True):
    """Run the full test suite."""
    cmd = ["zig", "build", "test"]
    if summary:
        cmd.extend(["--summary", "all"])
    return run_cmd(cmd, "Running test suite")


def run_memory_planner(model="bitnet_0_7b", target="stm32n6"):
    """Run the memory planner with a predefined config."""
    return run_cmd(
        ["python3", "tools/memory_planner.py", "--model", model, "--target", target],
        f"Memory planner ({model} on {target})",
    )


def generate_test_model():
    """Generate a test .zink model file."""
    return run_cmd(
        ["python3", "tools/zink_pack.py", "--test", "-o", "test_model.zink"],
        "Generating test .zink model",
    )


def get_binary_size(board):
    """Get the binary size for a board build."""
    ok, _, dur = build_board(board)
    if not ok:
        return None, dur

    # Look for the output library.
    lib_path = "zig-out/lib/libzink.a"
    if os.path.exists(lib_path):
        size = os.path.getsize(lib_path)
        return size, dur
    return None, dur


def main():
    parser = argparse.ArgumentParser(description="Zink benchmark runner")
    parser.add_argument("--build-only", action="store_true", help="Only build, skip tests")
    parser.add_argument("--target", type=str, default=None, help="Cross-compilation target triple")
    parser.add_argument("--cpu", type=str, default=None, help="CPU for cross-compilation")
    parser.add_argument("--optimize", type=str, default="Debug", help="Optimization level")
    parser.add_argument("--all-boards", action="store_true",
                        help="Cross-compile for all supported boards")
    parser.add_argument("--generate-test-model", action="store_true",
                        help="Generate test .zink model file")
    parser.add_argument("--binary-sizes", action="store_true",
                        help="Report binary sizes for all boards")
    args = parser.parse_args()

    print("=" * 60)
    print("Zink Benchmark Runner")
    print("=" * 60)
    print()

    results = []
    step = 1

    # Generate test model if requested.
    if args.generate_test_model:
        print(f"[{step}] Test Model Generation")
        ok, _, dur = generate_test_model()
        results.append(("Test model gen", ok, dur))
        step += 1
        if not ok:
            print("\nTest model generation failed.")

    # Native build.
    print(f"\n[{step}] Native Build")
    ok, _, dur = build_native(args.optimize)
    results.append(("Native build", ok, dur))
    step += 1
    if not ok:
        print("\nNative build failed. Aborting.")
        sys.exit(1)

    if not args.build_only:
        # Tests.
        print(f"\n[{step}] Test Suite")
        ok, output, dur = run_tests()
        results.append(("Test suite", ok, dur))
        step += 1

        # Memory planner.
        print(f"\n[{step}] Memory Planner")
        ok, _, dur = run_memory_planner("bitnet_0_7b", "stm32n6")
        results.append(("Memory planner (0.7B)", ok, dur))
        step += 1

    # Cross-compilation for specific target.
    if args.target:
        print(f"\n[{step}] Cross-Compilation ({args.target})")
        ok, _, dur = build_cross(args.target, args.cpu)
        results.append(("Cross-compile", ok, dur))
        step += 1

    # Cross-compile all boards.
    if args.all_boards:
        boards = ["stm32n6", "rp2350", "generic_m55"]
        print(f"\n[{step}] Cross-Compilation (all boards)")
        for board in boards:
            ok, _, dur = build_board(board)
            results.append((f"Board: {board}", ok, dur))
        step += 1

    # Binary sizes.
    if args.binary_sizes:
        boards = ["stm32n6", "rp2350", "generic_m55"]
        print(f"\n[{step}] Binary Sizes")
        for optimize in ["Debug", "ReleaseFast", "ReleaseSmall"]:
            for board in boards:
                ok, _, dur = run_cmd(
                    ["zig", "build", f"-Dboard={board}", f"-Doptimize={optimize}"],
                    f"Building {board} ({optimize})",
                )
                if ok:
                    lib_path = "zig-out/lib/libzink.a"
                    if os.path.exists(lib_path):
                        size = os.path.getsize(lib_path)
                        results.append((
                            f"Size {board}/{optimize}",
                            True,
                            size,
                        ))
                        print(f"    → {size:,} bytes ({size / 1024:.1f} KB)")
        step += 1

    # Summary.
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    all_passed = True
    for name, ok, dur in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        if "Size" in name:
            print(f"  {name:<30} {dur:>10,} bytes")
        else:
            print(f"  {name:<30} {status:<6} {dur:>8.0f}ms")

    print()
    if all_passed:
        print("All benchmarks passed.")
    else:
        print("Some benchmarks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
