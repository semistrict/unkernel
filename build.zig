// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
const std = @import("std");

// The guest CPU is AArch64: src/a64 on top of the MMU, block cache and JIT
// core in src/emu and the shared types and softfloat in src/cpu.

/// AArch64 guests (static aarch64-linux-musl C, test/guest/a64/), run under kwemu's
/// kw_a64_run and under qemu-aarch64 by test/a64-test.mjs (outputs must match).
const a64_guests = [_][]const u8{
    "hello",
    "inttest",
    "simdmovetest",
    "cryptotest",
    "atomictest",
    "lsetest",
    "simdtest",
    "fptest",
    "simdfptest",
    "bench",
};
/// A64 builds of portable C guests outside test/guest/a64 (name, source).
const a64_ports = [_][2][]const u8{
    .{ "kbench", "test/guest/kbench.c" },
};
pub fn build(b: *std.Build) void {
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "Optimization mode for the native library and harness (default: ReleaseFast)") orelse .ReleaseFast;
    const native = b.standardTargetOptions(.{});

    // ---- kwemu: the CPU core as a library for the real kernel -----------------
    // wasm64 relocatable object + static archive with the C ABI of
    // include/kw_emu.h. wasm features: memory64 plus what the kernel's C
    // objects (clang's generic cpu) need anyway: bulk memory, sign-ext,
    // nontrapping-fptoint and mutable globals. No threads/atomics/TLS,
    // unless -Dkwemu-atomics (below).
    const lib_optimize = b.option(std.builtin.OptimizeMode, "lib-optimize", "Optimization mode for libkwemu (default: ReleaseFast)") orelse .ReleaseFast;
    const stats = b.option(bool, "kwemu-stats", "kwemu: block-cache counters (kw_debug_stats export, printed by the harness; default: false)") orelse false;
    // JIT tier (src/emu/jitcore.zig, src/a64/jit): hot blocks are compiled to wasm through the
    // optional host imports unkernel.kw_host_jit_{caps,compile,free}. wasm64
    // only; off by default (the object then has no `unkernel` imports). The
    // tests build a JIT-enabled harness and unit-test module regardless.
    const jit = b.option(bool, "kwemu-jit", "kwemu: wasm JIT tier, needs the unkernel.kw_host_jit_* host imports (default: false)") orelse false;
    const jit_threshold = b.option(u32, "kwemu-jit-threshold", "kwemu: retired instructions in a block before it is compiled (default: 131072)") orelse 1 << 17;
    const jit_grow = b.option(u32, "kwemu-jit-grow", "kwemu: exits from a region towards a compiled block before it joins the region (default: 32768 per KiB of the region's module; 0 = never)") orelse 32768;
    // wasm atomics in the wasm64 objects, so that a kernel linked with
    // --shared-memory (wasm-ld refuses objects without atomics and bulk
    // memory) can include kwemu. With them (core.smp) exclusives and LSE atomics
    // are atomic, a store-exclusive a compare-and-swap against the value the
    // load-exclusive read, and barriers full barriers, for an SMP kernel's
    // CPUs on one memory.
    const atomics = b.option(bool, "kwemu-atomics", "kwemu: build the wasm64 objects with the atomics feature, for a shared-memory kernel (default: false)") orelse false;
    // kw_run = the A64 reference loop (src/a64/run.zig runRef) instead of
    // the decoded-op tier: the oracle, for comparisons in the kernel.
    const a64_ref = b.option(bool, "kwemu-a64-ref", "kwemu: kw_run runs the A64 reference interpreter instead of the decoded-op tier (default: false)") orelse false;
    // Programs of the A64 tier's randomized block fuzz (src/a64/tier_tests.zig)
    // in `zig build test`, and the first seed.
    const a64_fuzz = b.option(u32, "kwemu-a64-fuzz", "kwemu: programs in the A64 tier-vs-reference block fuzz of zig build test (default: 0 = 300)") orelse 0;
    const a64_fuzz_seed = b.option(u64, "kwemu-a64-fuzz-seed", "kwemu: first seed of the A64 block fuzz (default: 1)") orelse 1;
    const kw_opts_mod = kwOptions(b, stats, jit, jit_threshold, jit_grow, a64_ref, a64_fuzz, a64_fuzz_seed);
    const kw_opts_jit_mod = if (jit) kw_opts_mod else kwOptions(b, stats, true, jit_threshold, jit_grow, a64_ref, a64_fuzz, a64_fuzz_seed);
    const wasm64 = b.resolveTargetQuery(.{
        .cpu_arch = .wasm64,
        .os_tag = .freestanding,
        .cpu_model = .{ .explicit = &std.Target.wasm.cpu.mvp },
        .cpu_features_add = std.Target.wasm.featureSet(if (atomics)
            &.{ .bulk_memory, .sign_ext, .nontrapping_fptoint, .mutable_globals, .atomics }
        else
            &.{ .bulk_memory, .sign_ext, .nontrapping_fptoint, .mutable_globals }),
    });
    const kw_h_wasm = kwHeader(b, wasm64, lib_optimize);
    const kw_h_native = kwHeader(b, native, optimize);
    const lib_wasm_mod = kwemuModule(b, wasm64, lib_optimize, kw_h_wasm, kw_opts_mod);
    const lib_wasm = b.addLibrary(.{ .name = "kwemu", .linkage = .static, .root_module = lib_wasm_mod });
    b.installArtifact(lib_wasm); // zig-out/lib/libkwemu.a
    const obj_wasm = b.addObject(.{ .name = "kwemu", .root_module = lib_wasm_mod });
    b.getInstallStep().dependOn(&b.addInstallLibFile(obj_wasm.getEmittedBin(), "kwemu.o").step);
    b.getInstallStep().dependOn(&b.addInstallHeaderFile(b.path("include/kw_emu.h"), "kw_emu.h").step);
    const lib_native = b.addLibrary(.{ .name = "kwemu", .linkage = .static, .root_module = kwemuModule(b, native, optimize, kw_h_native, kw_opts_mod) });
    b.getInstallStep().dependOn(&b.addInstallArtifact(lib_native, .{ .dest_dir = .{ .override = .{ .custom = "lib/native" } } }).step);

    // Test harness ("kernel" driving the C ABI only), native and wasm64.
    const harness_native = b.addExecutable(.{
        .name = "kwemu-harness",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/emu/harness_native.zig"),
            .target = native,
            .optimize = optimize,
            .imports = &.{ .{ .name = "kw_h", .module = kw_h_native }, .{ .name = "kwemu_options", .module = kw_opts_mod } },
        }),
    });
    harness_native.root_module.linkLibrary(lib_native);
    b.installArtifact(harness_native);
    const harness_wasm = b.addExecutable(.{
        .name = "kwemu-harness",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/emu/harness_wasm.zig"),
            .target = wasm64,
            .optimize = lib_optimize,
            .strip = lib_optimize != .Debug,
            .imports = &.{ .{ .name = "kw_h", .module = kw_h_wasm }, .{ .name = "kwemu_options", .module = kw_opts_mod } },
        }),
    });
    harness_wasm.entry = .disabled;
    harness_wasm.rdynamic = true;
    harness_wasm.root_module.linkLibrary(lib_wasm);
    b.installArtifact(harness_wasm); // zig-out/bin/kwemu-harness.wasm
    // The wasm64 harness with the JIT tier (test/a64-test.mjs runs it with the
    // JS host of test/kwemu-jit-host.mjs): imports unkernel.kw_host_jit_* and
    // exports its function table.
    const lib_wasm_jit = if (jit) lib_wasm else b.addLibrary(.{
        .name = "kwemu-jit",
        .linkage = .static,
        .root_module = kwemuModule(b, wasm64, lib_optimize, kw_h_wasm, kw_opts_jit_mod),
    });
    const harness_jit = b.addExecutable(.{
        .name = "kwemu-harness-jit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/emu/harness_wasm.zig"),
            .target = wasm64,
            .optimize = lib_optimize,
            .strip = lib_optimize != .Debug,
            .imports = &.{ .{ .name = "kw_h", .module = kw_h_wasm }, .{ .name = "kwemu_options", .module = kw_opts_jit_mod } },
        }),
    });
    harness_jit.entry = .disabled;
    harness_jit.rdynamic = true;
    harness_jit.export_table = true;
    harness_jit.root_module.linkLibrary(lib_wasm_jit);
    b.installArtifact(harness_jit); // zig-out/bin/kwemu-harness-jit.wasm
    // The object a kernel links with the JIT (kernel/build.sh's default), so
    // that test/a64-test.mjs can check its imports too.
    const obj_wasm_jit = if (jit) obj_wasm else b.addObject(.{ .name = "kwemu-jit", .root_module = kwemuModule(b, wasm64, lib_optimize, kw_h_wasm, kw_opts_jit_mod) });
    b.getInstallStep().dependOn(&b.addInstallLibFile(obj_wasm_jit.getEmittedBin(), "kwemu-jit.o").step);

    // A64 guests: ARMv8.0-A (the generic cpu: FP and AdvSIMD, no LSE), no
    // auto-vectorization in the test code itself (libc is what it is).
    const a64_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .musl,
        .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.generic },
    });
    for (a64_guests) |name| addA64Guest(b, a64_target, name, b.fmt("test/guest/a64/{s}.c", .{name}));
    for (a64_ports) |p| addA64Guest(b, a64_target, p[0], p[1]);
    addA64Guest(b, a64_target, "paravirt_bench", "test/guest/a64/paravirt_bench.c");

    // A64 coverage classifier (test/a64-coverage.mjs): which raw words the
    // interpreter rejects as ILLEGAL, and the group each one is routed to.
    const a64_classify = b.addExecutable(.{
        .name = "kwemu-a64-classify",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/a64_classify.zig"),
            .target = native,
            .optimize = optimize,
            .imports = &.{ .{ .name = "kw_h", .module = kw_h_native }, .{ .name = "kwemu_options", .module = kw_opts_mod } },
        }),
    });
    b.installArtifact(a64_classify);

    // ---- tests -------------------------------------------------------------------
    // A64 guests under kwemu (native and wasm64) against qemu-aarch64 (the
    // comparison is skipped, with a note, without qemu-aarch64 on PATH).
    const a64_e2e = b.addSystemCommand(&.{"node"});
    a64_e2e.addFileArg(b.path("test/a64-test.mjs"));
    a64_e2e.addArg(b.getInstallPath(.prefix, ""));
    a64_e2e.step.dependOn(b.getInstallStep());
    a64_e2e.has_side_effects = true;
    const unit_a64 = b.addTest(.{
        .name = "a64-unit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests_a64.zig"),
            .target = native,
            // ReleaseSafe: the tier's block fuzz runs millions of
            // instructions on two machines; safety checks stay on.
            .optimize = .ReleaseSafe,
            .imports = &.{ .{ .name = "kw_h", .module = kw_h_native }, .{ .name = "kwemu_options", .module = kw_opts_mod } },
        }),
    });
    // The A64 unit tests that run code (the tier's and the JIT's) again as
    // wasm64 with the JIT tier, against an eager JS JIT host
    // (test/kwemu-unit-wasm.mjs): every block compiles on its first entry.
    const unit_a64_wasm = b.addTest(.{
        .name = "a64-unit-jit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests_a64_wasm.zig"),
            .target = wasm64,
            .optimize = .ReleaseSafe,
            .single_threaded = true,
            .imports = &.{ .{ .name = "kw_h", .module = kw_h_wasm }, .{ .name = "kwemu_options", .module = kw_opts_jit_mod } },
        }),
        .test_runner = .{ .path = b.path("src/test_runner_wasm.zig"), .mode = .simple },
        // Only these: the rest (softfloat, AdvSIMD primitives) is pulled in
        // through imports and does not depend on the JIT.
        .filters = &.{ "a64 tier", "a64 jit" },
    });
    unit_a64_wasm.entry = .disabled;
    unit_a64_wasm.rdynamic = true;
    unit_a64_wasm.export_table = true;
    const unit_a64_wasm_run = b.addSystemCommand(&.{"node"});
    unit_a64_wasm_run.addFileArg(b.path("test/kwemu-unit-wasm.mjs"));
    unit_a64_wasm_run.addFileArg(unit_a64_wasm.getEmittedBin());
    unit_a64_wasm_run.addArgs(&.{ "--name", "a64" });
    unit_a64_wasm_run.has_side_effects = true;
    // Again with a host that offers no SIMD (jitcore.zig cap_simd): the
    // i64 register locals and generic calls instead of v128 code.
    const unit_a64_wasm_nosimd = b.addSystemCommand(&.{"node"});
    unit_a64_wasm_nosimd.addFileArg(b.path("test/kwemu-unit-wasm.mjs"));
    unit_a64_wasm_nosimd.addFileArg(unit_a64_wasm.getEmittedBin());
    unit_a64_wasm_nosimd.addArgs(&.{ "--name", "a64-nosimd" });
    unit_a64_wasm_nosimd.setEnvironmentVariable("KWEMU_JIT_SIMD", "0");
    unit_a64_wasm_nosimd.has_side_effects = true;
    const test_step = b.step("test", "Run the A64 unit tests (native, and wasm64 with the JIT) and the A64 guests (native, wasm64, wasm64 with the JIT) against qemu-aarch64");
    test_step.dependOn(&b.addRunArtifact(unit_a64).step);
    test_step.dependOn(&unit_a64_wasm_run.step);
    test_step.dependOn(&unit_a64_wasm_nosimd.step);
    test_step.dependOn(&a64_e2e.step);
}

fn kwOptions(b: *std.Build, stats: bool, jit: bool, jit_threshold: u32, jit_grow: u32, a64_ref: bool, a64_fuzz: u32, a64_fuzz_seed: u64) *std.Build.Module {
    const o = b.addOptions();
    o.addOption(bool, "a64_ref", a64_ref);
    o.addOption(u32, "a64_fuzz", a64_fuzz);
    o.addOption(u64, "a64_fuzz_seed", a64_fuzz_seed);
    o.addOption(bool, "stats", stats);
    o.addOption(bool, "jit", jit);
    o.addOption(u32, "jit_threshold", jit_threshold);
    o.addOption(u32, "jit_grow", jit_grow);
    return o.createModule();
}

fn kwHeader(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const tc = b.addTranslateC(.{
        .root_source_file = b.path("include/kw_emu.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = false,
    });
    return tc.createModule();
}

fn kwemuModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, kw_h: *std.Build.Module, opts: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("src/libkwemu.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = true,
        .stack_protector = false,
        .strip = optimize != .Debug,
        .imports = &.{ .{ .name = "kw_h", .module = kw_h }, .{ .name = "kwemu_options", .module = opts } },
    });
}

fn addA64Guest(b: *std.Build, target: std.Build.ResolvedTarget, name: []const u8, src: []const u8) void {
    const mod = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libc = true });
    mod.addIncludePath(b.path("include"));
    mod.addCSourceFile(.{ .file = b.path(src), .flags = &.{ "-std=gnu11", "-Wall", "-fno-vectorize", "-fno-slp-vectorize" } });
    mod.linkSystemLibrary("m", .{});
    const exe = b.addExecutable(.{ .name = b.fmt("a64-{s}", .{name}), .root_module = mod, .linkage = .static });
    installGuest(b, exe, b.fmt("a64/{s}", .{name}));
}

fn installGuest(b: *std.Build, exe: *std.Build.Step.Compile, name: []const u8) void {
    const inst = b.addInstallArtifact(exe, .{
        .dest_dir = .{ .override = .{ .custom = "guest" } },
        .dest_sub_path = b.fmt("{s}.elf", .{name}),
    });
    b.getInstallStep().dependOn(&inst.step);
}
