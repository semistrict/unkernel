// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! Unit tests of the A64 JIT core (emit.zig, the JIT parts of tier.zig and
//! jitcore.zig as A64 uses it): hand-backs resume at op k with XZR = 0,
//! compiled self-loops keep the budget exact and poll *irq_pending and the
//! code epoch at their back edges, regions across pages, a flush asked for
//! during a compile. Each test also runs natively (the tier alone) where it
//! does not depend on compiled code. The per-kind tests are in the family
//! files (kindtest.zig), the block fuzz in tier_tests.zig.

const std = @import("std");
const t = std.testing;
const c = @import("../common.zig");
const kwemu = @import("../../emu/kwemu.zig");
const jc = @import("../../emu/jitcore.zig");
const mach = @import("../machine.zig");
const tier = @import("../tier.zig");
const Machine = mach.Machine;
const Cpu = kwemu.Cpu;

/// The JIT host calls `kwemu_test_compile_hook` (tests_a64_wasm.zig) after
/// every compile, inside kwemu's compile window: what "another CPU" does
/// then.
pub const Hook = struct {
    /// Set this word to 1 (an interrupt raised during the compile).
    irq: ?*u32 = null,
    /// Ask this CPU for a full flush during every `flush_every`-th compile
    /// (0: never), as a CPU flushing TLBs would (jitcore.zig, windowClose).
    cpu: ?*Cpu = null,
    flush_every: u32 = 0,
    compiles: u32 = 0,
    flushes: u32 = 0,
    /// Called after every compile (after the above), with the compile's
    /// number (1, 2, ...): a family test's own stand-in for another CPU.
    call: ?*const fn (n: u32) void = null,
};
pub var hook: Hook = .{};

pub fn compileHook() void {
    hook.compiles += 1;
    if (hook.irq) |p| p.* = 1;
    if (hook.cpu) |cpu| {
        if (hook.flush_every != 0 and hook.compiles % hook.flush_every == 0) {
            // Another CPU sees the window open and asks instead of waiting.
            if (@atomicLoad(u32, &cpu.jit_busy, .seq_cst) == 1) {
                @atomicStore(u32, &cpu.flush_all, 1, .seq_cst);
                hook.flushes += 1;
            }
        }
    }
    if (hook.call) |f| f(hook.compiles);
}

const movz_x0_1: u32 = 0xd280_0020;
const movz_x0_7: u32 = 0xd280_00e0;
const movz_x1_5: u32 = 0xd280_00a1;
const movz_x2_3: u32 = 0xd280_0062;
const add_x4_x1_xzr: u32 = 0x8b1f_0024; // add x4, x1, xzr (shifted register): reads slot 32
const svc = mach.svc0;

fn b(from: u64, to: u64) u32 {
    return 0x1400_0000 | (@as(u32, @truncate((to -% from) >> 2)) & 0x3ff_ffff);
}

fn jitStats(m: *Machine) if (jc.enabled) *jc.State else void {
    if (jc.enabled) return &m.cpu.priv.jit;
}

test "a64 jit: a hand-back resumes at op k of the same decode with XZR = 0" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    // movi, movi (compiled), add_rr reading XZR (handed back), svc.
    m.code(0x10000, &.{ movz_x0_1, movz_x1_5, add_x4_x1_xzr, svc });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(@as(u64, 5), m.regs.x[4]);
    try t.expectEqual(@as(u64, 1), m.regs.x[0]);
    try t.expectEqual(@as(u64, 0x10010), m.regs.pc);
    try t.expectEqual(@as(u64, 4), m.cpu.icount);
    if (jc.enabled) {
        const st = jitStats(&m);
        try t.expect(st.compiles >= 1);
        try t.expect(st.handbacks >= 1);
    }
}

/// movz x0, #7; b . - 4: a self-loop of two instructions at 0x10000.
fn selfLoop(m: *Machine) void {
    m.code(0x10000, &.{ movz_x0_7, b(0x10004, 0x10000) });
}

test "a64 jit: a compiled self-loop retires exactly its budget" {
    for ([_]u64{ 1, 2, 3, 999, 1000, 1001 }) |budget| {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        selfLoop(&m);
        try t.expectEqual(@as(u32, mach.EV_BUDGET), m.run(budget));
        try t.expectEqual(budget, m.cpu.icount);
        try t.expectEqual(@as(u64, if (budget % 2 == 0) 0x10000 else 0x10004), m.regs.pc);
        try t.expectEqual(@as(u64, 7), m.regs.x[0]);
        if (jc.enabled and budget >= 1000) try t.expect(jitStats(&m).entries <= 2);
    }
}

test "a64 jit: a compiled self-loop polls *irq_pending at its back edge" {
    if (!jc.enabled) return error.SkipZigTest;
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    var irq: u32 = 0;
    m.cpu.irq_pending = &irq;
    selfLoop(&m);
    hook = .{ .irq = &irq };
    defer hook = .{};
    // The interrupt is raised during the compile: the first pass runs, the
    // back edge sees it.
    try t.expectEqual(@as(u32, mach.EV_INTERRUPT), m.run(100_000));
    try t.expectEqual(@as(u64, 2), m.cpu.icount);
    try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
}

test "a64 jit: a compiled self-loop polls the code epoch at its back edge" {
    if (!jc.enabled) return error.SkipZigTest;
    for ([_]bool{ false, true }) |changed| {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        selfLoop(&m);
        // A stand-in for the global epoch that another CPU bumped: compiled
        // code reads it through epoch_ptr; `run` reads the real one (which
        // did not change), so each stop resumes at the next lookup.
        var fake: u64 = m.cpu.priv.a64.seen_epoch +% @intFromBool(changed);
        m.cpu.priv.a64.epoch_ptr = @intFromPtr(&fake);
        try t.expectEqual(@as(u32, mach.EV_BUDGET), m.run(1000));
        try t.expectEqual(@as(u64, 1000), m.cpu.icount);
        try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
        const entries = jitStats(&m).entries;
        if (changed) try t.expect(entries >= 450) else try t.expectEqual(@as(u64, 1), entries);
    }
}

test "a64 jit: two blocks on two pages that branch to each other form a region" {
    // Every budget from 1 to 40 stops at some member boundary or inside a
    // member (single-stepped); registers start nonzero, so a member's
    // registers written back without having been loaded would show.
    var budgets: [44]u64 = undefined;
    for (0..40) |i| budgets[i] = i + 1;
    budgets[40..].* = .{ 1000, 9999, 10000, 10001 };
    for (budgets) |budget| {
        var ms: [2]Machine = .{ try Machine.init(t.allocator), try Machine.init(t.allocator) };
        defer for (&ms) |*m| m.deinit();
        for (&ms) |*m| {
            m.code(0x20000, &.{ movz_x1_5, b(0x20004, 0x10000) });
            m.code(0x10000, &.{ movz_x0_1, movz_x2_3, b(0x10008, 0x20000) });
            for (0..3) |k| m.regs.x[k] = 0x1111 * (k + 7);
            if (jc.enabled) m.cpu.priv.jit.grow_threshold = 1;
        }
        const kt = ms[0].run(budget);
        const kr = ms[1].runRef(budget);
        try t.expectEqual(kr, kt);
        try t.expectEqual(ms[1].cpu.icount, ms[0].cpu.icount);
        try t.expect(std.mem.eql(u8, std.mem.asBytes(&ms[0].regs), std.mem.asBytes(&ms[1].regs)));
        if (jc.enabled and budget >= 1000) {
            const st = jitStats(&ms[0]);
            // The first region grew along its exit to the other page.
            try t.expect(st.exit_grows >= 1);
            try t.expect(st.entries < 100);
        }
    }
}

test "a64 jit: a flush asked for during a compile looks the block up again" {
    if (!jc.enabled) return error.SkipZigTest;
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.code(0x10000, &.{ movz_x0_1, movz_x1_5, svc });
    hook = .{ .cpu = m.cpu, .flush_every = 1 };
    defer hook = .{};
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(@as(u64, 5), m.regs.x[1]);
    try t.expectEqual(@as(u64, 3), m.cpu.icount);
    try t.expect(hook.flushes >= 1);
}

/// ADDs per chain page: with the branch, one block (`tier.max_block`).
const chain_adds = @min(40, tier.max_block - 1);

/// An outer loop of `iters` passes over a self-loop at 0x10000 (x1 counts
/// down from 100) and a chain of 19 pages from 0x40000, each `chain_adds`
/// ADDs of x3 and a branch to the next; a 20th page counts x2 down and
/// branches back to 0x10000, then SVC. Each pass decodes 800 to 1000 arena
/// slots.
fn chainProgram(m: *Machine, iters: u16) void {
    const add_x0_1: u32 = 0x9100_0400; // add x0, x0, #1
    const add_x3_1: u32 = 0x9100_0463; // add x3, x3, #1
    const subs_x1_1: u32 = 0xf100_0421; // subs x1, x1, #1
    const subs_x2_1: u32 = 0xf100_0442; // subs x2, x2, #1
    const bne_back3: u32 = 0x5400_0000 | ((@as(u32, @bitCast(@as(i32, -2))) & 0x7ffff) << 5) | 1; // b.ne . - 8
    const movz_x1_100: u32 = 0xd280_0c81;
    m.code(0x10000, &.{ add_x0_1, subs_x1_1, bne_back3, movz_x1_100, b(0x10010, 0x40000) });
    var w: [chain_adds + 1]u32 = undefined;
    for (0..chain_adds) |i| w[i] = add_x3_1;
    for (0..19) |p| {
        const va = 0x40000 + p * 4096;
        w[chain_adds] = b(va + 4 * chain_adds, va + 4096);
        m.code(va, &w);
    }
    const last = 0x40000 + 19 * 4096;
    m.code(last, &.{ subs_x2_1, 0x5400_0000 | ((@as(u32, @truncate((0x10000 -% (last + 4)) >> 2)) & 0x7ffff) << 5) | 1, svc });
    m.regs.x[1] = 100;
    m.regs.x[2] = iters;
    m.regs.pc = 0x10000;
}

fn shrinkArena(m: *Machine, nursery: u64, tenured: u64) void {
    m.cpu.priv.nursery = nursery;
    m.cpu.priv.ten_mask = tenured - 1;
    m.cpu.resetCache();
}

test "a64 jit: nursery resets and the tenured ring keep running the right code" {
    // Nursery of 600 slots (a reset every pass); tenured rings that hold
    // less than everything compiled (the oldest go), or all of it.
    for ([_]u64{ 2048, 4096, 65536 }) |ten| {
        var ms: [2]Machine = .{ try Machine.init(t.allocator), try Machine.init(t.allocator) };
        defer for (&ms) |*m| m.deinit();
        for (&ms) |*m| {
            chainProgram(m, 6);
            shrinkArena(m, 600, ten);
        }
        const kt = ms[0].run(1 << 30);
        const kr = ms[1].runRef(1 << 30);
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), kr);
        try t.expectEqual(kr, kt);
        try t.expectEqual(ms[1].cpu.icount, ms[0].cpu.icount);
        try t.expect(std.mem.eql(u8, std.mem.asBytes(&ms[0].regs), std.mem.asBytes(&ms[1].regs)));
        try t.expectEqual(@as(u64, 6 * 19 * chain_adds), ms[0].regs.x[3]);
        if (jc.enabled and ten >= 4096) {
            // Everything is compiled once (eager host: at its first pass) and
            // stays compiled across the nursery resets of later passes.
            const st = jitStats(&ms[0]);
            try t.expect(st.compiles <= 23);
            try t.expect(st.entries >= 6 * 20);
        }
    }
}

test "a64 jit: a block decoded again after a nursery reset goes on counting" {
    if (!jc.enabled) return error.SkipZigTest;
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    chainProgram(&m, 5);
    shrinkArena(&m, 600, 60000);
    m.cpu.priv.jit.threshold = 1000;
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(1 << 30));
    // The first chain page (one block) ran 5 times, each pass decoded anew
    // after a reset: 5 passes charged in all, not just the last pass.
    const pc: u64 = 0x40000;
    const set = kwemu.blockSet(pc);
    const blk = for (m.cpu.priv.blocks[set..][0..2]) |*x| {
        if (x.pc == pc) break x;
    } else return error.TestUnexpectedResult;
    try t.expectEqual(@as(u32, 1000 - 5 * (chain_adds + 1)), blk.jx.st);
}

fn runCompare(prog: []const u32, init_regs: *const c.Regs) !c.Regs {
    var ms: [2]Machine = .{ try Machine.init(t.allocator), try Machine.init(t.allocator) };
    defer for (&ms) |*m| m.deinit();
    for (&ms) |*m| {
        m.code(0x10000, prog);
        m.regs = init_regs.*;
        m.regs.pc = 0x10000;
    }
    const kt = ms[0].run(10_000);
    const kr = ms[1].runRef(10_000);
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), kr);
    try t.expectEqual(kr, kt);
    try t.expectEqual(ms[1].cpu.icount, ms[0].cpu.icount);
    try t.expect(std.mem.eql(u8, std.mem.asBytes(&ms[0].regs), std.mem.asBytes(&ms[1].regs)));
    return ms[0].regs;
}

test "a64 jit: strict relaxed-SIMD FMA (scalar FMADD/FMSUB/FNMADD/FNMSUB and vector 2S/4S/2D FMLA/FMLS)" {
    // Inputs where single-rounded FMA differs from double-rounded mul+add:
    // f64: (1 + 2^-52)^2 - (1 + 2^-51) = 2^-104 (0x3970_0000_0000_0000); double-rounded gives 0.0
    // f32: (1 + 2^-23)^2 - (1 + 2^-22) = 2^-46  (0x2880_0000);           double-rounded gives 0.0
    const a64: u64 = 0x3ff0_0000_0000_0001; // 1 + 2^-52
    const c64: u64 = 0xbff0_0000_0000_0002; // -1 - 2^-51
    const nc64: u64 = 0x3ff0_0000_0000_0002; // +1 + 2^-51
    const e64: u64 = 0x3970_0000_0000_0000; // 2^-104
    const ne64: u64 = 0xb970_0000_0000_0000; // -2^-104

    const a32: u32 = 0x3f80_0001; // 1 + 2^-23
    const c32: u32 = 0xbf80_0002; // -1 - 2^-22
    const nc32: u32 = 0x3f80_0002; // +1 + 2^-22
    const e32: u32 = 0x2880_0000; // 2^-46
    const ne32: u32 = 0xa880_0000; // -2^-46

    // Test with both FPSR.IXC set (hits relaxed_fma fast path) and FPSR = 0 (strict IXC tracking)
    for ([_]u32{ c.FPSR_IXC, 0 }) |init_fpsr| {
        var r: c.Regs = .{};
        r.fpsr = init_fpsr;
        r.v[0] = c.v128(a64, a64);
        r.v[1] = c.v128(c64, c64);
        r.v[2] = c.v128(nc64, nc64);
        r.v[3] = c.fromLanes(u32, .{ a32, a32, a32, a32 });
        r.v[4] = c.fromLanes(u32, .{ c32, c32, c32, c32 });
        r.v[5] = c.fromLanes(u32, .{ nc32, nc32, nc32, nc32 });
        // Dirty upper half for Q = 0 (2S) tests (NaN in lanes 2..3)
        r.v[6] = c.fromLanes(u32, .{ c32, c32, 0x7fc0_1234, 0x7f80_0001 });
        r.v[7] = c.fromLanes(u32, .{ nc32, nc32, 0x7fc0_5678, 0xff80_0001 });
        // By-element source for 2D (lane 0 = 2.0, lane 1 = a64)
        r.v[8] = c.v128(0x4000_0000_0000_0000, a64);
        r.v[9] = c.v128(c64, c64);
        r.v[10] = c.v128(nc64, nc64);
        r.v[11] = c.v128(c64, c64);
        r.v[12] = c.v128(nc64, nc64);
        // Underflow inputs: 2^-600 * 2^-600 + 0.0 -> underflow (UFC | IXC)
        r.v[13] = c.v128(0x1a70_0000_0000_0000, 0x1a70_0000_0000_0000);
        r.v[14] = c.v128(0, 0);
        // Inf * 0 + qNaN -> default NaN + IOC
        r.v[15] = c.v128(0x7ff0_0000_0000_0000, 0);
        r.v[16] = c.v128(0x7ff8_0000_0000_0042, 0);

        const out = try runCompare(&.{
            // Scalar double: FMADD d20, d0, d0, d1 -> +2^-104
            0x1f40_0414,
            // Scalar double: FMSUB d21, d0, d0, d2 -> -2^-104
            0x1f40_8815,
            // Scalar double: FNMADD d22, d0, d0, d1 -> -2^-104 (-(d1) - d0*d0 = +1+2^-51 - (1+2^-51+2^-104))
            0x1f60_0416,
            // Scalar double: FNMSUB d23, d0, d0, d2 -> +2^-104 (-(d2) + d0*d0)
            0x1f60_8817,
            // Scalar single: FMADD s24, s3, s3, s4 -> +2^-46
            0x1f03_1078,
            // Scalar single: FMSUB s25, s3, s3, s5 -> -2^-46
            0x1f03_9479,
            // Vector 2D: FMLA v9.2d, v0.2d, v0.2d -> (+2^-104, +2^-104)
            0x4e60_cc09,
            // Vector 2D: FMLS v10.2d, v0.2d, v0.2d -> (-2^-104, -2^-104)
            0x4ee0_cc0a,
            // Vector 2D by-element: FMLA v11.2d, v0.2d, v8.d[1] -> (+2^-104, +2^-104)
            0x4fc8_180b,
            // Vector 2D by-element: FMLS v12.2d, v0.2d, v8.d[1] -> (-2^-104, -2^-104)
            0x4fc8_580c,
            // Vector 4S: FMLA v4.4s, v3.4s, v3.4s -> (+2^-46 x4)
            0x4e23_cc64,
            // Vector 4S: FMLS v5.4s, v3.4s, v3.4s -> (-2^-46 x4)
            0x4ea3_cc65,
            // Vector 2S (Q = 0 with NaNs in upper half of Vd): FMLA v6.2s, v3.2s, v3.2s
            0x0e23_cc66,
            // Vector 2S (Q = 0 with NaNs in upper half of Vd): FMLS v7.2s, v3.2s, v3.2s
            0x0ea3_cc67,
            // Save FPSR before edge cases into x0
            0xd53b_4420, // mrs x0, fpsr
            // Underflow 2D FMLA: v14 += v13 * v13 -> UFC | IXC
            0x4e6d_cdbe,
            0xd53b_4421, // mrs x1, fpsr
            // Inf * 0 + qNaN scalar FMADD d26, d15, d14, d16 -> default NaN, IOC
            0x1f4e_41fa,
            0xd53b_4422, // mrs x2, fpsr
            svc,
        }, &r);

        try t.expectEqual(@as(u128, e64), out.v[20]);
        try t.expectEqual(@as(u128, ne64), out.v[21]);
        try t.expectEqual(@as(u128, ne64), out.v[22]);
        try t.expectEqual(@as(u128, e64), out.v[23]);
        try t.expectEqual(@as(u128, e32), out.v[24]);
        try t.expectEqual(@as(u128, ne32), out.v[25]);
        try t.expectEqual(c.v128(e64, e64), out.v[9]);
        try t.expectEqual(c.v128(ne64, ne64), out.v[10]);
        try t.expectEqual(c.v128(e64, e64), out.v[11]);
        try t.expectEqual(c.v128(ne64, ne64), out.v[12]);
        try t.expectEqual(c.fromLanes(u32, .{ e32, e32, e32, e32 }), out.v[4]);
        try t.expectEqual(c.fromLanes(u32, .{ ne32, ne32, ne32, ne32 }), out.v[5]);
        try t.expectEqual(c.fromLanes(u32, .{ e32, e32, 0, 0 }), out.v[6]);
        try t.expectEqual(c.fromLanes(u32, .{ ne32, ne32, 0, 0 }), out.v[7]);
        // All exact FMAs above must not raise IXC when init_fpsr == 0
        try t.expectEqual(@as(u64, init_fpsr), out.x[0]);
        // Underflow raised UFC | IXC
        try t.expect(out.x[1] & (c.FPSR_UFC | c.FPSR_IXC) == (c.FPSR_UFC | c.FPSR_IXC));
        // Inf * 0 + qNaN raised IOC and gave default NaN
        try t.expect(out.x[2] & c.FPSR_IOC != 0);
        try t.expectEqual(@as(u128, 0x7ff8_0000_0000_0000), out.v[26]);
    }
}

test "a64 jit: FPCR_WASM_FP opt-in mode (MSR/MRS, no-FPSR, subnormals, NaNs, FMA, cmp, cvt)" {
    var r: c.Regs = .{};
    r.x[0] = c.FPCR_WASM_FP;
    r.x[1] = 1234567890123456789; // inexact for f64 conversion
    r.x[2] = c.FPCR_WASM_FP | c.FPCR_FZ; // FZ disables Wasm-FP mode
    r.x[3] = c.FPCR_WASM_FP | (1 << c.FPCR_RMODE_SHIFT); // RP disables Wasm-FP mode
    // Subnormal f64 and f32 values
    const sub64_a: u64 = 0x0000_0000_0001_2345;
    const sub64_b: u64 = 0x0000_0000_0000_5432;
    const sub32_a: u32 = 0x0000_1234;
    const sub32_b: u32 = 0x0000_0567;
    const qnan64: u64 = 0x7ff8_0000_0000_0042;
    const f1_5: u64 = @bitCast(@as(f64, 1.5));
    const f3_0: u64 = @bitCast(@as(f64, 3.0));
    const f2_0: u64 = @bitCast(@as(f64, 2.0));
    const f0_0: u64 = 0;

    r.v[0] = c.v128(f1_5, sub64_a);
    r.v[1] = c.v128(f3_0, sub64_b);
    r.v[2] = c.v128(f2_0, f0_0);
    r.v[3] = c.v128(qnan64, f2_0);
    r.v[4] = c.fromLanes(u32, .{ sub32_a, 0x3f80_0000, 0x4040_0000, 0x7fc0_0001 });
    r.v[5] = c.fromLanes(u32, .{ sub32_b, 0x4040_0000, 0x0000_0000, 0x3f80_0000 });
    r.v[6] = c.v128(f2_0, f2_0);
    r.v[7] = c.v128(f3_0, f3_0);
    r.v[8] = c.fromLanes(u32, .{ @bitCast(@as(f32, 2.5)), @bitCast(@as(f32, -3.5)), @bitCast(@as(f32, 1e30)), 0x7fc0_0000 });

    const out = try runCompare(&.{
        // Enable FPCR_WASM_FP and read it back
        0xd51b_4400, // msr fpcr, x0
        0xd53b_4404, // mrs x4, fpcr
        0xd51b_443f, // msr fpsr, xzr
        // Scalar inexact / div-by-zero / subnormal / sqrt / cvt / fcmp-NaN
        0x1e61_180a, // fdiv d10, d0, d1 (1.5 / 3.0 = 0.5)
        0x1e62_280b, // fadd d11, d0, d2 (1.5 + 2.0)
        0x1e61_080c, // fmul d12, d0, d1
        0x1e61_c02d, // fsqrt d13, d1 (sqrt(3.0): inexact)
        0x9e62_002e, // scvtf d14, x1 (inexact)
        0x9e78_0065, // fcvtzs x5, d3 (NaN -> 0, no IOC!)
        0x9e78_0006, // fcvtzs x6, d0 (1.5 -> 1, no IXC!)
        0x1e63_2010, // fcmpe d0, d3 (NaN -> unordered C|V, no IOC!)
        0xd53b_4207, // mrs x7, nzcv
        0x1e61_2000, // fcmp d0, d1 (1.5 < 3.0 -> N)
        0xd53b_4208, // mrs x8, nzcv
        // Scalar FMADD with exact product (1.5 * 3.0 + 2.0 = 6.5)
        0x1f41_080f, // fmadd d15, d0, d1, d2
        // Vector 2D FADD (includes subnormal in lane 1!), FMLA.2D, FCMEQ.2D (includes NaN!)
        0x4e61_d410, // fadd v16.2d, v0.2d, v1.2d
        0x4e67_cce6, // fmla v6.2d, v7.2d, v7.2d (2 + 3*3 = 11)
        0x4e63_e411, // fcmeq v17.2d, v0.2d, v3.2d
        0x4ee0_d812, // fcmle v18.2d, v0.2d, #0.0
        // Vector 4S FCVTZS (fraction, negative, overflow, NaN -> saturates/0, no FPSR)
        0x4ea1_b913, // fcvtzs v19.4s, v8.4s
        // Vector 2S subnormal FADD (Q = 0, lanes 2..3 zeroed, no FPSR)
        0x0e25_d494, // fadd v20.2s, v4.2s, v5.2s
        // Read FPSR: MUST still be 0!
        0xd53b_4429, // mrs x9, fpsr
        // Now test fallback when FPCR_FZ is also set (Wasm-FP disabled):
        // fadd d21, d0, d1 with subnormal in d21... use v16.d[1] via scalar fsqrt(3.0) to raise IXC
        0xd51b_4402, // msr fpcr, x2 (FPCR_WASM_FP | FPCR_FZ)
        0x1e61_c035, // fsqrt d21, d1 (raises IXC because FZ=1 disables Wasm-FP!)
        0xd53b_442a, // mrs x10, fpsr
        // Clear FPSR and test non-RNE rounding with FPCR_WASM_FP set (RP disables Wasm-FP):
        0xd51b_443f, // msr fpsr, xzr
        0xd51b_4403, // msr fpcr, x3 (FPCR_WASM_FP | RP)
        0x1e61_c036, // fsqrt d22, d1 (rounds toward +Inf and raises IXC!)
        0xd53b_442b, // mrs x11, fpsr
        svc,
    }, &r);

    try t.expectEqual(@as(u64, c.FPCR_WASM_FP), out.x[4]);
    try t.expectEqual(@as(u64, 0), out.x[5]); // fcvtzs(NaN) == 0
    try t.expectEqual(@as(u64, 1), out.x[6]); // fcvtzs(1.5) == 1
    try t.expectEqual(@as(u64, c.C | c.V), out.x[7]); // fcmpe NaN -> unordered
    try t.expectEqual(@as(u64, c.N), out.x[8]); // fcmp 1.5 < 3.0 -> N
    try t.expectEqual(@as(u64, 0), out.x[9]); // FPSR stayed 0 throughout Wasm-FP mode!
    try t.expectEqual(@as(u128, @as(u64, @bitCast(@as(f64, 6.5)))), out.v[15]);
    try t.expectEqual(c.v128(@bitCast(@as(f64, 11.0)), @bitCast(@as(f64, 11.0))), out.v[6]);
    try t.expectEqual(c.v128(0, 0), out.v[17]); // NaN == 1.5 is false, sub != 2.0 is false
    try t.expectEqual(c.fromLanes(u32, .{ 2, @bitCast(@as(i32, -3)), 0x7fff_ffff, 0 }), out.v[19]);
    // Subnormal 2S add matched exact IEEE and zeroed upper half
    try t.expectEqual(c.fromLanes(u32, .{ sub32_a + sub32_b, @bitCast(@as(f32, 4.0)), 0, 0 }), out.v[20]);
    // With FZ or RP set alongside FPCR_WASM_FP, strict ARM64 mode took over and set IXC
    try t.expect(out.x[10] & c.FPSR_IXC != 0);
    try t.expect(out.x[11] & c.FPSR_IXC != 0);
}
