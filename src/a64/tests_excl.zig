// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! Unit tests of exclusives, load-acquire/store-release (excl.zig) and the
//! system space (branch.zig): monitor rules, alignment and precise faults,
//! the counter, ID registers, register masks, and the encodings that must
//! stay ILLEGAL. test/guest/a64/atomictest.c covers the same against qemu.

const std = @import("std");
const t = std.testing;
const c = @import("common.zig");
const core = @import("../cpu/core.zig");
const kwemu = @import("../emu/kwemu.zig");
const excl = @import("excl.zig");
const branch = @import("branch.zig");
const run = @import("run.zig");
const mach = @import("machine.zig");
const Machine = mach.Machine;

/// size 001000 o2 L o1 Rs o0 Rt2 Rn Rt
fn ex(size: u32, o2: u32, l: u32, o1: u32, rs: u32, o0: u32, rt2: u32, rn: u32, rt: u32) u32 {
    return (size << 30) | 0x0800_0000 | (o2 << 23) | (l << 22) | (o1 << 21) | (rs << 16) | (o0 << 15) | (rt2 << 10) | (rn << 5) | rt;
}
fn ldxr(size: u32, rt: u32, rn: u32) u32 {
    return ex(size, 0, 1, 0, 31, 0, 31, rn, rt);
}
fn ldaxr(size: u32, rt: u32, rn: u32) u32 {
    return ex(size, 0, 1, 0, 31, 1, 31, rn, rt);
}
fn stxr(size: u32, rs: u32, rt: u32, rn: u32) u32 {
    return ex(size, 0, 0, 0, rs, 0, 31, rn, rt);
}
fn stlxr(size: u32, rs: u32, rt: u32, rn: u32) u32 {
    return ex(size, 0, 0, 0, rs, 1, 31, rn, rt);
}
fn ldxp(x: bool, rt: u32, rt2: u32, rn: u32, acq: u32) u32 {
    return ex(if (x) 3 else 2, 0, 1, 1, 31, acq, rt2, rn, rt);
}
fn stxp(x: bool, rs: u32, rt: u32, rt2: u32, rn: u32, rel: u32) u32 {
    return ex(if (x) 3 else 2, 0, 0, 1, rs, rel, rt2, rn, rt);
}
fn ldar(size: u32, rt: u32, rn: u32) u32 {
    return ex(size, 1, 1, 0, 31, 1, 31, rn, rt);
}
fn stlr(size: u32, rt: u32, rn: u32) u32 {
    return ex(size, 1, 0, 0, 31, 1, 31, rn, rt);
}
/// MRS (l = 1) / MSR (l = 0) Xt, S<op0>_<op1>_C<crn>_C<crm>_<op2>
fn sysreg(l: u32, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) u32 {
    return 0xd510_0000 | (l << 21) | ((op0 & 1) << 19) | (op1 << 16) | (crn << 12) | (crm << 8) | (op2 << 5) | rt;
}
fn mrs(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) u32 {
    return sysreg(1, op0, op1, crn, crm, op2, rt);
}
fn msr(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) u32 {
    return sysreg(0, op0, op1, crn, crm, op2, rt);
}
const svc0 = mach.svc0;
const nop: u32 = 0xd503201f;
const clrex: u32 = 0xd503305f;

fn get64(m: *Machine, va: u64) u64 {
    return std.mem.readInt(u64, m.at(va)[0..8], .little);
}
fn put64(m: *Machine, va: u64, v: u64) void {
    std.mem.writeInt(u64, m.at(va)[0..8], v, .little);
}

test "a64 excl: encodings match the assembler" {
    try t.expectEqual(@as(u32, 0xc85f7c20), ldxr(3, 0, 1)); // ldxr x0, [x1]
    try t.expectEqual(@as(u32, 0x885ffc20), ldaxr(2, 0, 1)); // ldaxr w0, [x1]
    try t.expectEqual(@as(u32, 0xc80f7c31), stxr(3, 15, 17, 1)); // stxr w15, x17, [x1]
    try t.expectEqual(@as(u32, 0x8800fc22), stlxr(2, 0, 2, 1)); // stlxr w0, w2, [x1]
    try t.expectEqual(@as(u32, 0xc8dffc00), ldar(3, 0, 0)); // ldar x0, [x0]
    try t.expectEqual(@as(u32, 0x889ffc01), stlr(2, 1, 0)); // stlr w1, [x0]
    try t.expectEqual(@as(u32, 0xd53bd040), mrs(3, 3, 13, 0, 2, 0)); // mrs x0, tpidr_el0
    try t.expectEqual(@as(u32, 0xd53be040), mrs(3, 3, 14, 0, 2, 0)); // mrs x0, cntvct_el0
    try t.expectEqual(@as(u32, 0xd51b4405), msr(3, 3, 4, 4, 0, 5)); // msr fpcr, x5
}

test "a64 excl: LDXR/STXR pass once, then fail; CLREX, address and size mismatches" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.mapData(0x20000, 1);
    put64(&m, 0x20010, 0x1122_3344_5566_7788);
    m.regs.x[1] = 0x20010;
    m.regs.x[13] = 0x20018;
    for ([_]u32{ 3, 4, 6, 8, 12, 15 }) |r| m.regs.x[r] = 77;
    m.code(0x10000, &.{
        ldxr(3, 0, 1), // ldxr x0, [x1]
        0x91000402, // add x2, x0, #1
        stxr(3, 3, 2, 1), // stxr w3, x2, [x1]: passes
        stxr(3, 4, 2, 1), // again: the monitor is closed, fails
        ldaxr(2, 5, 1), // ldaxr w5, [x1]
        clrex,
        stlxr(2, 6, 5, 1), // fails after CLREX
        ldxr(0, 7, 1), // ldxrb w7, [x1]
        stxr(1, 8, 7, 1), // stxrh: another size, fails
        ldxr(3, 11, 1),
        stxr(3, 12, 2, 13), // another address, fails
        ldxr(2, 14, 1), // ldxr w14, [x1]
        stxr(2, 15, 31, 1), // stxr w15, wzr, [x1]: passes, low word only
        svc0,
    });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(@as(u64, 0x1122_3344_5566_7788), m.regs.x[0]);
    try t.expectEqual(@as(u64, 0), m.regs.x[3]);
    try t.expectEqual(@as(u64, 1), m.regs.x[4]);
    try t.expectEqual(@as(u64, 0x5566_7789), m.regs.x[5]); // zero-extended W
    try t.expectEqual(@as(u64, 1), m.regs.x[6]);
    try t.expectEqual(@as(u64, 0x89), m.regs.x[7]);
    try t.expectEqual(@as(u64, 1), m.regs.x[8]);
    try t.expectEqual(@as(u64, 1), m.regs.x[12]);
    try t.expectEqual(@as(u64, 0x5566_7789), m.regs.x[14]);
    try t.expectEqual(@as(u64, 0), m.regs.x[15]);
    try t.expectEqual(@as(u64, 0x1122_3344_0000_0000), get64(&m, 0x20010));
    try t.expectEqual(core.no_resv, m.cpu.priv.hart.resv);

    // kw_clear_reservation between LDXR and STXR (a trap return) fails it
    m.code(0x11000, &.{ ldxr(3, 0, 1), stxr(3, 3, 0, 1), svc0 });
    try t.expectEqual(@as(u32, mach.EV_BUDGET), m.run(1));
    try t.expectEqual(@as(u64, 0x20010), m.cpu.priv.hart.resv);
    kwemu.clearReservation(m.cpu);
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(10));
    try t.expectEqual(@as(u64, 1), m.regs.x[3]);
    // ... and so does kw_set_pgd
    m.code(0x12000, &.{ ldxr(3, 0, 1), stxr(3, 3, 0, 1), svc0 });
    try t.expectEqual(@as(u32, mach.EV_BUDGET), m.run(1));
    kwemu.setPgd(m.cpu, m.root);
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(10));
    try t.expectEqual(@as(u64, 1), m.regs.x[3]);
    // Rs == Rt (CONSTRAINED UNPREDICTABLE): stores the old Rt, then the status
    m.code(0x13000, &.{ ldxr(3, 0, 1), stxr(3, 0, 0, 1), svc0 });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(10));
    try t.expectEqual(@as(u64, 0), m.regs.x[0]);
    try t.expectEqual(@as(u64, 0x1122_3344_0000_0000), get64(&m, 0x20010));
}

test "a64 excl: pairs (W and X), acquire/release forms" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.mapData(0x20000, 1);
    put64(&m, 0x20020, 0x2222_2222_1111_1111);
    put64(&m, 0x20030, 0xaaaa_aaaa_aaaa_aaaa);
    put64(&m, 0x20038, 0xbbbb_bbbb_bbbb_bbbb);
    m.regs.x[2] = 0x20020;
    m.regs.x[6] = 0x20030;
    m.code(0x10000, &.{
        ldxp(false, 0, 1, 2, 0), // ldxp w0, w1, [x2]
        stxp(false, 3, 1, 0, 2, 0), // stxp w3, w1, w0, [x2]: swap
        ldxp(true, 4, 5, 6, 1), // ldaxp x4, x5, [x6]
        stxp(true, 7, 5, 4, 6, 1), // stlxp w7, x5, x4, [x6]: swap
        stxp(true, 8, 5, 4, 6, 0), // closed: fails
        ldxp(false, 9, 10, 2, 0),
        stxp(true, 11, 9, 10, 2, 0), // other size, fails
        ldxr(3, 12, 2), // ldxr x12, [x2]
        stxp(false, 13, 31, 12, 2, 0), // stxp w13, wzr, w12, [x2]: same 8 bytes, passes
        svc0,
    });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(@as(u64, 0x1111_1111), m.regs.x[0]);
    try t.expectEqual(@as(u64, 0x2222_2222), m.regs.x[1]);
    try t.expectEqual(@as(u64, 0), m.regs.x[3]);
    try t.expectEqual(@as(u64, 0xaaaa_aaaa_aaaa_aaaa), m.regs.x[4]);
    try t.expectEqual(@as(u64, 0xbbbb_bbbb_bbbb_bbbb), m.regs.x[5]);
    try t.expectEqual(@as(u64, 0), m.regs.x[7]);
    try t.expectEqual(@as(u64, 1), m.regs.x[8]);
    try t.expectEqual(@as(u64, 0x2222_2222), m.regs.x[9]);
    try t.expectEqual(@as(u64, 0x1111_1111), m.regs.x[10]);
    try t.expectEqual(@as(u64, 1), m.regs.x[11]);
    try t.expectEqual(@as(u64, 0x1111_1111_2222_2222), m.regs.x[12]);
    try t.expectEqual(@as(u64, 0), m.regs.x[13]);
    try t.expectEqual(@as(u64, 0x2222_2222_0000_0000), get64(&m, 0x20020));
    try t.expectEqual(@as(u64, 0xbbbb_bbbb_bbbb_bbbb), get64(&m, 0x20030));
    try t.expectEqual(@as(u64, 0xaaaa_aaaa_aaaa_aaaa), get64(&m, 0x20038));

    // LDAR/STLR of every size
    m.regs.x[1] = 0x20040;
    m.regs.x[2] = 0x8877_6655_4433_2211;
    m.code(0x11000, &.{
        stlr(3, 2, 1), ldar(0, 3, 1), ldar(1, 4, 1), ldar(2, 5, 1), ldar(3, 6, 1),
        0x91002021, // add x1, x1, #8
        stlr(0, 2, 1), // stlrb
        0x91000821, // add x1, x1, #2
        stlr(1, 2, 1), // stlrh
        0x91000821, // add x1, x1, #2
        stlr(2, 2, 1), // stlr w
        svc0,
    });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(@as(u64, 0x11), m.regs.x[3]);
    try t.expectEqual(@as(u64, 0x2211), m.regs.x[4]);
    try t.expectEqual(@as(u64, 0x4433_2211), m.regs.x[5]);
    try t.expectEqual(@as(u64, 0x8877_6655_4433_2211), m.regs.x[6]);
    try t.expectEqual(@as(u64, 0x4433_2211_2211_0011), get64(&m, 0x20048));
}

test "a64 excl: alignment faults and precise page faults" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.mapData(0x20000, 1);
    const cases = [_]struct { inst: u32, addr: u64 }{
        .{ .inst = ldxr(3, 0, 1), .addr = 0x20004 },
        .{ .inst = ldxr(1, 0, 1), .addr = 0x20001 },
        .{ .inst = stxr(2, 3, 0, 1), .addr = 0x20002 }, // even with the monitor closed
        .{ .inst = ldar(2, 0, 1), .addr = 0x20002 },
        .{ .inst = stlr(3, 0, 1), .addr = 0x20008 + 4 },
        .{ .inst = stlr(1, 0, 1), .addr = 0x20003 },
        .{ .inst = ldxp(true, 0, 3, 1, 0), .addr = 0x20008 }, // an X pair needs 16
        .{ .inst = ldxp(false, 0, 3, 1, 0), .addr = 0x20004 }, // a W pair needs 8
        .{ .inst = stxp(true, 4, 0, 3, 1, 0), .addr = 0x20028 },
    };
    for (cases) |cs| {
        m.regs.x[0] = 0xa0;
        m.regs.x[3] = 0xa3;
        m.regs.x[1] = cs.addr;
        m.code(0x10000, &.{cs.inst});
        try t.expectEqual(@as(u32, mach.EV_MISALIGNED), m.run(10));
        try t.expectEqual(cs.addr, m.cpu.ev.addr);
        try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
        try t.expectEqual(@as(u64, 0xa0), m.regs.x[0]);
        try t.expectEqual(@as(u64, 0xa3), m.regs.x[3]);
        try t.expectEqual(core.no_resv, m.cpu.priv.hart.resv);
    }
    // byte accesses are always aligned
    m.regs.x[1] = 0x20003;
    m.code(0x10000, &.{ ldar(0, 0, 1), ldxr(0, 0, 1), stxr(0, 3, 0, 1), svc0 });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(10));
    try t.expectEqual(@as(u64, 0), m.regs.x[3]);

    // STXR to a read-only page: a write fault with no effect (status and
    // monitor unchanged); once writable the same STXR passes.
    const ro = m.frame();
    m.map(0x30000, ro, 2); // R
    m.bytes(ro)[16] = 0x5a;
    m.regs.x[1] = 0x30010;
    m.regs.x[2] = 0x1234;
    m.regs.x[3] = 77;
    m.code(0x11000, &.{ ldxr(3, 0, 1), stxr(3, 3, 2, 1), svc0 });
    try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(10));
    try t.expectEqual(@as(u32, 1), m.cpu.ev.access);
    try t.expectEqual(@as(u64, 0x11004), m.regs.pc);
    try t.expectEqual(@as(u64, 0x5a), m.regs.x[0]);
    try t.expectEqual(@as(u64, 77), m.regs.x[3]);
    try t.expectEqual(@as(u64, 0x30010), m.cpu.priv.hart.resv);
    m.map(0x30000, ro, 2 | 4); // RW
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(10));
    try t.expectEqual(@as(u64, 0), m.regs.x[3]);
    try t.expectEqual(@as(u64, 0x1234), get64(&m, 0x30010));
    // STXP of two X registers: same (one page, all or nothing)
    m.map(0x30000, ro, 2);
    m.regs.x[4] = 0x1111;
    m.regs.x[5] = 0x2222;
    m.code(0x12000, &.{ ldxp(true, 6, 7, 1, 0), stxp(true, 3, 4, 5, 1, 0), svc0 });
    m.regs.x[3] = 77;
    try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(10));
    try t.expectEqual(@as(u64, 77), m.regs.x[3]);
    try t.expectEqual(@as(u64, 0x1234), get64(&m, 0x30010));
    // LDXR of an unmapped page: a read fault, no register or monitor change
    kwemu.clearReservation(m.cpu);
    m.regs.x[1] = 0x40000;
    m.regs.x[0] = 0xa0;
    m.code(0x13000, &.{ldxr(3, 0, 1)});
    try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(10));
    try t.expectEqual(@as(u32, 0), m.cpu.ev.access);
    try t.expectEqual(@as(u64, 0xa0), m.regs.x[0]);
    try t.expectEqual(core.no_resv, m.cpu.priv.hart.resv);
}

test "a64 excl: reserved neighbours and non-v8.0 forms are ILLEGAL" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.mapData(0x20000, 1);
    m.regs.x[1] = 0x20000;
    const words = [_]u32{
        ex(3, 0, 1, 0, 0, 0, 31, 1, 0), // ldxr with Rs != 11111
        ex(3, 0, 1, 0, 31, 0, 0, 1, 0), // ldxr with Rt2 != 11111
        ex(3, 0, 0, 0, 2, 0, 0, 1, 0), // stxr with Rt2 != 11111
        ex(3, 0, 1, 1, 0, 0, 2, 1, 0), // ldxp with Rs != 11111
        ex(2, 1, 1, 0, 0, 1, 31, 1, 0), // ldar with Rs != 11111
        0x888f969d, // "stlr w29, [x20]" with Rs, Rt2 != 11111 (static corpus)
        0x484f565d, // "ldxrh w29, [x18]" with Rs, Rt2 != 11111 (static corpus)
        ex(2, 1, 1, 0, 31, 0, 31, 1, 0), // ldlar (FEAT_LOR)
        ex(2, 1, 0, 0, 31, 0, 31, 1, 0), // stllr (FEAT_LOR)
        ex(2, 1, 0, 1, 2, 0, 0, 1, 0), // cas with Rt2 != 11111
        ex(0, 0, 0, 1, 3, 0, 31, 1, 4), // casp with an odd Rs
    };
    for (words) |w| {
        m.code(0x10000, &.{w});
        try t.expectEqual(@as(u32, mach.EV_ILLEGAL), m.run(10));
        try t.expectEqual(@as(u64, w), m.cpu.ev.insn);
    }
}

test "a64 excl: 128-bit STXP is atomic between threads (shared-memory scheme)" {
    // The SMP path of STXP of two X registers, run natively on host threads:
    // a 128-bit counter incremented by LDXP-style snapshots (two 64-bit
    // atomic loads) and storePair128Shared; carries cross the halves.
    const per = 20000;
    const nthreads = 4;
    var cell: [2]u64 align(16) = .{ std.math.maxInt(u64) - per * nthreads / 2, 7 };
    const Worker = struct {
        fn work(p: *[2]u64) void {
            var i: usize = 0;
            while (i < per) {
                const lo = @atomicLoad(u64, &p[0], .seq_cst);
                const hi = @atomicLoad(u64, &p[1], .seq_cst);
                const r = @addWithOverflow(lo, 1);
                if (excl.storePair128Shared(p, lo, hi, r[0], hi + r[1])) i += 1;
            }
        }
    };
    var th: [nthreads]std.Thread = undefined;
    for (&th) |*h| h.* = try std.Thread.spawn(.{}, Worker.work, .{&cell});
    for (th) |h| h.join();
    try t.expectEqual(@as(u64, per * nthreads / 2 - 1), cell[0]);
    try t.expectEqual(@as(u64, 8), cell[1]);
    // a stale expectation fails and leaves memory alone
    try t.expect(!excl.storePair128Shared(&cell, cell[0], 0, 1, 2));
    try t.expect(!excl.storePair128Shared(&cell, 0, cell[1], 1, 2));
    try t.expectEqual(@as(u64, 8), cell[1]);
}

// ---- system space (branch.zig) ---------------------------------------------------

test "a64 system: CNTVCT_EL0 counts retired instructions, CNTFRQ_EL0" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.code(0x10000, &.{
        nop, nop, nop,
        mrs(3, 3, 14, 0, 2, 0), // mrs x0, cntvct_el0
        mrs(3, 3, 14, 0, 2, 1),
        mrs(3, 3, 14, 0, 0, 2), // mrs x2, cntfrq_el0
        mrs(3, 3, 14, 0, 2, 31), // into xzr
        mrs(3, 3, 14, 0, 2, 3),
        svc0,
    });
    try t.expectEqual(@as(u32, mach.EV_BUDGET), m.run(5)); // stops after x1's read
    try t.expectEqual(@as(u64, 3), m.regs.x[0]);
    try t.expectEqual(@as(u64, 4), m.regs.x[1]);
    try t.expectEqual(@as(u64, 0x10014), m.regs.pc);
    try t.expectEqual(@as(u64, 5), m.cpu.icount);
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(@as(u64, branch.cntfrq), m.regs.x[2]);
    try t.expectEqual(@as(u64, 7), m.regs.x[3]);
    try t.expectEqual(@as(u64, 9), m.cpu.icount);
    // straight-line counter reads still poll for interrupts at least every
    // 64 instructions (the guest raises it: irq_pending is in its memory)
    m.mapData(0x20000, 1);
    m.cpu.irq_pending = @ptrCast(@alignCast(m.at(0x20000).ptr));
    var words: [200]u32 = @splat(mrs(3, 3, 14, 0, 2, 0));
    words[0] = 0xd2800021; // mov x1, #1
    words[1] = 0xd2a00042; // mov x2, #0x20000
    words[2] = 0xb9000041; // str w1, [x2]
    m.code(0x11000, &words);
    try t.expectEqual(@as(u32, mach.EV_INTERRUPT), m.run(1000));
    try t.expect(m.regs.pc > 0x1100c and m.regs.pc <= 0x1100c + 64 * 4);
    m.cpu.irq_pending = null;
}

test "a64 system: ID registers, read-only registers, FPCR/FPSR masks" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.regs.x[9] = std.math.maxInt(u64);
    m.code(0x10000, &.{
        mrs(3, 0, 0, 0, 0, 0), // midr_el1
        mrs(3, 0, 0, 0, 5, 1), // mpidr_el1
        mrs(3, 0, 0, 0, 6, 2), // revidr_el1
        mrs(3, 0, 0, 4, 0, 3), // id_aa64pfr0_el1
        mrs(3, 0, 0, 6, 0, 4), // id_aa64isar0_el1
        mrs(3, 0, 0, 7, 0, 5), // id_aa64mmfr0_el1
        mrs(3, 0, 0, 5, 0, 6), // id_aa64dfr0_el1
        mrs(3, 0, 0, 7, 7, 7), // unallocated: RAZ
        mrs(3, 0, 0, 2, 0, 8), // id_isar0_el1 (AArch32): RAZ
        msr(3, 3, 4, 4, 0, 9), // msr fpcr, x9 (all ones)
        mrs(3, 3, 4, 4, 0, 10),
        msr(3, 3, 4, 4, 1, 9), // msr fpsr, x9
        mrs(3, 3, 4, 4, 1, 11),
        msr(3, 3, 4, 2, 0, 9), // msr nzcv, x9
        mrs(3, 3, 4, 2, 0, 12),
        mrs(3, 3, 13, 0, 3, 13), // tpidrro_el0
        mrs(3, 3, 0, 0, 1, 14), // ctr_el0
        mrs(3, 3, 0, 0, 7, 15), // dczid_el0
        svc0,
    });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(branch.midr, m.regs.x[0]);
    try t.expectEqual(@as(u64, 0x8000_0000), m.regs.x[1]);
    try t.expectEqual(@as(u64, 0), m.regs.x[2]);
    try t.expectEqual(@as(u64, 0x11), m.regs.x[3]);
    // ISAR0 follows the enabled features, like AT_HWCAP
    var isar0: u64 = 0;
    const isa = m.cpu.priv.a64.isa;
    const fb = c.featureBit;
    if (isa & fb(.aes) != 0) isar0 |= @as(u64, if (isa & fb(.pmull) != 0) 2 else 1) << 4;
    if (isa & fb(.sha1) != 0) isar0 |= 1 << 8;
    if (isa & fb(.sha2) != 0) isar0 |= @as(u64, if (isa & fb(.sha512) != 0) 2 else 1) << 12;
    if (isa & fb(.crc32) != 0) isar0 |= 1 << 16;
    if (isa & fb(.lse) != 0) isar0 |= 2 << 20;
    if (isa & fb(.sha3) != 0) isar0 |= 1 << 32;
    try t.expectEqual(isar0, m.regs.x[4]);
    try t.expectEqual(@as(u64, 0x0000_0111_ff00_0000), m.regs.x[5]);
    try t.expectEqual(@as(u64, 6), m.regs.x[6]);
    try t.expectEqual(@as(u64, 0), m.regs.x[7]);
    try t.expectEqual(@as(u64, 0), m.regs.x[8]);
    try t.expectEqual(@as(u64, c.FPCR_MASK), m.regs.x[10]); // AHP, DN, FZ, RMode, WASM_FP
    try t.expectEqual(@as(u64, 0x0800_009f), m.regs.x[11]); // QC, IDC, IXC..IOC
    try t.expectEqual(@as(u64, 0xf000_0000), m.regs.x[12]);
    try t.expectEqual(@as(u64, 0), m.regs.x[13]);
    try t.expectEqual(branch.ctr, m.regs.x[14]);
    try t.expectEqual(@as(u64, 4), m.regs.x[15]);
    // with every optional feature disabled ISAR0 is 0
    _ = run.setIsa(m.cpu, 0);
    m.code(0x11000, &.{ mrs(3, 0, 0, 6, 0, 4), svc0 });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(10));
    try t.expectEqual(@as(u64, 0), m.regs.x[4]);
}

test "a64 system: hints and barriers are NOPs, IC IVAU and DC by VA" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.mapData(0x20000, 1);
    m.regs.x[30] = 0x12345;
    m.regs.x[16] = 0x16;
    m.regs.x[17] = 0x17;
    m.regs.x[1] = 0x20040;
    m.regs.x[2] = 0x50000; // unmapped
    m.code(0x10000, &.{
        nop, 0xd503203f, 0xd503205f, 0xd503207f, // yield, wfe, wfi
        0xd503209f, 0xd50320bf, 0xd503229f, // sev, sevl, csdb
        0xd503233f, 0xd50323bf, 0xd50320ff, // paciasp, autiasp, xpaclri
        0xd503211f, 0xd503219f, // pacia1716, autia1716
        0xd503245f, 0xd503241f, // bti c, bti
        0xd5032fff, // hint #0x7f
        0xd5033bbf, 0xd50339bf, 0xd5033abf, 0xd5033f9f, 0xd5033fdf, // dmb ish, ishld, ishst; dsb sy; isb
        0xd503309f, 0xd503349f, clrex, 0xd503315f, // ssbb, pssbb, clrex #1
        0xd50b7b21, // dc cvau, x1
        0xd50b7a21, // dc cvac, x1
        0xd50b7e21, // dc civac, x1
        0xd50b7521, // ic ivau, x1
        0xd50b7522, // ic ivau, x2 (unmapped: no check)
        svc0,
    });
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(@as(u64, 0x12345), m.regs.x[30]);
    try t.expectEqual(@as(u64, 0x16), m.regs.x[16]);
    try t.expectEqual(@as(u64, 0x17), m.regs.x[17]);
    try t.expectEqual(@as(u64, 0x10000 + 29 * 4 + 4), m.regs.pc);
    // DC CVAU of an unmapped address: a read fault, as on hardware
    m.code(0x11000, &.{ 0xd50b7b22, svc0 });
    try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(10));
    try t.expectEqual(@as(u64, 0x50000), m.cpu.ev.addr & ~@as(u64, 0xfff));
    try t.expectEqual(@as(u32, 0), m.cpu.ev.access);
    // DC ZVA on a read-only page: a write fault, nothing written
    const ro = m.frame();
    m.map(0x30000, ro, 2);
    @memset(m.bytes(ro)[0..128], 0xee);
    m.regs.x[3] = 0x30047;
    m.code(0x12000, &.{ 0xd50b7423, svc0 });
    try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(10));
    try t.expectEqual(@as(u32, 1), m.cpu.ev.access);
    try t.expectEqual(@as(u8, 0xee), m.bytes(ro)[0x47]);
    m.map(0x30000, ro, 6);
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(10));
    try t.expectEqualSlices(u8, &(.{0} ** 64), m.bytes(ro)[0x40..0x80]);
    try t.expectEqual(@as(u8, 0xee), m.bytes(ro)[0x3f]);
}

test "a64 system: what EL0 may not do is ILLEGAL" {
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    const words = [_]u32{
        0x00000000, // udf #0
        0xd4000002, 0xd4000003, // hvc, smc
        0xd4400000, 0xd4a00001, 0xd4a00002, 0xd4a00003, // hlt, dcps1..3
        0xd69f03e0, 0xd6bf03e0, // eret, drps
        0xd65f0bff, 0xd71f0820, 0xd63f081f, // retaa, braa, blraaz
        0xd50330ff, // sb (FEAT_SB)
        0xd5033a3f, // dsb nXS (FEAT_XS)
        0xd503307f, // tcommit
        0xd503301f, // barrier op2 = 0 (unallocated)
        0xd50340df, // msr daifset, #0
        0xd500401f, // cfinv
        0xd5031000, // wfet x0
        0xd503201e, // hint with Rt != 11111
        0xd5280000, // sysl
        0xd50b7c20, 0xd50b7d20, // dc cvap, dc cvadp
        0xd5087620, 0xd508751f, // dc ivac, ic iallu (EL1)
        0xd50b7460, // dc gva (MTE)
        mrs(3, 3, 2, 4, 0, 0), mrs(3, 3, 2, 4, 1, 0), // rndr, rndrrs
        mrs(3, 3, 14, 0, 1, 0), mrs(3, 3, 14, 0, 6, 0), // cntpct, cntvctss
        mrs(3, 3, 14, 3, 1, 0), mrs(3, 3, 14, 2, 1, 0), // cntv_ctl, cntp_ctl
        mrs(3, 3, 4, 2, 1, 0), // daif
        mrs(3, 0, 4, 2, 2, 0), // currentel
        mrs(3, 3, 9, 12, 0, 0), // pmcr_el0
        mrs(3, 3, 13, 0, 5, 0), // tpidr2_el0
        mrs(3, 3, 4, 2, 6, 0), // ssbs
        mrs(3, 3, 4, 2, 5, 0), // dit
        mrs(2, 3, 0, 1, 0, 0), // mdccsr_el0 (debug)
        mrs(3, 0, 0, 1, 0, 0), // id_pfr0_el1 (CRm = 1: not emulated by Linux)
        mrs(3, 0, 0, 0, 1, 0), // CRm = 0, op2 = 1
        mrs(3, 0, 0, 0, 4, 0), // CRm = 0, op2 = 4
        mrs(3, 1, 0, 0, 0, 0), // ccsidr_el1
        mrs(3, 0, 1, 0, 0, 0), // sctlr_el1
        msr(3, 0, 0, 4, 0, 0), // msr id_aa64pfr0_el1
        msr(3, 0, 0, 0, 0, 0), // msr midr_el1
        msr(3, 3, 14, 0, 2, 0), // msr cntvct_el0
        msr(3, 3, 14, 0, 0, 0), // msr cntfrq_el0
        msr(3, 3, 0, 0, 1, 0), // msr ctr_el0
        msr(3, 3, 13, 0, 3, 0), // msr tpidrro_el0
        0x54000010, // bc.eq (FEAT_HBC)
        0xf4fb0209, // cbne (FEAT_CMPBR, static corpus)
    };
    for (words) |w| {
        m.regs.x[0] = 0xa0;
        m.code(0x10000, &.{w});
        try t.expectEqual(@as(u32, mach.EV_ILLEGAL), m.run(10));
        try t.expectEqual(@as(u64, w), m.cpu.ev.insn);
        try t.expectEqual(@as(u64, 0xa0), m.regs.x[0]);
    }
    // SVC #imm retires, BRK does not
    m.code(0x11000, &.{0xd4001fe1}); // svc #0xff
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(10));
    try t.expectEqual(@as(u64, 0x11004), m.regs.pc);
    m.code(0x12000, &.{0xd4200000}); // brk #0
    try t.expectEqual(@as(u32, mach.EV_BREAKPOINT), m.run(10));
    try t.expectEqual(@as(u64, 0x12000), m.regs.pc);
}
