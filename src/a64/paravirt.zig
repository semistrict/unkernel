// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! Paravirt bulk-op instructions in the permanently-undefined AArch64 UDF
//! range `0x0000_ff00 .. 0x0000_ff0f`:
//!   0x0000_ff00  KW_MEMCPY  x0=dst, x1=src, x2=len          -> x0=dst
//!   0x0000_ff01  KW_MEMSET  x0=dst, x1=byte(u8), x2=len     -> x0=dst
//!   0x0000_ff02  KW_MEMCMP  x0=a,   x1=b,   x2=len          -> x0=signed diff
//!   0x0000_ff03  KW_DAXPY   x0=y,   x1=x,   x2=n, d0=a      -> y[i] += a*x[i]
//!   0x0000_ff04  KW_DDOT    x0=a,   x1=b,   x2=n            -> d0=sum(a[i]*b[i])
//!   0x0000_ff05 .. 0x0000_ff0f  reserved (ILLEGAL)
//!
//! Each op translates guest pages once per 4 KiB chunk through Cpu.readHost /
//! Cpu.writeHost, probes all pages before mutating memory for precise faults
//! (so demand paging and JIT hand-backs never double-apply or leave partial
//! writes), and notifies SMC code-page tracking via Cpu.writeHost.

const std = @import("std");
const c = @import("common.zig");
const core = @import("../cpu/core.zig");
const Cpu = c.Cpu;

pub const base: u32 = 0x0000_ff00;
pub const mask: u32 = 0xffff_fff0;

pub const KW_MEMCPY: u32 = 0x0000_ff00;
pub const KW_MEMSET: u32 = 0x0000_ff01;
pub const KW_MEMCMP: u32 = 0x0000_ff02;
pub const KW_DAXPY: u32 = 0x0000_ff03;
pub const KW_DDOT: u32 = 0x0000_ff04;

const page_size: u64 = 4096;
const page_mask: u64 = page_size - 1;

pub inline fn isParavirt(inst: u32) bool {
    return inst & mask == base;
}

pub inline fn isDefined(inst: u32) bool {
    return inst >= KW_MEMCPY and inst <= KW_DDOT;
}

pub fn exec(s: c.St, inst: u32) c.Error!c.Exit {
    return switch (inst) {
        KW_MEMCPY => execMemcpy(s),
        KW_MEMSET => execMemset(s),
        KW_MEMCMP => execMemcmp(s),
        KW_DAXPY => execDaxpy(s),
        KW_DDOT => execDdot(s),
        else => c.ill(s, inst),
    };
}

fn probeRange(cpu: *Cpu, va: u64, len: u64, access: core.Access) c.Error!void {
    return cpu.probeRange(va, len, access);
}

inline fn hostMemmove(dst: u64, src: u64, n: usize) void {
    const d: [*]allowzero u8 = @ptrFromInt(@as(usize, @intCast(dst)));
    const s: [*]allowzero const u8 = @ptrFromInt(@as(usize, @intCast(src)));
    @memmove(d[0..n], s[0..n]);
}

inline fn hostMemset(dst: u64, val: u8, n: usize) void {
    const d: [*]allowzero u8 = @ptrFromInt(@as(usize, @intCast(dst)));
    @memset(d[0..n], val);
}

fn compareChunk(a_addr: u64, b_addr: u64, n: usize) ?i64 {
    const pa: [*]allowzero const u8 = @ptrFromInt(@as(usize, @intCast(a_addr)));
    const pb: [*]allowzero const u8 = @ptrFromInt(@as(usize, @intCast(b_addr)));
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const va = @as(*allowzero align(1) const u64, @ptrCast(pa + i)).*;
        const vb = @as(*allowzero align(1) const u64, @ptrCast(pb + i)).*;
        if (va != vb) {
            // Little-endian: lowest differing bit lies in the first differing byte.
            const diff_byte: usize = @ctz(va ^ vb) >> 3;
            return @as(i64, pa[i + diff_byte]) - @as(i64, pb[i + diff_byte]);
        }
    }
    while (i < n) : (i += 1) {
        if (pa[i] != pb[i]) return @as(i64, pa[i]) - @as(i64, pb[i]);
    }
    return null;
}

/// KW_MEMCPY: x0 = dst, x1 = src, x2 = len -> x0 = dst.
/// Handles arbitrary overlap safely (memmove semantics).
fn execMemcpy(s: c.St) c.Error!c.Exit {
    const dst = s.r.x[0];
    const src = s.r.x[1];
    const len = s.r.x[2];
    if (len == 0) return .next;
    const cpu = s.m;

    // Single-page fast path for both src and dst: one TLB check each, no pre-probe.
    if (len <= page_size and (src & page_mask) + len <= page_size and (dst & page_mask) + len <= page_size) {
        const shost = try cpu.readHost(src);
        const dhost = try cpu.writeHost(dst, len);
        if (dhost != shost) hostMemmove(dhost, shost, @intCast(len));
        return .next;
    }

    // Multi-page: probe all pages before modifying memory for precise faults.
    try probeRange(cpu, src, len, .read);
    try probeRange(cpu, dst, len, .write);
    if (dst == src) return .next;

    if (dst > src and dst - src < len) {
        // Overlapping with dst > src: copy backward page-chunk by page-chunk.
        var d_end = dst + len;
        var s_end = src + len;
        var rem = len;
        while (rem > 0) {
            const d_off = ((d_end - 1) & page_mask) + 1;
            const s_off = ((s_end - 1) & page_mask) + 1;
            const chunk = @min(rem, @min(d_off, s_off));
            d_end -= chunk;
            s_end -= chunk;
            const shost = try cpu.readHost(s_end);
            const dhost = try cpu.writeHost(d_end, chunk);
            if (dhost != shost) hostMemmove(dhost, shost, @intCast(chunk));
            rem -= chunk;
        }
    } else {
        // Non-overlapping or dst < src: copy forward page-chunk by page-chunk.
        var d_cur = dst;
        var s_cur = src;
        var rem = len;
        while (rem > 0) {
            const d_avail = page_size - (d_cur & page_mask);
            const s_avail = page_size - (s_cur & page_mask);
            const chunk = @min(rem, @min(d_avail, s_avail));
            const shost = try cpu.readHost(s_cur);
            const dhost = try cpu.writeHost(d_cur, chunk);
            if (dhost != shost) hostMemmove(dhost, shost, @intCast(chunk));
            d_cur += chunk;
            s_cur += chunk;
            rem -= chunk;
        }
    }
    return .next;
}

/// KW_MEMSET: x0 = dst, x1 = byte (low 8 bits), x2 = len -> x0 = dst.
fn execMemset(s: c.St) c.Error!c.Exit {
    const dst = s.r.x[0];
    const byte: u8 = @truncate(s.r.x[1]);
    const len = s.r.x[2];
    if (len == 0) return .next;
    const cpu = s.m;

    if (len <= page_size and (dst & page_mask) + len <= page_size) {
        const dhost = try cpu.writeHost(dst, len);
        hostMemset(dhost, byte, @intCast(len));
        return .next;
    }

    try probeRange(cpu, dst, len, .write);
    var cur = dst;
    var rem = len;
    while (rem > 0) {
        const chunk = @min(rem, page_size - (cur & page_mask));
        const dhost = try cpu.writeHost(cur, chunk);
        hostMemset(dhost, byte, @intCast(chunk));
        cur += chunk;
        rem -= chunk;
    }
    return .next;
}

/// KW_MEMCMP: x0 = a, x1 = b, x2 = len -> x0 = signed diff (0 if equal).
fn execMemcmp(s: c.St) c.Error!c.Exit {
    const a = s.r.x[0];
    const b = s.r.x[1];
    const len = s.r.x[2];
    if (len == 0) {
        s.r.x[0] = 0;
        return .next;
    }
    const cpu = s.m;
    if (a == b) {
        try probeRange(cpu, a, len, .read);
        s.r.x[0] = 0;
        return .next;
    }
    var a_cur = a;
    var b_cur = b;
    var rem = len;
    while (rem > 0) {
        const a_avail = page_size - (a_cur & page_mask);
        const b_avail = page_size - (b_cur & page_mask);
        const chunk = @min(rem, @min(a_avail, b_avail));
        const ahost = try cpu.readHost(a_cur);
        const bhost = try cpu.readHost(b_cur);
        if (compareChunk(ahost, bhost, @intCast(chunk))) |diff| {
            s.r.x[0] = @bitCast(diff);
            return .next;
        }
        a_cur +%= chunk;
        b_cur +%= chunk;
        rem -= chunk;
    }
    s.r.x[0] = 0;
    return .next;
}

inline fn daxpyChunk(yhost: u64, xhost: u64, count: usize, a: f64) void {
    const py: [*]allowzero u8 = @ptrFromInt(@as(usize, @intCast(yhost)));
    const px: [*]allowzero const u8 = @ptrFromInt(@as(usize, @intCast(xhost)));
    var j: usize = 0;
    while (j < count) : (j += 1) {
        const off = j * 8;
        const xv = @as(*allowzero align(1) const f64, @ptrCast(px + off)).*;
        const yp = @as(*allowzero align(1) f64, @ptrCast(py + off));
        yp.* = yp.* + a * xv;
    }
}

/// KW_DAXPY: x0 = y (double*), x1 = x (const double*), x2 = n, d0 = a -> y[i] += a * x[i].
fn execDaxpy(s: c.St) c.Error!c.Exit {
    const y = s.r.x[0];
    const x = s.r.x[1];
    const n = s.r.x[2];
    if (n == 0) return .next;
    const a: f64 = @bitCast(c.vd(s, 0));
    const cpu = s.m;
    const bytes = if (n >= (@as(u64, 1) << 45)) std.math.maxInt(u64) else n << 3;

    if (bytes <= page_size and (x & page_mask) + bytes <= page_size and (y & page_mask) + bytes <= page_size) {
        const xhost = try cpu.readHost(x);
        const yhost = try cpu.writeHost(y, bytes);
        daxpyChunk(yhost, xhost, @intCast(n), a);
        return .next;
    }

    // Multi-page: probe all pages before modifying y (daxpy is not idempotent).
    try probeRange(cpu, x, bytes, .read);
    try probeRange(cpu, y, bytes, .write);

    var y_cur = y;
    var x_cur = x;
    var rem = n;
    while (rem > 0) {
        const y_avail = page_size - (y_cur & page_mask);
        const x_avail = page_size - (x_cur & page_mask);
        const avail = @min(y_avail, x_avail);
        if (avail >= 8) {
            const count: usize = @intCast(@min(rem, avail >> 3));
            const chunk_bytes: u64 = @as(u64, count) * 8;
            const xhost = try cpu.readHost(x_cur);
            const yhost = try cpu.writeHost(y_cur, chunk_bytes);
            daxpyChunk(yhost, xhost, count, a);
            y_cur += chunk_bytes;
            x_cur += chunk_bytes;
            rem -= count;
        } else {
            // Unaligned element straddling a page boundary.
            const xv: f64 = @bitCast(try cpu.load(u64, x_cur));
            const yv: f64 = @bitCast(try cpu.load(u64, y_cur));
            try cpu.store(u64, y_cur, @bitCast(yv + a * xv));
            y_cur += 8;
            x_cur += 8;
            rem -= 1;
        }
    }
    return .next;
}

inline fn ddotChunk(ahost: u64, bhost: u64, count: usize, sum0: f64) f64 {
    const pa: [*]allowzero const u8 = @ptrFromInt(@as(usize, @intCast(ahost)));
    const pb: [*]allowzero const u8 = @ptrFromInt(@as(usize, @intCast(bhost)));
    var sum = sum0;
    var j: usize = 0;
    while (j < count) : (j += 1) {
        const off = j * 8;
        const av = @as(*allowzero align(1) const f64, @ptrCast(pa + off)).*;
        const bv = @as(*allowzero align(1) const f64, @ptrCast(pb + off)).*;
        sum += av * bv;
    }
    return sum;
}

/// KW_DDOT: x0 = a (const double*), x1 = b (const double*), x2 = n -> d0 = sum(a[i] * b[i]).
fn execDdot(s: c.St) c.Error!c.Exit {
    const a = s.r.x[0];
    const b = s.r.x[1];
    const n = s.r.x[2];
    if (n == 0) {
        c.setVd(s, 0, 0);
        return .next;
    }
    const cpu = s.m;
    var sum: f64 = 0.0;
    var a_cur = a;
    var b_cur = b;
    var rem = n;
    while (rem > 0) {
        const a_avail = page_size - (a_cur & page_mask);
        const b_avail = page_size - (b_cur & page_mask);
        const avail = @min(a_avail, b_avail);
        if (avail >= 8) {
            const count: usize = @intCast(@min(rem, avail >> 3));
            const chunk_bytes: u64 = @as(u64, count) * 8;
            const ahost = try cpu.readHost(a_cur);
            const bhost = try cpu.readHost(b_cur);
            sum = ddotChunk(ahost, bhost, count, sum);
            a_cur +%= chunk_bytes;
            b_cur +%= chunk_bytes;
            rem -= count;
        } else {
            const av: f64 = @bitCast(try cpu.load(u64, a_cur));
            const bv: f64 = @bitCast(try cpu.load(u64, b_cur));
            sum += av * bv;
            a_cur +%= 8;
            b_cur +%= 8;
            rem -= 1;
        }
    }
    c.setVd(s, 0, @bitCast(sum));
    return .next;
}

// ---- unit tests (run natively and on wasm64+JIT via "a64 jit:" filter) --------

const t = std.testing;
const decode = @import("decode.zig");
const mach = @import("machine.zig");
const Machine = mach.Machine;

const R: u64 = 2;
const W: u64 = 4;
const X: u64 = 8;

fn writeGuestBytes(m: *Machine, va: u64, data: []const u8) void {
    var cur = va;
    var off: usize = 0;
    while (off < data.len) {
        const dst = m.at(cur);
        const n = @min(data.len - off, dst.len);
        @memcpy(dst[0..n], data[off..][0..n]);
        cur += n;
        off += n;
    }
}

fn readGuestBytes(m: *Machine, va: u64, out: []u8) void {
    var cur = va;
    var off: usize = 0;
    while (off < out.len) {
        const src = m.at(cur);
        const n = @min(out.len - off, src.len);
        @memcpy(out[off..][0..n], src[0..n]);
        cur += n;
        off += n;
    }
}

fn runLockstep(code: []const u32, setup: anytype, check: anytype) !void {
    var m_tier = try Machine.init(t.allocator);
    defer m_tier.deinit();
    var m_ref = try Machine.init(t.allocator);
    defer m_ref.deinit();
    setup(&m_tier, code);
    setup(&m_ref, code);
    const ev_tier = m_tier.run(100_000);
    const ev_ref = m_ref.runRef(100_000);
    try t.expectEqual(ev_ref, ev_tier);
    try t.expectEqual(m_ref.cpu.icount, m_tier.cpu.icount);
    try t.expectEqual(m_ref.cpu.ev, m_tier.cpu.ev);
    try t.expectEqual(m_ref.regs, m_tier.regs);
    try check(&m_tier, &m_ref);
}

test "a64 jit: paravirt decode routing and reserved UDF opcodes" {
    try t.expectEqual(decode.Group.unallocated, decode.route(0x0000_0000));
    try t.expectEqual(decode.Group.unallocated, decode.route(0x0000_feff));
    try t.expectEqual(decode.Group.unallocated, decode.route(0x0000_ff10));
    try t.expectEqual(decode.Group.unallocated, decode.route(0x0000_dead));
    for (0..16) |i| {
        try t.expectEqual(decode.Group.paravirt, decode.route(base + @as(u32, @intCast(i))));
    }
    // Reserved slots 0x0000_ff05..0x0000_ff0f and outside UDFs must raise EV_ILLEGAL.
    for ([_]u32{ 0x0000_0000, 0x0000_feff, 0x0000_ff05, 0x0000_ff0f, 0x0000_ff10 }) |w| {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.code(0x10000, &.{ w, mach.svc0 });
        try t.expectEqual(@as(u32, mach.EV_ILLEGAL), m.run(100));
        try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
        try t.expectEqual(@as(u64, w), m.cpu.ev.insn);
        try t.expectEqual(@as(u64, 0), m.cpu.icount);
    }
}

test "a64 jit: paravirt KW_MEMCPY multi-page, unaligned, overlap, zero-len, lockstep" {
    const src_va: u64 = 0x20000;
    const dst_va: u64 = 0x40000;
    // Unaligned copy crossing 3 page boundaries with different page offsets.
    const len: usize = 9000;
    var pattern: [len]u8 = undefined;
    for (&pattern, 0..) |*b, i| b.* = @truncate((i * 73 + 19) ^ (i >> 5));

    const S = struct {
        pat: []const u8,
        fn setup(self: @This(), m: *Machine, code: []const u32) void {
            m.mapData(src_va, 4);
            m.mapData(dst_va, 4);
            writeGuestBytes(m, src_va + 17, self.pat);
            m.code(0x10000, code);
            m.regs.x[0] = dst_va + 333;
            m.regs.x[1] = src_va + 17;
            m.regs.x[2] = self.pat.len;
        }
        fn check(self: @This(), m_tier: *Machine, m_ref: *Machine) !void {
            try t.expectEqual(@as(u64, dst_va + 333), m_tier.regs.x[0]);
            var got_t: [len]u8 = undefined;
            var got_r: [len]u8 = undefined;
            readGuestBytes(m_tier, dst_va + 333, &got_t);
            readGuestBytes(m_ref, dst_va + 333, &got_r);
            try t.expectEqualSlices(u8, self.pat, &got_t);
            try t.expectEqualSlices(u8, self.pat, &got_r);
        }
    };
    {
        const ctx: S = .{ .pat = &pattern };
        const code = [_]u32{ KW_MEMCPY, mach.svc0 };
        var m_tier = try Machine.init(t.allocator);
        defer m_tier.deinit();
        var m_ref = try Machine.init(t.allocator);
        defer m_ref.deinit();
        ctx.setup(&m_tier, &code);
        ctx.setup(&m_ref, &code);
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m_tier.run(100));
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m_ref.runRef(100));
        try t.expectEqual(m_ref.cpu.icount, m_tier.cpu.icount);
        try t.expectEqual(m_ref.regs, m_tier.regs);
        try ctx.check(&m_tier, &m_ref);
    }

    // Overlapping forward (dst > src) across multiple pages.
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(src_va, 5);
        writeGuestBytes(&m, src_va + 50, &pattern);
        m.code(0x10000, &.{ KW_MEMCPY, mach.svc0 });
        m.regs.x[0] = src_va + 50 + 500;
        m.regs.x[1] = src_va + 50;
        m.regs.x[2] = len;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(@as(u64, src_va + 550), m.regs.x[0]);
        var got: [len]u8 = undefined;
        readGuestBytes(&m, src_va + 550, &got);
        try t.expectEqualSlices(u8, &pattern, &got);
    }

    // Overlapping backward (dst < src) across multiple pages.
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(src_va, 5);
        writeGuestBytes(&m, src_va + 700, &pattern);
        m.code(0x10000, &.{ KW_MEMCPY, mach.svc0 });
        m.regs.x[0] = src_va + 123;
        m.regs.x[1] = src_va + 700;
        m.regs.x[2] = len;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        var got: [len]u8 = undefined;
        readGuestBytes(&m, src_va + 123, &got);
        try t.expectEqualSlices(u8, &pattern, &got);
    }

    // Zero length with NULL pointers must not fault.
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.code(0x10000, &.{ KW_MEMCPY, mach.svc0 });
        m.regs.x[0] = 0;
        m.regs.x[1] = 0;
        m.regs.x[2] = 0;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(@as(u64, 0), m.regs.x[0]);
    }
}

test "a64 jit: paravirt KW_MEMSET multi-page, unaligned, zero-len, lockstep" {
    const dst_va: u64 = 0x30000;
    const len: usize = 8500;
    const code = [_]u32{ KW_MEMSET, mach.svc0 };

    const setup = struct {
        fn f(m: *Machine, w: []const u32) void {
            m.mapData(dst_va, 4);
            m.code(0x10000, w);
            m.regs.x[0] = dst_va + 77;
            m.regs.x[1] = 0xdead_beef_cafe_00a5; // only low byte 0xa5 used
            m.regs.x[2] = len;
        }
    }.f;
    const check = struct {
        fn f(m_tier: *Machine, m_ref: *Machine) !void {
            try t.expectEqual(@as(u64, dst_va + 77), m_tier.regs.x[0]);
            var got_t: [len + 2]u8 = undefined;
            var got_r: [len + 2]u8 = undefined;
            readGuestBytes(m_tier, dst_va + 76, &got_t);
            readGuestBytes(m_ref, dst_va + 76, &got_r);
            try t.expectEqual(@as(u8, 0), got_t[0]);
            try t.expectEqual(@as(u8, 0), got_t[len + 1]);
            for (got_t[1 .. len + 1]) |b| try t.expectEqual(@as(u8, 0xa5), b);
            try t.expectEqualSlices(u8, &got_r, &got_t);
        }
    }.f;
    try runLockstep(&code, setup, check);

    // Zero length with unmapped pointer does not fault.
    var m = try Machine.init(t.allocator);
    defer m.deinit();
    m.code(0x10000, &.{ KW_MEMSET, mach.svc0 });
    m.regs.x[0] = 0xdead_0000;
    m.regs.x[1] = 0x55;
    m.regs.x[2] = 0;
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
    try t.expectEqual(@as(u64, 0xdead_0000), m.regs.x[0]);
}

test "a64 jit: paravirt KW_MEMCMP equal, signed diff, unaligned, early stop, lockstep" {
    const a_va: u64 = 0x20000;
    const b_va: u64 = 0x40000;
    const len: usize = 8500;
    var buf: [len]u8 = undefined;
    for (&buf, 0..) |*x, i| x.* = @truncate(i * 31 + 7);

    // Equal multi-page unaligned buffers -> 0.
    {
        var m_t = try Machine.init(t.allocator);
        defer m_t.deinit();
        var m_r = try Machine.init(t.allocator);
        defer m_r.deinit();
        for ([_]*Machine{ &m_t, &m_r }) |m| {
            m.mapData(a_va, 4);
            m.mapData(b_va, 4);
            writeGuestBytes(m, a_va + 3, &buf);
            writeGuestBytes(m, b_va + 4091, &buf);
            m.code(0x10000, &.{ KW_MEMCMP, mach.svc0 });
            m.regs.x[0] = a_va + 3;
            m.regs.x[1] = b_va + 4091;
            m.regs.x[2] = len;
        }
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m_t.run(100));
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m_r.runRef(100));
        try t.expectEqual(@as(u64, 0), m_t.regs.x[0]);
        try t.expectEqual(m_r.regs, m_t.regs);
    }

    // Differing bytes at various positions (unsigned byte diff sign-extended to i64).
    const diff_cases = [_]struct { pos: usize, av: u8, bv: u8, want: i64 }{
        .{ .pos = 0, .av = 0xff, .bv = 0x00, .want = 255 },
        .{ .pos = 5, .av = 0x00, .bv = 0xff, .want = -255 },
        .{ .pos = 4095, .av = 0x10, .bv = 0x25, .want = -21 },
        .{ .pos = 4096, .av = 0x80, .bv = 0x7f, .want = 1 },
        .{ .pos = len - 1, .av = 0x01, .bv = 0x02, .want = -1 },
    };
    for (diff_cases) |dc| {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(a_va, 4);
        m.mapData(b_va, 4);
        writeGuestBytes(&m, a_va + 13, &buf);
        writeGuestBytes(&m, b_va + 29, &buf);
        writeGuestBytes(&m, a_va + 13 + dc.pos, &.{dc.av});
        writeGuestBytes(&m, b_va + 29 + dc.pos, &.{dc.bv});
        m.code(0x10000, &.{ KW_MEMCMP, mach.svc0 });
        m.regs.x[0] = a_va + 13;
        m.regs.x[1] = b_va + 29;
        m.regs.x[2] = len;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(dc.want, @as(i64, @bitCast(m.regs.x[0])));
    }

    // Early stop: differing on page 0 must NOT fault on unmapped page 1.
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(a_va, 1);
        m.mapData(b_va, 1);
        writeGuestBytes(&m, a_va, &.{ 1, 2, 10 });
        writeGuestBytes(&m, b_va, &.{ 1, 2, 20 });
        m.code(0x10000, &.{ KW_MEMCMP, mach.svc0 });
        m.regs.x[0] = a_va;
        m.regs.x[1] = b_va;
        m.regs.x[2] = 8192; // page 1 unmapped
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(@as(i64, -10), @as(i64, @bitCast(m.regs.x[0])));
    }

    // Zero length with NULL pointers -> 0.
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.code(0x10000, &.{ KW_MEMCMP, mach.svc0 });
        m.regs.x[0] = 0;
        m.regs.x[1] = 0;
        m.regs.x[2] = 0;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(@as(u64, 0), m.regs.x[0]);
    }
}

test "a64 jit: paravirt KW_DAXPY and KW_DDOT multi-page, unaligned straddle, zero-len, lockstep" {
    const y_va: u64 = 0x20000;
    const x_va: u64 = 0x40000;
    const n: usize = 1100; // > 2 pages (512 doubles/page)
    var y_init: [n]f64 = undefined;
    var x_init: [n]f64 = undefined;
    var y_want: [n]f64 = undefined;
    const alpha: f64 = -2.375;
    var dot_want: f64 = 0.0;
    for (0..n) |i| {
        const fi: f64 = @floatFromInt(i);
        y_init[i] = fi * 0.25 - 10.0;
        x_init[i] = (fi * 0.125) + 1.5;
        y_want[i] = y_init[i] + alpha * x_init[i];
        dot_want += y_want[i] * x_init[i];
    }

    // Test both 8-byte aligned and 4-byte unaligned (page-straddling doubles!).
    for ([_]struct { yoff: u64, xoff: u64 }{
        .{ .yoff = 0, .xoff = 0 },
        .{ .yoff = 4, .xoff = 12 }, // forces doubles to straddle 4096-byte page boundaries!
        .{ .yoff = 4092, .xoff = 4090 }, // immediate page straddle on element 0!
    }) |offs| {
        var m_t = try Machine.init(t.allocator);
        defer m_t.deinit();
        var m_r = try Machine.init(t.allocator);
        defer m_r.deinit();
        // Run KW_DAXPY then KW_DDOT in one block!
        const code = [_]u32{ KW_DAXPY, KW_DDOT, mach.svc0 };
        for ([_]*Machine{ &m_t, &m_r }) |m| {
            m.mapData(y_va, 4);
            m.mapData(x_va, 4);
            writeGuestBytes(m, y_va + offs.yoff, std.mem.sliceAsBytes(&y_init));
            writeGuestBytes(m, x_va + offs.xoff, std.mem.sliceAsBytes(&x_init));
            m.code(0x10000, &code);
            m.regs.x[0] = y_va + offs.yoff;
            m.regs.x[1] = x_va + offs.xoff;
            m.regs.x[2] = n;
            m.regs.v[0] = @as(u64, @bitCast(alpha)) | (@as(u128, 0xdeadbeef) << 64);
        }
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m_t.run(100));
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m_r.runRef(100));
        try t.expectEqual(m_r.cpu.icount, m_t.cpu.icount);
        try t.expectEqual(m_r.regs, m_t.regs);
        // Verify y[] bit-for-bit and d0 bit-for-bit (and upper 64 bits of v0 cleared by KW_DDOT).
        var y_got: [n]f64 = undefined;
        readGuestBytes(&m_t, y_va + offs.yoff, std.mem.sliceAsBytes(&y_got));
        for (0..n) |i| {
            try t.expectEqual(@as(u64, @bitCast(y_want[i])), @as(u64, @bitCast(y_got[i])));
        }
        try t.expectEqual(@as(u128, @as(u64, @bitCast(dot_want))), m_t.regs.v[0]);
    }

    // Zero length for KW_DAXPY and KW_DDOT.
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.code(0x10000, &.{ KW_DAXPY, KW_DDOT, mach.svc0 });
        m.regs.x[0] = 0;
        m.regs.x[1] = 0;
        m.regs.x[2] = 0;
        m.regs.v[0] = 0xffff_ffff_ffff_ffff_1234_5678_9abc_def0;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(@as(u128, 0), m.regs.v[0]);
    }
}

test "a64 jit: paravirt precise faults leave memory and registers untouched" {
    const src_va: u64 = 0x20000;
    const dst_va: u64 = 0x40000;

    // 1. KW_MEMCPY: page 0 of dst mapped RW, page 1 of dst unmapped -> page 0 must stay zero!
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(src_va, 2);
        m.mapData(dst_va, 1); // page 1 unmapped
        @memset(m.at(src_va), 0x77);
        @memset(m.at(src_va + 4096), 0x88);
        m.code(0x10000, &.{ KW_MEMCPY, mach.svc0 });
        m.regs.x[0] = dst_va;
        m.regs.x[1] = src_va;
        m.regs.x[2] = 6000;
        try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(100));
        try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
        try t.expectEqual(@as(u32, @intFromEnum(core.Access.write)), m.cpu.ev.access);
        try t.expectEqual(dst_va + 4096, m.cpu.ev.addr);
        for (m.at(dst_va)) |b| try t.expectEqual(@as(u8, 0), b);
    }

    // 2. KW_MEMSET: page 0 mapped RW, page 1 mapped read-only -> page 0 must stay untouched!
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(dst_va, 1);
        m.map(dst_va + 4096, m.frame(), R); // read-only
        m.code(0x10000, &.{ KW_MEMSET, mach.svc0 });
        m.regs.x[0] = dst_va;
        m.regs.x[1] = 0x99;
        m.regs.x[2] = 5000;
        try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(100));
        try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
        try t.expectEqual(@as(u32, @intFromEnum(core.Access.write)), m.cpu.ev.access);
        try t.expectEqual(dst_va + 4096, m.cpu.ev.addr);
        for (m.at(dst_va)) |b| try t.expectEqual(@as(u8, 0), b);
    }

    // 3. KW_DAXPY: page 0 mapped RW, page 1 unmapped -> page 0 must stay untouched!
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(src_va, 2);
        m.mapData(dst_va, 1);
        const one: f64 = 1.0;
        const two: f64 = 2.0;
        writeGuestBytes(&m, src_va, std.mem.asBytes(&one));
        writeGuestBytes(&m, dst_va, std.mem.asBytes(&two));
        m.code(0x10000, &.{ KW_DAXPY, mach.svc0 });
        m.regs.x[0] = dst_va;
        m.regs.x[1] = src_va;
        m.regs.x[2] = 600; // > 512 doubles
        m.regs.v[0] = @as(u64, @bitCast(@as(f64, 3.0)));
        try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(100));
        try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
        var got: f64 = 0;
        readGuestBytes(&m, dst_va, std.mem.asBytes(&got));
        try t.expectEqual(two, got);
    }

    // 4. KW_DDOT: page 1 unmapped -> v0 must stay untouched!
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(src_va, 1);
        m.mapData(dst_va, 2);
        m.code(0x10000, &.{ KW_DDOT, mach.svc0 });
        m.regs.x[0] = dst_va;
        m.regs.x[1] = src_va;
        m.regs.x[2] = 600;
        const sentinel: u128 = 0x1122334455667788_99aabbccddeeff00;
        m.regs.v[0] = sentinel;
        try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(100));
        try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
        try t.expectEqual(sentinel, m.regs.v[0]);
    }

    // 5. KW_DAXPY / KW_DDOT with huge n (n * 8 overflows u64) faults immediately.
    for ([_]u32{ KW_DAXPY, KW_DDOT }) |op| {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        m.mapData(src_va, 1);
        m.mapData(dst_va, 1);
        m.code(0x10000, &.{ op, mach.svc0 });
        m.regs.x[0] = dst_va;
        m.regs.x[1] = src_va;
        m.regs.x[2] = @as(u64, 1) << 62;
        try t.expectEqual(@as(u32, mach.EV_FAULT), m.run(100));
        try t.expectEqual(@as(u64, 0x10000), m.regs.pc);
    }
}

test "a64 jit: paravirt SMC invalidation when KW_MEMCPY or KW_MEMSET overwrites code" {
    const addi_x0_1: u32 = 0x9100_0400; // add x0, x0, #1
    const addi_x0_9: u32 = 0x9100_2400; // add x0, x0, #9
    const movz_x0_0: u32 = 0xd280_0000;

    // 1. KW_MEMCPY overwrites an instruction later in the SAME decoded/JIT block!
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        const code_va: u64 = 0x10000;
        const data_va: u64 = 0x20000;
        m.map(code_va, m.frame(), R | W | X);
        m.mapData(data_va, 1);
        writeGuestBytes(&m, data_va, std.mem.asBytes(&addi_x0_9));
        const words = [_]u32{
            KW_MEMCPY, // overwrites word 1 (code_va + 4) with add x0, x0, #9
            addi_x0_1, // originally add x0, x0, #1
            mach.svc0,
        };
        writeGuestBytes(&m, code_va, std.mem.sliceAsBytes(&words));
        m.regs.pc = code_va;
        m.regs.x[0] = code_va + 4;
        m.regs.x[1] = data_va;
        m.regs.x[2] = 4;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(@as(u64, code_va + 4 + 9), m.regs.x[0]);
    }

    // 2. KW_MEMCPY overwrites a previously executed/compiled code page via a writable alias.
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        const target_pa = m.frame();
        m.map(0x10000, target_pa, R | X);
        m.map(0x30000, target_pa, R | W); // writable alias
        m.mapData(0x40000, 1);
        const orig = [_]u32{ movz_x0_0, addi_x0_1, mach.svc0 };
        const patched = [_]u32{ movz_x0_0, addi_x0_9, mach.svc0 };
        writeGuestBytes(&m, 0x30000, std.mem.sliceAsBytes(&orig));
        writeGuestBytes(&m, 0x40000, std.mem.sliceAsBytes(&patched));
        m.code(0x50000, &.{ KW_MEMCPY, mach.svc0 });

        // First run caches/compiles 0x10000 -> x0 = 1.
        m.regs.pc = 0x10000;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(@as(u64, 1), m.regs.x[0]);

        // Overwrite via KW_MEMCPY through alias 0x30000.
        m.regs.pc = 0x50000;
        m.regs.x[0] = 0x30000;
        m.regs.x[1] = 0x40000;
        m.regs.x[2] = 12;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));

        // Second run of 0x10000 must see the new code -> x0 = 9.
        m.regs.pc = 0x10000;
        try t.expectEqual(@as(u32, mach.EV_SYSCALL), m.run(100));
        try t.expectEqual(@as(u64, 9), m.regs.x[0]);
    }

    // 3. KW_MEMSET zeroes the next instruction in the same block -> EV_ILLEGAL (udf #0).
    {
        var m = try Machine.init(t.allocator);
        defer m.deinit();
        const code_va: u64 = 0x10000;
        m.map(code_va, m.frame(), R | W | X);
        const words = [_]u32{ KW_MEMSET, addi_x0_1, mach.svc0 };
        writeGuestBytes(&m, code_va, std.mem.sliceAsBytes(&words));
        m.regs.pc = code_va;
        m.regs.x[0] = code_va + 4;
        m.regs.x[1] = 0;
        m.regs.x[2] = 4;
        try t.expectEqual(@as(u32, mach.EV_ILLEGAL), m.run(100));
        try t.expectEqual(@as(u64, code_va + 4), m.regs.pc);
        try t.expectEqual(@as(u64, 0), m.cpu.ev.insn);
    }
}

test "a64 jit: paravirt loop with KW_DDOT stays lockstep with reference" {
    // Tight loop calling KW_DDOT 8 times in a row (self-loop containing a paravirt op).
    // loop: KW_DDOT; add x3, x3, #1; subs x4, x4, #1; b.ne loop; svc #0
    const y_va: u64 = 0x20000;
    const x_va: u64 = 0x40000;
    const n: usize = 64;
    var a: [n]f64 = undefined;
    var b: [n]f64 = undefined;
    var want: f64 = 0.0;
    for (0..n) |i| {
        a[i] = @floatFromInt(i + 1);
        b[i] = 0.5;
        want += a[i] * b[i];
    }
    const code = [_]u32{
        KW_DDOT, // x0=a, x1=b, x2=n -> d0
        0x9100_0463, // add x3, x3, #1
        0xf100_0484, // subs x4, x4, #1
        0x54ff_ffa1, // b.ne -12 (back to KW_DDOT)
        mach.svc0,
    };
    const setup = struct {
        aa: []const f64,
        bb: []const f64,
        fn f(self: @This(), m: *Machine) void {
            m.mapData(y_va, 1);
            m.mapData(x_va, 1);
            writeGuestBytes(m, y_va, std.mem.sliceAsBytes(self.aa));
            writeGuestBytes(m, x_va, std.mem.sliceAsBytes(self.bb));
            m.code(0x10000, &code);
            m.regs.x[0] = y_va;
            m.regs.x[1] = x_va;
            m.regs.x[2] = n;
            m.regs.x[3] = 0;
            m.regs.x[4] = 8;
        }
    }{ .aa = &a, .bb = &b };

    var m_t = try Machine.init(t.allocator);
    defer m_t.deinit();
    var m_r = try Machine.init(t.allocator);
    defer m_r.deinit();
    setup.f(&m_t);
    setup.f(&m_r);
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m_t.run(1000));
    try t.expectEqual(@as(u32, mach.EV_SYSCALL), m_r.runRef(1000));
    try t.expectEqual(m_r.cpu.icount, m_t.cpu.icount);
    try t.expectEqual(@as(u64, 8), m_t.regs.x[3]);
    try t.expectEqual(@as(u128, @as(u64, @bitCast(want))), m_t.regs.v[0]);
    try t.expectEqual(m_r.regs, m_t.regs);
}
