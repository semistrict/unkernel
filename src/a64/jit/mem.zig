// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! A64 JIT, memory family (emit.zig; stream "memory"). Semantics:
//! opexec.zig (`ld`, `st`, `ld2`, `stN`, `ldrV`), field layout: ops.zig.
//! Every kind is translated.
//!
//! The rule: compute the address into
//! cm.l_va, check the data TLB inline (`e.tlbCheck`: tag == vpn of the
//! access's last byte, so a pair or a Q access checks its whole range
//! once), and on a miss hand the op back BEFORE it executes (no register,
//! no writeback, no store done): faults, page straddles (also an access
//! that wraps past 2^64), stores into pages holding decoded code
//! (code_flag), and an LDP/STP Q whose 32 bytes cross a page end all go
//! back to the interpreter, which keeps them exact. Past the check nothing
//! can fail, so the reference's order of register writes (writeback, then
//! Rt, then Rt2) matters only where registers coincide: a writeback whose
//! base is also a loaded register is dropped, Rt2 is written after Rt.
//! Store data is read before the writeback (STR Xn, [Xn], #8 stores the
//! old base). Q registers move as one v128 where the host has SIMD, else as
//! two i64 locals (a64/jit.md, Memory). Guest data takes plain wasm loads and stores, also on a shared
//! memory, as the tier's fast path takes plain host accesses.

const std = @import("std");
const ops = @import("../ops.zig");
const jc = @import("../../emu/jitcore.zig");
const jit = @import("emit.zig");
const cm = @import("common.zig");
const gen_mod = @import("generic.zig");
const atomic_mod = @import("atomic.zig");
const vec_mod = @import("vec.zig");
const Op = ops.Op;
const H = ops.H;
const Use = jit.Use;
const Emit = jit.Emit;
const At = jit.At;
const W = jc.W;

pub const kinds = [_]ops.H{
    .ldr8_i,  .ldr16_i, .ldr32_i, .ldr64_i, .ldr8_w,  .ldr16_w, .ldr32_w, .ldr64_w, .ldr8_r,  .ldr16_r,
    .ldr32_r, .ldr64_r, .str8_i,  .str16_i, .str32_i, .str64_i, .str8_w,  .str16_w, .str32_w, .str64_w,
    .str8_r,  .str16_r, .str32_r, .str64_r, .ldp32,   .ldp64,   .ldp32_w, .ldp64_w, .stp32,   .stp64,
    .stp32_w, .stp64_w, .ldr_v,   .ldr_q_i, .ldr_q_w, .ldr_q_r, .str_q_i, .str_q_w, .str_q_r, .ldp_q,
    .ldp_q_w, .stp_q,   .stp_q_w,
};

const max_ops = 65; // tier.max_block + trailing goto
const max_groups = 16;
const max_slots = cm.n_mem;
const max_cache = cm.n_mcache / 3;
const max_span: i64 = 256;
const max_delta: i64 = 1 << 28;
const no_cache: u8 = 0xff;

// op_tag encoding per op index k:
//   0                                    : single (no loop cache)
//   1 .. max_cache                       : single with loop cache slot (tag - 1)
//   tag_head .. tag_head + max_groups-1  : head of coalesced group (tag - tag_head)
//   tag_follow ..                        : follower of coalesced group (tag - tag_follow)
const tag_head: u8 = 1 + max_cache;
const tag_follow: u8 = tag_head + max_groups;

const flag_write: u8 = 1;
const flag_reg: u8 = 2;

const Group = struct {
    va_off: i32,
    span_m1: u8,
    slot: u8,
    cache: u8,
    flags: u8,
};

/// This family's part of the module's emission state (`e.x.mem`).
/// Arrays are left `undefined` by default so default-initializing `Emit`
/// emits no bulk zeroing/memcpy; `beginMember` initializes `op_tag[0..n_ops]`.
pub const Ext = struct {
    n_ops: u8 = 0,
    op_tag: [max_ops]u8 = undefined,
    op_delta: [max_ops]u8 = undefined,
    groups: [max_groups]Group = undefined,
};

/// Host-address local + wasm memory offset for multi-part accesses.
pub const Host = struct {
    local: u32,
    off: u64,
};

/// Scratch local for an uncoalesced multi-part access (never live across ops).
const l_host_tmp: u32 = cm.l_t;

/// Bytes of one element of kind h (one register of a pair, a Q half: 8).
fn width(h: H) u64 {
    return switch (h) {
        .ldr8_i, .ldr8_w, .ldr8_r, .str8_i, .str8_w, .str8_r => 1,
        .ldr16_i, .ldr16_w, .ldr16_r, .str16_i, .str16_w, .str16_r => 2,
        .ldr32_i, .ldr32_w, .ldr32_r, .str32_i, .str32_w, .str32_r, .ldp32, .ldp32_w, .stp32, .stp32_w => 4,
        else => 8,
    };
}

pub fn supported(o: Op) bool {
    return switch (o.h) {
        // What the decoder makes: zero extension (d = 0) or sign extension
        // from the access width (d = 64 - width), a W mask by g (0 or 1).
        .ldr8_i, .ldr16_i, .ldr32_i, .ldr64_i, .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w, .ldr8_r, .ldr16_r, .ldr32_r, .ldr64_r => (o.d == 0 or o.d == 64 - 8 * width(o.h)) and o.g <= 1,
        .ldp32, .ldp32_w => o.d == 0 or o.d == 32,
        .ldr_v => o.d <= 3 and o.e <= 2,
        else => true,
    };
}

/// Data slot of a store or load: an X/SP slot or a V slot.
fn data(slot: u8) Use {
    return .{ .read = cm.xbit(slot), .vread = cm.vbit(slot) };
}

pub fn usage(o: Op) Use {
    const base = cm.xbit(o.b);
    return switch (o.h) {
        .ldr8_i, .ldr16_i, .ldr32_i, .ldr64_i => .{ .read = base, .write = cm.xbit(o.a) },
        .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w => .{ .read = base, .write = base | cm.xbit(o.a) },
        .ldr8_r, .ldr16_r, .ldr32_r, .ldr64_r => .{ .read = base | cm.xbit(o.c), .write = cm.xbit(o.a) },
        .str8_i, .str16_i, .str32_i, .str64_i => blk: {
            var u = data(o.a);
            u.read |= base;
            break :blk u;
        },
        .str8_w, .str16_w, .str32_w, .str64_w => blk: {
            var u = data(o.a);
            u.read |= base;
            u.write = base;
            break :blk u;
        },
        .str8_r, .str16_r, .str32_r, .str64_r => blk: {
            var u = data(o.a);
            u.read |= base | cm.xbit(o.c);
            break :blk u;
        },
        .ldp32, .ldp64 => .{ .read = base, .write = cm.xbit(o.a) | cm.xbit(o.c) },
        .ldp32_w, .ldp64_w => .{ .read = base, .write = base | cm.xbit(o.a) | cm.xbit(o.c) },
        .stp32, .stp64, .stp32_w, .stp64_w => blk: {
            var u = data(o.a);
            u.add(data(o.c));
            u.read |= base;
            if (o.h == .stp32_w or o.h == .stp64_w) u.write = base;
            break :blk u;
        },
        .ldr_v, .ldr_q_i, .ldr_q_r => .{ .read = base | (if (o.h == .ldr_q_r or (o.h == .ldr_v and o.e == 2)) cm.xbit(o.c) else 0), .write = if (o.h == .ldr_v and o.e == 1) base else 0, .vwrite = cm.vbit(o.a) },
        .ldr_q_w => .{ .read = base, .write = base, .vwrite = cm.vbit(o.a) },
        .str_q_i, .str_q_r => .{ .read = base | (if (o.h == .str_q_r) cm.xbit(o.c) else 0), .vread = cm.vbit(o.a) },
        .str_q_w => .{ .read = base, .write = base, .vread = cm.vbit(o.a) },
        .ldp_q => .{ .read = base, .vwrite = cm.vbit(o.a) | cm.vbit(o.c) },
        .ldp_q_w => .{ .read = base, .write = base, .vwrite = cm.vbit(o.a) | cm.vbit(o.c) },
        .stp_q => .{ .read = base, .vread = cm.vbit(o.a) | cm.vbit(o.c) },
        .stp_q_w => .{ .read = base, .write = base, .vread = cm.vbit(o.a) | cm.vbit(o.c) },
        else => unreachable,
    };
}

// ---- addresses ----------------------------------------------------------------

/// l_va = slot b + off (b = XZR: a literal's absolute address).
fn vaImm(e: *Emit, b: u8, off: u64) void {
    if (b == ops.zr) {
        e.k64(off);
    } else {
        cm.get(e, b);
        if (off != 0) {
            e.k64(off);
            e.op(W.i64_add);
        }
    }
    e.set(cm.l_va);
}

/// l_va = slot b + op2(slot c), op2(v) = ((v << s1) >>a s2) & mask.
fn vaReg(e: *Emit, b: u8, c: u8, s1: u8, s2: u8, mask: u64) void {
    cm.get(e, b);
    cm.get(e, c);
    if (s1 & 63 != 0) {
        e.k64(s1 & 63);
        e.op(W.i64_shl);
    }
    if (s2 & 63 != 0) {
        e.k64(s2 & 63);
        e.op(W.i64_shr_s);
    }
    if (mask != std.math.maxInt(u64)) {
        e.k64(mask);
        e.op(W.i64_and);
    }
    e.op(W.i64_add);
    e.set(cm.l_va);
}

/// Pre/post index: l_va = b + (pre ? imm : 0).
fn vaIndex(e: *Emit, o: Op, pre: u8) void {
    vaImm(e, o.b, if (pre != 0) o.imm else 0);
}

/// b += imm (after the access).
fn writeback(e: *Emit, o: Op) void {
    if (o.imm == 0) return;
    cm.get(e, o.b);
    e.k64(o.imm);
    e.op(W.i64_add);
    cm.set(e, o.b);
}

/// Push the kept host address in `local` as a memory operand.
fn host(e: *Emit, local: u32) void {
    e.get(local);
    e.addr();
}

/// Emit a 64-bit TLB check (cached when in a standalone self-loop with a
/// cache slot, else direct), leaving the 64-bit host address on the stack.
fn emitTlb64(e: *Emit, at: At, is_write: bool, size: u64, cache: u8) void {
    const tlb_off: u64 = if (is_write) jc.tlb_w_off else jc.tlb_r_off;
    if (e.rg == null and at.self_loop and cache < max_cache) {
        const base = cm.l_mcache + @as(u32, cache) * 3;
        e.tlbCheckCached64(at.k, at.pc, tlb_off, size, base, base + 1, base + 2);
    } else {
        e.tlbCheck64(at.k, at.pc, tlb_off, size);
    }
}

fn tagOf(e: *const Emit, k: usize) u8 {
    return if (k < e.x.mem.n_ops) e.x.mem.op_tag[k] else 0;
}

/// Prepare a host address on the stack (wrapped by `e.addr()`) for a
/// base+imm access (used by vec.zig `ldstSingle`), returning its wasm
/// memory offset.
pub fn prepareAddr(e: *Emit, at: At, base: u8, imm: u64, is_write: bool, size: u64) u64 {
    const tag = tagOf(e, at.k);
    if (tag >= tag_follow) {
        const g = e.x.mem.groups[tag - tag_follow];
        host(e, cm.l_mem + g.slot);
        return e.x.mem.op_delta[at.k];
    }
    if (tag >= tag_head) {
        const g = e.x.mem.groups[tag - tag_head];
        vaImm(e, base, @bitCast(@as(i64, g.va_off)));
        emitTlb64(e, at, g.flags & flag_write != 0, @as(u64, g.span_m1) + 1, g.cache);
        e.tee(cm.l_mem + g.slot);
        e.addr();
        return e.x.mem.op_delta[at.k];
    }
    vaImm(e, base, imm);
    emitTlb64(e, at, is_write, size, if (tag > 0) tag - 1 else no_cache);
    e.addr();
    return 0;
}

/// Prepare a host address in an i64 local for a multi-part base+imm access
/// (used by vec.zig `ldst1` and `pairEmit`), returning `(local, off)`.
pub fn prepareHost(e: *Emit, at: At, base: u8, imm: u64, is_write: bool, size: u64) Host {
    const tag = tagOf(e, at.k);
    if (tag >= tag_follow) {
        const g = e.x.mem.groups[tag - tag_follow];
        return .{ .local = cm.l_mem + g.slot, .off = e.x.mem.op_delta[at.k] };
    }
    if (tag >= tag_head) {
        const g = e.x.mem.groups[tag - tag_head];
        vaImm(e, base, @bitCast(@as(i64, g.va_off)));
        emitTlb64(e, at, g.flags & flag_write != 0, @as(u64, g.span_m1) + 1, g.cache);
        const l = cm.l_mem + g.slot;
        e.set(l);
        return .{ .local = l, .off = e.x.mem.op_delta[at.k] };
    }
    vaImm(e, base, imm);
    emitTlb64(e, at, is_write, size, if (tag > 0) tag - 1 else no_cache);
    e.set(l_host_tmp);
    return .{ .local = l_host_tmp, .off = 0 };
}

fn emitOpVa(e: *Emit, o: Op) void {
    switch (o.h) {
        .ldr8_r, .ldr16_r, .ldr32_r, .ldr64_r, .str8_r, .str16_r, .str32_r, .str64_r, .ldr_q_r, .str_q_r => vaReg(e, o.b, o.c, o.e, o.f, o.imm),
        .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w, .str8_w, .str16_w, .str32_w, .str64_w, .ldr_q_w, .str_q_w => vaIndex(e, o, o.f),
        .ldp32_w, .ldp64_w, .stp32_w, .stp64_w, .ldp_q_w, .stp_q_w => vaIndex(e, o, o.e),
        .ldr_v => switch (o.e) {
            0 => vaImm(e, o.b, o.imm),
            1 => vaIndex(e, o, o.f),
            else => vaReg(e, o.b, o.c, o.f, o.g, o.imm),
        },
        else => vaImm(e, o.b, o.imm),
    }
}

fn prepareAddrOp(e: *Emit, o: Op, at: At, is_write: bool, size: u64) u64 {
    const tag = tagOf(e, at.k);
    if (tag >= tag_follow) {
        const g = e.x.mem.groups[tag - tag_follow];
        host(e, cm.l_mem + g.slot);
        return e.x.mem.op_delta[at.k];
    }
    if (tag >= tag_head) {
        const g = e.x.mem.groups[tag - tag_head];
        if (g.flags & flag_reg != 0) emitOpVa(e, o) else vaImm(e, o.b, @bitCast(@as(i64, g.va_off)));
        emitTlb64(e, at, g.flags & flag_write != 0, @as(u64, g.span_m1) + 1, g.cache);
        e.tee(cm.l_mem + g.slot);
        e.addr();
        return e.x.mem.op_delta[at.k];
    }
    emitOpVa(e, o);
    emitTlb64(e, at, is_write, size, if (tag > 0) tag - 1 else no_cache);
    e.addr();
    return 0;
}

fn prepareHostOp(e: *Emit, o: Op, at: At, is_write: bool, size: u64) Host {
    const tag = tagOf(e, at.k);
    if (tag >= tag_follow) {
        const g = e.x.mem.groups[tag - tag_follow];
        return .{ .local = cm.l_mem + g.slot, .off = e.x.mem.op_delta[at.k] };
    }
    if (tag >= tag_head) {
        const g = e.x.mem.groups[tag - tag_head];
        if (g.flags & flag_reg != 0) emitOpVa(e, o) else vaImm(e, o.b, @bitCast(@as(i64, g.va_off)));
        emitTlb64(e, at, g.flags & flag_write != 0, @as(u64, g.span_m1) + 1, g.cache);
        const l = cm.l_mem + g.slot;
        e.set(l);
        return .{ .local = l, .off = e.x.mem.op_delta[at.k] };
    }
    emitOpVa(e, o);
    emitTlb64(e, at, is_write, size, if (tag > 0) tag - 1 else no_cache);
    e.set(l_host_tmp);
    return .{ .local = l_host_tmp, .off = 0 };
}

/// With SIMD: a v128 load or store at the address operand on the stack.
fn vmem(e: *Emit, code: u32, off: u64) void {
    e.vop(code);
    e.uleb(0);
    e.uleb(off);
}
/// The V register whose low half is slot s.
fn vreg(s: u8) u8 {
    return (s - ops.v0_slot) / 2;
}

fn loadOp(size: u64, signed: bool) u8 {
    return switch (size) {
        1 => if (signed) W.i64_load8_s else W.i64_load8_u,
        2 => if (signed) W.i64_load16_s else W.i64_load16_u,
        4 => if (signed) W.i64_load32_s else W.i64_load32_u,
        else => W.i64_load,
    };
}
fn storeOp(size: u64) u8 {
    return switch (size) {
        1 => W.i64_store8,
        2 => W.i64_store16,
        4 => W.i64_store32,
        else => W.i64_store,
    };
}

// ---- emission -----------------------------------------------------------------

pub fn emit(e: *Emit, o: Op, at: At) void {
    const size = width(o.h);
    switch (o.h) {
        .ldr8_i, .ldr16_i, .ldr32_i, .ldr64_i, .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w, .ldr8_r, .ldr16_r, .ldr32_r, .ldr64_r => {
            const wb = switch (o.h) {
                .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w => true,
                else => false,
            };
            const off = prepareAddrOp(e, o, at, false, size);
            const signed = o.d != 0;
            e.mem(loadOp(size, signed), 0, off);
            if (signed and o.g & 1 != 0) {
                e.k64(0xffff_ffff);
                e.op(W.i64_and);
            }
            // Writeback, then Rt: with Rt == Rn the loaded value stays.
            if (wb and o.a != o.b) writeback(e, o);
            cm.set(e, o.a);
        },
        .str8_i, .str16_i, .str32_i, .str64_i, .str8_w, .str16_w, .str32_w, .str64_w, .str8_r, .str16_r, .str32_r, .str64_r => {
            const wb = switch (o.h) {
                .str8_w, .str16_w, .str32_w, .str64_w => true,
                else => false,
            };
            const off = prepareAddrOp(e, o, at, true, size);
            cm.get(e, o.a);
            e.mem(storeOp(size), 0, off);
            if (wb) writeback(e, o);
        },
        .ldp32, .ldp64, .ldp32_w, .ldp64_w => {
            const wb = o.h == .ldp32_w or o.h == .ldp64_w;
            const a = prepareHostOp(e, o, at, false, 2 * size);
            const signed = o.d != 0; // LDPSW
            host(e, a.local);
            e.mem(loadOp(size, signed), 0, a.off);
            cm.set(e, o.a);
            host(e, a.local);
            e.mem(loadOp(size, signed), 0, a.off + size);
            cm.set(e, o.c); // after Rt: Rt2 wins when they are equal
            if (wb and o.b != o.a and o.b != o.c) writeback(e, o);
        },
        .stp32, .stp64, .stp32_w, .stp64_w => {
            const wb = o.h == .stp32_w or o.h == .stp64_w;
            const a = prepareHostOp(e, o, at, true, 2 * size);
            host(e, a.local);
            cm.get(e, o.a);
            e.mem(storeOp(size), 0, a.off);
            host(e, a.local);
            cm.get(e, o.c);
            e.mem(storeOp(size), 0, a.off + size);
            if (wb) writeback(e, o);
        },
        .ldr_v => {
            const n: u64 = @as(u64, 1) << @intCast(o.d);
            const off = prepareAddrOp(e, o, at, false, n);
            e.mem(loadOp(n, false), 0, off);
            if (o.e == 1) writeback(e, o); // the base is an X register
            cm.setScalar(e, o.a);
        },
        .ldr_q_i, .ldr_q_w, .ldr_q_r => {
            if (e.simd) {
                const off = prepareAddrOp(e, o, at, false, 16);
                vmem(e, cm.S.v128_load, off);
                cm.setV(e, vreg(o.a));
            } else {
                const a = prepareHostOp(e, o, at, false, 16);
                host(e, a.local);
                e.mem(W.i64_load, 0, a.off);
                cm.set(e, o.a);
                host(e, a.local);
                e.mem(W.i64_load, 0, a.off + 8);
                cm.set(e, o.a + 1);
            }
            if (o.h == .ldr_q_w) writeback(e, o);
        },
        .str_q_i, .str_q_w, .str_q_r => {
            if (e.simd) {
                const off = prepareAddrOp(e, o, at, true, 16);
                cm.getV(e, vreg(o.a));
                vmem(e, cm.S.v128_store, off);
            } else {
                const a = prepareHostOp(e, o, at, true, 16);
                host(e, a.local);
                cm.get(e, o.a);
                e.mem(W.i64_store, 0, a.off);
                host(e, a.local);
                cm.get(e, o.a + 1);
                e.mem(W.i64_store, 0, a.off + 8);
            }
            if (o.h == .str_q_w) writeback(e, o);
        },
        .ldp_q, .ldp_q_w => {
            const a = prepareHostOp(e, o, at, false, 32);
            if (e.simd) {
                // Rt, then Rt2: Rt2 wins when they are equal.
                host(e, a.local);
                vmem(e, cm.S.v128_load, a.off);
                cm.setV(e, vreg(o.a));
                host(e, a.local);
                vmem(e, cm.S.v128_load, a.off + 16);
                cm.setV(e, vreg(o.c));
            } else {
                const parts = [4]struct { u64, u8 }{ .{ 0, o.a }, .{ 8, o.a + 1 }, .{ 16, o.c }, .{ 24, o.c + 1 } };
                for (parts) |p| {
                    host(e, a.local);
                    e.mem(W.i64_load, 0, a.off + p[0]);
                    cm.set(e, p[1]);
                }
            }
            if (o.h == .ldp_q_w) writeback(e, o);
        },
        .stp_q, .stp_q_w => {
            const a = prepareHostOp(e, o, at, true, 32);
            if (e.simd) {
                host(e, a.local);
                cm.getV(e, vreg(o.a));
                vmem(e, cm.S.v128_store, a.off);
                host(e, a.local);
                cm.getV(e, vreg(o.c));
                vmem(e, cm.S.v128_store, a.off + 16);
            } else {
                const parts = [4]struct { u64, u8 }{ .{ 0, o.a }, .{ 8, o.a + 1 }, .{ 16, o.c }, .{ 24, o.c + 1 } };
                for (parts) |p| {
                    host(e, a.local);
                    cm.get(e, p[1]);
                    e.mem(W.i64_store, 0, a.off + p[0]);
                }
            }
            if (o.h == .stp_q_w) writeback(e, o);
        },
        else => unreachable,
    }
}

// ---- intra-block TLB coalescing & self-loop page cache analysis ---------------

const AccessKind = enum { base_imm, reg_off };

const Access = struct {
    kind: AccessKind,
    base: u8, // 0..31
    is_write: bool,
    size: u8,
    acc_off: i64 = 0,
    wb: bool = false,
    wb_imm: i64 = 0,
    wb_kill: bool = false,
    dest_mask: u32 = 0, // X registers overwritten by load data (excl. writeback)
    idx_reg: u8 = ops.zr,
    s1: u8 = 0,
    s2: u8 = 0,
    mask: u64 = 0,
};

fn classifyMemOp(o: Op) ?Access {
    const sz: u8 = @intCast(width(o.h));
    switch (o.h) {
        .ldr8_i, .ldr16_i, .ldr32_i, .ldr64_i => {
            if (o.b >= 32) return null; // literal
            return .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = sz, .acc_off = @bitCast(o.imm), .dest_mask = cm.xbit(o.a) };
        },
        .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w => {
            if (o.b >= 32) return null;
            const imm: i64 = @bitCast(o.imm);
            return .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = sz, .acc_off = if (o.f != 0) imm else 0, .wb = true, .wb_imm = imm, .dest_mask = cm.xbit(o.a) };
        },
        .ldr8_r, .ldr16_r, .ldr32_r, .ldr64_r => {
            if (o.b >= 32) return null;
            return .{ .kind = .reg_off, .base = o.b, .is_write = false, .size = sz, .dest_mask = cm.xbit(o.a), .idx_reg = o.c, .s1 = o.e & 63, .s2 = o.f & 63, .mask = o.imm };
        },
        .str8_i, .str16_i, .str32_i, .str64_i => {
            if (o.b >= 32) return null;
            return .{ .kind = .base_imm, .base = o.b, .is_write = true, .size = sz, .acc_off = @bitCast(o.imm) };
        },
        .str8_w, .str16_w, .str32_w, .str64_w => {
            if (o.b >= 32) return null;
            const imm: i64 = @bitCast(o.imm);
            return .{ .kind = .base_imm, .base = o.b, .is_write = true, .size = sz, .acc_off = if (o.f != 0) imm else 0, .wb = true, .wb_imm = imm };
        },
        .str8_r, .str16_r, .str32_r, .str64_r => {
            if (o.b >= 32) return null;
            return .{ .kind = .reg_off, .base = o.b, .is_write = true, .size = sz, .idx_reg = o.c, .s1 = o.e & 63, .s2 = o.f & 63, .mask = o.imm };
        },
        .ldp32, .ldp64 => {
            if (o.b >= 32) return null;
            return .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = 2 * sz, .acc_off = @bitCast(o.imm), .dest_mask = cm.xbit(o.a) | cm.xbit(o.c) };
        },
        .ldp32_w, .ldp64_w => {
            if (o.b >= 32) return null;
            const imm: i64 = @bitCast(o.imm);
            return .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = 2 * sz, .acc_off = if (o.e != 0) imm else 0, .wb = true, .wb_imm = imm, .dest_mask = cm.xbit(o.a) | cm.xbit(o.c) };
        },
        .stp32, .stp64 => {
            if (o.b >= 32) return null;
            return .{ .kind = .base_imm, .base = o.b, .is_write = true, .size = 2 * sz, .acc_off = @bitCast(o.imm) };
        },
        .stp32_w, .stp64_w => {
            if (o.b >= 32) return null;
            const imm: i64 = @bitCast(o.imm);
            return .{ .kind = .base_imm, .base = o.b, .is_write = true, .size = 2 * sz, .acc_off = if (o.e != 0) imm else 0, .wb = true, .wb_imm = imm };
        },
        .ldr_v => {
            if (o.b >= 32) return null;
            const n: u8 = @as(u8, 1) << @intCast(o.d);
            return switch (o.e) {
                0 => .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = n, .acc_off = @bitCast(o.imm) },
                1 => blk: {
                    const imm: i64 = @bitCast(o.imm);
                    break :blk .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = n, .acc_off = if (o.f != 0) imm else 0, .wb = true, .wb_imm = imm };
                },
                else => .{ .kind = .reg_off, .base = o.b, .is_write = false, .size = n, .idx_reg = o.c, .s1 = o.f & 63, .s2 = o.g & 63, .mask = o.imm },
            };
        },
        .ldr_q_i => {
            if (o.b >= 32) return null;
            return .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = 16, .acc_off = @bitCast(o.imm) };
        },
        .ldr_q_w => {
            if (o.b >= 32) return null;
            const imm: i64 = @bitCast(o.imm);
            return .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = 16, .acc_off = if (o.f != 0) imm else 0, .wb = true, .wb_imm = imm };
        },
        .ldr_q_r => {
            if (o.b >= 32) return null;
            return .{ .kind = .reg_off, .base = o.b, .is_write = false, .size = 16, .idx_reg = o.c, .s1 = o.e & 63, .s2 = o.f & 63, .mask = o.imm };
        },
        .str_q_i => {
            if (o.b >= 32) return null;
            return .{ .kind = .base_imm, .base = o.b, .is_write = true, .size = 16, .acc_off = @bitCast(o.imm) };
        },
        .str_q_w => {
            if (o.b >= 32) return null;
            const imm: i64 = @bitCast(o.imm);
            return .{ .kind = .base_imm, .base = o.b, .is_write = true, .size = 16, .acc_off = if (o.f != 0) imm else 0, .wb = true, .wb_imm = imm };
        },
        .str_q_r => {
            if (o.b >= 32) return null;
            return .{ .kind = .reg_off, .base = o.b, .is_write = true, .size = 16, .idx_reg = o.c, .s1 = o.e & 63, .s2 = o.f & 63, .mask = o.imm };
        },
        .ldp_q => {
            if (o.b >= 32) return null;
            return .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = 32, .acc_off = @bitCast(o.imm) };
        },
        .ldp_q_w => {
            if (o.b >= 32) return null;
            const imm: i64 = @bitCast(o.imm);
            return .{ .kind = .base_imm, .base = o.b, .is_write = false, .size = 32, .acc_off = if (o.e != 0) imm else 0, .wb = true, .wb_imm = imm };
        },
        .stp_q => {
            if (o.b >= 32) return null;
            return .{ .kind = .base_imm, .base = o.b, .is_write = true, .size = 32, .acc_off = @bitCast(o.imm) };
        },
        .stp_q_w => {
            if (o.b >= 32) return null;
            const imm: i64 = @bitCast(o.imm);
            return .{ .kind = .base_imm, .base = o.b, .is_write = true, .size = 32, .acc_off = if (o.e != 0) imm else 0, .wb = true, .wb_imm = imm };
        },
        else => return null,
    }
}

fn classifyVecOp(v: vec_mod.V) ?Access {
    return switch (v.k) {
        .ldpv, .stpv => blk: {
            const n: u8 = @as(u8, 1) << v.size;
            const imm: i64 = @bitCast(v.imm);
            break :blk .{
                .kind = .base_imm,
                .base = v.n,
                .is_write = v.k == .stpv,
                .size = 2 * n,
                .acc_off = if (v.x != 1) imm else 0,
                .wb = v.x == 1 or v.x == 3,
                .wb_imm = imm,
            };
        },
        .ld1, .st1, .ldn, .stn => .{
            .kind = .base_imm,
            .base = v.n,
            .is_write = v.k == .st1 or v.k == .stn,
            .size = (if (v.q) @as(u8, 16) else 8) * v.sh,
            .wb = v.x == 1,
            .wb_imm = @bitCast(v.imm),
            .wb_kill = v.x == 2,
        },
        .ld1s, .st1s, .ld1r => .{
            .kind = .base_imm,
            .base = v.n,
            .is_write = v.k == .st1s,
            .size = @as(u8, 1) << v.size,
            .wb = v.x == 1,
            .wb_imm = @bitCast(v.imm),
            .wb_kill = v.x == 2,
        },
        else => null,
    };
}

const RegVal = struct {
    root: u8,
    ver: u16,
    delta: i64,
    exact_ver: u16,
};

const Cand = struct {
    kind: AccessKind,
    root: u8,
    ver: u16,
    idx_reg: u8,
    idx_ver: u16,
    s1: u8,
    s2: u8,
    mask: u64,
    seg_id: u16,
    head_k: u8,
    last_k: u8,
    head_delta: i64,
    min_off: i64,
    max_end: i64,
    is_write: bool,
    count: u8,
};

const max_cand = 32;
const no_cand: u8 = 0xff;

fn inBounds(v: i64) bool {
    return v >= -max_delta and v <= max_delta;
}

pub fn beginMember(e: *Emit, o: []const Op) void {
    const n: usize = @min(o.len, max_ops);
    e.x.mem.n_ops = @intCast(n);
    for (0..n) |k| {
        e.x.mem.op_tag[k] = 0;
        e.x.mem.op_delta[k] = 0;
    }

    var reg: [32]RegVal = undefined;
    for (0..32) |r| reg[r] = .{ .root = @intCast(r), .ver = 0, .delta = 0, .exact_ver = 0 };
    var next_ver: u16 = 1;
    var seg_id: u16 = 0;

    var cands: [max_cand]Cand = undefined;
    var n_cands: usize = 0;
    var min_cand: usize = 0;
    var op_cand: [max_ops]u8 = @splat(no_cand);
    var op_off: [max_ops]i64 = @splat(0);
    var is_mem_op: [max_ops]bool = @splat(false);
    var has_generic_call = false;

    for (o[0..n], 0..) |op, k| {
        const fam = jit.family(op.h);
        var acc: ?Access = null;
        var generic_call = false;
        var wmask: u32 = 0;

        if (fam == .mem) {
            is_mem_op[k] = true;
            acc = classifyMemOp(op);
            if (acc == null) wmask = usage(op).write; // literal load
        } else if (fam == .none) {
            const fast = gen_mod.decode(op.aux());
            const aa: ?atomic_mod.A = if (fast == null) atomic_mod.decode(op.aux()) else null;
            const aok = if (aa) |a| atomic_mod.enabled(e, a) else false;
            const zl: ?Op = if (fast == null and aa == null) jit.xzrLoad(op) else null;
            const vv: ?vec_mod.V = if (fast == null and aa == null and zl == null) vec_mod.decode(op.aux()) else null;
            const vok = if (vv) |v| vec_mod.enabled(e, v) else false;
            if (fast) |f| {
                wmask = gen_mod.usage(f).write;
            } else if (aok) {
                wmask = atomic_mod.usage(aa.?).write;
            } else if (zl) |l| {
                is_mem_op[k] = true;
                acc = classifyMemOp(l);
                if (acc == null) wmask = usage(l).write;
            } else if (vok) {
                acc = classifyVecOp(vv.?);
                if (acc != null) {
                    is_mem_op[k] = true;
                } else {
                    wmask = vec_mod.usage(vv.?).write;
                }
            } else {
                generic_call = true;
            }
        } else if (fam == .branch) {
            if (ops.isSide(op.h)) {
                seg_id += 1;
            } else {
                wmask = jit.usage(op).write;
            }
        } else if (fam == .alu) {
            if (op.h == .add_i and op.d == 0 and op.a < 32 and op.b < 32) {
                const imm_s: i64 = @bitCast(op.imm);
                const src = reg[op.b];
                if (inBounds(imm_s) and inBounds(src.delta + imm_s)) {
                    reg[op.a] = .{ .root = src.root, .ver = src.ver, .delta = src.delta + imm_s, .exact_ver = next_ver };
                    next_ver += 1;
                } else {
                    reg[op.a] = .{ .root = op.a, .ver = next_ver, .delta = 0, .exact_ver = next_ver };
                    next_ver += 1;
                }
                continue;
            } else if (op.h == .mov and op.imm == std.math.maxInt(u64) and op.a < 32 and op.b < 32) {
                const src = reg[op.b];
                reg[op.a] = .{ .root = src.root, .ver = src.ver, .delta = src.delta, .exact_ver = next_ver };
                next_ver += 1;
                continue;
            } else {
                wmask = jit.usage(op).write;
            }
        } else {
            wmask = jit.usage(op).write;
        }

        if (generic_call) {
            has_generic_call = true;
            seg_id += 1;
            min_cand = n_cands;
            for (0..32) |r| {
                reg[r] = .{ .root = @intCast(r), .ver = next_ver, .delta = 0, .exact_ver = next_ver };
                next_ver += 1;
            }
            continue;
        }

        if (acc) |a| {
            switch (a.kind) {
                .base_imm => {
                    const rv = reg[a.base];
                    if (inBounds(a.acc_off) and inBounds(rv.delta + a.acc_off)) {
                        const off = rv.delta + a.acc_off;
                        const end = off + @as(i64, a.size);
                        var matched = false;
                        var j = n_cands;
                        while (j > min_cand) {
                            j -= 1;
                            const c = &cands[j];
                            if (c.kind != .base_imm or c.root != rv.root or c.ver != rv.ver) continue;
                            if (c.seg_id == seg_id) {
                                const new_min = @min(c.min_off, off);
                                const new_max = @max(c.max_end, end);
                                if (new_max - new_min <= max_span) {
                                    c.min_off = new_min;
                                    c.max_end = new_max;
                                    c.is_write = c.is_write or a.is_write;
                                    c.last_k = @intCast(k);
                                    c.count += 1;
                                    op_cand[k] = @intCast(j);
                                    op_off[k] = off;
                                    matched = true;
                                    break;
                                }
                            } else {
                                // After a side exit: reuse only if already covered (no widening or write promotion).
                                if (off >= c.min_off and end <= c.max_end and (!a.is_write or c.is_write)) {
                                    c.last_k = @intCast(k);
                                    c.count += 1;
                                    op_cand[k] = @intCast(j);
                                    op_off[k] = off;
                                    matched = true;
                                    break;
                                }
                            }
                        }
                        if (!matched and n_cands < max_cand) {
                            cands[n_cands] = .{
                                .kind = .base_imm,
                                .root = rv.root,
                                .ver = rv.ver,
                                .idx_reg = ops.zr,
                                .idx_ver = 0,
                                .s1 = 0,
                                .s2 = 0,
                                .mask = 0,
                                .seg_id = seg_id,
                                .head_k = @intCast(k),
                                .last_k = @intCast(k),
                                .head_delta = rv.delta,
                                .min_off = off,
                                .max_end = end,
                                .is_write = a.is_write,
                                .count = 1,
                            };
                            op_cand[k] = @intCast(n_cands);
                            op_off[k] = off;
                            n_cands += 1;
                        }
                    }
                },
                .reg_off => {
                    const bver = reg[a.base].exact_ver;
                    const iver: u16 = if (a.idx_reg < 32) reg[a.idx_reg].exact_ver else 0;
                    const end: i64 = a.size;
                    var matched = false;
                    var j = n_cands;
                    while (j > min_cand) {
                        j -= 1;
                        const c = &cands[j];
                        if (c.kind != .reg_off or c.root != a.base or c.ver != bver or
                            c.idx_reg != a.idx_reg or c.idx_ver != iver or
                            c.s1 != a.s1 or c.s2 != a.s2 or c.mask != a.mask) continue;
                        if (c.seg_id == seg_id) {
                            const new_max = @max(c.max_end, end);
                            if (new_max <= max_span) {
                                c.max_end = new_max;
                                c.is_write = c.is_write or a.is_write;
                                c.last_k = @intCast(k);
                                c.count += 1;
                                op_cand[k] = @intCast(j);
                                op_off[k] = 0;
                                matched = true;
                                break;
                            }
                        } else if (end <= c.max_end and (!a.is_write or c.is_write)) {
                            c.last_k = @intCast(k);
                            c.count += 1;
                            op_cand[k] = @intCast(j);
                            op_off[k] = 0;
                            matched = true;
                            break;
                        }
                    }
                    if (!matched and n_cands < max_cand) {
                        cands[n_cands] = .{
                            .kind = .reg_off,
                            .root = a.base,
                            .ver = bver,
                            .idx_reg = a.idx_reg,
                            .idx_ver = iver,
                            .s1 = a.s1,
                            .s2 = a.s2,
                            .mask = a.mask,
                            .seg_id = seg_id,
                            .head_k = @intCast(k),
                            .last_k = @intCast(k),
                            .head_delta = 0,
                            .min_off = 0,
                            .max_end = end,
                            .is_write = a.is_write,
                            .count = 1,
                        };
                        op_cand[k] = @intCast(n_cands);
                        op_off[k] = 0;
                        n_cands += 1;
                    }
                },
            }
            // Apply base writeback, then kill any X load destinations.
            if (a.wb and !a.wb_kill) {
                const src = reg[a.base];
                if (inBounds(a.wb_imm) and inBounds(src.delta + a.wb_imm)) {
                    reg[a.base] = .{ .root = src.root, .ver = src.ver, .delta = src.delta + a.wb_imm, .exact_ver = next_ver };
                    next_ver += 1;
                } else {
                    reg[a.base] = .{ .root = a.base, .ver = next_ver, .delta = 0, .exact_ver = next_ver };
                    next_ver += 1;
                }
            } else if (a.wb_kill) {
                reg[a.base] = .{ .root = a.base, .ver = next_ver, .delta = 0, .exact_ver = next_ver };
                next_ver += 1;
            }
            wmask = a.dest_mask;
        }

        // Kill any X registers written by this op.
        var m = wmask;
        while (m != 0) {
            const r: u8 = @intCast(@ctz(m));
            m &= m - 1;
            reg[r] = .{ .root = r, .ver = next_ver, .delta = 0, .exact_ver = next_ver };
            next_ver += 1;
        }
    }

    // Materialize coalesced groups (count >= 2) with linear-scan slot allocation.
    var cand_to_group: [max_cand]u8 = @splat(no_cand);
    var slot_end: [max_slots]u8 = @splat(0);
    var slot_used: [max_slots]bool = @splat(false);
    var n_groups: u8 = 0;
    for (0..n_cands) |ci| {
        const c = &cands[ci];
        if (c.count < 2 or n_groups >= max_groups) continue;
        var found: ?u8 = null;
        for (0..max_slots) |s| {
            if (!slot_used[s] or slot_end[s] < c.head_k) {
                found = @intCast(s);
                break;
            }
        }
        const s = found orelse continue;
        slot_used[s] = true;
        slot_end[s] = c.last_k;
        const span = c.max_end - c.min_off;
        std.debug.assert(span >= 1 and span <= max_span);
        e.x.mem.groups[n_groups] = .{
            .va_off = @intCast(c.min_off - c.head_delta),
            .span_m1 = @intCast(span - 1),
            .slot = s,
            .cache = no_cache,
            .flags = (if (c.is_write) flag_write else 0) | (if (c.kind == .reg_off) flag_reg else 0),
        };
        cand_to_group[ci] = n_groups;
        n_groups += 1;
    }

    for (0..n) |k| {
        const ci = op_cand[k];
        if (ci == no_cand) continue;
        const gi = cand_to_group[ci];
        if (gi == no_cand) continue;
        const c = &cands[ci];
        e.x.mem.op_delta[k] = @intCast(op_off[k] - c.min_off);
        e.x.mem.op_tag[k] = if (k == c.head_k) tag_head + gi else tag_follow + gi;
    }

    // Assign loop-carried page cache slots for standalone self-loops without generic calls.
    if (e.rg == null and !has_generic_call) {
        var next_cache: u8 = 0;
        for (0..n) |k| {
            if (!is_mem_op[k] or next_cache >= max_cache) continue;
            const tag = e.x.mem.op_tag[k];
            if (tag == 0) {
                e.x.mem.op_tag[k] = 1 + next_cache;
                next_cache += 1;
            } else if (tag >= tag_head and tag < tag_follow) {
                e.x.mem.groups[tag - tag_head].cache = next_cache;
                next_cache += 1;
            }
        }
    }
}

pub fn sync(e: *Emit) void {
    _ = e;
}

// ---- tests (kindtest.zig: one line per kind) ----------------------------------
//
// Each generator makes only its kind, in every variant the decoder gives it
// (sizes, extensions, W/X and B/H/S/D/Q data, every addressing mode), with
// a setup word or two before it (ADR for the base, MOVZ/MOVN for an index:
// `movi` ops, so the compiled prefix reaches the access). The base is
// chosen so the access lands where the generator wants: mostly inside the
// RW pages, else at a page end (straddling into the next RW page, which
// the interpreter performs and the JIT must hand back, or into the
// unmapped page), on the read-only page (stores fault), on the code page
// (stores fault, or with `rwx_code` hit decoded code), unmapped, or
// wrapping past 2^64. Rt is now and then the base or the index register
// (writeback with Rn == Rt, Rt == Rt2), store data XZR.

const kt = @import("kindtest.zig");

const Mode = enum { uimm, unscaled, unpriv, post, pre, regoff, literal, pair_off, pair_nt, pair_post, pair_pre };

/// One encoding of a kind: its fixed fields and how it addresses memory.
const Form = struct {
    /// The word without Rt, Rn, Rt2/Rm and the offset fields.
    word: u32,
    mode: Mode,
    /// log2 of the immediate scale (single: the access size, pairs: one
    /// register).
    scale: u5,
    /// Bytes accessed.
    n: u64,
    load: bool,
    v: bool,
};

fn one(comptime T: type, r: std.Random, xs: []const T) T {
    return xs[r.uintLessThan(usize, xs.len)];
}

/// A random encoding that decodes to kind h.
fn form(comptime h: H, r: std.Random) Form {
    const single = struct {
        fn f(size: u32, v: u32, opc: u32, m: Mode, load: bool) Form {
            const scale: u5 = if (v == 1 and opc >= 2) 4 else @intCast(size);
            const base: u32 = switch (m) {
                .uimm => 0x3900_0000,
                .unscaled => 0x3800_0000,
                .post => 0x3800_0400,
                .unpriv => 0x3800_0800,
                .pre => 0x3800_0c00,
                .regoff => 0x3820_0800,
                else => unreachable,
            };
            return .{ .word = base | (size << 30) | (v << 26) | (opc << 22), .mode = m, .scale = scale, .n = @as(u64, 1) << scale, .load = load, .v = v == 1 };
        }
        fn lit(opc: u32, v: u32, n: u64) Form {
            return .{ .word = 0x1800_0000 | (opc << 30) | (v << 26), .mode = .literal, .scale = 0, .n = n, .load = true, .v = v == 1 };
        }
        fn pair(opc: u32, v: u32, l: u32, m: Mode, scale: u5) Form {
            const idx: u32 = switch (m) {
                .pair_nt => 0,
                .pair_post => 1,
                .pair_off => 2,
                .pair_pre => 3,
                else => unreachable,
            };
            return .{ .word = 0x2800_0000 | (opc << 30) | (v << 26) | (idx << 23) | (l << 22), .mode = m, .scale = scale, .n = @as(u64, 2) << scale, .load = l == 1, .v = v == 1 };
        }
    };
    const S = single;
    const size: u32 = switch (width(h)) {
        1 => 0,
        2 => 1,
        4 => 2,
        else => 3,
    };
    // Integer load opc: 1 zero-extend, 2 sign-extend to X (not 64-bit), 3
    // sign-extend to W (8 and 16 bits).
    const ld_opc: []const u32 = switch (size) {
        0, 1 => &.{ 1, 2, 3 },
        2 => &.{ 1, 2 },
        else => &.{1},
    };
    return switch (h) {
        .ldr8_i, .ldr16_i, .ldr32_i, .ldr64_i => blk: {
            if (size >= 2 and r.uintLessThan(u32, 5) == 0) break :blk if (size == 2)
                (if (r.boolean()) S.lit(0, 0, 4) else S.lit(2, 0, 4)) // LDR W, LDRSW (literal)
            else
                S.lit(1, 0, 8);
            break :blk S.f(size, 0, one(u32, r, ld_opc), one(Mode, r, &.{ .uimm, .uimm, .unscaled, .unpriv }), true);
        },
        .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w => S.f(size, 0, one(u32, r, ld_opc), one(Mode, r, &.{ .post, .pre }), true),
        .ldr8_r, .ldr16_r, .ldr32_r, .ldr64_r => S.f(size, 0, one(u32, r, ld_opc), .regoff, true),
        // Stores: integer, or STR B/H/S/D (V = 1, opc 0).
        .str8_i, .str16_i, .str32_i, .str64_i => blk: {
            // STTR exists for integer data only (V = 1 there is generic).
            const v = r.uintLessThan(u32, 2);
            break :blk S.f(size, v, 0, if (v == 1) one(Mode, r, &.{ .uimm, .unscaled }) else one(Mode, r, &.{ .uimm, .uimm, .unscaled, .unpriv }), false);
        },
        .str8_w, .str16_w, .str32_w, .str64_w => S.f(size, r.uintLessThan(u32, 2), 0, one(Mode, r, &.{ .post, .pre }), false),
        .str8_r, .str16_r, .str32_r, .str64_r => S.f(size, r.uintLessThan(u32, 2), 0, .regoff, false),
        .ldp32 => if (r.boolean()) S.pair(1, 0, 1, .pair_off, 2) else S.pair(0, 0, 1, one(Mode, r, &.{ .pair_off, .pair_nt }), 2),
        .ldp32_w => S.pair(r.uintLessThan(u32, 2), 0, 1, one(Mode, r, &.{ .pair_post, .pair_pre }), 2),
        .ldp64 => S.pair(2, 0, 1, one(Mode, r, &.{ .pair_off, .pair_nt }), 3),
        .ldp64_w => S.pair(2, 0, 1, one(Mode, r, &.{ .pair_post, .pair_pre }), 3),
        // STP W / S (opc 0), X / D (opc 2 / V opc 1).
        .stp32 => S.pair(0, r.uintLessThan(u32, 2), 0, one(Mode, r, &.{ .pair_off, .pair_nt }), 2),
        .stp32_w => S.pair(0, r.uintLessThan(u32, 2), 0, one(Mode, r, &.{ .pair_post, .pair_pre }), 2),
        .stp64, .stp64_w => blk: {
            const v = r.uintLessThan(u32, 2);
            const m: Mode = if (h == .stp64) one(Mode, r, &.{ .pair_off, .pair_nt }) else one(Mode, r, &.{ .pair_post, .pair_pre });
            break :blk S.pair(if (v == 1) 1 else 2, v, 0, m, 3);
        },
        .ldr_v => blk: {
            if (r.uintLessThan(u32, 6) == 0) break :blk if (r.boolean()) S.lit(0, 1, 4) else S.lit(1, 1, 8);
            break :blk S.f(r.uintLessThan(u32, 4), 1, 1, one(Mode, r, &.{ .uimm, .unscaled, .post, .pre, .regoff }), true);
        },
        .ldr_q_i => if (r.uintLessThan(u32, 5) == 0) S.lit(2, 1, 16) else S.f(0, 1, 3, one(Mode, r, &.{ .uimm, .unscaled }), true),
        .ldr_q_w => S.f(0, 1, 3, one(Mode, r, &.{ .post, .pre }), true),
        .ldr_q_r => S.f(0, 1, 3, .regoff, true),
        .str_q_i => S.f(0, 1, 2, one(Mode, r, &.{ .uimm, .unscaled }), false),
        .str_q_w => S.f(0, 1, 2, one(Mode, r, &.{ .post, .pre }), false),
        .str_q_r => S.f(0, 1, 2, .regoff, false),
        .ldp_q => S.pair(2, 1, 1, one(Mode, r, &.{ .pair_off, .pair_nt }), 4),
        .ldp_q_w => S.pair(2, 1, 1, one(Mode, r, &.{ .pair_post, .pair_pre }), 4),
        .stp_q => S.pair(2, 1, 0, one(Mode, r, &.{ .pair_off, .pair_nt }), 4),
        .stp_q_w => S.pair(2, 1, 0, one(Mode, r, &.{ .pair_post, .pair_pre }), 4),
        else => comptime unreachable,
    };
}

/// Where an access of n bytes should land (see above).
fn target(r: std.Random, n: u64) u64 {
    const page = r.uintLessThan(u64, 4);
    const cut = if (n > 1) r.intRangeAtMost(u64, 1, n - 1) else 0;
    return switch (r.uintLessThan(u32, 100)) {
        0...57 => kt.data_va + page * 4096 + r.uintLessThan(u64, 4096 - n + 1),
        58...63 => kt.data_va + page * 4096 + (if (r.boolean()) 4096 - n else 0), // a page's last or first bytes
        64...71 => kt.data_va + r.uintLessThan(u64, 3) * 4096 + 4096 - cut, // straddles two RW pages
        72...74 => kt.lazy_va - cut, // straddles into the unmapped page
        75...80 => kt.ro_va + r.uintLessThan(u64, 4096 - n + 1),
        81...82 => kt.ro_va + 4096 - cut, // straddles out of the read-only page
        83...89 => kt.code_va + r.uintLessThan(u64, 2 * 4096 - n + 1),
        90...92 => kt.lazy_va + r.uintLessThan(u64, 4096),
        93...96 => 0 -% (if (r.boolean()) cut else r.uintLessThan(u64, 64)), // wraps past 2^64
        else => r.uintLessThan(u64, 4096), // page 0
    };
}

/// ADR xd, t (t within 1 MiB of pc).
fn adr(rd: u32, pc: u64, t: u64) ?u32 {
    const d: i64 = @bitCast(t -% pc);
    if (d < -(1 << 20) or d >= 1 << 20) return null;
    const u: u32 = @truncate(@as(u64, @bitCast(d)));
    return 0x1000_0000 | ((u & 3) << 29) | (((u >> 2) & 0x7ffff) << 5) | rd;
}

fn sext(v: u64, comptime bits: u7) u64 {
    const s: u6 = @intCast(64 - bits);
    return @bitCast(@as(i64, @bitCast(v << s)) >> s);
}

/// After the n words at out (from pc0), now and then B code_va: the block
/// (from code_va) becomes a compiled self-loop, whose later passes find the
/// data TLB filled by the first one (a program's first access to a page
/// always misses and hands back). Returns the word count.
fn closeLoop(r: std.Random, out: []u32, pc0: u64, n: usize) usize {
    if (r.uintLessThan(u32, 8) != 0) return n;
    out[n] = 0x1400_0000 | (@as(u32, @truncate((kt.code_va -% (pc0 + 4 * n)) >> 2)) & 0x3ff_ffff);
    return n + 1;
}

/// The generator of kind h's test.
fn gen(comptime h: H) kt.Gen {
    return struct {
        fn g(r: std.Random, out: []u32, pc0: u64) usize {
            const f = form(h, r);
            var i: usize = 0;
            const t = target(r, f.n);
            // Registers: B the base (set by ADR), I an index (MOVZ/MOVN).
            const b_reg = kt.dst(r);
            var i_reg = kt.dst(r);
            while (i_reg == b_reg) i_reg = kt.dst(r);
            // Rt, Rt2: X loads never 31 (a load into XZR is generic).
            const pick = struct {
                fn rt(rr: std.Random, ff: Form, bb: u32, ii: u32) u32 {
                    return switch (rr.uintLessThan(u32, 10)) {
                        0 => bb,
                        1 => if (ff.mode == .regoff) ii else bb,
                        2 => if (ff.load and !ff.v) kt.dst(rr) else 31,
                        else => if (ff.v) rr.uintLessThan(u32, 32) else kt.dst(rr),
                    };
                }
            }.rt;
            const rt = pick(r, f, b_reg, i_reg);
            var w = f.word | rt;
            // Offset: the access is at base + off (post-index: at base).
            var off: u64 = 0;
            var use_reg_base = true;
            switch (f.mode) {
                .literal => {
                    const d: i64 = @bitCast((t & ~@as(u64, 3)) -% pc0);
                    // Out of reach: offset 0, the literal is the word itself.
                    if (d >= -(1 << 20) and d < 1 << 20) w |= (@as(u32, @truncate(@as(u64, @bitCast(d)) >> 2)) & 0x7ffff) << 5;
                    out[0] = w;
                    return closeLoop(r, out, pc0, 1);
                },
                .uimm => {
                    const imm = if (r.boolean()) r.uintLessThan(u32, 64) else r.uintLessThan(u32, 4096);
                    off = @as(u64, imm) << f.scale;
                    w |= imm << 10;
                },
                .unscaled, .unpriv, .pre, .post => {
                    const imm = r.intRangeAtMost(i32, -256, 255);
                    w |= (@as(u32, @bitCast(imm)) & 0x1ff) << 12;
                    if (f.mode != .post) off = @bitCast(@as(i64, imm));
                },
                .pair_off, .pair_nt, .pair_pre, .pair_post => {
                    const imm = r.intRangeAtMost(i32, -64, 63);
                    // Rt2: like Rt, sometimes equal to it.
                    const rt2 = if (r.uintLessThan(u32, 6) == 0) rt else pick(r, f, b_reg, i_reg);
                    w |= ((@as(u32, @bitCast(imm)) & 0x7f) << 15) | (rt2 << 10);
                    if (f.mode != .pair_post) off = @as(u64, @bitCast(@as(i64, imm))) << f.scale;
                },
                .regoff => {
                    const option = one(u32, r, &.{ 2, 3, 6, 7 }); // UXTW, LSL, SXTW, SXTX
                    const s = r.uintLessThan(u32, 2);
                    const xzr_index = r.uintLessThan(u32, 10) == 0;
                    var v: u64 = 0;
                    if (!xzr_index) {
                        const k = r.uintLessThan(u32, 300);
                        const neg = option != 2 and r.boolean() or r.uintLessThan(u32, 20) == 0;
                        const wide = r.uintLessThan(u32, 4) != 0; // else a W MOVN: 0x0000_0000_ffff_xxxx
                        out[i] = if (neg)
                            (if (wide) @as(u32, 0x9280_0000) else 0x1280_0000) | (k << 5) | i_reg // movn
                        else
                            0xd280_0000 | (k << 5) | i_reg; // movz
                        v = if (!neg) k else if (wide) ~@as(u64, k) else ~@as(u64, k) & 0xffff_ffff;
                        i += 1;
                    }
                    const shift: u6 = if (s == 1) f.scale else 0;
                    off = switch (option) {
                        2 => (v & 0xffff_ffff) << shift,
                        6 => sext(v, 32) << shift,
                        else => v << shift,
                    };
                    w |= ((if (xzr_index) 31 else i_reg) << 16) | (option << 13) | (s << 12);
                },
            }
            // The base: ADR to t - off, or (now and then, and when that is
            // out of reach) one of the fixed bases with this offset.
            if (r.uintLessThan(u32, 7) != 0) {
                if (adr(b_reg, pc0 + 4 * i, t -% off)) |a| {
                    out[i] = a;
                    i += 1;
                } else use_reg_base = false;
            } else use_reg_base = false;
            const rn = if (use_reg_base) b_reg else kt.base(r);
            out[i] = w | (rn << 5);
            return closeLoop(r, out, pc0, i + 1);
        }
    }.g;
}

// Loops end at the budget (or a fault). Half the programs start with
// filled data TLBs, so that also a first access runs compiled (a page end
// straddle into an unmapped page must still hand back).
const mem_opt: kt.Opt = .{ .rounds = 200, .budget = 3000 };
const warm_opt: kt.Opt = .{ .rounds = 200, .budget = 3000, .warm_tlb = true, .seed = 3 };

fn both(comptime h: H) !void {
    try kt.kind(h, gen(h), mem_opt);
    try kt.kind(h, gen(h), warm_opt);
}

test "a64 jit mem: ldr8_i" {
    try both(.ldr8_i);
}
test "a64 jit mem: ldr16_i" {
    try both(.ldr16_i);
}
test "a64 jit mem: ldr32_i" {
    try both(.ldr32_i);
}
test "a64 jit mem: ldr64_i" {
    try both(.ldr64_i);
}
test "a64 jit mem: ldr8_w" {
    try both(.ldr8_w);
}
test "a64 jit mem: ldr16_w" {
    try both(.ldr16_w);
}
test "a64 jit mem: ldr32_w" {
    try both(.ldr32_w);
}
test "a64 jit mem: ldr64_w" {
    try both(.ldr64_w);
}
test "a64 jit mem: ldr8_r" {
    try both(.ldr8_r);
}
test "a64 jit mem: ldr16_r" {
    try both(.ldr16_r);
}
test "a64 jit mem: ldr32_r" {
    try both(.ldr32_r);
}
test "a64 jit mem: ldr64_r" {
    try both(.ldr64_r);
}
test "a64 jit mem: str8_i" {
    try both(.str8_i);
}
test "a64 jit mem: str16_i" {
    try both(.str16_i);
}
test "a64 jit mem: str32_i" {
    try both(.str32_i);
}
test "a64 jit mem: str64_i" {
    try both(.str64_i);
}
test "a64 jit mem: str8_w" {
    try both(.str8_w);
}
test "a64 jit mem: str16_w" {
    try both(.str16_w);
}
test "a64 jit mem: str32_w" {
    try both(.str32_w);
}
test "a64 jit mem: str64_w" {
    try both(.str64_w);
}
test "a64 jit mem: str8_r" {
    try both(.str8_r);
}
test "a64 jit mem: str16_r" {
    try both(.str16_r);
}
test "a64 jit mem: str32_r" {
    try both(.str32_r);
}
test "a64 jit mem: str64_r" {
    try both(.str64_r);
}
test "a64 jit mem: ldp32" {
    try both(.ldp32);
}
test "a64 jit mem: ldp64" {
    try both(.ldp64);
}
test "a64 jit mem: ldp32_w" {
    try both(.ldp32_w);
}
test "a64 jit mem: ldp64_w" {
    try both(.ldp64_w);
}
test "a64 jit mem: stp32" {
    try both(.stp32);
}
test "a64 jit mem: stp64" {
    try both(.stp64);
}
test "a64 jit mem: stp32_w" {
    try both(.stp32_w);
}
test "a64 jit mem: stp64_w" {
    try both(.stp64_w);
}
test "a64 jit mem: ldr_v" {
    try both(.ldr_v);
}
test "a64 jit mem: ldr_q_i" {
    try both(.ldr_q_i);
}
test "a64 jit mem: ldr_q_w" {
    try both(.ldr_q_w);
}
test "a64 jit mem: ldr_q_r" {
    try both(.ldr_q_r);
}
test "a64 jit mem: str_q_i" {
    try both(.str_q_i);
}
test "a64 jit mem: str_q_w" {
    try both(.str_q_w);
}
test "a64 jit mem: str_q_r" {
    try both(.str_q_r);
}
test "a64 jit mem: ldp_q" {
    try both(.ldp_q);
}
test "a64 jit mem: ldp_q_w" {
    try both(.ldp_q_w);
}
test "a64 jit mem: stp_q" {
    try both(.stp_q);
}
test "a64 jit mem: stp_q_w" {
    try both(.stp_q_w);
}

// Stores into the code pages, mapped writable: a store into the decoded
// block (or the page it is on) hands back, and the interpreter's store ends
// the block (SMC); the next instruction is the stored word.
const smc_opt: kt.Opt = .{ .rounds = 300, .budget = 3000, .rwx_code = true, .warm_tlb = true, .seed = 2 };

test "a64 jit mem: str32_i into decoded code" {
    try kt.kind(.str32_i, gen(.str32_i), smc_opt);
}
test "a64 jit mem: str64_r into decoded code" {
    try kt.kind(.str64_r, gen(.str64_r), smc_opt);
}
test "a64 jit mem: stp64_w into decoded code" {
    try kt.kind(.stp64_w, gen(.stp64_w), smc_opt);
}
test "a64 jit mem: stp_q into decoded code" {
    try kt.kind(.stp_q, gen(.stp_q), smc_opt);
}

// Adversarial generator for intra-block TLB coalescing & self-loop page cache:
// emits multi-access sequences off shared base registers (spanning page ends,
// RW->unmapped straddles, RO pages, RWX code pages, 2^64 wraps, page 0),
// pre/post writeback chains, ADD/MOV base aliasing, base clobbers, RMW pairs,
// reg-offset RMW, side exits between accesses, and multi-page self-loops.
fn genCoalesce(r: std.Random, out: []u32, pc0: u64) usize {
    const pickBase = struct {
        fn b(rr: std.Random) u32 {
            return switch (rr.uintLessThan(u32, 8)) {
                0, 1, 2 => 20, // RW page 0
                3 => 21, // RW page 1
                4 => 31, // SP
                5 => 24, // RO page
                6 => 23, // unmapped page
                else => 2, // scratch/clobbered/edge base
            };
        }
    }.b;
    const rn = pickBase(r);
    const rt1 = kt.dst(r);
    const rt2 = kt.dst(r);
    const off1 = r.uintLessThan(u32, 8); // * 8 = 0..56
    const off2 = r.uintLessThan(u32, 8);
    var i: usize = 0;
    switch (r.uintLessThan(u32, 9)) {
        0 => { // Two loads off same base (sometimes near a page end)
            if (r.uintLessThan(u32, 3) == 0) {
                const t = target(r, 64);
                if (adr(2, pc0, t)) |a| {
                    out[i] = a;
                    i += 1;
                    out[i] = 0xf940_0000 | (off1 << 10) | (2 << 5) | rt1; // ldr x, [x2, #off1*8]
                    out[i + 1] = 0xf940_0000 | (off2 << 10) | (2 << 5) | rt2; // ldr x, [x2, #off2*8]
                    return closeLoop(r, out, pc0, i + 2);
                }
            }
            out[0] = 0xf940_0000 | (off1 << 10) | (rn << 5) | rt1;
            out[1] = 0xf940_0000 | (off2 << 10) | (rn << 5) | rt2;
            return closeLoop(r, out, pc0, 2);
        },
        1 => { // RMW: load then store at same or nearby offset (hits RO & RWX SMC too)
            const b = if (r.uintLessThan(u32, 3) == 0) @as(u32, 19) else rn; // x19 = code_va
            out[0] = 0xf940_0000 | (off1 << 10) | (b << 5) | rt1;
            out[1] = 0x9100_0400 | (rt1 << 5) | rt1; // add x, x, #1
            out[2] = 0xf900_0000 | ((if (r.boolean()) off1 else off2) << 10) | (b << 5) | rt1;
            return closeLoop(r, out, pc0, 3);
        },
        2 => { // Pre-index writeback + offset follow (prologue pattern)
            const b: u32 = if (r.boolean()) 31 else 2;
            if (b == 2) {
                if (adr(2, pc0, target(r, 32))) |a| {
                    out[i] = a;
                    i += 1;
                }
            }
            out[i] = 0xa9be_0000 | (rt2 << 10) | (b << 5) | rt1; // stp rt1, rt2, [b, #-32]!
            out[i + 1] = 0xa901_0000 | (rt2 << 10) | (b << 5) | rt1; // stp rt1, rt2, [b, #16]
            return i + 2;
        },
        3 => { // Offset + post-index writeback (epilogue pattern)
            const b: u32 = if (r.boolean()) 31 else 20;
            out[0] = 0xa941_0000 | (rt2 << 10) | (b << 5) | rt1; // ldp rt1, rt2, [b, #16]
            out[1] = 0xa8c2_0000 | (rt2 << 10) | (b << 5) | rt1; // ldp rt1, rt2, [b], #32
            return 2;
        },
        4 => { // ADD/MOV base alias + base overwrite
            out[0] = 0x9100_4000 | (rn << 5) | 3; // add x3, rn, #16
            out[1] = 0xf940_0000 | (off1 << 10) | (rn << 5) | rt1;
            out[2] = 0xf940_0000 | (off2 << 10) | (3 << 5) | rt2;
            out[3] = 0xf940_0000 | (off1 << 10) | (3 << 5) | 3; // overwrite x3!
            return 4;
        },
        5 => { // Side exit between load and store/load (RO & straddle precision)
            out[0] = 0xf940_0000 | (off1 << 10) | (rn << 5) | rt1;
            out[1] = 0xb400_0060 | rt1; // cbz rt1, +12 (side exit)
            out[2] = (if (r.boolean()) @as(u32, 0xf900_0000) else 0xf940_0000) | (off2 << 10) | (rn << 5) | rt2;
            return 3;
        },
        6 => { // Reg-offset RMW: ldr [rn, x4] then str [rn, x4]
            const idx_k = r.uintLessThan(u32, 8) * 8;
            out[0] = 0xd280_0000 | (idx_k << 5) | 4; // movz x4, #idx_k
            out[1] = 0xf864_6800 | (rn << 5) | rt1; // ldr rt1, [rn, x4]
            out[2] = 0xf824_6800 | (rn << 5) | rt1; // str rt1, [rn, x4]
            return closeLoop(r, out, pc0, 3);
        },
        7 => { // Self-loop walking across pages (hits page crossings & page 0)
            out[0] = 0xf841_0400 | (rn << 5) | rt1; // ldr rt1, [rn], #16
            out[1] = 0xf940_0400 | (rn << 5) | rt2; // ldr rt2, [rn, #8]
            out[2] = 0x1400_0000 | (@as(u32, @truncate((kt.code_va -% (pc0 + 8)) >> 2)) & 0x3ff_ffff);
            return 3;
        },
        else => { // Q-register RMW pair (ldp q + stp q)
            const v1 = r.uintLessThan(u32, 32);
            const v2 = r.uintLessThan(u32, 32);
            out[0] = 0xad40_0000 | (v2 << 10) | (rn << 5) | v1; // ldp q1, q2, [rn]
            out[1] = 0xad00_0000 | (v2 << 10) | (rn << 5) | v1; // stp q1, q2, [rn]
            return closeLoop(r, out, pc0, 2);
        },
    }
}

test "a64 jit mem: intra-block TLB coalescing and self-loop page cache" {
    try kt.kind(.ldr64_i, genCoalesce, .{ .rounds = 300, .budget = 3000, .seed = 10 });
    try kt.kind(.ldr64_i, genCoalesce, .{ .rounds = 300, .budget = 3000, .warm_tlb = true, .seed = 11 });
    try kt.kind(.str64_i, genCoalesce, .{ .rounds = 200, .budget = 3000, .rwx_code = true, .warm_tlb = true, .seed = 12 });
}
