// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! Helpers the A64 JIT family files share (emit.zig): the local layout,
//! register slots in locals, `Use` bits per slot. Owned by the core stream.
//!
//! Register slots are the tier's (ops.zig): 0..30 X, 31 SP, 32 XZR (regs.pc
//! in memory; compiled code never reads or writes it as a register), 33
//! NZCV, 35 TPIDR_EL0, 36 + 2n / 37 + 2n the low / high half of Vn. Every
//! slot a block uses lives in a local from the prologue to the single
//! epilogue: an i64 per X register and, without SIMD, per V half (`local`);
//! with SIMD (`Emit.simd`, jitcore.zig cap_simd) a v128 per V register
//! (`vloc`), whose halves `get` and `set` extract and replace. XZR reads as
//! the constant 0 and writes to it are dropped (`get`, `set`).

const std = @import("std");
const ops = @import("../ops.zig");
const jc = @import("../../emu/jitcore.zig");
const jit = @import("emit.zig");
const Use = jit.Use;
const Emit = jit.Emit;
const W = jc.W;

// ---- locals --------------------------------------------------------------------
//
//   0..3     the parameters: x, cpu, irqp, jl (jitcore.zig)
//   4..7     core: next pc, result code, a data address, its TLB entry
//   8..11    core: region state (jitcore.zig, n_region_locals)
//   12       NZCV (bits 31..28, as regs.nzcv); alu.zig decides what it holds
//            between exits (lazy flags), the epilogue stores it
//   13..20   l_t + 0..7: scratch of one op's code (any family; never live
//            from one op to the next)
//   21..24   l_alu + 0..3: alu.zig's (lazy flags)
//   25..33   l_mcache + 0..8: mem.zig's self-loop page cache (3 x page/add/lim)
//   34..36   l_mem + 0..2: mem.zig's coalesced group host addresses
//   37..68   X0..X30, SP (slot s: 37 + s)
//   69       TPIDR_EL0
//   70       l_vtmp: with SIMD, the lane value `set` puts into a V register
//   71..78   l_u + 0..7: more i64 scratch of one op's code (vec.zig, generic.zig)
//   without SIMD (all i64):
//   79..142  V halves (slot 36 + j: 79 + j)
//   with SIMD: i64 up to 78, then v128 and i32 locals:
//   79..110  V0..V31 (v128)
//   111..118 l_q + 0..7: v128 scratch of one op's code
//   119..134 l_w + 0..15: i32 scratch of one op's code
//
// A module declares locals only up to the highest one its code uses
// (emit.zig, `locals`), so they are in the order of how often a block
// needs them; with SIMD a group of a later type makes the earlier groups
// full.

pub const l_npc: u32 = 4;
pub const l_code: u32 = 5;
pub const l_va: u32 = 6;
pub const l_e: u32 = 7;
pub const l_region: u32 = 8;
pub const l_nzcv: u32 = 12;
pub const l_t: u32 = 13;
pub const n_t = 8;
pub const l_alu: u32 = 21;
pub const n_alu = 4;
pub const l_mcache: u32 = 25;
pub const n_mcache = 9;
pub const l_mem: u32 = 34;
pub const n_mem = 3;
const l_xbase: u32 = 37;
const l_tpidr: u32 = 69;
pub const l_vtmp: u32 = 70;
pub const l_u: u32 = 71;
pub const n_u = 8;
pub const l_vbase: u32 = l_u + n_u;
/// One past the last local without SIMD.
pub const n_locals: u32 = l_vbase + 64;
/// With SIMD: v128 scratch, i32 scratch (each held only within one op),
/// and one past the last local.
pub const l_q: u32 = l_vbase + 32;
pub const n_q = 8;
pub const l_w: u32 = l_q + n_q;
pub const n_w = 16;
pub const n_locals_simd: u32 = l_w + n_w;

comptime {
    std.debug.assert(l_region + jc.n_region_locals == l_nzcv);
    std.debug.assert(l_t + n_t == l_alu and l_alu + n_alu == l_mcache and l_mcache + n_mcache == l_mem and l_mem + n_mem == l_xbase);
    std.debug.assert(l_xbase + 32 == l_tpidr);
}

/// Byte offset of a slot in struct kw_regs (from x = &regs.x).
pub inline fn off(slot: u8) u64 {
    return @as(u64, slot) * 8;
}

/// Whether slot is a V half.
pub fn isV(slot: u8) bool {
    return slot >= ops.v0_slot and slot < ops.v0_slot + 64;
}

/// With SIMD: the v128 local of V register n.
pub fn vloc(n: u8) u32 {
    return l_vbase + n;
}

/// The i64 local of a register slot (X, SP, TPIDR_EL0, a V half without
/// SIMD; NZCV: l_nzcv; not XZR).
pub fn local(slot: u8) u32 {
    if (slot < ops.zr) return l_xbase + slot;
    if (slot == ops.tpidr_slot) return l_tpidr;
    if (slot == ops.nzcv_slot) return l_nzcv;
    std.debug.assert(slot >= ops.v0_slot and slot < ops.v0_slot + 64);
    return l_vbase + (slot - ops.v0_slot);
}

/// Push the 64-bit value of slot `slot` (XZR: 0).
pub fn get(e: *Emit, slot: u8) void {
    if (e.simd and isV(slot)) {
        e.get(vloc((slot - ops.v0_slot) / 2));
        e.vlane(S.i64x2_extract_lane, (slot - ops.v0_slot) & 1);
    } else if (slot == ops.zr) e.k64(0) else e.get(local(slot));
}

/// Pop the i64 on the stack into slot `slot` (XZR: dropped).
pub fn set(e: *Emit, slot: u8) void {
    if (e.simd and isV(slot)) {
        const l = vloc((slot - ops.v0_slot) / 2);
        e.set(l_vtmp);
        e.get(l);
        e.get(l_vtmp);
        e.vlane(S.i64x2_replace_lane, (slot - ops.v0_slot) & 1);
        e.set(l);
    } else if (slot == ops.zr) e.op(W.drop) else e.set(local(slot));
}

/// Pop the i64 on the stack into the low half of the V register whose low
/// half is slot `lo`, and zero its high half (a scalar FP or SIMD result).
pub fn setScalar(e: *Emit, lo: u8) void {
    if (e.simd) {
        e.vop(S.i64x2_splat);
        e.k64(0);
        e.vlane(S.i64x2_replace_lane, 1);
        e.set(vloc((lo - ops.v0_slot) / 2));
    } else {
        e.set(local(lo));
        e.k64(0);
        e.set(local(lo + 1));
    }
}

/// With SIMD: push V register n (v128), pop into it, pop the low 64 bits
/// into it with the high half zeroed (a Q = 0 result).
pub fn getV(e: *Emit, n: u8) void {
    e.get(vloc(n));
}
pub fn setV(e: *Emit, n: u8) void {
    e.set(vloc(n));
}
pub fn setV64(e: *Emit, n: u8) void {
    e.k64(0);
    e.vlane(S.i64x2_replace_lane, 1);
    e.set(vloc(n));
}
/// Pop a v128 into Vn, the high half zeroed unless q.
pub fn setVq(e: *Emit, n: u8, q: bool) void {
    if (q) setV(e, n) else setV64(e, n);
}

/// Load / store V register n from / to struct kw_regs (both halves; one
/// v128 access with SIMD).
pub fn loadV(e: *Emit, n: u8) void {
    const lo: u8 = ops.v0_slot + 2 * n;
    if (e.simd) {
        e.get(jc.l_x);
        e.addr();
        e.vop(S.v128_load);
        e.uleb(3);
        e.uleb(off(lo));
        e.set(vloc(n));
    } else for (0..2) |h| {
        e.get(jc.l_x);
        e.addr();
        e.mem(W.i64_load, 3, off(lo + @as(u8, @intCast(h))));
        e.set(local(lo + @as(u8, @intCast(h))));
    }
}
pub fn storeV(e: *Emit, n: u8) void {
    const lo: u8 = ops.v0_slot + 2 * n;
    if (e.simd) {
        e.get(jc.l_x);
        e.addr();
        e.get(vloc(n));
        e.vop(S.v128_store);
        e.uleb(3);
        e.uleb(off(lo));
    } else for (0..2) |h| {
        e.get(jc.l_x);
        e.addr();
        e.get(local(lo + @as(u8, @intCast(h))));
        e.mem(W.i64_store, 3, off(lo + @as(u8, @intCast(h))));
    }
}

/// Wasm SIMD opcodes (after the 0xfd prefix; `Emit.vop`, `Emit.vlane`).
pub const S = struct {
    pub const v128_load = 0x00;
    pub const v128_load8x8_s = 0x01;
    pub const v128_load8x8_u = 0x02;
    pub const v128_load8_splat = 0x07;
    pub const v128_load16_splat = 0x08;
    pub const v128_load32_splat = 0x09;
    pub const v128_load64_splat = 0x0a;
    pub const v128_store = 0x0b;
    pub const v128_const = 0x0c;
    pub const i8x16_shuffle = 0x0d;
    pub const i8x16_swizzle = 0x0e;
    pub const i8x16_splat = 0x0f;
    pub const i16x8_splat = 0x10;
    pub const i32x4_splat = 0x11;
    pub const i64x2_splat = 0x12;
    pub const i8x16_extract_lane_s = 0x15;
    pub const i8x16_extract_lane_u = 0x16;
    pub const i8x16_replace_lane = 0x17;
    pub const i16x8_extract_lane_s = 0x18;
    pub const i16x8_extract_lane_u = 0x19;
    pub const i16x8_replace_lane = 0x1a;
    pub const i32x4_extract_lane = 0x1b;
    pub const i32x4_replace_lane = 0x1c;
    pub const i64x2_extract_lane = 0x1d;
    pub const i64x2_replace_lane = 0x1e;
    pub const i8x16_eq = 0x23;
    pub const i8x16_ne = 0x24;
    pub const i8x16_lt_s = 0x25;
    pub const i8x16_lt_u = 0x26;
    pub const i8x16_gt_s = 0x27;
    pub const i8x16_gt_u = 0x28;
    pub const i8x16_le_s = 0x29;
    pub const i8x16_le_u = 0x2a;
    pub const i8x16_ge_s = 0x2b;
    pub const i8x16_ge_u = 0x2c;
    pub const i16x8_eq = 0x2d;
    pub const i32x4_eq = 0x37;
    pub const i32x4_lt_u = 0x3a;
    pub const f32x4_eq = 0x41;
    pub const f32x4_ne = 0x42;
    pub const f32x4_lt = 0x43;
    pub const f32x4_gt = 0x44;
    pub const f32x4_le = 0x45;
    pub const f32x4_ge = 0x46;
    pub const f64x2_eq = 0x47;
    pub const f64x2_ne = 0x48;
    pub const f64x2_lt = 0x49;
    pub const f64x2_gt = 0x4a;
    pub const f64x2_le = 0x4b;
    pub const f64x2_ge = 0x4c;
    pub const v128_not = 0x4d;
    pub const v128_and = 0x4e;
    pub const v128_andnot = 0x4f;
    pub const v128_or = 0x50;
    pub const v128_xor = 0x51;
    pub const v128_bitselect = 0x52;
    pub const v128_any_true = 0x53;
    pub const v128_load8_lane = 0x54;
    pub const v128_load16_lane = 0x55;
    pub const v128_load32_lane = 0x56;
    pub const v128_load64_lane = 0x57;
    pub const v128_store8_lane = 0x58;
    pub const v128_store16_lane = 0x59;
    pub const v128_store32_lane = 0x5a;
    pub const v128_store64_lane = 0x5b;
    pub const v128_load32_zero = 0x5c;
    pub const v128_load64_zero = 0x5d;
    pub const f32x4_demote_f64x2_zero = 0x5e;
    pub const f64x2_promote_low_f32x4 = 0x5f;
    pub const i8x16_popcnt = 0x62;
    pub const i8x16_shl = 0x6b;
    pub const i8x16_shr_s = 0x6c;
    pub const i8x16_shr_u = 0x6d;
    pub const i8x16_add = 0x6e;
    pub const i8x16_sub = 0x71;
    pub const i16x8_shl = 0x8b;
    pub const i32x4_all_true = 0xa3;
    pub const i32x4_shl = 0xab;
    pub const i32x4_shr_s = 0xac;
    pub const i32x4_shr_u = 0xad;
    pub const i32x4_add = 0xae;
    pub const i32x4_sub = 0xb1;
    pub const i64x2_all_true = 0xc3;
    pub const i64x2_shl = 0xcb;
    pub const i64x2_shr_u = 0xcd;
    pub const i64x2_add = 0xce;
    pub const i64x2_eq = 0xd6;
    pub const i64x2_ne = 0xd7;
    pub const i64x2_lt_s = 0xd8;
    pub const i64x2_gt_s = 0xd9;
    pub const f32x4_abs = 0xe0;
    pub const f32x4_neg = 0xe1;
    pub const f32x4_sqrt = 0xe3;
    pub const f32x4_add = 0xe4;
    pub const f32x4_sub = 0xe5;
    pub const f32x4_mul = 0xe6;
    pub const f32x4_div = 0xe7;
    pub const f32x4_min = 0xe8;
    pub const f32x4_max = 0xe9;
    pub const f64x2_abs = 0xec;
    pub const f64x2_neg = 0xed;
    pub const f64x2_sqrt = 0xef;
    pub const f64x2_add = 0xf0;
    pub const f64x2_sub = 0xf1;
    pub const f64x2_mul = 0xf2;
    pub const f64x2_div = 0xf3;
    pub const f64x2_min = 0xf4;
    pub const f64x2_max = 0xf5;
    pub const i32x4_trunc_sat_f32x4_s = 0xf8;
    pub const i32x4_trunc_sat_f32x4_u = 0xf9;
    pub const f32x4_convert_i32x4_s = 0xfa;
    pub const f32x4_convert_i32x4_u = 0xfb;
    pub const f32x4_relaxed_madd = 0x105;
    pub const f32x4_relaxed_nmadd = 0x106;
    pub const f64x2_relaxed_madd = 0x107;
    pub const f64x2_relaxed_nmadd = 0x108;
};

/// Use bit of an X/SP slot (0..31); none for XZR (slot 32) and other slots.
pub fn xbit(slot: u8) u32 {
    return if (slot < 32) @as(u32, 1) << @intCast(slot) else 0;
}
/// Use bit of a V slot (36 + 2n, either half: the whole register).
pub fn vbit(slot: u8) u32 {
    return if (slot >= ops.v0_slot and slot < ops.v0_slot + 64) @as(u32, 1) << @intCast((slot - ops.v0_slot) / 2) else 0;
}
/// What reading slot `slot` uses (any slot; XZR: nothing).
pub fn rd(slot: u8) Use {
    return .{
        .read = xbit(slot),
        .vread = vbit(slot),
        .tpidr_read = slot == ops.tpidr_slot,
        .flags_read = slot == ops.nzcv_slot,
    };
}
/// What writing slot `slot` uses (any slot; XZR: nothing).
pub fn wr(slot: u8) Use {
    return .{
        .write = xbit(slot),
        .vwrite = vbit(slot),
        .tpidr_write = slot == ops.tpidr_slot,
        .flags_write = slot == ops.nzcv_slot,
    };
}
/// Reads b and c, writes a (the common shape).
pub fn abc(o: ops.Op) Use {
    return rd(o.b).with(rd(o.c)).with(wr(o.a));
}
