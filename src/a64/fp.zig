// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! Scalar floating point (op0<31:28> = x0x1 in the SIMD&FP space):
//! conversions between FP and fixed point and between FP and integer,
//! FP data-processing with 1, 2 and 3 sources, compare, conditional
//! compare, conditional select, and FP immediate (FMOV).
//!
//! Everything of ARMv8.0 is implemented for single and double precision,
//! plus FCVT to and from half precision. The arithmetic and all Arm rules
//! (NaNs, FPCR.FZ/DN/AHP/RMode, flags) are fparith.zig. ILLEGAL: half
//! precision arithmetic and moves (ftype 11: FEAT_FP16), FRINT32/64[ZX]
//! (FEAT_FRINTTS), FJCVTZS (FEAT_JSCVT), BFCVT (FEAT_BF16) and every
//! unallocated encoding. A scalar result zeroes the rest of the V
//! register; FPSR is updated once, at the end.

const std = @import("std");
const c = @import("common.zig");
const fa = @import("fparith.zig");

pub fn exec(s: c.St, inst: u32) c.Error!c.Exit {
    if (c.bit(inst, 29)) return c.ill(s, inst); // S
    // Bit 31 is sf in the conversions, M (must be 0) everywhere else.
    if (!c.bit(inst, 24) and !c.bit(inst, 21)) return fixedConv(s, inst);
    if (!c.bit(inst, 24) and c.bits(inst, 15, 10) == 0) return intConv(s, inst);
    if (c.bit(inst, 31)) return c.ill(s, inst);
    if (c.bit(inst, 24)) return dp3(s, inst);
    if (c.bits(inst, 14, 10) == 0b10000) return dp1(s, inst);
    if (c.bits(inst, 13, 10) == 0b1000) return compare(s, inst);
    if (c.bits(inst, 12, 10) == 0b100) return imm(s, inst);
    return switch (c.bits(inst, 11, 10)) {
        0b01 => condCompare(s, inst),
        0b10 => dp2(s, inst),
        0b11 => condSelect(s, inst),
        else => c.ill(s, inst),
    };
}

// ---- operands ------------------------------------------------------------------

/// The scalar FP register n in format B (the low bits of Vn).
inline fn get(comptime B: type, s: c.St, n: u32) B {
    return @truncate(c.vq(s, n));
}
/// Write a scalar FP result; the rest of Vd becomes zero.
inline fn put(comptime B: type, s: c.St, d: u32, v: B) void {
    c.setVq(s, d, v);
}
inline fn fpcr(s: c.St) u32 {
    return s.r.fpcr;
}
inline fn raise(s: c.St, fl: u32) void {
    if (!c.wasmFpMode(s.r.fpcr)) s.r.fpsr |= fl;
}
/// FPCR.RMode, for instructions that round as the FPCR says.
inline fn rmode(s: c.St) fa.Rounding {
    return fa.fpcrRounding(s.r.fpcr);
}

// ---- conversions ---------------------------------------------------------------

/// Conversion between FP and integer: SCVTF, UCVTF, FCVT[NPMZA][SU], FMOV
/// (general), FJCVTZS (`sf 0 S 11110 ftype 1 rmode opcode 000000 Rn Rd`).
fn intConv(s: c.St, inst: u32) c.Error!c.Exit {
    const sf = c.bit(inst, 31);
    const ftype = c.bits(inst, 23, 22);
    const rm = c.bits(inst, 20, 19);
    const opcode = c.bits(inst, 18, 16);
    const d = c.rd(inst);
    const n = c.rn(inst);
    if (opcode == 6 or opcode == 7) {
        // FMOV (general): to general register (6) or from it (7)
        const to_gpr = opcode == 6;
        switch ((@as(u32, @intFromBool(sf)) << 4) | (ftype << 2) | rm) {
            0b0_00_00 => if (to_gpr) c.setX(s, d, c.vs(s, n)) else c.setVs(s, d, c.wr(s, n)),
            0b1_01_00 => if (to_gpr) c.setX(s, d, c.vd(s, n)) else c.setVd(s, d, c.xr(s, n)),
            0b1_10_01 => if (to_gpr) c.setX(s, d, c.lane(u64, c.vq(s, n), 1)) else c.setVq(s, d, c.setLane(u64, c.vq(s, d), 1, c.xr(s, n))),
            else => return c.ill(s, inst), // half precision (FEAT_FP16), FJCVTZS, unallocated
        }
        return .next;
    }
    if (ftype > 1) return c.ill(s, inst); // 10 unallocated, 11 FEAT_FP16
    switch (opcode) {
        0, 1 => { // FCVT[NPMZ][SU]: rmode is the rounding
            const rnd: fa.Rounding = switch (rm) {
                0 => .tieeven,
                1 => .posinf,
                2 => .neginf,
                else => .zero,
            };
            toInt(s, ftype, sf, d, n, 0, opcode == 1, rnd);
        },
        2, 3 => { // SCVTF, UCVTF
            if (rm != 0) return c.ill(s, inst);
            fromInt(s, ftype, sf, d, n, 0, opcode == 3);
        },
        4, 5 => { // FCVTA[SU]
            if (rm != 0) return c.ill(s, inst);
            toInt(s, ftype, sf, d, n, 0, opcode == 5, .tieaway);
        },
        else => unreachable,
    }
    return .next;
}

/// Conversion between FP and fixed point: SCVTF, UCVTF, FCVTZS, FCVTZU with
/// #fbits (`sf 0 S 11110 ftype 0 rmode opcode scale Rn Rd`, fbits = 64 - scale).
fn fixedConv(s: c.St, inst: u32) c.Error!c.Exit {
    const sf = c.bit(inst, 31);
    const ftype = c.bits(inst, 23, 22);
    const scale = c.bits(inst, 15, 10);
    if (!sf and scale < 32) return c.ill(s, inst); // more fraction bits than a W register has
    if (ftype > 1) return c.ill(s, inst); // 10 unallocated, 11 FEAT_FP16
    const fbits = 64 - scale;
    const d = c.rd(inst);
    const n = c.rn(inst);
    const unsigned = c.bit(inst, 16);
    switch (c.bits(inst, 20, 16)) {
        0b00_010, 0b00_011 => fromInt(s, ftype, sf, d, n, fbits, unsigned), // SCVTF, UCVTF
        0b11_000, 0b11_001 => toInt(s, ftype, sf, d, n, fbits, unsigned, .zero), // FCVTZS, FCVTZU
        else => return c.ill(s, inst),
    }
    return .next;
}

/// FCVT*[SU], FCVTZ[SU] (fixed point): Sn or Dn to Wd or Xd.
fn toInt(s: c.St, ftype: u32, sf: bool, d: u32, n: u32, fbits: u32, unsigned: bool, rnd: fa.Rounding) void {
    var fl: u32 = 0;
    const size: u7 = if (sf) 64 else 32;
    const r = if (ftype == 0)
        fa.toFixed(u32, get(u32, s, n), fbits, unsigned, size, rnd, fpcr(s), &fl)
    else
        fa.toFixed(u64, get(u64, s, n), fbits, unsigned, size, rnd, fpcr(s), &fl);
    c.setX(s, d, r); // zero-extended from 32 bits for Wd
    raise(s, fl);
}

/// SCVTF, UCVTF (integer and fixed point): Wn or Xn (/ 2^fbits) to Sd or
/// Dd, rounded as FPCR says.
fn fromInt(s: c.St, ftype: u32, sf: bool, d: u32, n: u32, fbits: u32, unsigned: bool) void {
    var fl: u32 = 0;
    const size: u7 = if (sf) 64 else 32;
    const v = c.xr(s, n);
    if (ftype == 0)
        put(u32, s, d, fa.fromFixed(u32, v, size, fbits, unsigned, rmode(s), fpcr(s), &fl))
    else
        put(u64, s, d, fa.fromFixed(u64, v, size, fbits, unsigned, rmode(s), fpcr(s), &fl));
    raise(s, fl);
}

// ---- 1 source ------------------------------------------------------------------

/// FP data-processing (1 source): FMOV (register), FABS, FNEG, FSQRT, FCVT,
/// FRINT* (`M 0 S 11110 ftype 1 opcode 10000 Rn Rd`).
fn dp1(s: c.St, inst: u32) c.Error!c.Exit {
    const ftype = c.bits(inst, 23, 22);
    const opcode = c.bits(inst, 20, 15);
    const d = c.rd(inst);
    const n = c.rn(inst);
    if (opcode & 0b111100 == 0b000100) { // FCVT: opcode<1:0> is the destination type
        const opc = opcode & 3;
        if (opc == ftype or opc == 2 or ftype == 2) return c.ill(s, inst); // BFCVT is opc 10
        var fl: u32 = 0;
        const fc = fpcr(s);
        const rnd = rmode(s);
        switch (ftype << 2 | opc) {
            0b00_01 => put(u64, s, d, fa.convert(u64, u32, get(u32, s, n), rnd, fc, &fl)),
            0b00_11 => put(u16, s, d, fa.convert(u16, u32, get(u32, s, n), rnd, fc, &fl)),
            0b01_00 => put(u32, s, d, fa.convert(u32, u64, get(u64, s, n), rnd, fc, &fl)),
            0b01_11 => put(u16, s, d, fa.convert(u16, u64, get(u64, s, n), rnd, fc, &fl)),
            0b11_00 => put(u32, s, d, fa.convert(u32, u16, get(u16, s, n), rnd, fc, &fl)),
            0b11_01 => put(u64, s, d, fa.convert(u64, u16, get(u16, s, n), rnd, fc, &fl)),
            else => unreachable,
        }
        raise(s, fl);
        return .next;
    }
    return switch (ftype) {
        0 => dp1T(u32, s, inst, opcode, d, n),
        1 => dp1T(u64, s, inst, opcode, d, n),
        else => c.ill(s, inst), // 10 unallocated, 11 FEAT_FP16
    };
}

fn dp1T(comptime B: type, s: c.St, inst: u32, opcode: u32, d: u32, n: u32) c.Exit {
    const a = get(B, s, n);
    const fc = fpcr(s);
    var fl: u32 = 0;
    const r: B = switch (opcode) {
        0b000000 => a, // FMOV
        0b000001 => fa.abs(B, a),
        0b000010 => fa.neg(B, a),
        0b000011 => fa.sqrt(B, a, fc, &fl),
        0b001000 => fa.roundInt(B, a, .tieeven, false, fc, &fl), // FRINTN
        0b001001 => fa.roundInt(B, a, .posinf, false, fc, &fl), // FRINTP
        0b001010 => fa.roundInt(B, a, .neginf, false, fc, &fl), // FRINTM
        0b001011 => fa.roundInt(B, a, .zero, false, fc, &fl), // FRINTZ
        0b001100 => fa.roundInt(B, a, .tieaway, false, fc, &fl), // FRINTA
        0b001110 => fa.roundInt(B, a, rmode(s), true, fc, &fl), // FRINTX
        0b001111 => fa.roundInt(B, a, rmode(s), false, fc, &fl), // FRINTI
        else => return c.ill(s, inst), // FRINT32/64 (FEAT_FRINTTS), unallocated
    };
    put(B, s, d, r);
    raise(s, fl);
    return .next;
}

// ---- compare -------------------------------------------------------------------

/// FP compare: FCMP, FCMPE, with a register or #0.0
/// (`M 0 S 11110 ftype 1 Rm op 1000 Rn opcode2`).
fn compare(s: c.St, inst: u32) c.Error!c.Exit {
    if (c.bits(inst, 15, 14) != 0 or c.bits(inst, 2, 0) != 0) return c.ill(s, inst);
    const zero = c.bit(inst, 3); // the #0.0 forms ignore Rm (should be zero)
    const signal = c.bit(inst, 4);
    const n = c.rn(inst);
    const m = c.rm(inst);
    var fl: u32 = 0;
    s.r.nzcv = switch (c.bits(inst, 23, 22)) {
        0 => fa.compare(u32, get(u32, s, n), if (zero) 0 else get(u32, s, m), signal, fpcr(s), &fl),
        1 => fa.compare(u64, get(u64, s, n), if (zero) 0 else get(u64, s, m), signal, fpcr(s), &fl),
        else => return c.ill(s, inst),
    };
    raise(s, fl);
    return .next;
}

/// FP conditional compare: FCCMP, FCCMPE
/// (`M 0 S 11110 ftype 1 Rm cond 01 Rn op nzcv`).
fn condCompare(s: c.St, inst: u32) c.Error!c.Exit {
    const ftype = c.bits(inst, 23, 22);
    if (ftype > 1) return c.ill(s, inst);
    if (!c.condHolds(s.r.nzcv, @intCast(c.bits(inst, 15, 12)))) {
        s.r.nzcv = c.bits(inst, 3, 0) << 28;
        return .next;
    }
    const signal = c.bit(inst, 4);
    const n = c.rn(inst);
    const m = c.rm(inst);
    var fl: u32 = 0;
    s.r.nzcv = if (ftype == 0)
        fa.compare(u32, get(u32, s, n), get(u32, s, m), signal, fpcr(s), &fl)
    else
        fa.compare(u64, get(u64, s, n), get(u64, s, m), signal, fpcr(s), &fl);
    raise(s, fl);
    return .next;
}

// ---- select, immediate ---------------------------------------------------------

/// FP conditional select: FCSEL (`M 0 S 11110 ftype 1 Rm cond 11 Rn Rd`).
fn condSelect(s: c.St, inst: u32) c.Error!c.Exit {
    const ftype = c.bits(inst, 23, 22);
    if (ftype > 1) return c.ill(s, inst);
    const src = if (c.condHolds(s.r.nzcv, @intCast(c.bits(inst, 15, 12)))) c.rn(inst) else c.rm(inst);
    if (ftype == 0) put(u32, s, c.rd(inst), get(u32, s, src)) else put(u64, s, c.rd(inst), get(u64, s, src));
    return .next;
}

/// FP immediate: FMOV (scalar, immediate) (`M 0 S 11110 ftype 1 imm8 100 imm5 Rd`).
fn imm(s: c.St, inst: u32) c.Error!c.Exit {
    if (c.bits(inst, 9, 5) != 0) return c.ill(s, inst);
    const imm8: u8 = @intCast(c.bits(inst, 20, 13));
    switch (c.bits(inst, 23, 22)) {
        0 => put(u32, s, c.rd(inst), fa.expandImm(u32, imm8)),
        1 => put(u64, s, c.rd(inst), fa.expandImm(u64, imm8)),
        else => return c.ill(s, inst),
    }
    return .next;
}

// ---- 2 and 3 sources -----------------------------------------------------------

/// FP data-processing (2 source): FMUL, FDIV, FADD, FSUB, FMAX, FMIN,
/// FMAXNM, FMINNM, FNMUL (`M 0 S 11110 ftype 1 Rm opcode 10 Rn Rd`).
fn dp2(s: c.St, inst: u32) c.Error!c.Exit {
    const opcode = c.bits(inst, 15, 12);
    if (opcode > 8) return c.ill(s, inst);
    return switch (c.bits(inst, 23, 22)) {
        0 => dp2T(u32, s, inst, opcode),
        1 => dp2T(u64, s, inst, opcode),
        else => c.ill(s, inst),
    };
}

fn dp2T(comptime B: type, s: c.St, inst: u32, opcode: u32) c.Exit {
    const a = get(B, s, c.rn(inst));
    const b = get(B, s, c.rm(inst));
    const fc = fpcr(s);
    var fl: u32 = 0;
    const r: B = switch (opcode) {
        0 => fa.mul(B, a, b, fc, &fl),
        1 => fa.div(B, a, b, fc, &fl),
        2 => fa.add(B, a, b, fc, &fl),
        3 => fa.sub(B, a, b, fc, &fl),
        4 => fa.max(B, a, b, fc, &fl),
        5 => fa.min(B, a, b, fc, &fl),
        6 => fa.maxNum(B, a, b, fc, &fl),
        7 => fa.minNum(B, a, b, fc, &fl),
        8 => fa.neg(B, fa.mul(B, a, b, fc, &fl)), // FNMUL: a NaN result is negated too
        else => unreachable,
    };
    put(B, s, c.rd(inst), r);
    raise(s, fl);
    return .next;
}

/// FP data-processing (3 source): FMADD, FMSUB, FNMADD, FNMSUB
/// (`M 0 S 11111 ftype o1 Rm o0 Ra Rn Rd`).
fn dp3(s: c.St, inst: u32) c.Error!c.Exit {
    return switch (c.bits(inst, 23, 22)) {
        0 => dp3T(u32, s, inst),
        1 => dp3T(u64, s, inst),
        else => c.ill(s, inst),
    };
}

fn dp3T(comptime B: type, s: c.St, inst: u32) c.Exit {
    const o1 = c.bit(inst, 21);
    const o0 = c.bit(inst, 15);
    var a = get(B, s, c.ra(inst));
    var x = get(B, s, c.rn(inst));
    const y = get(B, s, c.rm(inst));
    if (o1) a = fa.neg(B, a); // FNMADD, FNMSUB
    if (o0 != o1) x = fa.neg(B, x); // FMSUB, FNMADD
    var fl: u32 = 0;
    put(B, s, c.rd(inst), fa.mulAdd(B, a, x, y, fpcr(s), &fl));
    raise(s, fl);
    return .next;
}

// ---- tests ---------------------------------------------------------------------

test "a64 fp: encodings outside ARMv8.0 FP stay ILLEGAL, results zero the V register" {
    const t = std.testing;
    const mach = @import("machine.zig");
    var m = try mach.Machine.init(t.allocator);
    defer m.deinit();
    for ([_]u32{
        0x1ee22800, // fadd h0, h0, h2 (FEAT_FP16)
        0x1ee04000, // fmov h0, h0 (FEAT_FP16)
        0x1ee20000, // scvtf h0, w0 (FEAT_FP16)
        0x1e7e0000, // fjcvtzs w0, d0 (FEAT_JSCVT)
        0x1e284000, // frint32z s0, s0 (FEAT_FRINTTS)
        0x1e634000, // bfcvt h0, s0 (FEAT_BF16)
        0x1e62c000, // fcvt d0, d0 (same type)
        0x1e234000, // fcvt, opc 10
        0x1e26c000, // frint, opcode 001101
        0x1e209020, // fmov s0, #imm with imm5 != 0
        0x1e209800, // dp2 opcode 1001
        0x1e2a0000, // scvtf with rmode 01
        0x1e020000, // scvtf s0, w0, #64 (scale < 32 for W)
        0x1e202001, // fcmp, opcode2<2:0> != 0
        0x1e206000, // fcmp, op != 00
        0x9e6e0000, // fmov x0, d0 with rmode 01
        0x1ea00000, // ftype 10
        0x3e220000, // S = 1
        0x9f000000, // fmadd with M = 1
    }) |w| {
        m.code(0x10000, &.{w});
        try t.expectEqual(@as(u32, mach.EV_ILLEGAL), m.run(10));
    }
    m.regs.v[1] = std.math.maxInt(u128);
    m.regs.v[2] = 0x4000_0000; // 2.0f
    m.code(0x11000, &.{0x1e222820}); // fadd s0, s1, s2
    _ = m.run(1);
    try t.expectEqual(@as(u128, 0xffff_ffff), m.regs.v[0]); // the quiet NaN, the rest zero
    try t.expectEqual(@as(u32, 0), m.regs.fpsr);
    m.regs.v[1] = 0x7f80_0001; // signalling
    m.regs.fpcr = c.FPCR_DN;
    m.code(0x12000, &.{0x1e222820});
    _ = m.run(1);
    try t.expectEqual(@as(u128, 0x7fc0_0000), m.regs.v[0]);
    try t.expectEqual(c.FPSR_IOC, m.regs.fpsr);
}
