// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! AdvSIMD floating point, vector and scalar, single and double precision
//! (ARMv8.0). simd_arith.zig, simd_shift.zig and simd_long.zig decode the
//! integer part of each AdvSIMD class and hand the FP part of the class,
//! undecoded beyond the listed fields, to one function here, which decodes
//! the rest strictly from the Arm ARM tables. ILLEGAL: the FP16 forms
//! (size 00 in by element, U = 0 in across lanes and scalar pairwise, immh
//! 001x in shift by immediate; the FP16 classes never reach this file),
//! FRINT32/64 (FEAT_FRINTTS), FMLAL/FMLSL (FEAT_FHM), FCMLA/FCADD
//! (FEAT_FCMA), BFCVTN (FEAT_BF16), and reserved size/Q/immh combinations.
//!
//! Every lane goes through fparith.zig (Arm NaN rules, FPCR.FZ/DN/RMode,
//! AHP for the half conversions of FCVTN/FCVTL): one `fl` collects the
//! FPSR bits of all lanes and is ORed into FPSR once, at the end. Vector
//! results with Q = 0 zero the upper half, scalar results the rest of the
//! register.

const std = @import("std");
const c = @import("common.zig");
const fa = @import("fparith.zig");
const si = @import("simd_int.zig");
const arith = @import("simd_arith.zig");

const Rounding = fa.Rounding;

// ---- lane operations -----------------------------------------------------------

/// Two-operand ops (three same, by element, pairwise, reductions); `d` is
/// the old destination element (FMLA, FMLS).
const Bin = enum {
    add, sub, mul, div, mulx, abd, max, min, maxnm, minnm,
    cmeq, cmge, cmgt, acge, acgt, recps, rsqrts, mla, mls,
};

fn bin(comptime B: type, op: Bin, a: B, b: B, d: B, fc: u32, fl: *u32) B {
    const all: B = std.math.maxInt(B);
    return switch (op) {
        .add => fa.add(B, a, b, fc, fl),
        .sub => fa.sub(B, a, b, fc, fl),
        .mul => fa.mul(B, a, b, fc, fl),
        .div => fa.div(B, a, b, fc, fl),
        .mulx => fa.mulx(B, a, b, fc, fl),
        .abd => fa.abs(B, fa.sub(B, a, b, fc, fl)),
        .max => fa.max(B, a, b, fc, fl),
        .min => fa.min(B, a, b, fc, fl),
        .maxnm => fa.maxNum(B, a, b, fc, fl),
        .minnm => fa.minNum(B, a, b, fc, fl),
        .cmeq => if (fa.cmpEq(B, a, b, fc, fl)) all else 0,
        .cmge => if (fa.cmpGe(B, a, b, fc, fl)) all else 0,
        .cmgt => if (fa.cmpGt(B, a, b, fc, fl)) all else 0,
        .acge => if (fa.cmpGe(B, fa.abs(B, a), fa.abs(B, b), fc, fl)) all else 0,
        .acgt => if (fa.cmpGt(B, fa.abs(B, a), fa.abs(B, b), fc, fl)) all else 0,
        .recps => fa.recipStep(B, a, b, fc, fl),
        .rsqrts => fa.rsqrtStep(B, a, b, fc, fl),
        .mla => fa.mulAdd(B, d, a, b, fc, fl),
        .mls => fa.mulAdd(B, d, fa.neg(B, a), b, fc, fl),
    };
}

/// One-operand ops (two-register misc). Conversions to integer produce an
/// integer of the element's width; `fbits` is the fixed-point scale (shift
/// by immediate) or 0.
const Un = union(enum) {
    abs,
    neg,
    sqrt,
    rint: struct { rnd: Rounding, exact: bool },
    to_int: struct { rnd: Rounding, unsigned: bool, fbits: u32 = 0 },
    from_int: struct { unsigned: bool, fbits: u32 = 0 },
    recpe,
    rsqrte,
    recpx,
    cmp0: Cmp0,
};
const Cmp0 = enum { gt, ge, eq, le, lt };

fn un(comptime B: type, op: Un, a: B, fc: u32, fl: *u32) B {
    const all: B = std.math.maxInt(B);
    const n: u7 = @bitSizeOf(B);
    return switch (op) {
        .abs => fa.abs(B, a),
        .neg => fa.neg(B, a),
        .sqrt => fa.sqrt(B, a, fc, fl),
        .rint => |r| fa.roundInt(B, a, r.rnd, r.exact, fc, fl),
        .to_int => |r| @truncate(fa.toFixed(B, a, r.fbits, r.unsigned, n, r.rnd, fc, fl)),
        .from_int => |r| fa.fromFixed(B, a, n, r.fbits, r.unsigned, fa.fpcrRounding(fc), fc, fl),
        .recpe => fa.recipEstimate(B, a, fc, fl),
        .rsqrte => fa.rsqrtEstimate(B, a, fc, fl),
        .recpx => fa.recpX(B, a, fc, fl),
        .cmp0 => |k| if (switch (k) {
            .gt => fa.cmpGt(B, a, 0, fc, fl),
            .ge => fa.cmpGe(B, a, 0, fc, fl),
            .eq => fa.cmpEq(B, a, 0, fc, fl),
            .le => fa.cmpGe(B, 0, a, fc, fl),
            .lt => fa.cmpGt(B, 0, a, fc, fl),
        }) all else 0,
    };
}

/// `bin` over the low `nl` lanes of n and m (d the accumulator); the other
/// lanes are zero. A by-element op passes `elem`, the Vm element, instead
/// of m's lanes.
fn mapBin(comptime B: type, op: Bin, n: u128, m: u128, d: u128, elem: ?B, nl: usize, fc: u32, fl: *u32) u128 {
    const a = c.lanes(B, n);
    const b = c.lanes(B, m);
    const acc = c.lanes(B, d);
    var r: [a.len]B = undefined;
    for (&r, 0..) |*x, i| x.* = if (i < nl) bin(B, op, a[i], elem orelse b[i], acc[i], fc, fl) else 0;
    return c.fromLanes(B, r);
}

/// Pairwise (FADDP, FMAXP, ...): op over adjacent pairs of the `nl` lanes
/// of n followed by the `nl` lanes of m.
fn mapPair(comptime B: type, op: Bin, n: u128, m: u128, nl: usize, fc: u32, fl: *u32) u128 {
    const a = c.lanes(B, n);
    const b = c.lanes(B, m);
    var r: [a.len]B = undefined;
    const h = nl / 2;
    for (&r, 0..) |*x, i| {
        if (i >= nl) {
            x.* = 0;
            continue;
        }
        const src = if (i < h) a[2 * i ..][0..2] else b[2 * (i - h) ..][0..2];
        x.* = bin(B, op, src[0], src[1], 0, fc, fl);
    }
    return c.fromLanes(B, r);
}

fn mapUn(comptime B: type, op: Un, n: u128, nl: usize, fc: u32, fl: *u32) u128 {
    const a = c.lanes(B, n);
    var r: [a.len]B = undefined;
    for (&r, 0..) |*x, i| x.* = if (i < nl) un(B, op, a[i], fc, fl) else 0;
    return c.fromLanes(B, r);
}

/// The FPCR, and the end of every FP instruction: FPSR |= fl.
inline fn fpcr(s: c.St) u32 {
    return s.r.fpcr;
}
inline fn raise(s: c.St, fl: u32) void {
    if (!c.wasmFpMode(s.r.fpcr)) s.r.fpsr |= fl;
}

// ---- three same ----------------------------------------------------------------

/// FP three same by U, size<1> (a) and opcode (inst<15:11>, 11000..11111),
/// vector and scalar (`scalar` false where the op has no scalar form).
const TsFp = struct { op: Bin, pair: bool = false, scalar: bool = false };

fn tsFp(u: bool, a: bool, opcode: u32) ?TsFp {
    const T = TsFp;
    const k = (@as(u32, @intFromBool(u)) << 6) | (@as(u32, @intFromBool(a)) << 5) | opcode;
    return switch (k) {
        0b0_0_11000 => T{ .op = .maxnm },
        0b0_0_11001 => T{ .op = .mla },
        0b0_0_11010 => T{ .op = .add },
        0b0_0_11011 => T{ .op = .mulx, .scalar = true },
        0b0_0_11100 => T{ .op = .cmeq, .scalar = true },
        0b0_0_11110 => T{ .op = .max },
        0b0_0_11111 => T{ .op = .recps, .scalar = true },
        0b0_1_11000 => T{ .op = .minnm },
        0b0_1_11001 => T{ .op = .mls },
        0b0_1_11010 => T{ .op = .sub },
        0b0_1_11110 => T{ .op = .min },
        0b0_1_11111 => T{ .op = .rsqrts, .scalar = true },
        0b1_0_11000 => T{ .op = .maxnm, .pair = true },
        0b1_0_11010 => T{ .op = .add, .pair = true },
        0b1_0_11011 => T{ .op = .mul },
        0b1_0_11100 => T{ .op = .cmge, .scalar = true },
        0b1_0_11101 => T{ .op = .acge, .scalar = true },
        0b1_0_11110 => T{ .op = .max, .pair = true },
        0b1_0_11111 => T{ .op = .div },
        0b1_1_11000 => T{ .op = .minnm, .pair = true },
        0b1_1_11010 => T{ .op = .abd, .scalar = true },
        0b1_1_11100 => T{ .op = .cmgt, .scalar = true },
        0b1_1_11101 => T{ .op = .acgt, .scalar = true },
        0b1_1_11110 => T{ .op = .min, .pair = true },
        // U=0 11101 FMLAL/FMLSL (FEAT_FHM), the rest unallocated
        else => null,
    };
}

fn threeSameAny(s: c.St, inst: u32, sc: bool) c.Exit {
    const q = c.bit(inst, 30);
    const sz = c.bit(inst, 22);
    const info = tsFp(c.bit(inst, 29), c.bit(inst, 23), c.bits(inst, 15, 11)) orelse return c.ill(s, inst);
    if (sc and !info.scalar) return c.ill(s, inst);
    if (!sc and sz and !q) return c.ill(s, inst);
    const n = c.vq(s, c.rn(inst));
    const m = c.vq(s, c.rm(inst));
    const d = c.rd(inst);
    const fc = fpcr(s);
    var fl: u32 = 0;
    const r = switch (sz) {
        inline else => |dbl| blk: {
            const B = if (dbl) u64 else u32;
            const nl: usize = if (sc) 1 else @as(usize, if (q) 16 else 8) / @sizeOf(B);
            break :blk if (info.pair) mapPair(B, info.op, n, m, nl, fc, &fl) else mapBin(B, info.op, n, m, c.vq(s, d), null, nl, fc, &fl);
        },
    };
    c.setVq(s, d, r);
    raise(s, fl);
    return .next;
}

/// Three same, vector (`0 Q U 01110 size 1 Rm opcode 1 Rn Rd`), opcode
/// 11000..11111: FMAXNM, FMLA, FADD, FMULX, FCMEQ, FMAX, FRECPS, FMINNM,
/// FMLS, FSUB, FMIN, FRSQRTS, FMAXNMP, FADDP, FMUL, FCMGE, FACGE, FMAXP,
/// FDIV, FMINNMP, FABD, FCMGT, FACGT, FMINP.
pub fn threeSame(s: c.St, inst: u32) c.Exit {
    return threeSameAny(s, inst, false);
}
/// Three same, scalar (`01 U 11110 size 1 Rm opcode 1 Rn Rd`), opcode
/// 11000..11111: FMULX, FCMEQ, FRECPS, FRSQRTS, FCMGE, FACGE, FABD, FCMGT,
/// FACGT.
pub fn scalarThreeSame(s: c.St, inst: u32) c.Exit {
    return threeSameAny(s, inst, true);
}

// ---- two-register misc ---------------------------------------------------------

const Where = enum { both, vector, scalar };
const Misc = struct { op: Un, where: Where = .both };

/// The same-size FP ops of two-register misc by U, size<1> (a) and opcode
/// (inst<16:12>), with the forms (vector, scalar) that exist. FCVTN,
/// FCVTXN and FCVTL are decoded by the callers.
fn miscFp(u: bool, a: bool, opcode: u32, fc: u32) ?Misc {
    const M = Misc;
    const rm = fa.fpcrRounding(fc);
    const k = (@as(u32, @intFromBool(u)) << 6) | (@as(u32, @intFromBool(a)) << 5) | opcode;
    return switch (k) {
        0b0_0_11000 => M{ .op = .{ .rint = .{ .rnd = .tieeven, .exact = false } }, .where = .vector }, // FRINTN
        0b0_0_11001 => M{ .op = .{ .rint = .{ .rnd = .neginf, .exact = false } }, .where = .vector }, // FRINTM
        0b0_0_11010 => M{ .op = .{ .to_int = .{ .rnd = .tieeven, .unsigned = false } } }, // FCVTNS
        0b0_0_11011 => M{ .op = .{ .to_int = .{ .rnd = .neginf, .unsigned = false } } }, // FCVTMS
        0b0_0_11100 => M{ .op = .{ .to_int = .{ .rnd = .tieaway, .unsigned = false } } }, // FCVTAS
        0b0_0_11101 => M{ .op = .{ .from_int = .{ .unsigned = false } } }, // SCVTF
        0b0_1_01100 => M{ .op = .{ .cmp0 = .gt } },
        0b0_1_01101 => M{ .op = .{ .cmp0 = .eq } },
        0b0_1_01110 => M{ .op = .{ .cmp0 = .lt } },
        0b0_1_01111 => M{ .op = .abs, .where = .vector },
        0b0_1_11000 => M{ .op = .{ .rint = .{ .rnd = .posinf, .exact = false } }, .where = .vector }, // FRINTP
        0b0_1_11001 => M{ .op = .{ .rint = .{ .rnd = .zero, .exact = false } }, .where = .vector }, // FRINTZ
        0b0_1_11010 => M{ .op = .{ .to_int = .{ .rnd = .posinf, .unsigned = false } } }, // FCVTPS
        0b0_1_11011 => M{ .op = .{ .to_int = .{ .rnd = .zero, .unsigned = false } } }, // FCVTZS
        0b0_1_11101 => M{ .op = .recpe },
        0b0_1_11111 => M{ .op = .recpx, .where = .scalar },
        0b1_0_11000 => M{ .op = .{ .rint = .{ .rnd = .tieaway, .exact = false } }, .where = .vector }, // FRINTA
        0b1_0_11001 => M{ .op = .{ .rint = .{ .rnd = rm, .exact = true } }, .where = .vector }, // FRINTX
        0b1_0_11010 => M{ .op = .{ .to_int = .{ .rnd = .tieeven, .unsigned = true } } }, // FCVTNU
        0b1_0_11011 => M{ .op = .{ .to_int = .{ .rnd = .neginf, .unsigned = true } } }, // FCVTMU
        0b1_0_11100 => M{ .op = .{ .to_int = .{ .rnd = .tieaway, .unsigned = true } } }, // FCVTAU
        0b1_0_11101 => M{ .op = .{ .from_int = .{ .unsigned = true } } }, // UCVTF
        0b1_1_01100 => M{ .op = .{ .cmp0 = .ge } },
        0b1_1_01101 => M{ .op = .{ .cmp0 = .le } },
        0b1_1_01111 => M{ .op = .neg, .where = .vector },
        0b1_1_11001 => M{ .op = .{ .rint = .{ .rnd = rm, .exact = false } }, .where = .vector }, // FRINTI
        0b1_1_11010 => M{ .op = .{ .to_int = .{ .rnd = .posinf, .unsigned = true } } }, // FCVTPU
        0b1_1_11011 => M{ .op = .{ .to_int = .{ .rnd = .zero, .unsigned = true } } }, // FCVTZU
        0b1_1_11101 => M{ .op = .rsqrte },
        0b1_1_11111 => M{ .op = .sqrt, .where = .vector },
        // 0x 1111x: FRINT32/64 (FEAT_FRINTTS); 1x 10110: BFCVTN (FEAT_BF16);
        // 1x 11100: URECPE/URSQRTE (simd_arith.zig); the rest unallocated
        else => null,
    };
}

/// FCVTN, FCVTXN (`odd`): the `nl` wide lanes of n narrowed, packed into
/// 64 bits (double to single, or single to half).
fn narrowCvt(comptime To: type, comptime From: type, n: u128, rnd: Rounding, fc: u32, fl: *u32) u64 {
    const a = c.lanes(From, n);
    var r: [8 / @sizeOf(To)]To = undefined;
    for (&r, a) |*x, e| x.* = fa.convert(To, From, e, rnd, fc, fl);
    return @bitCast(r);
}
/// FCVTL: the elements of the 64-bit half `part` of n widened.
fn longCvt(comptime To: type, comptime From: type, n: u128, part: bool, fc: u32, fl: *u32) u128 {
    const a = si.half(From, n, part);
    var r: [a.len]To = undefined;
    for (&r, a) |*x, e| x.* = fa.convert(To, From, e, fa.fpcrRounding(fc), fc, fl);
    return c.fromLanes(To, r);
}

/// Two-register misc, vector (`0 Q U 01110 size 10000 opcode 10 Rn Rd`),
/// opcode 01100..01111 and 10110..11111 except URECPE/URSQRTE (11100 with
/// size 1x): FCMGT/FCMEQ/FCMLT/FCMGE/FCMLE (zero), FABS, FNEG, FCVTN(2),
/// FCVTXN(2), FCVTL(2), FRINTN/M/P/Z/A/X/I, FCVTNS/MS/AS/PS/ZS, FCVTNU/MU/
/// AU/PU/ZU, SCVTF, UCVTF, FRECPE, FRSQRTE, FSQRT.
pub fn twoRegMisc(s: c.St, inst: u32) c.Exit {
    const q = c.bit(inst, 30);
    const u = c.bit(inst, 29);
    const a = c.bit(inst, 23);
    const sz = c.bit(inst, 22);
    const opcode = c.bits(inst, 16, 12);
    const n = c.vq(s, c.rn(inst));
    const d = c.rd(inst);
    const fc = fpcr(s);
    var fl: u32 = 0;
    if (!a and (opcode == 0b10110 or opcode == 0b10111)) {
        if (opcode == 0b10111) { // FCVTL, FCVTL2
            if (u) return c.ill(s, inst);
            c.setVq(s, d, if (sz) longCvt(u64, u32, n, q, fc, &fl) else longCvt(u32, u16, n, q, fc, &fl));
        } else if (u) { // FCVTXN, FCVTXN2
            if (!sz) return c.ill(s, inst);
            arith.setHalf(s, d, q, narrowCvt(u32, u64, n, .odd, fc, &fl));
        } else { // FCVTN, FCVTN2
            const rnd = fa.fpcrRounding(fc);
            arith.setHalf(s, d, q, if (sz) narrowCvt(u32, u64, n, rnd, fc, &fl) else narrowCvt(u16, u32, n, rnd, fc, &fl));
        }
        raise(s, fl);
        return .next;
    }
    const info = miscFp(u, a, opcode, fc) orelse return c.ill(s, inst);
    if (info.where == .scalar or (sz and !q)) return c.ill(s, inst);
    const r = if (sz) mapUn(u64, info.op, n, if (q) 2 else 1, fc, &fl) else mapUn(u32, info.op, n, if (q) 4 else 2, fc, &fl);
    c.setVq(s, d, r);
    raise(s, fl);
    return .next;
}

/// Two-register misc, scalar (`01 U 11110 size 10000 opcode 10 Rn Rd`),
/// opcode 01100..01111 and 10110..11111: FCMGT/FCMEQ/FCMLT/FCMGE/FCMLE
/// (zero), FCVTXN, FCVTNS/MS/AS/PS/ZS, FCVTNU/MU/AU/PU/ZU, SCVTF, UCVTF,
/// FRECPE, FRSQRTE, FRECPX.
pub fn scalarTwoRegMisc(s: c.St, inst: u32) c.Exit {
    const u = c.bit(inst, 29);
    const a = c.bit(inst, 23);
    const sz = c.bit(inst, 22);
    const opcode = c.bits(inst, 16, 12);
    const n = c.vq(s, c.rn(inst));
    const d = c.rd(inst);
    const fc = fpcr(s);
    var fl: u32 = 0;
    if (u and !a and opcode == 0b10110) { // FCVTXN (scalar): Sd from Dn
        if (!sz) return c.ill(s, inst);
        c.setVs(s, d, fa.convert(u32, u64, @truncate(n), .odd, fc, &fl));
        raise(s, fl);
        return .next;
    }
    const info = miscFp(u, a, opcode, fc) orelse return c.ill(s, inst);
    if (info.where == .vector) return c.ill(s, inst);
    c.setVq(s, d, if (sz) mapUn(u64, info.op, n, 1, fc, &fl) else mapUn(u32, info.op, n, 1, fc, &fl));
    raise(s, fl);
    return .next;
}

// ---- across lanes, scalar pairwise ---------------------------------------------

/// Across lanes (`0 Q U 01110 size 11000 opcode 10 Rn Rd`), opcode 01100 and
/// 01111: FMAXNMV, FMINNMV, FMAXV, FMINV. Only 4S exists in ARMv8.0 (U = 0
/// is FEAT_FP16). The pseudocode's Reduce halves the vector recursively:
/// op(op(e0, e1), op(e2, e3)).
pub fn acrossLanes(s: c.St, inst: u32) c.Exit {
    if (!c.bit(inst, 29) or !c.bit(inst, 30) or c.bit(inst, 22)) return c.ill(s, inst);
    const a = c.bit(inst, 23);
    const op: Bin = if (c.bits(inst, 16, 12) == 0b01100) (if (a) .minnm else .maxnm) else (if (a) .min else .max);
    const e = c.lanes(u32, c.vq(s, c.rn(inst)));
    const fc = fpcr(s);
    var fl: u32 = 0;
    const lo = bin(u32, op, e[0], e[1], 0, fc, &fl);
    const hi = bin(u32, op, e[2], e[3], 0, fc, &fl);
    c.setVs(s, c.rd(inst), bin(u32, op, lo, hi, 0, fc, &fl));
    raise(s, fl);
    return .next;
}

/// Scalar pairwise (`01 U 11110 size 11000 opcode 10 Rn Rd`), opcode 01100,
/// 01101, 01111: FMAXNMP, FMINNMP, FADDP, FMAXP, FMINP (scalar): op(Vn[0],
/// Vn[1]) of 2S or 2D. U = 0 is FEAT_FP16.
pub fn scalarPairwise(s: c.St, inst: u32) c.Exit {
    if (!c.bit(inst, 29)) return c.ill(s, inst);
    const a = c.bit(inst, 23);
    const op: Bin = switch (c.bits(inst, 16, 12)) {
        0b01100 => if (a) .minnm else .maxnm,
        0b01101 => if (a) return c.ill(s, inst) else .add,
        0b01111 => if (a) .min else .max,
        else => return c.ill(s, inst),
    };
    const n = c.vq(s, c.rn(inst));
    const fc = fpcr(s);
    var fl: u32 = 0;
    if (c.bit(inst, 22)) {
        const e = c.lanes(u64, n);
        c.setVd(s, c.rd(inst), bin(u64, op, e[0], e[1], 0, fc, &fl));
    } else {
        const e = c.lanes(u32, n);
        c.setVs(s, c.rd(inst), bin(u32, op, e[0], e[1], 0, fc, &fl));
    }
    raise(s, fl);
    return .next;
}

// ---- shift by immediate --------------------------------------------------------

/// Shift by immediate, vector and scalar (`0 Q U 011110 immh immb opcode 1
/// Rn Rd`, `01 U 111110 ...`), opcode 11100 and 11111: SCVTF, UCVTF,
/// FCVTZS, FCVTZU (fixed-point). immh 1xxx is double with fbits = 128 -
/// immh:immb, 01xx single with fbits = 64 - immh:immb; 001x is FEAT_FP16,
/// 0001 reserved.
pub fn shiftImm(s: c.St, inst: u32) c.Exit {
    const sc = c.bit(inst, 28);
    const q = c.bit(inst, 30);
    const u = c.bit(inst, 29);
    const immh = c.bits(inst, 22, 19);
    const immhb = c.bits(inst, 22, 16);
    const dbl = immh & 0b1000 != 0;
    if (!dbl and immh & 0b0100 == 0) return c.ill(s, inst);
    if (dbl and !sc and !q) return c.ill(s, inst);
    const fbits: u32 = (if (dbl) @as(u32, 128) else 64) - immhb;
    const op: Un = if (c.bits(inst, 15, 11) == 0b11100)
        .{ .from_int = .{ .unsigned = u, .fbits = fbits } }
    else
        .{ .to_int = .{ .rnd = .zero, .unsigned = u, .fbits = fbits } };
    const n = c.vq(s, c.rn(inst));
    const fc = fpcr(s);
    var fl: u32 = 0;
    const r = if (dbl)
        mapUn(u64, op, n, if (sc) 1 else 2, fc, &fl)
    else
        mapUn(u32, op, n, if (sc) 1 else if (q) 4 else 2, fc, &fl);
    c.setVq(s, c.rd(inst), r);
    raise(s, fl);
    return .next;
}

// ---- by element ----------------------------------------------------------------

/// By element, vector and scalar (`0 Q U 01111 size L M Rm opcode H 0 Rn
/// Rd`, `01 U 11111 ...`), opcode 0001, 0101, 1001: FMLA, FMLS, FMUL, FMULX
/// (by element). size 1x only (00 is FEAT_FP16); single: index H:L, double:
/// index H with L = 0; Vm is M:Rm. U = 1 with 0001/0101 is FCMLA (FEAT_FCMA).
pub fn indexed(s: c.St, inst: u32) c.Exit {
    const sc = c.bit(inst, 28);
    const q = c.bit(inst, 30);
    const u = c.bit(inst, 29);
    const op: Bin = switch (c.bits(inst, 15, 12)) {
        0b0001 => if (u) return c.ill(s, inst) else .mla,
        0b0101 => if (u) return c.ill(s, inst) else .mls,
        0b1001 => if (u) .mulx else .mul,
        else => return c.ill(s, inst),
    };
    if (!c.bit(inst, 23)) return c.ill(s, inst);
    const dbl = c.bit(inst, 22);
    const h: usize = @intFromBool(c.bit(inst, 11));
    const l: usize = @intFromBool(c.bit(inst, 21));
    if (dbl and (l != 0 or (!sc and !q))) return c.ill(s, inst);
    const n = c.vq(s, c.rn(inst));
    const m = c.vq(s, c.bits(inst, 20, 16));
    const d = c.rd(inst);
    const fc = fpcr(s);
    var fl: u32 = 0;
    // a lane by a run-time index becomes a variable u128 shift (__lshrti3
    // on wasm): select the 64-bit half with a mask, then shift that
    const hm = c.lanes(u64, m);
    const sel: u64 = 0 -% @as(u64, h);
    const half = (hm[0] & ~sel) | (hm[1] & sel);
    const r = if (dbl)
        mapBin(u64, op, n, 0, c.vq(s, d), half, if (sc) 1 else 2, fc, &fl)
    else
        mapBin(u32, op, n, 0, c.vq(s, d), @truncate(half >> @intCast(32 * l)), if (sc) 1 else if (q) 4 else 2, fc, &fl);
    c.setVq(s, d, r);
    raise(s, fl);
    return .next;
}

// ---- tests ---------------------------------------------------------------------

const t = std.testing;

test "a64 simd_fp: estimates and steps, known answers" {
    var fl: u32 = 0;
    try t.expectEqual(@as(u32, 0x3f7f_8000), fa.recipEstimate(u32, 0x3f80_0000, 0, &fl)); // 1.0
    try t.expectEqual(@as(u32, 0x3f7f_8000), fa.rsqrtEstimate(u32, 0x3f80_0000, 0, &fl));
    try t.expectEqual(@as(u32, 0), fl);
    try t.expectEqual(@as(u32, 0x7f80_0000), fa.recipEstimate(u32, 0, 0, &fl));
    try t.expectEqual(fa.DZC, fl);
    fl = 0;
    try t.expectEqual(@as(u32, 0x7fc0_0000), fa.rsqrtEstimate(u32, 0xbf80_0000, 0, &fl));
    try t.expectEqual(fa.IOC, fl);
    fl = 0;
    try t.expectEqual(@as(u32, 0x4000_0000), fa.recipStep(u32, 0x7f80_0000, 0x8000_0000, 0, &fl)); // inf * -0: +2
    try t.expectEqual(@as(u32, 0x3fc0_0000), fa.rsqrtStep(u32, 0, 0xff80_0000, 0, &fl)); // +1.5
    try t.expectEqual(@as(u32, 0x8000_0000), fa.recipStep(u32, 0x3f80_0000, 0x4000_0000, 2 << c.FPCR_RMODE_SHIFT, &fl)); // 2 - 1*2 = -0 (RM)
    try t.expectEqual(@as(u32, 0), fa.recipStep(u32, 0x3f80_0000, 0x4000_0000, 0, &fl));
    try t.expectEqual(@as(u32, 0), fl);
    // (3 - 2^64 * -2^64) / 2 = 2^127 + 1.5: 3 + x*y alone would overflow
    try t.expectEqual(@as(u32, 0x7f00_0000), fa.rsqrtStep(u32, 0x5f80_0000, 0xdf80_0000, 0, &fl));
    try t.expectEqual(fa.IXC, fl);
    fl = 0;
    try t.expectEqual(@as(u64, 0x7fe0_0000_0000_0000), fa.recpX(u64, 1, 0, &fl));
    try t.expectEqual(@as(u32, 0xc000_0000), fa.recpX(u32, 0xbf80_0000, 0, &fl)); // -1.0: exponent 127 -> 128
}

test "a64 simd_fp: estimate tables agree with URECPE/URSQRTE and stay in range" {
    for (256..512) |a| {
        const r = fa.recipEstimateInt(@intCast(a));
        try t.expect(r >= 256 and r < 512);
        try t.expectEqual(si.urecpe(@as(u32, @intCast(a)) << 23), r << 23);
    }
    for (128..512) |a| {
        const r = fa.rsqrtEstimateInt(@intCast(a));
        try t.expect(r >= 256 and r < 512);
        try t.expectEqual(si.ursqrte(@as(u32, @intCast(a)) << 23), r << 23);
    }
}

test "a64 simd_fp: reserved and post-v8.0 encodings raise ILLEGAL" {
    const mach = @import("machine.zig");
    var m = try mach.Machine.init(t.allocator);
    defer m.deinit();
    for ([_]u32{
        0x0e60d400, // fadd .1d (Q = 0)
        0x0ea0dc00, // three same U=0 a=1 11011
        0x0e20ec00, // fmlal (FEAT_FHM)
        0x5e20d400, // scalar fadd (no scalar form)
        0x0e61a800, // fcvtns .1d
        0x0e21e800, // frint32z (FEAT_FRINTTS)
        0x0ea16800, // bfcvtn (FEAT_BF16)
        0x2e216800, // fcvtxn .4h (sz = 0)
        0x2e217800, // FCVTL with U = 1
        0x5ea0f800, // fabs (scalar): not in AdvSIMD scalar
        0x7e216800, // fcvtxn h, s
        0x0e30c800, // fmaxnmv .4h (FEAT_FP16)
        0x2e30c800, // fmaxnmv .2s (Q = 0)
        0x6e70c800, // fmaxnmv .2d
        0x5e30d800, // faddp h (FEAT_FP16)
        0x7eb0d800, // scalar pairwise U=1 a=1 01101
        0x0f10e400, // scvtf .4h, #imm (FEAT_FP16)
        0x0f40e400, // scvtf .1d, #imm (Q = 0)
        0x0f001000, // fmla .4h by element (FEAT_FP16)
        0x0f401000, // fmla size 01
        0x4fe01000, // fmla .2d by element, L = 1
        0x0fc01000, // fmla .1d by element (Q = 0)
        0x2f801000, // fcmla (FEAT_FCMA)
        0x7f801000, // scalar U = 1 opcode 0001
    }) |w| {
        m.code(0x10000, &.{w});
        try t.expectEqual(@as(u32, mach.EV_ILLEGAL), m.run(10));
        try t.expectEqual(@as(u64, w), m.cpu.ev.insn);
    }
}
