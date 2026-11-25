// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa
//! A64 JIT: AdvSIMD and crypto `generic` ops translated inline with wasm
//! SIMD, where the host compiles it (jitcore.zig cap_simd; the V registers
//! are then v128 locals, common.zig). The tier keeps them `generic`: without
//! SIMD, and while an op's feature is off (kw_set_isa), they stay calls of
//! the interpreter (emit.zig, `generic`). generic.zig's inline forms (FMOV,
//! UMOV, INS/DUP general, MOVI/MVNI, AND..EOR) are decoded first.
//!
//!   decode(raw) ?V      the word's form, only for encodings the reference
//!                       accepts with exactly that meaning (routed by
//!                       decode.route, then each group file's own masks and
//!                       legality rules)
//!   usage(v) Use        what it reads and writes
//!   enabled(e, v)       its feature (crypto) is on for this CPU
//!   emit(e, v, at)      its code
//!
//! Integer AdvSIMD maps to wasm SIMD lane ops (lane-wise arithmetic and
//! compares, shifts, extending multiplies, popcnt, bitselect); every fixed
//! byte permutation (UZP, ZIP, TRN, EXT, REV, DUP and INS of elements, the
//! narrowing moves) is one i8x16.shuffle whose indices are computed from
//! the reference's own functions on numbered bytes; TBL/TBX are swizzles.
//!
//! Crypto: AES rounds are table lookups (jitcore.zig AesTab, cpu-relative:
//! SubBytes a byte table, SubBytes + MixColumns of a column four word
//! loads rotated by row; AESE followed by AESMC of the same register is one
//! T-table round, see `Ext`); SHA1, SHA256 and SHA512 run their rounds on
//! i32/i64 locals; PMULL (64-bit) is a Karatsuba of three 32x32 carry-less
//! products, each core.bitmanip's multiply-and-mask sequence. Unlike the
//! interpreter's bitsliced AES (crypto.zig), the AES tables are indexed by
//! data (a64/jit.md).

const std = @import("std");
const c = @import("../common.zig");
const ops = @import("../ops.zig");
const kwemu = @import("../../emu/kwemu.zig");
const jc = @import("../../emu/jitcore.zig");
const dec = @import("../decode.zig");
const simd_move = @import("../simd_move.zig");
const opexec = @import("../opexec.zig");
const jit = @import("emit.zig");
const cm = @import("common.zig");
const mem = @import("mem.zig");
const Use = jit.Use;
const Emit = jit.Emit;
const At = jit.At;
const W = jc.W;
const Cpu = kwemu.Cpu;

pub const Kind = enum(u8) {
    /// Vd = code(Vn, Vm), one lane op
    bin,
    /// USHL (x 0), SSHL (x 1) by register on S or D lanes (vector, scalar
    /// D): each lane by the signed low byte of Vm's lane, lane by lane in
    /// i64 (wasm shifts take one count for all lanes)
    shlreg,
    /// MLA, MLS: Vd = code2(Vd, code(Vn, Vm))
    mla,
    /// CMTST: (Vn & Vm) != 0 per lane
    cmtst,
    /// CMHI (x 0), CMHS (x 1) on 64-bit lanes (wasm has no unsigned i64x2
    /// compares: both sides with the sign bit flipped, signed)
    cmhu64,
    /// BSL (x 0), BIT (1), BIF (2)
    bsl,
    /// Vd = code(Vn) (code 0: NOT)
    un,
    /// Vd = code(Vn, 0), a compare with zero
    cmz,
    /// Vd = code(Vn, sh) (x 1: the result is 0, USHR by the element size)
    shift,
    /// Vd = Vd + code(Vn, sh) (code2: the add; x 1: the shifted part is 0)
    sra,
    /// SLI, SRI: Vn shifted by code into Vd under the mask in imm (both
    /// halves; x 1: SRI by the element size, Vd unchanged)
    ins_shift,
    /// USHLL, SSHLL (2): code extends, code2 shifts left by sh (0: none)
    shll,
    /// XTN, SHRN (2): code shifts the wide lanes right by sh (0: none),
    /// idx packs their low halves; q: into the upper half, keeping Vd's lower
    narrow,
    /// shuffle(a, b) by idx: x 0 (Vn, Vm), 1 (Vn, Vn), 2 (Vd, Vn), 3 (Vn, 0)
    shuf,
    /// TBL (x 0), TBX (x 1) with sh registers
    tbl,
    /// ADDV, [SU]MAXV, [SU]MINV: code over the lanes, into a scalar
    across,
    /// ADDP, [SU]MAXP, [SU]MINP (vector): code(evens, odds)
    pair,
    /// ADDP (scalar): Dd = Vn.d[0] + Vn.d[1]
    paird,
    /// three different: x 0 MULL, 1 MLAL, 2 MLSL (code: extmul, code2: add
    /// or sub), 3 ADDL/SUBL, 4 ADDW/SUBW (code: extend, code2: add or sub)
    long,
    /// ORR (x 0), BIC (x 1) (vector, immediate)
    immlog,
    /// SMOV: Xd/Wd = lane sh of Vn (code: extract_lane_s or i32x4 extract;
    /// x 1: to an X register)
    smov,
    /// INS (general) of a B, H or S lane: lane sh of Vd = Wn (code: replace)
    insx,
    /// LD1, ST1 (multiple structures, sh registers); x 0: no writeback, 1:
    /// post-index by imm, 2: by Xm
    ld1,
    st1,
    /// LD1, ST1 (single structure: lane `idx[0]` of Vt, 1 << size bytes),
    /// LD1R (Q: both halves); x as for ld1
    ld1s,
    st1s,
    ld1r,
    /// LDP, STP (and the no-allocate forms) of S, D, Q registers: Rt in d,
    /// Rt2 in m, Xn in n; x: 0 no-allocate, 1 post-index, 2 offset, 3
    /// pre-index; imm the scaled offset; 4 << size bytes each (size 2..4)
    ldpv,
    stpv,
    aese,
    aesd,
    aesmc,
    aesimc,
    /// PMULL, PMULL2 (x: the half) of 64-bit lanes
    pmull,
    sha1h,
    sha1c,
    sha1p,
    sha1m,
    sha1su0,
    sha1su1,
    sha256h,
    sha256h2,
    sha256su0,
    sha256su1,
    sha512h,
    sha512h2,
    sha512su0,
    sha512su1,
    /// FP binary ops (FADD, FSUB, FMUL, FDIV, vector & by-element FMUL): x
    /// 0 add, 1 sub, 2 mul, 3 div; size 2 S, 3 D
    fp_bin,
    /// FP min/max (FMAX, FMIN, FMAXNM, FMINNM): size 2 S, 3 D
    fp_minmax,
    /// FMLA (x 0), FMLS (x 1) (vector & by-element, S lanes only)
    fp_mla,
    /// FSQRT (vector): size 2 S, 3 D
    fp_sqrt,
    /// FP compare (FCMEQ, FCMGE, FCMGT): size 2 S, 3 D
    fp_cmp,
    /// FP compare with zero (FCMGT/FCMEQ/FCMLT/FCMGE/FCMLE #0): size 2 S, 3 D
    fp_cmp0,
    /// SCVTF (x 0), UCVTF (x 1) (vector, S lanes)
    fp_cvtf,
    /// FCVTZS (x 0), FCVTZU (x 1) (vector, S lanes)
    fp_cvtz,
};

pub const V = struct {
    k: Kind,
    d: u8 = 0,
    n: u8 = 0,
    m: u8 = 0,
    q: bool = false,
    size: u2 = 0,
    x: u8 = 0,
    sh: u8 = 0,
    code: u16 = 0,
    code2: u16 = 0,
    /// Per kind: a mask, constant, step or offset; for the AdvSIMD FP
    /// kinds with a slow path, the raw word its call passes.
    imm: u64 = 0,
    idx: [16]u8 = @splat(0),
    /// By element: Vm is its element broadcast by `bidx`.
    bc: bool = false,
    bidx: [16]u8 = @splat(0),
    feat: ?c.Feature = null,
};

// ---- wasm SIMD opcodes by element size (0xffff: none) ----------------------------

const none: u16 = 0xffff;
const op_add = [4]u16{ 0x6e, 0x8e, 0xae, 0xce };
const op_sub = [4]u16{ 0x71, 0x91, 0xb1, 0xd1 };
const op_mul = [4]u16{ none, 0x95, 0xb5, none };
const op_eq = [4]u16{ 0x23, 0x2d, 0x37, 0xd6 };
const op_ne = [4]u16{ 0x24, 0x2e, 0x38, 0xd7 };
const op_lt_s = [4]u16{ 0x25, 0x2f, 0x39, 0xd8 };
const op_gt_s = [4]u16{ 0x27, 0x31, 0x3b, 0xd9 };
const op_le_s = [4]u16{ 0x29, 0x33, 0x3d, 0xda };
const op_ge_s = [4]u16{ 0x2b, 0x35, 0x3f, 0xdb };
const op_gt_u = [4]u16{ 0x28, 0x32, 0x3c, none };
const op_ge_u = [4]u16{ 0x2c, 0x36, 0x40, none };
const op_min_s = [4]u16{ 0x76, 0x96, 0xb6, none };
const op_min_u = [4]u16{ 0x77, 0x97, 0xb7, none };
const op_max_s = [4]u16{ 0x78, 0x98, 0xb8, none };
const op_max_u = [4]u16{ 0x79, 0x99, 0xb9, none };
const op_avgr_u = [4]u16{ 0x7b, 0x9b, none, none };
const op_shl = [4]u16{ 0x6b, 0x8b, 0xab, 0xcb };
const op_shr_s = [4]u16{ 0x6c, 0x8c, 0xac, 0xcc };
const op_shr_u = [4]u16{ 0x6d, 0x8d, 0xad, 0xcd };
const op_abs = [4]u16{ 0x60, 0x80, 0xa0, 0xc0 };
const op_neg = [4]u16{ 0x61, 0x81, 0xa1, 0xc1 };
/// By the narrow element size: extend low s, high s, low u, high u to twice
/// the width; extmul the same.
const op_ext = [3][4]u16{ .{ 0x87, 0x88, 0x89, 0x8a }, .{ 0xa7, 0xa8, 0xa9, 0xaa }, .{ 0xc7, 0xc8, 0xc9, 0xca } };
const op_extmul = [3][4]u16{ .{ 0x9c, 0x9d, 0x9e, 0x9f }, .{ 0xbc, 0xbd, 0xbe, 0xbf }, .{ 0xdc, 0xdd, 0xde, 0xdf } };
const v128_not = 0x4d;
const v128_and = 0x4e;
const v128_andnot = 0x4f;
const v128_or = 0x50;
const v128_xor = 0x51;
const v128_bitselect = 0x52;
const i8x16_popcnt = 0x62;
const i8x16_swizzle = 0x0e;
const i8x16_lt_u = 0x26;
const i32x4_splat = 0x11;
const i64x2_splat = 0x12;
const i8x16_extract_u = 0x16;
const i8x16_extract_s = 0x15;
const i16x8_extract_u = 0x19;
const i16x8_extract_s = 0x18;
const i32x4_extract = 0x1b;
const i64x2_extract = 0x1d;
const i8x16_replace = 0x17;
const i16x8_replace = 0x1a;
const i32x4_replace = 0x1c;
const i64x2_replace = 0x1e;

fn bits(x: u32, comptime hi: u5, comptime lo: u5) u32 {
    return c.bits(x, hi, lo);
}
fn bit(x: u32, comptime b: u5) bool {
    return (x >> b) & 1 != 0;
}

// ---- decode ------------------------------------------------------------------------

pub fn decode(raw: u32) ?V {
    if (!jc.simd_ok) return null;
    return switch (dec.route(raw)) {
        .simd_arith => arith(raw),
        .simd_move => move(raw),
        .simd_ldst => ldst(raw),
        .crypto => crypto(raw),
        .ldst => pairV(raw),
        else => null,
    };
}

/// ldst.pair of SIMD&FP registers (LDP, STP, LDNP, STNP of S, D, Q; the
/// tier has ops for most Q forms): idx in x (0 no-allocate, 1 post, 2
/// offset, 3 pre), the element size in size, the offset in imm, Rt2 in m.
fn pairV(raw: u32) ?V {
    if (raw & 0x3e00_0000 != 0x2c00_0000) return null;
    const opc = bits(raw, 31, 30);
    if (opc == 3) return null;
    var v = regs3(raw, if (bit(raw, 22)) .ldpv else .stpv);
    v.m = @intCast(bits(raw, 14, 10));
    v.x = @intCast(bits(raw, 24, 23));
    v.size = @intCast(2 + opc);
    v.imm = @bitCast(@as(i64, @as(i7, @bitCast(@as(u7, @intCast(bits(raw, 21, 15)))))) << @intCast(v.size));
    return v;
}

fn regs3(raw: u32, k: Kind) V {
    return .{ .k = k, .d = @intCast(raw & 31), .n = @intCast((raw >> 5) & 31), .m = @intCast((raw >> 16) & 31), .q = bit(raw, 30), .size = @intCast(bits(raw, 23, 22)) };
}

fn arith(raw: u32) ?V {
    if (bit(raw, 28)) {
        if (raw & 0xdf20_0400 == 0x5e20_0400) return threeSame(raw, true);
        if (raw & 0xdf3e_0c00 == 0x5e30_0800) {
            // ADDP (scalar): opcode 11011, U = 0, size 11
            if (bits(raw, 16, 12) != 0b11011 or bit(raw, 29) or bits(raw, 23, 22) != 3) return null;
            return regs3(raw, .paird);
        }
        if (raw & 0xdf80_0400 == 0x5f00_0400) return shiftImm(raw, true);
        return null;
    }
    if (raw & 0x9f20_0400 == 0x0e20_0400) return threeSame(raw, false);
    if (raw & 0x9f3e_0c00 == 0x0e30_0800) return across(raw);
    if (raw & 0x9f3e_0c00 == 0x0e20_0800) return misc(raw);
    if (raw & 0x9f20_0c00 == 0x0e20_0000) return threeDiff(raw);
    if (raw & 0x9f80_0400 == 0x0f00_0400) return shiftImm(raw, false);
    if (raw & 0x9f00_0400 == 0x0f00_0000) return byElem(raw);
    return null;
}

/// simd_long.indexed (vector): MUL, MLA, MLS, [SU]MULL, [SU]MLAL, [SU]MLSL,
/// and FP FMUL, FMLA, FMLS by element, as the vector op with the element
/// of Vm broadcast.
fn byElem(raw: u32) ?V {
    var v = regs3(raw, .mla);
    const u = bit(raw, 29);
    const z = v.size;
    const opcode = bits(raw, 15, 12);
    const h: u8 = @intFromBool(bit(raw, 11));
    const l: u8 = @intFromBool(bit(raw, 21));
    if (opcode == 0b0001 or opcode == 0b0101 or opcode == 0b1001) {
        if (u or !bit(raw, 23)) return null;
        const dbl = bit(raw, 22);
        if (dbl and (l != 0 or !v.q)) return null;
        const index: u8 = if (dbl) h else (h << 1) | l;
        const esz: u8 = if (dbl) 8 else 4;
        v.imm = raw;
        v.size = if (dbl) 3 else 2;
        v.bc = true;
        for (0..16) |i| v.bidx[i] = index * esz + @as(u8, @intCast(i)) % esz;
        switch (opcode) {
            0b1001 => { // FMUL by element
                v.k = .fp_bin;
                v.x = 2;
                v.code = if (dbl) cm.S.f64x2_mul else cm.S.f32x4_mul;
            },
            0b0001, 0b0101 => { // FMLA, FMLS by element (single only)
                if (dbl) return null;
                v.k = .fp_mla;
                v.x = @intFromBool(opcode == 0b0101);
            },
            else => unreachable,
        }
        return v;
    }
    const index: u8 = switch (z) {
        1 => blk: {
            v.m = @intCast(bits(raw, 19, 16));
            break :blk (h << 2) | (l << 1) | @intFromBool(bit(raw, 20));
        },
        2 => (h << 1) | l,
        else => return null,
    };
    const esz = @as(u8, 1) << z;
    v.bc = true;
    for (0..16) |i| v.bidx[i] = index * esz + @as(u8, @intCast(i)) % esz;
    const sel: usize = @as(usize, @intFromBool(v.q)) + 2 * @as(usize, @intFromBool(u));
    switch (opcode) {
        0b0000, 0b0100 => { // MLA, MLS
            if (!u) return null;
            v.code = op_mul[z];
            v.code2 = if (opcode == 0b0000) op_add[z] else op_sub[z];
        },
        0b1000 => { // MUL
            if (u) return null;
            v.k = .bin;
            v.code = op_mul[z];
        },
        0b0010, 0b0110, 0b1010 => { // [SU]MLAL, [SU]MLSL, [SU]MULL
            v.k = .long;
            v.x = if (opcode == 0b1010) 0 else if (opcode == 0b0010) 1 else 2;
            v.code = op_extmul[z][sel];
            v.code2 = if (opcode == 0b0110) op_sub[z + 1] else op_add[z + 1];
        },
        else => return null,
    }
    return v;
}

/// simd_arith.threeSame's integer ops that map to one wasm op (scalar: the
/// D forms of ADD, SUB and the compares).
fn threeSame(raw: u32, sc: bool) ?V {
    var v = regs3(raw, .bin);
    const u = bit(raw, 29);
    const opcode = bits(raw, 15, 11);
    if (opcode >= 0b11000) return if (sc) null else threeSameFp(raw);
    const z = v.size;
    if (sc) {
        if (z != 3) return null;
        v.q = false; // one 64-bit lane, the rest zeroed
    } else if (z == 3 and !v.q) return null;
    switch (opcode) {
        0b10000 => v.code = if (u) op_sub[z] else op_add[z],
        0b01000 => { // USHL, SSHL (B and H lanes stay calls)
            if (z < 2) return null;
            v.k = .shlreg;
            v.x = @intFromBool(!u);
        },
        0b10001 => if (u) {
            v.code = op_eq[z];
        } else {
            v.k = .cmtst;
        },
        0b00110, 0b00111 => {
            const ge = opcode == 0b00111;
            if (u and z == 3) {
                v.k = .cmhu64;
                v.x = @intFromBool(ge);
            } else v.code = if (u) (if (ge) op_ge_u[z] else op_gt_u[z]) else (if (ge) op_ge_s[z] else op_gt_s[z]);
        },
        0b01100, 0b01101 => {
            if (sc or z == 3) return null;
            v.code = if (opcode == 0b01100) (if (u) op_max_u[z] else op_max_s[z]) else (if (u) op_min_u[z] else op_min_s[z]);
        },
        0b10011 => {
            if (sc or u or z == 0 or z == 3) return null; // MUL h, s (no i8x16.mul; PMUL)
            v.code = op_mul[z];
        },
        0b10010 => { // MLA, MLS h, s
            if (sc or z == 0 or z == 3) return null;
            v.k = .mla;
            v.code = op_mul[z];
            v.code2 = if (u) op_sub[z] else op_add[z];
        },
        0b00010 => {
            if (sc or !u or z > 1) return null; // URHADD b, h
            v.code = op_avgr_u[z];
        },
        0b10100, 0b10101, 0b10111 => { // [SU]MAXP, [SU]MINP, ADDP
            if (sc) return null;
            v.k = .pair;
            if (opcode == 0b10111) {
                if (u) return null;
                v.code = op_add[z];
            } else {
                if (z == 3) return null;
                v.code = if (opcode == 0b10100) (if (u) op_max_u[z] else op_max_s[z]) else (if (u) op_min_u[z] else op_min_s[z]);
            }
        },
        else => return null,
    }
    return v;
}

/// simd_fp.threeSame (vector only): FADD, FSUB, FMUL, FDIV, FMAX, FMIN,
/// FMAXNM, FMINNM, FMLA, FMLS, FCMEQ, FCMGE, FCMGT (2S, 4S, 2D; FMLA/FMLS S
/// only).
fn threeSameFp(raw: u32) ?V {
    var v = regs3(raw, .fp_bin);
    const u = bit(raw, 29);
    const a = bit(raw, 23);
    const dbl = bit(raw, 22);
    if (dbl and !v.q) return null;
    v.imm = raw;
    v.size = if (dbl) 3 else 2;
    const k = (@as(u32, @intFromBool(u)) << 6) | (@as(u32, @intFromBool(a)) << 5) | bits(raw, 15, 11);
    switch (k) {
        0b0_0_11010 => {
            v.x = 0;
            v.code = if (dbl) cm.S.f64x2_add else cm.S.f32x4_add;
        },
        0b0_1_11010 => {
            v.x = 1;
            v.code = if (dbl) cm.S.f64x2_sub else cm.S.f32x4_sub;
        },
        0b1_0_11011 => {
            v.x = 2;
            v.code = if (dbl) cm.S.f64x2_mul else cm.S.f32x4_mul;
        },
        0b1_0_11111 => {
            v.x = 3;
            v.code = if (dbl) cm.S.f64x2_div else cm.S.f32x4_div;
        },
        0b0_0_11000, 0b0_0_11110, 0b0_1_11000, 0b0_1_11110 => {
            v.k = .fp_minmax;
            v.code = if (a) (if (dbl) cm.S.f64x2_min else cm.S.f32x4_min) else (if (dbl) cm.S.f64x2_max else cm.S.f32x4_max);
        },
        0b0_0_11001, 0b0_1_11001 => {
            if (dbl) return null;
            v.k = .fp_mla;
            v.x = @intFromBool(a);
        },
        0b0_0_11100 => {
            v.k = .fp_cmp;
            v.code = if (dbl) cm.S.f64x2_eq else cm.S.f32x4_eq;
        },
        0b1_0_11100 => {
            v.k = .fp_cmp;
            v.code = if (dbl) cm.S.f64x2_ge else cm.S.f32x4_ge;
        },
        0b1_1_11100 => {
            v.k = .fp_cmp;
            v.code = if (dbl) cm.S.f64x2_gt else cm.S.f32x4_gt;
        },
        else => return null,
    }
    return v;
}

/// USHL/SSHL lane by lane (simd_int.shlReg without rounding or
/// saturation): s = the signed low byte of Vm's lane; s >= 0: a << s (0 from
/// the element size on); s < 0: a >> -s, logical (0 from the element size
/// on) or arithmetic (the sign from the element size on). S lanes go through
/// i64 zero- or sign-extended, which gives the same low 32 bits.
fn shlReg(e: *Emit, v: V) void {
    const d64 = v.size == 3;
    const lanes: u8 = if (d64) (if (v.q) 2 else 1) else (if (v.q) 4 else 2);
    const es: u64 = if (d64) 64 else 32;
    const signed = v.x == 1;
    for (0..lanes) |i| {
        const li: u8 = @intCast(i);
        get(e, v.n);
        if (d64) e.vlane(cm.S.i64x2_extract_lane, li) else {
            e.vlane(cm.S.i32x4_extract_lane, li);
            e.op(if (signed) W.i64_ext_s else W.i64_ext_u);
        }
        e.set(lu); // a
        get(e, v.m);
        if (d64) e.vlane(cm.S.i64x2_extract_lane, li) else {
            e.vlane(cm.S.i32x4_extract_lane, li);
            e.op(W.i64_ext_u);
        }
        e.k64(56);
        e.op(W.i64_shl);
        e.k64(56);
        e.op(W.i64_shr_s);
        e.set(lu + 1); // s
        // left: s < es ? a << s : 0
        e.get(lu);
        e.get(lu + 1);
        e.op(W.i64_shl);
        e.k64(0);
        e.get(lu + 1);
        e.k64(es);
        e.op(W.i64_lt_s);
        e.op(W.select);
        // right, by n = -s
        e.get(lu);
        e.k64(0);
        e.get(lu + 1);
        e.op(W.i64_sub);
        e.tee(lu + 2);
        if (signed) {
            e.k64(es - 1);
            e.get(lu + 2);
            e.k64(es);
            e.op(W.i64_lt_s);
            e.op(W.select);
            e.op(W.i64_shr_s);
        } else {
            e.op(W.i64_shr_u);
            e.k64(0);
            e.get(lu + 2);
            e.k64(es);
            e.op(W.i64_lt_s);
            e.op(W.select);
        }
        e.get(lu + 1);
        e.k64(0);
        e.op(W.i64_ge_s);
        e.op(W.select);
        e.set(lu + 3 + @as(u32, @intCast(i)));
    }
    k128(e, 0, 0);
    for (0..lanes) |i| {
        e.get(lu + 3 + @as(u32, @intCast(i)));
        if (d64) e.vlane(cm.S.i64x2_replace_lane, @intCast(i)) else {
            e.op(W.i32_wrap);
            e.vlane(cm.S.i32x4_replace_lane, @intCast(i));
        }
    }
    cm.setVq(e, v.d, v.q);
}

fn across(raw: u32) ?V {
    var v = regs3(raw, .across);
    const u = bit(raw, 29);
    const z = v.size;
    if (z == 3 or (z == 2 and !v.q)) return null;
    v.code = switch (bits(raw, 16, 12)) {
        0b11011 => if (u) return null else op_add[z],
        0b01010 => if (u) op_max_u[z] else op_max_s[z],
        0b11010 => if (u) op_min_u[z] else op_min_s[z],
        else => return null,
    };
    return v;
}

/// simd_arith.twoRegMisc (vector): REV*, CNT, NOT, the compares with zero,
/// ABS, NEG, XTN.
fn misc(raw: u32) ?V {
    var v = regs3(raw, .un);
    const u = bit(raw, 29);
    const z = v.size;
    const opcode = bits(raw, 16, 12);
    switch (opcode) {
        0b00000, 0b00001 => {
            if (opcode == 1 and u) return null;
            const cont: u2 = if (opcode == 1) 1 else if (u) 2 else 3;
            if (z >= cont) return null;
            const esz = @as(u8, 1) << z;
            const per = @as(u8, 1) << (cont - z);
            v.k = .shuf;
            v.x = 1;
            for (0..16) |i| {
                const e: u8 = @intCast(i / esz);
                v.idx[i] = (e ^ (per - 1)) * esz + @as(u8, @intCast(i % esz));
            }
        },
        0b00101 => {
            if (z != 0) return null; // RBIT (size 01) stays a call
            v.code = if (u) 0 else i8x16_popcnt;
        },
        0b01000, 0b01001, 0b01010, 0b01011 => {
            if (z == 3 and !v.q) return null;
            v.k = .cmz;
            switch (opcode) {
                0b01000 => v.code = if (u) op_ge_s[z] else op_gt_s[z],
                0b01001 => v.code = if (u) op_le_s[z] else op_eq[z],
                0b01010 => v.code = if (u) return null else op_lt_s[z],
                else => {
                    v.k = .un;
                    v.code = if (u) op_neg[z] else op_abs[z];
                },
            }
        },
        0b10010 => { // XTN, XTN2
            if (u or z == 3) return null;
            v.k = .narrow;
            narrowIdx(&v);
        },
        else => return miscFp(raw),
    }
    return v;
}

/// simd_fp.twoRegMisc (vector): FABS, FNEG, FSQRT, FCM{GT,EQ,LT,GE,LE} #0,
/// SCVTF/UCVTF (S), FCVTZS/FCVTZU (S).
fn miscFp(raw: u32) ?V {
    var v = regs3(raw, .un);
    const u = bit(raw, 29);
    const a = bit(raw, 23);
    const dbl = bit(raw, 22);
    if (dbl and !v.q) return null;
    v.imm = raw;
    v.size = if (dbl) 3 else 2;
    const k = (@as(u32, @intFromBool(u)) << 6) | (@as(u32, @intFromBool(a)) << 5) | bits(raw, 16, 12);
    switch (k) {
        0b0_1_01111 => v.code = if (dbl) cm.S.f64x2_abs else cm.S.f32x4_abs,
        0b1_1_01111 => v.code = if (dbl) cm.S.f64x2_neg else cm.S.f32x4_neg,
        0b1_1_11111 => {
            v.k = .fp_sqrt;
            v.code = if (dbl) cm.S.f64x2_sqrt else cm.S.f32x4_sqrt;
        },
        0b0_1_01100 => {
            v.k = .fp_cmp0;
            v.code = if (dbl) cm.S.f64x2_gt else cm.S.f32x4_gt;
        },
        0b0_1_01101 => {
            v.k = .fp_cmp0;
            v.code = if (dbl) cm.S.f64x2_eq else cm.S.f32x4_eq;
        },
        0b0_1_01110 => {
            v.k = .fp_cmp0;
            v.code = if (dbl) cm.S.f64x2_lt else cm.S.f32x4_lt;
        },
        0b1_1_01100 => {
            v.k = .fp_cmp0;
            v.code = if (dbl) cm.S.f64x2_ge else cm.S.f32x4_ge;
        },
        0b1_1_01101 => {
            v.k = .fp_cmp0;
            v.code = if (dbl) cm.S.f64x2_le else cm.S.f32x4_le;
        },
        0b0_0_11101, 0b1_0_11101 => {
            if (dbl) return null;
            v.k = .fp_cvtf;
            v.x = @intFromBool(u);
            v.code = if (u) cm.S.f32x4_convert_i32x4_u else cm.S.f32x4_convert_i32x4_s;
        },
        0b0_1_11011, 0b1_1_11011 => {
            if (dbl) return null;
            v.k = .fp_cvtz;
            v.x = @intFromBool(u);
            v.code = if (u) cm.S.i32x4_trunc_sat_f32x4_u else cm.S.i32x4_trunc_sat_f32x4_s;
        },
        else => return null,
    }
    return v;
}

/// The bytes of the low halves of the wide (2 << size bytes) lanes, packed.
fn narrowIdx(v: *V) void {
    const esz = @as(u8, 1) << v.size;
    for (0..8) |j| {
        const b: u8 = @intCast(j);
        const src = (b / esz) * 2 * esz + b % esz;
        if (v.q) {
            v.idx[j] = b; // Vd's lower half
            v.idx[8 + j] = 16 + src;
        } else {
            v.idx[j] = src;
            v.idx[8 + j] = src;
        }
    }
}

fn threeDiff(raw: u32) ?V {
    var v = regs3(raw, .long);
    const u = bit(raw, 29);
    const z = v.size;
    const opcode = bits(raw, 15, 12);
    if (opcode == 0b1110) { // PMULL
        if (u or z != 3) return null;
        v.k = .pmull;
        v.x = @intFromBool(v.q);
        v.feat = .pmull;
        return v;
    }
    if (z == 3) return null;
    const sel: usize = @as(usize, @intFromBool(v.q)) + 2 * @as(usize, @intFromBool(u)); // low/high, s/u
    switch (opcode) {
        0b1100, 0b1000, 0b1010 => {
            v.x = if (opcode == 0b1100) 0 else if (opcode == 0b1000) 1 else 2;
            v.code = op_extmul[z][sel];
            v.code2 = if (opcode == 0b1010) op_sub[z + 1] else op_add[z + 1];
        },
        0b0000, 0b0010, 0b0001, 0b0011 => {
            v.x = if (opcode & 1 == 0) 3 else 4;
            v.code = op_ext[z][sel];
            v.code2 = if (opcode & 2 != 0) op_sub[z + 1] else op_add[z + 1];
        },
        else => return null,
    }
    return v;
}

fn shiftImm(raw: u32, sc: bool) ?V {
    var v = regs3(raw, .shift);
    const u = bit(raw, 29);
    const immh = bits(raw, 22, 19);
    if (immh == 0) return null;
    const opcode = bits(raw, 15, 11);
    const z: u2 = @intCast(31 - @clz(immh));
    v.size = z;
    const esize: u32 = @as(u32, 8) << z;
    const immhb = bits(raw, 22, 16);
    const right = 2 * esize - immhb;
    const left = immhb - esize;
    switch (opcode) {
        0b00000, 0b00010, 0b01000, 0b01010 => {
            if (sc) {
                if (z != 3) return null;
                v.q = false;
            } else if (z == 3 and !v.q) return null;
            const ones: u64 = if (esize == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(esize)) - 1;
            switch (opcode) {
                0b00000, 0b00010 => { // [SU]SHR, [SU]SRA
                    v.k = if (opcode == 0) .shift else .sra;
                    v.code2 = op_add[z];
                    if (right == esize) {
                        if (u) v.x = 1 else v.sh = @intCast(esize - 1);
                    } else v.sh = @intCast(right);
                    v.code = if (u) op_shr_u[z] else op_shr_s[z];
                },
                0b01000 => { // SRI
                    if (!u) return null;
                    v.k = .ins_shift;
                    v.code = op_shr_u[z];
                    v.sh = @intCast(right);
                    if (right == esize) v.x = 1 else v.imm = c.replicate(ones >> @intCast(right), @intCast(esize));
                },
                else => { // SHL, SLI
                    v.k = if (u) .ins_shift else .shift;
                    v.code = op_shl[z];
                    v.sh = @intCast(left);
                    v.imm = c.replicate((ones << @intCast(left)) & ones, @intCast(esize));
                },
            }
        },
        0b10100 => { // SSHLL, USHLL (2)
            if (sc or z == 3) return null;
            v.k = .shll;
            v.code = op_ext[z][@as(usize, @intFromBool(v.q)) + 2 * @as(usize, @intFromBool(u))];
            v.code2 = op_shl[z + 1];
            v.sh = @intCast(left);
        },
        0b10000 => { // SHRN (2)
            if (sc or u or z == 3) return null;
            v.k = .narrow;
            v.code = op_shr_u[z + 1];
            v.sh = @intCast(right);
            narrowIdx(&v);
        },
        else => return null,
    }
    return v;
}

fn move(raw: u32) ?V {
    var v = regs3(raw, .shuf);
    const imm5 = bits(raw, 20, 16);
    if (raw & 0xdfe0_8400 == 0x5e00_0400) { // DUP (element), scalar
        if (bit(raw, 29) or bits(raw, 14, 11) != 0 or imm5 & 15 == 0) return null;
        const z: u3 = @intCast(@ctz(imm5));
        const esz = @as(u8, 1) << z;
        const index: u8 = @intCast(imm5 >> (z + 1));
        v.x = 3;
        v.q = true;
        for (0..16) |i| v.idx[i] = if (i < esz) index * esz + @as(u8, @intCast(i)) else 16;
        return v;
    }
    if (raw & 0x9fe0_8400 == 0x0e00_0400) { // copy
        if (imm5 & 15 == 0) return null;
        const z: u2 = @intCast(@ctz(imm5));
        const esz = @as(u8, 1) << z;
        const index: u8 = @intCast(imm5 >> (@as(u3, z) + 1));
        const imm4 = bits(raw, 14, 11);
        if (bit(raw, 29)) { // INS (element)
            if (!v.q) return null;
            const src: u8 = @intCast(imm4 >> z);
            v.x = 2;
            for (0..16) |i| {
                const b: u8 = @intCast(i);
                v.idx[i] = if (b / esz == index) 16 + src * esz + b % esz else b;
            }
            return v;
        }
        switch (imm4) {
            0b0000 => { // DUP (element)
                if (z == 3 and !v.q) return null;
                v.x = 1;
                for (0..16) |i| v.idx[i] = index * esz + @as(u8, @intCast(i)) % esz;
                return v;
            },
            0b0011 => { // INS (general), B/H/S (D is generic.zig's)
                if (!v.q or z == 3) return null;
                v.k = .insx;
                v.code = ([3]u16{ i8x16_replace, i16x8_replace, i32x4_replace })[z];
                v.sh = index;
                return v;
            },
            0b0101 => { // SMOV
                if (z == 3 or (z == 2 and !v.q)) return null;
                v.k = .smov;
                v.code = ([3]u16{ i8x16_extract_s, i16x8_extract_s, i32x4_extract })[z];
                v.sh = index;
                v.x = @intFromBool(v.q);
                return v;
            },
            else => return null,
        }
    }
    if (raw & 0xbf20_8c00 == 0x0e00_0000) { // TBL, TBX
        if (bits(raw, 23, 22) != 0) return null;
        v.k = .tbl;
        v.sh = @intCast(bits(raw, 14, 13) + 1);
        v.x = @intFromBool(bit(raw, 12));
        return v;
    }
    if (raw & 0xbf20_8c00 == 0x0e00_0800) { // UZP, TRN, ZIP
        const opcode: u3 = @intCast(bits(raw, 14, 12));
        if (opcode & 3 == 0 or (v.size == 3 and !v.q)) return null;
        const r = switch (v.size) {
            0 => simd_move.permuteOp(u8, opcode, v.q, numbered(1), numbered(17)),
            1 => simd_move.permuteOp(u16, opcode, v.q, numbered(1), numbered(17)),
            2 => simd_move.permuteOp(u32, opcode, v.q, numbered(1), numbered(17)),
            3 => simd_move.permuteOp(u64, opcode, v.q, numbered(1), numbered(17)),
        };
        fromNumbered(&v, r);
        return v;
    }
    if (raw & 0xbf20_8400 == 0x2e00_0000) { // EXT
        const imm4 = bits(raw, 14, 11);
        if (bits(raw, 23, 22) != 0 or (!v.q and imm4 >= 8)) return null;
        fromNumbered(&v, simd_move.ext(v.q, imm4, numbered(1), numbered(17)));
        return v;
    }
    if (raw & 0x9ff8_0400 == 0x0f00_0400) { // ORR, BIC (vector, immediate)
        if (bit(raw, 11)) return null;
        const cmode: u4 = @intCast(bits(raw, 15, 12));
        if (!(cmode & 1 == 1 and cmode < 12)) return null; // MOVI, MVNI, FMOV: generic.zig
        const op = bit(raw, 29);
        const imm8: u8 = @intCast((bits(raw, 18, 16) << 5) | bits(raw, 9, 5));
        v.k = .immlog;
        v.x = @intFromBool(op);
        v.imm = simd_move.expandImm(op, cmode, imm8, v.q) orelse return null;
        return v;
    }
    if (raw & 0x9f20_fc00 == 0x0e20_1c00) { // BSL, BIT, BIF
        const op = (bits(raw, 29, 29) << 2) | bits(raw, 23, 22);
        if (op < 5) return null; // AND..EOR: generic.zig
        v.k = .bsl;
        v.x = @intCast(op - 5);
        return v;
    }
    return null;
}

/// 16 bytes numbered first, first + 1, ... (to trace a permutation).
fn numbered(first: u8) u128 {
    var b: [16]u8 = undefined;
    for (&b, 0..) |*x, i| x.* = first + @as(u8, @intCast(i));
    return @bitCast(b);
}
/// The shuffle of (Vn, Vm) that moves numbered bytes as `r` shows (0: a
/// zeroed upper half, which the Q = 0 write gives).
fn fromNumbered(v: *V, r: u128) void {
    const b: [16]u8 = @bitCast(r);
    for (b, 0..) |x, i| v.idx[i] = if (x == 0) 0 else x - 1;
    v.x = 0;
}

fn ldst(raw: u32) ?V {
    if (bit(raw, 24)) return single(raw);
    const post = bit(raw, 23);
    if (if (post) bit(raw, 21) else bits(raw, 21, 16) != 0) return null;
    const nregs: u8 = switch (bits(raw, 15, 12)) {
        0b0111 => 1,
        0b1010 => 2,
        0b0110 => 3,
        0b0010 => 4,
        else => return null, // LD2-LD4, ST2-ST4: calls
    };
    var v = regs3(raw, if (bit(raw, 22)) .ld1 else .st1);
    v.sh = nregs;
    if (post) {
        if (v.m == 31) {
            v.x = 1;
            v.imm = @as(u64, nregs) * (if (v.q) @as(u64, 16) else 8);
        } else v.x = 2;
    }
    return v;
}

/// simd_ldst.single with one structure element: LD1/ST1 (one lane), LD1R.
fn single(raw: u32) ?V {
    const post = bit(raw, 23);
    if (!post and bits(raw, 20, 16) != 0) return null;
    const opcode = bits(raw, 15, 13);
    if (opcode & 1 != 0 or bit(raw, 21)) return null; // LD2-LD4, ST2-ST4: calls
    const load = bit(raw, 22);
    var v = regs3(raw, if (load) .ld1s else .st1s);
    const qb: u8 = @intFromBool(v.q);
    const sbit: u8 = @intFromBool(bit(raw, 12));
    const size: u8 = @intCast(bits(raw, 11, 10));
    var index: u8 = 0;
    switch (opcode >> 1) {
        0 => {
            v.size = 0;
            index = (qb << 3) | (sbit << 2) | size;
        },
        1 => {
            if (size & 1 != 0) return null;
            v.size = 1;
            index = (qb << 2) | (sbit << 1) | (size >> 1);
        },
        2 => if (size == 0) {
            v.size = 2;
            index = (qb << 1) | sbit;
        } else {
            if (size != 1 or sbit != 0) return null;
            v.size = 3;
            index = qb;
        },
        else => {
            if (!load or sbit != 0) return null;
            v.k = .ld1r;
            v.size = @intCast(size);
        },
    }
    v.idx[0] = index;
    v.sh = 1;
    if (post) {
        if (v.m == 31) {
            v.x = 1;
            v.imm = @as(u64, 1) << v.size;
        } else v.x = 2;
    }
    return v;
}

fn crypto(raw: u32) ?V {
    var v = regs3(raw, .aese);
    if (raw & 0xff3e_0c00 == 0x4e28_0800) {
        if (bits(raw, 23, 22) != 0) return null;
        v.feat = .aes;
        v.k = switch (bits(raw, 16, 12)) {
            0b00100 => .aese,
            0b00101 => .aesd,
            0b00110 => .aesmc,
            0b00111 => .aesimc,
            else => return null,
        };
        return v;
    }
    if (raw & 0xff20_8c00 == 0x5e00_0000) {
        if (bits(raw, 23, 22) != 0) return null;
        const op = bits(raw, 14, 12);
        v.feat = if (op < 4) .sha1 else .sha2;
        v.k = switch (op) {
            0 => .sha1c,
            1 => .sha1p,
            2 => .sha1m,
            3 => .sha1su0,
            4 => .sha256h,
            5 => .sha256h2,
            6 => .sha256su1,
            else => return null,
        };
        return v;
    }
    if (raw & 0xff3e_0c00 == 0x5e28_0800) {
        if (bits(raw, 23, 22) != 0) return null;
        switch (bits(raw, 16, 12)) {
            0 => {
                v.k = .sha1h;
                v.feat = .sha1;
            },
            1 => {
                v.k = .sha1su1;
                v.feat = .sha1;
            },
            2 => {
                v.k = .sha256su0;
                v.feat = .sha2;
            },
            else => return null,
        }
        return v;
    }
    if (raw & 0xffe0_b000 == 0xce60_8000) {
        if (bit(raw, 14)) return null;
        v.feat = .sha512;
        v.k = switch (bits(raw, 11, 10)) {
            0 => .sha512h,
            1 => .sha512h2,
            2 => .sha512su1,
            else => return null, // RAX1: a call
        };
        return v;
    }
    if (raw & 0xffff_f000 == 0xcec0_8000) {
        if (bits(raw, 11, 10) != 0) return null;
        v.k = .sha512su0;
        v.feat = .sha512;
        return v;
    }
    return null;
}

// ---- usage, feature --------------------------------------------------------------

fn vr(r: u8) Use {
    return cm.rd(ops.v0_slot + 2 * (r & 31));
}
fn vw(r: u8) Use {
    return cm.wr(ops.v0_slot + 2 * (r & 31));
}
fn xs(r: u8) u8 {
    return if (r == 31) ops.zr else r;
}

pub fn usage(v: V) Use {
    const nm = vr(v.n).with(vr(v.m));
    return switch (v.k) {
        .bin, .shlreg, .cmtst, .cmhu64, .pair, .pmull, .fp_bin, .fp_minmax, .fp_cmp => nm.with(vw(v.d)),
        .un, .cmz, .shift, .shll, .across, .paird, .aesmc, .aesimc, .sha1h, .fp_sqrt, .fp_cmp0, .fp_cvtf, .fp_cvtz => vr(v.n).with(vw(v.d)),
        .sra, .ins_shift, .narrow, .aese, .aesd, .sha256su0, .sha1su1, .sha512su0 => vr(v.n).with(vr(v.d)).with(vw(v.d)),
        .bsl, .long, .mla, .fp_mla, .sha1c, .sha1p, .sha1m, .sha1su0, .sha256h, .sha256h2, .sha256su1, .sha512h, .sha512h2, .sha512su1 => nm.with(vr(v.d)).with(vw(v.d)),
        .shuf => (switch (v.x) {
            0 => nm,
            1, 3 => vr(v.n),
            else => vr(v.n).with(vr(v.d)),
        }).with(vw(v.d)),
        .tbl => blk: {
            var u = vr(v.m).with(vw(v.d));
            if (v.x == 1) u = u.with(vr(v.d));
            for (0..v.sh) |j| u = u.with(vr(v.n + @as(u8, @intCast(j))));
            break :blk u;
        },
        .immlog => vr(v.d).with(vw(v.d)),
        .smov => vr(v.n).with(cm.wr(xs(v.d))),
        .insx => cm.rd(xs(v.n)).with(vr(v.d)).with(vw(v.d)),
        .ldpv, .stpv => blk: {
            var u = cm.rd(v.n);
            if (v.x == 1 or v.x == 3) u = u.with(cm.wr(v.n));
            break :blk u.with(if (v.k == .ldpv) vw(v.d).with(vw(v.m)) else vr(v.d).with(vr(v.m)));
        },
        .ld1, .st1, .ld1s, .st1s, .ld1r => blk: {
            var u = cm.rd(v.n); // Rn = 31 is SP (slot 31)
            if (v.x != 0) u = u.with(cm.wr(v.n));
            if (v.x == 2) u = u.with(cm.rd(v.m));
            for (0..v.sh) |j| u = u.with(if (v.k == .ld1 or v.k == .ld1r) vw(v.d + @as(u8, @intCast(j))) else vr(v.d + @as(u8, @intCast(j))));
            if (v.k == .ld1s) u = u.with(vr(v.d)).with(vw(v.d));
            break :blk u;
        },
    };
}

/// The op's feature is implemented and on (kw_set_isa) for e's CPU; else
/// the interpreter runs it (and raises ILLEGAL).
pub fn enabled(e: *Emit, v: V) bool {
    if (!e.simd) return false;
    const f = v.feat orelse return true;
    return c.implemented & c.featureBit(f) != 0 and e.cpu().priv.a64.isa & c.featureBit(f) != 0;
}

// ---- emission helpers ---------------------------------------------------------------

fn k128(e: *Emit, lo: u64, hi: u64) void {
    e.vop(cm.S.v128_const);
    for (0..8) |i| e.op(@truncate(lo >> @intCast(8 * i)));
    for (0..8) |i| e.op(@truncate(hi >> @intCast(8 * i)));
}
fn k32x4(e: *Emit, x: u32) void {
    const p: u64 = (@as(u64, x) << 32) | x;
    k128(e, p, p);
}
fn k64x2(e: *Emit, x: u64) void {
    k128(e, x, x);
}
fn shuffle(e: *Emit, idx: [16]u8) void {
    e.vop(cm.S.i8x16_shuffle);
    for (idx) |b| e.op(b);
}
fn get(e: *Emit, r: u8) void {
    cm.getV(e, r & 31);
}
/// Push Vm, or for a by-element op its element broadcast.
fn getM(e: *Emit, v: V) void {
    get(e, v.m);
    if (!v.bc) return;
    get(e, v.m);
    shuffle(e, v.bidx);
}
fn vop(e: *Emit, code: u16) void {
    std.debug.assert(code != none);
    e.vop(code);
}
/// Rotate every 32-bit (64-bit) lane of the v128 in local l right by k.
fn ror(e: *Emit, l: u32, k: u8, wide: bool) void {
    const w: u8 = if (wide) 64 else 32;
    e.get(l);
    e.k32(k);
    e.vop(if (wide) op_shr_u[3] else op_shr_u[2]);
    e.get(l);
    e.k32(w - k);
    e.vop(if (wide) op_shl[3] else op_shl[2]);
    e.vop(v128_or);
}
/// The v128 in local l shifted right (logically) by k in 32/64-bit lanes.
fn shrLanes(e: *Emit, l: u32, k: u8, wide: bool) void {
    e.get(l);
    e.k32(k);
    e.vop(if (wide) op_shr_u[3] else op_shr_u[2]);
}

const lq = cm.l_q;
const lw = cm.l_w;
const lu = cm.l_u;
const lt = cm.l_t;
/// The v128 scratch local that holds a pending AESE/AESD's state (`Ext`).
const l_aes = cm.l_q + cm.n_q - 1;

pub fn emit(e: *Emit, v: V, at: At) void {
    switch (v.k) {
        .bin => {
            get(e, v.n);
            getM(e, v);
            vop(e, v.code);
            cm.setVq(e, v.d, v.q);
        },
        .mla => {
            get(e, v.d);
            get(e, v.n);
            getM(e, v);
            vop(e, v.code);
            vop(e, v.code2);
            cm.setVq(e, v.d, v.q);
        },
        .shlreg => shlReg(e, v),
        .cmtst => {
            get(e, v.n);
            get(e, v.m);
            vop(e, v128_and);
            k128(e, 0, 0);
            vop(e, op_ne[v.size]);
            cm.setVq(e, v.d, v.q);
        },
        .cmhu64 => {
            const sign: u64 = 1 << 63;
            get(e, v.n);
            k128(e, sign, sign);
            vop(e, v128_xor);
            get(e, v.m);
            k128(e, sign, sign);
            vop(e, v128_xor);
            vop(e, if (v.x == 1) op_ge_s[3] else op_gt_s[3]);
            cm.setVq(e, v.d, v.q);
        },
        .bsl => {
            // bitselect(a, b, mask) = a & mask | b & ~mask
            switch (v.x) {
                0 => { // BSL: n where d, m elsewhere
                    get(e, v.n);
                    get(e, v.m);
                    get(e, v.d);
                },
                1 => { // BIT: n where m, d elsewhere
                    get(e, v.n);
                    get(e, v.d);
                    get(e, v.m);
                },
                else => { // BIF: d where m, n elsewhere
                    get(e, v.d);
                    get(e, v.n);
                    get(e, v.m);
                },
            }
            vop(e, v128_bitselect);
            cm.setVq(e, v.d, v.q);
        },
        .un => {
            get(e, v.n);
            vop(e, if (v.code == 0) v128_not else v.code);
            cm.setVq(e, v.d, v.q);
        },
        .cmz => {
            get(e, v.n);
            k128(e, 0, 0);
            vop(e, v.code);
            cm.setVq(e, v.d, v.q);
        },
        .shift => {
            if (v.x == 1) k128(e, 0, 0) else {
                get(e, v.n);
                if (v.sh != 0) {
                    e.k32(v.sh);
                    vop(e, v.code);
                }
            }
            cm.setVq(e, v.d, v.q);
        },
        .sra => {
            get(e, v.d);
            if (v.x != 1) {
                get(e, v.n);
                e.k32(v.sh);
                vop(e, v.code);
                vop(e, v.code2);
            }
            cm.setVq(e, v.d, v.q);
        },
        .ins_shift => {
            if (v.x == 1) get(e, v.d) else {
                get(e, v.n);
                if (v.sh != 0) {
                    e.k32(v.sh);
                    vop(e, v.code);
                }
                get(e, v.d);
                k128(e, v.imm, v.imm);
                vop(e, v128_bitselect);
            }
            cm.setVq(e, v.d, v.q);
        },
        .shll => {
            get(e, v.n);
            vop(e, v.code);
            if (v.sh != 0) {
                e.k32(v.sh);
                vop(e, v.code2);
            }
            cm.setV(e, v.d);
        },
        .narrow => {
            get(e, v.n);
            if (v.sh != 0) {
                e.k32(v.sh);
                vop(e, v.code);
            }
            e.set(lq);
            if (v.q) get(e, v.d) else e.get(lq);
            e.get(lq);
            shuffle(e, v.idx);
            cm.setVq(e, v.d, v.q);
        },
        .shuf => {
            switch (v.x) {
                0 => {
                    get(e, v.n);
                    get(e, v.m);
                },
                1 => {
                    get(e, v.n);
                    get(e, v.n);
                },
                2 => {
                    get(e, v.d);
                    get(e, v.n);
                },
                else => {
                    get(e, v.n);
                    k128(e, 0, 0);
                },
            }
            shuffle(e, v.idx);
            cm.setVq(e, v.d, v.q);
        },
        .tbl => {
            for (0..v.sh) |j| {
                get(e, v.n + @as(u8, @intCast(j)));
                get(e, v.m);
                if (j > 0) {
                    const k = c.replicate(16 * @as(u64, j), 8);
                    k128(e, k, k);
                    vop(e, op_sub[0]);
                }
                vop(e, i8x16_swizzle);
                if (j > 0) vop(e, v128_or);
            }
            if (v.x == 1) {
                get(e, v.d);
                get(e, v.m);
                const k = c.replicate(16 * @as(u64, v.sh), 8);
                k128(e, k, k);
                vop(e, i8x16_lt_u);
                vop(e, v128_bitselect);
            }
            cm.setVq(e, v.d, v.q);
        },
        .across => {
            const esz = @as(u8, 1) << v.size;
            var lanes: u8 = (if (v.q) @as(u8, 16) else 8) / esz;
            get(e, v.n);
            while (lanes > 1) {
                lanes /= 2;
                e.set(lq);
                e.get(lq);
                e.get(lq);
                e.get(lq);
                var idx: [16]u8 = undefined;
                for (&idx, 0..) |*b, i| b.* = @intCast((i + lanes * esz) % 16);
                shuffle(e, idx);
                vop(e, v.code);
            }
            e.vlane(switch (v.size) {
                0 => i8x16_extract_u,
                1 => i16x8_extract_u,
                else => i32x4_extract,
            }, 0);
            e.op(W.i64_ext_u);
            cm.setScalar(e, ops.v0_slot + 2 * v.d);
        },
        .pair => {
            const esz = @as(u8, 1) << v.size;
            const nl: u8 = (if (v.q) @as(u8, 16) else 8) / esz; // lanes per source
            for (0..2) |odd| {
                get(e, v.n);
                get(e, v.m);
                var idx: [16]u8 = undefined;
                for (&idx, 0..) |*b, i| {
                    const lane: u8 = @intCast(i / esz);
                    const byte: u8 = @intCast(i % esz);
                    const l = lane % nl; // Q = 0: the upper half repeats (zeroed)
                    const src = if (l < nl / 2) (2 * l + @as(u8, @intCast(odd))) * esz else 16 + (2 * (l - nl / 2) + @as(u8, @intCast(odd))) * esz;
                    b.* = src + byte;
                }
                shuffle(e, idx);
                if (odd == 0) e.set(lq);
            }
            e.set(lq + 1);
            e.get(lq);
            e.get(lq + 1);
            vop(e, v.code);
            cm.setVq(e, v.d, v.q);
        },
        .paird => {
            get(e, v.n);
            get(e, v.n);
            get(e, v.n);
            shuffle(e, .{ 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7 });
            vop(e, op_add[3]);
            cm.setV64(e, v.d);
        },
        .long => {
            switch (v.x) {
                0, 1, 2 => {
                    if (v.x != 0) get(e, v.d);
                    get(e, v.n);
                    getM(e, v);
                    vop(e, v.code);
                    if (v.x != 0) vop(e, v.code2);
                },
                3 => {
                    get(e, v.n);
                    vop(e, v.code);
                    get(e, v.m);
                    vop(e, v.code);
                    vop(e, v.code2);
                },
                else => {
                    get(e, v.n);
                    get(e, v.m);
                    vop(e, v.code);
                    vop(e, v.code2);
                },
            }
            cm.setV(e, v.d);
        },
        .immlog => {
            get(e, v.d);
            k128(e, v.imm, v.imm);
            vop(e, if (v.x == 1) v128_andnot else v128_or);
            cm.setVq(e, v.d, v.q);
        },
        .smov => {
            get(e, v.n);
            e.vlane(v.code, v.sh);
            e.op(if (v.x == 1) W.i64_ext_s else W.i64_ext_u);
            cm.set(e, xs(v.d));
        },
        .insx => {
            get(e, v.d);
            cm.get(e, xs(v.n));
            e.op(W.i32_wrap);
            e.vlane(v.code, v.sh);
            cm.setV(e, v.d);
        },
        .ld1, .st1 => ldst1(e, v, at),
        .ld1s, .st1s, .ld1r => ldstSingle(e, v, at),
        .ldpv, .stpv => pairEmit(e, v, at),
        .aese, .aesd => {
            // AddRoundKey now; the rest waits for an AESMC/AESIMC (`Ext`).
            get(e, v.d);
            get(e, v.n);
            vop(e, v128_xor);
            e.set(l_aes);
            e.x.vec.pend = v.d;
            e.x.vec.dec = v.k == .aesd;
        },
        .aesmc, .aesimc => {
            if (e.x.vec.pend == v.n and v.d == v.n and e.x.vec.dec == (v.k == .aesimc)) {
                // (Inv)SubBytes, (Inv)ShiftRows and (Inv)MixColumns of the
                // pending state: one T-table round.
                e.x.vec.pend = no_pend;
                columns(e, l_aes, if (v.k == .aesimc) tab_td else tab_te, if (v.k == .aesimc) inv_shift_rows else shift_rows);
            } else {
                get(e, v.n);
                e.set(lq);
                columns(e, lq, if (v.k == .aesimc) tab_im else tab_mc, identity);
            }
            cm.setV(e, v.d);
        },
        .pmull => pmull(e, v),
        .sha1h => {
            get(e, v.n);
            e.vlane(i32x4_extract, 0);
            e.k32(30);
            e.op(W.i32_rotl);
            e.op(W.i64_ext_u);
            cm.setScalar(e, ops.v0_slot + 2 * v.d);
        },
        .sha1c, .sha1p, .sha1m => sha1(e, v),
        .sha1su0 => {
            get(e, v.d);
            get(e, v.n);
            shuffle(e, .{ 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
            get(e, v.d);
            vop(e, v128_xor);
            get(e, v.m);
            vop(e, v128_xor);
            cm.setV(e, v.d);
        },
        .sha1su1 => {
            // T = d ^ (n >> 32); each word rotated left by 1, the top one
            // also xored with T<31:0> rotated left by 2
            get(e, v.d);
            get(e, v.n);
            k128(e, 0, 0);
            shuffle(e, .{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 });
            vop(e, v128_xor);
            e.set(lq);
            ror(e, lq, 31, false);
            e.get(lq);
            k128(e, 0, 0);
            shuffle(e, .{ 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 0, 1, 2, 3 });
            e.set(lq + 1);
            ror(e, lq + 1, 30, false);
            vop(e, v128_xor);
            cm.setV(e, v.d);
        },
        .sha256h, .sha256h2 => sha256(e, v),
        .sha256su0 => {
            // d + sigma0(n<31:0> : d<127:32>) per word
            get(e, v.d);
            get(e, v.n);
            shuffle(e, .{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 });
            e.set(lq);
            get(e, v.d);
            sigmaV(e, lq, .{ 7, 18, 3 }, false);
            vop(e, op_add[2]);
            cm.setV(e, v.d);
        },
        .sha256su1 => {
            // a = d + (m<31:0> : n<127:32>); words 0-1 + sigma1(m<127:64>),
            // words 2-3 + sigma1 of the new words 0-1
            get(e, v.d);
            get(e, v.n);
            get(e, v.m);
            shuffle(e, .{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 });
            vop(e, op_add[2]);
            e.set(lq);
            get(e, v.m);
            get(e, v.m);
            shuffle(e, .{ 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15 });
            e.set(lq + 1);
            e.get(lq);
            sigmaV(e, lq + 1, .{ 17, 19, 10 }, false);
            vop(e, op_add[2]);
            e.set(lq + 2); // words 0-1 right
            e.get(lq + 2);
            e.get(lq + 2);
            shuffle(e, .{ 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 });
            e.set(lq + 1);
            e.get(lq + 2);
            e.get(lq);
            sigmaV(e, lq + 1, .{ 17, 19, 10 }, false);
            vop(e, op_add[2]); // words 2-3 right
            shuffle(e, .{ 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31 });
            cm.setV(e, v.d);
        },
        .sha512h, .sha512h2 => sha512(e, v),
        .sha512su0 => {
            // (w0 + sigma0(w1), w1 + sigma0(x0))
            get(e, v.d);
            get(e, v.n);
            shuffle(e, .{ 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 });
            e.set(lq);
            get(e, v.d);
            sigmaV(e, lq, .{ 1, 8, 7 }, true);
            vop(e, op_add[3]);
            cm.setV(e, v.d);
        },
        .sha512su1 => {
            // d + sigma1(n) + m per lane
            get(e, v.n);
            e.set(lq);
            get(e, v.d);
            sigmaV(e, lq, .{ 19, 61, 6 }, true);
            vop(e, op_add[3]);
            get(e, v.m);
            vop(e, op_add[3]);
            cm.setV(e, v.d);
        },
        .fp_bin => emitFpBin(e, v, at),
        .fp_minmax => emitFpMinmax(e, v, at),
        .fp_mla => emitFpMla(e, v, at),
        .fp_sqrt => emitFpSqrt(e, v, at),
        .fp_cmp => emitFpCmp(e, v, at),
        .fp_cmp0 => emitFpCmp0(e, v, at),
        .fp_cvtf => emitFpCvtf(e, v, at),
        .fp_cvtz => emitFpCvtz(e, v, at),
    }
}

// ---- AdvSIMD FP fast paths --------------------------------------------------------

const fpcr_off = cm.off(ops.nzcv_slot) + 4;
const fpsr_off = fpcr_off + 4;
const fpcr_fast_mask: i32 = @bitCast(c.FPCR_FZ | (3 << c.FPCR_RMODE_SHIFT));

fn fpcrFast(e: *Emit) void {
    e.get(jc.l_x);
    e.addr();
    e.mem(W.i32_load, 2, fpcr_off);
    e.k32(fpcr_fast_mask);
    e.op(W.i32_and);
    e.op(W.i32_eqz);
}
fn fzClear(e: *Emit) void {
    e.get(jc.l_x);
    e.addr();
    e.mem(W.i32_load, 2, fpcr_off);
    e.k32(@bitCast(c.FPCR_FZ));
    e.op(W.i32_and);
    e.op(W.i32_eqz);
}
fn ixcSet(e: *Emit) void {
    e.get(jc.l_x);
    e.addr();
    e.mem(W.i32_load, 2, fpsr_off);
    e.k32(@ctz(c.FPSR_IXC));
    e.op(W.i32_shr_u);
    e.k32(1);
    e.op(W.i32_and);
}

/// Slow path for AdvSIMD FP: flush only the V registers `v` reads, run the
/// interpreter once, and reload `Vd` without disturbing unrelated locals.
fn vecFpSlow(e: *Emit, v: V, at: At) void {
    if (!e.table) {
        e.handBack(at.k, at.pc);
        return;
    }
    cm.storeV(e, v.n);
    if ((v.k == .fp_bin or v.k == .fp_minmax or v.k == .fp_mla or v.k == .fp_cmp) and v.m != v.n) cm.storeV(e, v.m);
    if (v.k == .fp_mla and v.d != v.n and v.d != v.m) cm.storeV(e, v.d);
    e.get(jc.l_x);
    e.get(jc.l_cpu);
    e.k64(v.imm);
    e.k64(at.pc);
    e.callTable(@intFromPtr(&opexec.jitGeneric));
    e.op(W.drop);
    cm.loadV(e, v.d);
}

/// Push Vn (`m = false`) or getM (`m = true`), replacing lanes 2..3 with
/// `neutral` when Q = 0 so inactive high lanes never poison vector checks.
fn normOp(e: *Emit, v: V, m: bool, neutral: u32) void {
    if (m) getM(e, v) else get(e, v.n);
    if (v.q) return;
    k32x4(e, neutral);
    shuffle(e, .{ 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31 });
}

fn inRangeMask(e: *Emit, l: u32, dbl: bool) void {
    if (dbl) {
        e.get(l);
        e.k32(1);
        e.vop(cm.S.i64x2_shl);
        e.k32(53);
        e.vop(cm.S.i64x2_shr_u);
        e.set(lq + 5);
        e.get(lq + 5);
        k64x2(e, 1);
        e.vop(cm.S.i64x2_gt_s);
        e.get(lq + 5);
        k64x2(e, 2046);
        e.vop(cm.S.i64x2_lt_s);
        e.vop(v128_and);
    } else {
        e.get(l);
        e.k32(1);
        e.vop(cm.S.i32x4_shl);
        e.k32(24);
        e.vop(cm.S.i32x4_shr_u);
        k32x4(e, 2);
        e.vop(cm.S.i32x4_sub);
        k32x4(e, 252);
        e.vop(cm.S.i32x4_lt_u);
    }
}

fn isZeroMask(e: *Emit, l: u32, dbl: bool) void {
    e.get(l);
    e.k32(1);
    e.vop(if (dbl) cm.S.i64x2_shl else cm.S.i32x4_shl);
    k128(e, 0, 0);
    e.vop(if (dbl) cm.S.i64x2_eq else cm.S.i32x4_eq);
}

fn emitFpBin(e: *Emit, v: V, at: At) void {
    const dbl = v.size == 3;
    const a_loc = lq;
    const b_loc = lq + 1;
    const r_loc = lq + 2;
    normOp(e, v, false, 0x4080_0000); // 4.0f
    e.set(a_loc);
    normOp(e, v, true, 0x3f80_0000); // 1.0f
    e.set(b_loc);
    e.get(a_loc);
    e.get(b_loc);
    vop(e, v.code);
    e.set(r_loc);
    inRangeMask(e, r_loc, dbl);
    isZeroMask(e, r_loc, dbl);
    if (v.x == 2) {
        isZeroMask(e, a_loc, dbl);
        isZeroMask(e, b_loc, dbl);
        e.vop(v128_or);
        e.vop(v128_and);
    } else if (v.x == 3) {
        isZeroMask(e, a_loc, dbl);
        e.vop(v128_and);
    }
    e.vop(v128_or);
    e.vop(if (dbl) cm.S.i64x2_all_true else cm.S.i32x4_all_true);
    fpcrFast(e);
    e.op(W.i32_and);
    ixcSet(e);
    e.op(W.i32_and);
    e.open(W.@"if", W.void_t);
    e.get(r_loc);
    cm.setVq(e, v.d, v.q);
    e.op(W.@"else");
    vecFpSlow(e, v, at);
    e.close();
}

fn emitFpSqrt(e: *Emit, v: V, at: At) void {
    const dbl = v.size == 3;
    const r_loc = lq + 1;
    normOp(e, v, false, 0x4080_0000);
    vop(e, v.code);
    e.set(r_loc);
    inRangeMask(e, r_loc, dbl);
    isZeroMask(e, r_loc, dbl);
    e.vop(v128_or);
    e.vop(if (dbl) cm.S.i64x2_all_true else cm.S.i32x4_all_true);
    fpcrFast(e);
    e.op(W.i32_and);
    ixcSet(e);
    e.op(W.i32_and);
    e.open(W.@"if", W.void_t);
    e.get(r_loc);
    cm.setVq(e, v.d, v.q);
    e.op(W.@"else");
    vecFpSlow(e, v, at);
    e.close();
}

fn emitFpMinmax(e: *Emit, v: V, at: At) void {
    const dbl = v.size == 3;
    const r_loc = lq;
    normOp(e, v, false, 0x4080_0000);
    normOp(e, v, true, 0x3f80_0000);
    vop(e, v.code);
    e.set(r_loc);
    e.get(r_loc);
    e.get(r_loc);
    e.vop(if (dbl) cm.S.f64x2_eq else cm.S.f32x4_eq);
    e.vop(if (dbl) cm.S.i64x2_all_true else cm.S.i32x4_all_true);
    fzClear(e);
    e.op(W.i32_and);
    e.open(W.@"if", W.void_t);
    e.get(r_loc);
    cm.setVq(e, v.d, v.q);
    e.op(W.@"else");
    vecFpSlow(e, v, at);
    e.close();
}

fn emitFpCmp(e: *Emit, v: V, at: At) void {
    const dbl = v.size == 3;
    const eq: u32 = if (dbl) cm.S.f64x2_eq else cm.S.f32x4_eq;
    const a_loc = lq;
    const b_loc = lq + 1;
    normOp(e, v, false, 0x4080_0000);
    e.set(a_loc);
    normOp(e, v, true, 0x3f80_0000);
    e.set(b_loc);
    e.get(a_loc);
    e.get(a_loc);
    e.vop(eq);
    e.get(b_loc);
    e.get(b_loc);
    e.vop(eq);
    e.vop(v128_and);
    e.vop(if (dbl) cm.S.i64x2_all_true else cm.S.i32x4_all_true);
    fzClear(e);
    e.op(W.i32_and);
    e.open(W.@"if", W.void_t);
    e.get(a_loc);
    e.get(b_loc);
    vop(e, v.code);
    cm.setVq(e, v.d, v.q);
    e.op(W.@"else");
    vecFpSlow(e, v, at);
    e.close();
}

fn emitFpCmp0(e: *Emit, v: V, at: At) void {
    const dbl = v.size == 3;
    const eq: u32 = if (dbl) cm.S.f64x2_eq else cm.S.f32x4_eq;
    const a_loc = lq;
    normOp(e, v, false, 0x4080_0000);
    e.set(a_loc);
    e.get(a_loc);
    e.get(a_loc);
    e.vop(eq);
    e.vop(if (dbl) cm.S.i64x2_all_true else cm.S.i32x4_all_true);
    fzClear(e);
    e.op(W.i32_and);
    e.open(W.@"if", W.void_t);
    e.get(a_loc);
    k128(e, 0, 0);
    vop(e, v.code);
    cm.setVq(e, v.d, v.q);
    e.op(W.@"else");
    vecFpSlow(e, v, at);
    e.close();
}

fn pushHalfF64(e: *Emit, l: u32, hi: bool) void {
    e.get(l);
    if (hi) {
        k128(e, 0, 0);
        shuffle(e, .{ 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7 });
    }
    e.vop(cm.S.f64x2_promote_low_f32x4);
}

/// Compute d ± a * b for one pair of single lanes via f64 (exact product),
/// demote to r_loc (upper 64 bits zeroed), and push the i64x2 "bit-exact" mask.
fn emitMlaPair(e: *Emit, l_a: u32, l_b: u32, l_d: u32, hi: bool, sub: bool, s_loc: u32, r_loc: u32) void {
    pushHalfF64(e, l_d, hi);
    pushHalfF64(e, l_a, hi);
    if (sub) e.vop(cm.S.f64x2_neg);
    pushHalfF64(e, l_b, hi);
    e.vop(cm.S.f64x2_mul);
    e.vop(cm.S.f64x2_add);
    e.tee(s_loc);
    e.vop(cm.S.f32x4_demote_f64x2_zero);
    e.set(r_loc);
    e.get(s_loc);
    e.k32(1);
    e.vop(cm.S.i64x2_shl);
    e.k32(53);
    e.vop(cm.S.i64x2_shr_u);
    e.set(lq + 5);
    e.get(lq + 5);
    k64x2(e, 897);
    e.vop(cm.S.i64x2_gt_s);
    e.get(lq + 5);
    k64x2(e, 1150);
    e.vop(cm.S.i64x2_lt_s);
    e.vop(v128_and);
    e.get(s_loc);
    k64x2(e, 0x1fff_ffff);
    e.vop(v128_and);
    k64x2(e, 0x1000_0000);
    e.vop(cm.S.i64x2_ne);
    e.vop(v128_and);
    isZeroMask(e, s_loc, true);
    e.vop(v128_or);
}

fn emitFpMla(e: *Emit, v: V, at: At) void {
    const sub = v.x == 1;
    const m_loc = lq;
    const s_loc = lq + 1;
    const r0_loc = lq + 2;
    const r1_loc = lq + 3;
    const ok_loc = lq + 4;
    const a_loc = cm.vloc(v.n & 31);
    const d_loc = cm.vloc(v.d & 31);
    getM(e, v);
    e.set(m_loc);
    emitMlaPair(e, a_loc, m_loc, d_loc, false, sub, s_loc, r0_loc);
    if (v.q) {
        e.set(ok_loc);
        emitMlaPair(e, a_loc, m_loc, d_loc, true, sub, s_loc, r1_loc);
        e.get(ok_loc);
        e.vop(v128_and);
    }
    e.vop(cm.S.i64x2_all_true);
    fpcrFast(e);
    e.op(W.i32_and);
    ixcSet(e);
    e.op(W.i32_and);
    e.open(W.@"if", W.void_t);
    e.get(r0_loc);
    if (v.q) {
        e.get(r1_loc);
        shuffle(e, .{ 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23 });
    }
    cm.setV(e, v.d);
    e.op(W.@"else");
    vecFpSlow(e, v, at);
    e.close();
}

fn emitFpCvtf(e: *Emit, v: V, at: At) void {
    fpcrFast(e);
    ixcSet(e);
    e.op(W.i32_and);
    e.open(W.@"if", W.void_t);
    get(e, v.n);
    vop(e, v.code);
    cm.setVq(e, v.d, v.q);
    e.op(W.@"else");
    vecFpSlow(e, v, at);
    e.close();
}

fn emitFpCvtz(e: *Emit, v: V, at: At) void {
    const a_loc = lq;
    normOp(e, v, false, 0x4080_0000);
    e.set(a_loc);
    e.get(a_loc);
    if (v.x == 1) {
        k32x4(e, 0xbf80_0000); // -1.0f
        e.vop(cm.S.f32x4_gt);
        e.get(a_loc);
        k32x4(e, 0x4f80_0000); // 2^32
        e.vop(cm.S.f32x4_lt);
    } else {
        k32x4(e, 0xcf00_0000); // -2^31
        e.vop(cm.S.f32x4_ge);
        e.get(a_loc);
        k32x4(e, 0x4f00_0000); // 2^31
        e.vop(cm.S.f32x4_lt);
    }
    e.vop(v128_and);
    e.vop(cm.S.i32x4_all_true);
    fzClear(e);
    e.op(W.i32_and);
    ixcSet(e);
    e.op(W.i32_and);
    e.open(W.@"if", W.void_t);
    e.get(a_loc);
    vop(e, v.code);
    cm.setVq(e, v.d, v.q);
    e.op(W.@"else");
    vecFpSlow(e, v, at);
    e.close();
}

/// Push ror(x, k0) ^ ror(x, k1) ^ (x >> k2) per 32/64-bit lane of local l.
fn sigmaV(e: *Emit, l: u32, k: [3]u8, wide: bool) void {
    ror(e, l, k[0], wide);
    ror(e, l, k[1], wide);
    vop(e, v128_xor);
    shrLanes(e, l, k[2], wide);
    vop(e, v128_xor);
}

// ---- LD1, ST1 (multiple structures) ---------------------------------------------------

fn ldst1(e: *Emit, v: V, at: At) void {
    const per: u64 = if (v.q) 16 else 8;
    const load = v.k == .ld1;
    const a = mem.prepareHost(e, at, v.n, 0, !load, per * v.sh);
    for (0..v.sh) |j| {
        const r: u8 = @intCast((v.d + j) & 31);
        e.get(a.local);
        e.addr();
        if (load) {
            e.vop(if (v.q) cm.S.v128_load else cm.S.v128_load64_zero);
            e.uleb(0);
            e.uleb(a.off + per * j);
            cm.setV(e, r);
        } else {
            get(e, r);
            e.vop(if (v.q) cm.S.v128_store else cm.S.v128_store64_lane);
            e.uleb(0);
            e.uleb(a.off + per * j);
            if (!v.q) e.op(0); // lane
        }
    }
    if (v.x != 0) {
        cm.get(e, v.n);
        if (v.x == 1) e.k64(v.imm) else cm.get(e, v.m);
        e.op(W.i64_add);
        cm.set(e, v.n);
    }
}

/// LDP/STP of S, D, Q registers: one TLB check over both, loads zero the
/// rest of each register; the writeback after the accesses (the base is
/// an X register, the data V registers: no overlap); Rt2 written after Rt.
fn pairEmit(e: *Emit, v: V, at: At) void {
    const n: u64 = @as(u64, 1) << v.size; // bytes per register (4 << (size - 2))
    const load = v.k == .ldpv;
    const a = mem.prepareHost(e, at, v.n, if (v.x != 1) v.imm else 0, !load, 2 * n);
    for (0..2) |j| {
        const r = if (j == 0) v.d else v.m;
        e.get(a.local);
        e.addr();
        if (load) {
            e.vop(switch (v.size) {
                2 => cm.S.v128_load32_zero,
                3 => cm.S.v128_load64_zero,
                else => cm.S.v128_load,
            });
            e.uleb(0);
            e.uleb(a.off + n * j);
            cm.setV(e, r);
        } else {
            get(e, r);
            e.vop(switch (v.size) {
                2 => cm.S.v128_store32_lane,
                3 => cm.S.v128_store64_lane,
                else => cm.S.v128_store,
            });
            e.uleb(0);
            e.uleb(a.off + n * j);
            if (v.size != 4) e.op(0);
        }
    }
    if (v.x == 1 or v.x == 3) {
        cm.get(e, v.n);
        e.k64(v.imm);
        e.op(W.i64_add);
        cm.set(e, v.n);
    }
}

/// LD1/ST1 of one lane, LD1R: one access of 1 << size bytes.
fn ldstSingle(e: *Emit, v: V, at: At) void {
    const n: u64 = @as(u64, 1) << v.size;
    const off = mem.prepareAddr(e, at, v.n, 0, v.k == .st1s, n);
    switch (v.k) {
        .ld1s => {
            get(e, v.d);
            e.vop(@as(u32, 0x54) + v.size); // v128.load{8,16,32,64}_lane
            e.uleb(0);
            e.uleb(off);
            e.op(v.idx[0]);
            cm.setV(e, v.d);
        },
        .st1s => {
            get(e, v.d);
            e.vop(@as(u32, 0x58) + v.size); // v128.store{8,16,32,64}_lane
            e.uleb(0);
            e.uleb(off);
            e.op(v.idx[0]);
        },
        else => {
            e.vop(@as(u32, 0x07) + v.size); // v128.load{8,16,32,64}_splat
            e.uleb(0);
            e.uleb(off);
            cm.setVq(e, v.d, v.q);
        },
    }
    if (v.x != 0) {
        cm.get(e, v.n);
        if (v.x == 1) e.k64(v.imm) else cm.get(e, v.m);
        e.op(W.i64_add);
        cm.set(e, v.n);
    }
}

// ---- AES ---------------------------------------------------------------------------

/// Emission state: an AESE or AESD whose AddRoundKey is done (the state in
/// `l_aes`) and whose (Inv)ShiftRows and (Inv)SubBytes are not, pending for
/// register `pend`: the next op, if it is AESMC (AESIMC) of that register
/// into itself, does all three in one T-table round; anything else writes
/// the pending register first (`flush`: before the next op, and on every
/// path out, `sync`). Nothing can leave between the two ops of a pair (both
/// are translated, neither can fault), so the fused pair is exact.
pub const Ext = struct {
    pend: u8 = no_pend,
    dec: bool = false,
};
const no_pend: u8 = 0xff;

pub fn beginMember(e: *Emit) void {
    e.x.vec = .{};
}

/// Whether op v consumes the pending AESE/AESD (emit.zig: else `flush`).
pub fn consumes(e: *Emit, v: V) bool {
    return (v.k == .aesmc or v.k == .aesimc) and e.x.vec.pend == v.n and v.d == v.n and e.x.vec.dec == (v.k == .aesimc);
}

/// Write the pending register (SubBytes and ShiftRows of the state), on
/// this path only (`sync`), or for good (`flush`).
pub fn sync(e: *Emit) void {
    const p = e.x.vec.pend;
    if (p == no_pend) return;
    // out byte i = S[state byte perm[i]], into a v128 lane by lane
    const dec_ = e.x.vec.dec;
    const perm = if (dec_) inv_shift_rows else shift_rows;
    k128(e, 0, 0);
    for (0..16) |i| {
        tabAddr(e, l_aes, perm[i], 0);
        e.mem(0x2d, 0, aes_off + (if (dec_) tab_is else tab_s)); // i32.load8_u
        e.vlane(i8x16_replace, @intCast(i));
    }
    cm.setV(e, p);
}
pub fn flush(e: *Emit) void {
    sync(e);
    e.x.vec.pend = no_pend;
}

const aes_off: u64 = jc.priv_off + @offsetOf(@FieldType(Cpu, "priv"), "jit") + @offsetOf(jc.State, "aes");
const tab_te: u64 = @offsetOf(jc.AesTab, "te");
const tab_td: u64 = @offsetOf(jc.AesTab, "td");
const tab_im: u64 = @offsetOf(jc.AesTab, "im");
const tab_mc: u64 = @offsetOf(jc.AesTab, "mc");
const tab_s: u64 = @offsetOf(jc.AesTab, "s");
const tab_is: u64 = @offsetOf(jc.AesTab, "is");

/// ShiftRows: out byte r + 4c = in byte r + 4((c + r) % 4); inverse c - r
/// (crypto.zig).
const shift_rows = rowShift(1);
const inv_shift_rows = rowShift(3);
const identity = rowShift(0);
fn rowShift(comptime dir: u32) [16]u8 {
    var m: [16]u8 = undefined;
    for (0..16) |i| m[i] = @intCast(i % 4 + 4 * ((i / 4 + dir * (i % 4)) % 4));
    return m;
}

/// Push the address operand of entry (byte `b` of the v128 in local l) of a
/// table with entries of 2^shift bytes, relative to aes_off (cpu-relative).
fn tabAddr(e: *Emit, l: u32, b: u8, shift: u5) void {
    e.get(l);
    e.vlane(i8x16_extract_u, b);
    if (shift != 0) {
        e.k32(shift);
        e.op(W.i32_shl);
    }
    if (e.mem32) {
        e.get(jc.l_cpu);
        e.op(W.i32_wrap);
        e.op(0x6a); // i32.add
    } else {
        e.op(W.i64_ext_u);
        e.get(jc.l_cpu);
        e.op(W.i64_add);
    }
}

/// Push the v128 whose column c is XOR over rows r of rotl(tab[state byte
/// perm[4c + r]], 8r) (state: the v128 in local l): SubBytes + MixColumns
/// (te), InvSubBytes + InvMixColumns (td), MixColumns (mc), InvMixColumns
/// (im) after the row shift `perm`.
fn columns(e: *Emit, l: u32, tab: u64, perm: [16]u8) void {
    for (0..4) |col| {
        for (0..4) |r| {
            tabAddr(e, l, perm[4 * col + r], 2);
            e.mem(W.i32_load, 2, aes_off + tab);
            if (r > 0) {
                e.k32(@intCast(8 * r));
                e.op(W.i32_rotl);
                e.op(W.i32_xor);
            }
        }
        e.set(lw + @as(u32, @intCast(col)));
    }
    buildI32x4(e, .{ lw, lw + 1, lw + 2, lw + 3 });
}

/// Push the v128 of the four i32 locals (lane 0 first).
fn buildI32x4(e: *Emit, l: [4]u32) void {
    e.get(l[0]);
    e.vop(i32x4_splat);
    for (1..4) |i| {
        e.get(l[i]);
        e.vlane(i32x4_replace, @intCast(i));
    }
}

// ---- PMULL (64-bit lanes) -------------------------------------------------------------

/// Push clmul(x, y) of two values below 2^32 (i64 locals xa ^ xb, ya ^ yb;
/// xb, yb: none = 0xffff_ffff): core.bitmanip.clmul's sequence on 32-bit
/// operands (each column of a product has at most 8 terms, and the 64-bit
/// product is exact). Scratch: l_t + 0..7.
fn clmul32(e: *Emit, xa: u32, xb: u32, ya: u32, yb: u32) void {
    const m0: u64 = 0x1111_1111;
    for (0..2) |s| {
        for (0..4) |i| {
            e.get(if (s == 0) xa else ya);
            const b = if (s == 0) xb else yb;
            if (b != 0xffff_ffff) {
                e.get(b);
                e.op(W.i64_xor);
            }
            e.k64(m0 << @intCast(i));
            e.op(W.i64_and);
            e.set(lt + @as(u32, @intCast(4 * s + i)));
        }
    }
    const mk: u64 = 0x1111_1111_1111_1111;
    for (0..4) |k| {
        for (0..4) |i| {
            e.get(lt + @as(u32, @intCast(i)));
            e.get(lt + 4 + @as(u32, @intCast((k + 4 - i) % 4)));
            e.op(W.i64_mul);
            if (i > 0) e.op(W.i64_xor);
        }
        e.k64(mk << @intCast(k));
        e.op(W.i64_and);
        if (k > 0) e.op(W.i64_or);
    }
}

/// PMULL/PMULL2 .1Q: the 128-bit carry-less product of lane x of Vn and Vm,
/// Karatsuba over 32-bit halves: lo*lo, hi*hi, (lo^hi)*(lo^hi).
fn pmull(e: *Emit, v: V) void {
    const nf = 0xffff_ffff;
    for (0..2) |s| {
        get(e, if (s == 0) v.n else v.m);
        e.vlane(i64x2_extract, v.x);
        e.set(lu + 4);
        e.get(lu + 4);
        e.k64(0xffff_ffff);
        e.op(W.i64_and);
        e.set(lu + 2 * @as(u32, @intCast(s)));
        e.get(lu + 4);
        e.k64(32);
        e.op(W.i64_shr_u);
        e.set(lu + 2 * @as(u32, @intCast(s)) + 1);
    }
    // lu + 0, 1: a0, a1; lu + 2, 3: b0, b1
    clmul32(e, lu, nf, lu + 2, nf);
    e.set(lu + 4); // L
    clmul32(e, lu + 1, nf, lu + 3, nf);
    e.set(lu + 5); // H
    clmul32(e, lu, lu + 1, lu + 2, lu + 3);
    e.get(lu + 4);
    e.op(W.i64_xor);
    e.get(lu + 5);
    e.op(W.i64_xor);
    e.set(lu + 6); // M = mid ^ L ^ H
    e.get(lu + 4);
    e.get(lu + 6);
    e.k64(32);
    e.op(W.i64_shl);
    e.op(W.i64_xor);
    e.vop(i64x2_splat);
    e.get(lu + 5);
    e.get(lu + 6);
    e.k64(32);
    e.op(W.i64_shr_u);
    e.op(W.i64_xor);
    e.vlane(i64x2_replace, 1);
    cm.setV(e, v.d);
}

// ---- SHA1, SHA256, SHA512 -------------------------------------------------------------

/// Extract the four 32-bit lanes of V register r into i32 locals l[0..3].
fn lanes32(e: *Emit, r: u8, l: [4]u32) void {
    for (0..4) |i| {
        get(e, r);
        e.vlane(i32x4_extract, @intCast(i));
        e.set(l[i]);
    }
}

/// SHA1C, SHA1P, SHA1M (crypto.sha1Hash): four rounds on X = Vd (a, b, c,
/// d), Y = Sn (e), W = Vm. Locals: the five state words rotate through
/// lw + 0..4 by renaming; W in lw + 8..11.
fn sha1(e: *Emit, v: V) void {
    var x = [4]u32{ lw, lw + 1, lw + 2, lw + 3 };
    var y: u32 = lw + 4;
    const w = [4]u32{ lw + 8, lw + 9, lw + 10, lw + 11 };
    lanes32(e, v.d, x);
    get(e, v.n);
    e.vlane(i32x4_extract, 0);
    e.set(y);
    lanes32(e, v.m, w);
    for (0..4) |r| {
        // y += rotl(x0, 5) + f(x1, x2, x3) + w[r]
        e.get(y);
        e.get(x[0]);
        e.k32(5);
        e.op(W.i32_rotl);
        e.op(0x6a); // i32.add
        switch (v.k) {
            .sha1c => { // ((x2 ^ x3) & x1) ^ x3
                e.get(x[2]);
                e.get(x[3]);
                e.op(W.i32_xor);
                e.get(x[1]);
                e.op(W.i32_and);
                e.get(x[3]);
                e.op(W.i32_xor);
            },
            .sha1p => {
                e.get(x[1]);
                e.get(x[2]);
                e.op(W.i32_xor);
                e.get(x[3]);
                e.op(W.i32_xor);
            },
            else => { // (x1 & x2) | ((x1 | x2) & x3)
                e.get(x[1]);
                e.get(x[2]);
                e.op(W.i32_and);
                e.get(x[1]);
                e.get(x[2]);
                e.op(W.i32_or);
                e.get(x[3]);
                e.op(W.i32_and);
                e.op(W.i32_or);
            },
        }
        e.op(0x6a);
        e.get(w[r]);
        e.op(0x6a);
        e.set(y);
        e.get(x[1]);
        e.k32(30);
        e.op(W.i32_rotl);
        e.set(x[1]);
        // <Y, X> = ROL(Y:X, 32), through a copy (an aggregate assigned to
        // x while it reads x would see its own writes)
        const nx = [4]u32{ y, x[0], x[1], x[2] };
        y = x[3];
        x = nx;
    }
    buildI32x4(e, x);
    cm.setV(e, v.d);
}

/// Push rotr(l, k0) ^ rotr(l, k1) ^ rotr(l, k2) (i32 local l).
fn bigSigma32(e: *Emit, l: u32, k: [3]u8) void {
    for (k, 0..) |r, i| {
        e.get(l);
        e.k32(r);
        e.op(W.i32_rotr);
        if (i > 0) e.op(W.i32_xor);
    }
}

/// SHA256H, SHA256H2 (crypto.sha256Hash): four rounds on X = (a, b, c, d),
/// Y = (e, f, g, h); H: X = Vd, Y = Vn, result X; H2: X = Vn, Y = Vd,
/// result Y. Locals: X lw + 0..3, Y lw + 4..7 (renamed per round), W
/// lw + 8..11, t1 lw + 12.
fn sha256(e: *Emit, v: V) void {
    const h1 = v.k == .sha256h;
    var x = [4]u32{ lw, lw + 1, lw + 2, lw + 3 };
    var y = [4]u32{ lw + 4, lw + 5, lw + 6, lw + 7 };
    const w = [4]u32{ lw + 8, lw + 9, lw + 10, lw + 11 };
    const t = lw + 12;
    lanes32(e, if (h1) v.d else v.n, x);
    lanes32(e, if (h1) v.n else v.d, y);
    lanes32(e, v.m, w);
    for (0..4) |r| {
        // t1 = y3 + Sigma1(y0) + Ch(y0, y1, y2) + w[r]
        e.get(y[3]);
        bigSigma32(e, y[0], .{ 6, 11, 25 });
        e.op(0x6a);
        e.get(y[1]);
        e.get(y[2]);
        e.op(W.i32_xor);
        e.get(y[0]);
        e.op(W.i32_and);
        e.get(y[2]);
        e.op(W.i32_xor);
        e.op(0x6a);
        e.get(w[r]);
        e.op(0x6a);
        e.tee(t);
        // x3 += t1
        e.get(x[3]);
        e.op(0x6a);
        e.set(x[3]);
        // y3 = t1 + Sigma0(x0) + Maj(x0, x1, x2)
        e.get(t);
        bigSigma32(e, x[0], .{ 2, 13, 22 });
        e.op(0x6a);
        e.get(x[0]);
        e.get(x[1]);
        e.op(W.i32_and);
        e.get(x[0]);
        e.get(x[1]);
        e.op(W.i32_or);
        e.get(x[2]);
        e.op(W.i32_and);
        e.op(W.i32_or);
        e.op(0x6a);
        e.set(y[3]);
        // <Y, X> = ROL(Y:X, 32)
        const nx = [4]u32{ y[3], x[0], x[1], x[2] };
        const ny = [4]u32{ x[3], y[0], y[1], y[2] };
        x = nx;
        y = ny;
    }
    buildI32x4(e, if (h1) x else y);
    cm.setV(e, v.d);
}

/// Push rotr(l, k0) ^ rotr(l, k1) ^ rotr(l, k2) (i64 local l).
fn bigSigma64(e: *Emit, l: u32, k: [3]u8) void {
    for (k, 0..) |r, i| {
        e.get(l);
        e.k64(r);
        e.op(W.i64_rotr);
        if (i > 0) e.op(W.i64_xor);
    }
}

/// SHA512H, SHA512H2 (crypto.sha512h, sha512h2; d = W, n = X, m = Y).
/// Locals: w0, w1, x0, x1, y0, y1 in lu + 0..5, hi lu + 6, tmp lu + 7.
fn sha512(e: *Emit, v: V) void {
    const src = [6]struct { u8, u8 }{ .{ v.d, 0 }, .{ v.d, 1 }, .{ v.n, 0 }, .{ v.n, 1 }, .{ v.m, 0 }, .{ v.m, 1 } };
    for (src, 0..) |s, i| {
        get(e, s[0]);
        e.vlane(i64x2_extract, s[1]);
        e.set(lu + @as(u32, @intCast(i)));
    }
    const w0 = lu;
    const w1 = lu + 1;
    const x0 = lu + 2;
    const x1 = lu + 3;
    const y0 = lu + 4;
    const y1 = lu + 5;
    const hi = lu + 6;
    const tmp = lu + 7;
    if (v.k == .sha512h) {
        // hi = Ch(y1, x0, x1) + Sigma1(y1) + w1
        ch64(e, y1, x0, x1);
        bigSigma64(e, y1, .{ 14, 18, 41 });
        e.op(W.i64_add);
        e.get(w1);
        e.op(W.i64_add);
        e.tee(hi);
        e.get(y0);
        e.op(W.i64_add);
        e.set(tmp);
        // lo = Ch(tmp, y1, x0) + Sigma1(tmp) + w0
        ch64(e, tmp, y1, x0);
        bigSigma64(e, tmp, .{ 14, 18, 41 });
        e.op(W.i64_add);
        e.get(w0);
        e.op(W.i64_add);
    } else {
        // hi = Maj(x0, y1, y0) + Sigma0(y0) + w1
        maj64(e, x0, y1, y0);
        bigSigma64(e, y0, .{ 28, 34, 39 });
        e.op(W.i64_add);
        e.get(w1);
        e.op(W.i64_add);
        e.set(hi);
        // lo = Maj(hi, y0, y1) + Sigma0(hi) + w0
        maj64(e, hi, y0, y1);
        bigSigma64(e, hi, .{ 28, 34, 39 });
        e.op(W.i64_add);
        e.get(w0);
        e.op(W.i64_add);
    }
    e.vop(i64x2_splat);
    e.get(hi);
    e.vlane(i64x2_replace, 1);
    cm.setV(e, v.d);
}
/// Push ((y ^ z) & x) ^ z.
fn ch64(e: *Emit, x: u32, y: u32, z: u32) void {
    e.get(y);
    e.get(z);
    e.op(W.i64_xor);
    e.get(x);
    e.op(W.i64_and);
    e.get(z);
    e.op(W.i64_xor);
}
/// Push (x & y) | ((x | y) & z).
fn maj64(e: *Emit, x: u32, y: u32, z: u32) void {
    e.get(x);
    e.get(y);
    e.op(W.i64_and);
    e.get(x);
    e.get(y);
    e.op(W.i64_or);
    e.get(z);
    e.op(W.i64_and);
    e.op(W.i64_or);
}

// ---- tests (kindtest.zig: against the reference, compiled eagerly in the wasm64 unit tests) ----

const kt = @import("kindtest.zig");

fn vreg(r: std.Random) u32 {
    // a few registers often, so that results feed later ops
    return if (r.uintLessThan(u32, 4) == 0) r.uintLessThan(u32, 32) else r.uintLessThan(u32, 6);
}
fn pick(r: std.Random, comptime T: type, xs_: []const T) T {
    return xs_[r.uintLessThan(usize, xs_.len)];
}

/// An edge-valued vector into a register: MOVI (bytes 0x00/0xff, Q at
/// random), MOVI/MVNI of a shifted byte (0x80, 0x7f, 0x01, 0xfe, ...).
fn edgeV(r: std.Random) u32 {
    const vd = vreg(r);
    const q = r.uintLessThan(u32, 2) << 30;
    if (r.boolean()) return 0x6f00_e400 | q | (r.uintLessThan(u32, 8) << 16) | (r.uintLessThan(u32, 32) << 5) | vd; // movi .2d, #bytes
    const imm8 = pick(r, u32, &.{ 0x80, 0x7f, 0x01, 0xfe, 0xff, 0x00, 0x40 });
    const cmode = pick(r, u32, &.{ 0, 2, 4, 6, 8, 10, 14 }); // 32-bit shifted, 16-bit shifted, 8-bit
    return 0x0f00_0400 | q | (r.uintLessThan(u32, 2) << 29) | ((imm8 >> 5) << 16) | (cmode << 12) | ((imm8 & 31) << 5) | vd;
}

/// Words of the AdvSIMD classes vec.zig translates, with random fields
/// (sizes, Q, U, opcodes of the class: also reserved and untranslated
/// neighbours, which are interpreter calls or ILLEGAL), between edge values.
fn genArith(r: std.Random, out: []u32, pc: u64) usize {
    _ = pc;
    const vd = vreg(r);
    const vn = vreg(r);
    const vm = vreg(r);
    const q = r.uintLessThan(u32, 2) << 30;
    const u = r.uintLessThan(u32, 2) << 29;
    const size = r.uintLessThan(u32, 4) << 22;
    const rr = (vm << 16) | (vn << 5) | vd;
    out[0] = switch (r.uintLessThan(u32, 13)) {
        0, 1 => 0x0e20_0400 | q | u | size | (pick(r, u32, &.{ 0b10000, 0b10001, 0b00110, 0b00111, 0b01100, 0b01101, 0b10011, 0b00010, 0b10100, 0b10101, 0b10111, 0b01000, r.uintLessThan(u32, 24) }) << 11) | rr, // three same
        2 => 0x5e20_0400 | u | (if (r.uintLessThan(u32, 4) == 0) size else 3 << 22) | (pick(r, u32, &.{ 0b10000, 0b10001, 0b00110, 0b00111, 0b01100, 0b01000 }) << 11) | rr, // scalar three same
        3, 4 => 0x0e20_0800 | q | u | size | (pick(r, u32, &.{ 0, 1, 5, 8, 9, 10, 11, 18, 4 }) << 12) | (vn << 5) | vd, // two-reg misc
        5 => 0x0e30_0800 | q | u | size | (pick(r, u32, &.{ 0x1b, 0x0a, 0x1a, 0x03 }) << 12) | (vn << 5) | vd, // across lanes
        6 => 0x0e20_0000 | q | u | (r.uintLessThan(u32, 3) << 22) | (pick(r, u32, &.{ 0xc, 0x8, 0xa, 0, 1, 2, 3, 4, 0xe }) << 12) | rr, // three different
        7, 8 => blk: { // shift by immediate, vector and scalar
            const sc = r.uintLessThan(u32, 4) == 0;
            const immhb = (@as(u32, 1) << @intCast(if (sc) 6 else 3 + r.uintLessThan(u32, 4))) | r.uintLessThan(u32, 64);
            const opc = pick(r, u32, &.{ 0, 2, 8, 0xa, 0x14, 0x10, 0x4 });
            break :blk (if (sc) @as(u32, 0x5f00_0400) else 0x0f00_0400 | q) | u | ((immhb & 0x7f) << 16) | (opc << 11) | (vn << 5) | vd;
        },
        9 => if (r.boolean()) 0x5ef1_b800 | (vn << 5) | vd // addp d
        else 0x0f00_0000 | q | u | (r.uintLessThan(u32, 4) << 22) | (r.uintLessThan(u32, 8) << 20) | (pick(r, u32, &.{ 0, 4, 8, 2, 6, 10, 1 }) << 12) | (r.uintLessThan(u32, 2) << 11) | (vm << 16) | (vn << 5) | vd, // by element
        10 => edgeV(r),
        11 => 0x0e20_1c00 | q | (1 << 29) | (r.uintLessThan(u32, 4) << 22) | rr, // EOR, BSL, BIT, BIF
        else => 0x0f00_0400 | q | (r.uintLessThan(u32, 2) << 29) | (r.uintLessThan(u32, 8) << 16) | ((r.uintLessThan(u32, 6) * 2 + 1) << 12) | (r.uintLessThan(u32, 32) << 5) | vd, // ORR, BIC (immediate)
    };
    return 1;
}

/// Copies, permutes, table lookups, extracts.
fn genMove(r: std.Random, out: []u32, pc: u64) usize {
    _ = pc;
    const vd = vreg(r);
    const vn = vreg(r);
    const vm = vreg(r);
    const q = r.uintLessThan(u32, 2) << 30;
    const rr = (vm << 16) | (vn << 5) | vd;
    out[0] = switch (r.uintLessThan(u32, 9)) {
        0, 1 => 0x0e00_0400 | q | (r.uintLessThan(u32, 2) << 29) | (r.uintLessThan(u32, 32) << 16) | (pick(r, u32, &.{ 0, 3, 5, r.uintLessThan(u32, 16) }) << 11) | (vn << 5) | (if (r.boolean()) vd else kt.dst(r)), // DUP/INS (element), INS (general), SMOV
        2 => 0x5e00_0400 | (r.uintLessThan(u32, 32) << 16) | (vn << 5) | vd, // DUP (element), scalar
        3, 4 => 0x0e00_0800 | q | (r.uintLessThan(u32, 4) << 22) | (r.uintLessThan(u32, 8) << 12) | rr, // UZP, TRN, ZIP
        5 => 0x2e00_0000 | q | (r.uintLessThan(u32, 16) << 11) | rr, // EXT
        6 => 0x0e00_0000 | q | (r.uintLessThan(u32, 4) << 13) | (r.uintLessThan(u32, 2) << 12) | rr, // TBL, TBX
        7 => edgeV(r),
        else => blk: { // small indices into Vm for TBL (bytes 0..0x47)
            break :blk 0x6f00_e400 | (r.uintLessThan(u32, 8) << 16) | (r.uintLessThan(u32, 32) << 5) | vm; // movi vm.2d
        },
    };
    if (out[0] & 0xbf20_8c00 == 0x0e00_0000 and r.boolean()) {
        // TBL with in-range indices: AND Vm with 0x3f bytes first (movi + and)
        out[1] = out[0];
        out[0] = 0x4f01_e7e0 | 30; // movi v30.16b, #0x3f
        out[2] = out[1];
        out[1] = 0x4e20_1c00 | (30 << 16) | (vm << 5) | vm; // and vm, vm, v30
        return 3;
    }
    return 1;
}

/// AES (AESE + AESMC and AESD + AESIMC pairs as OpenSSL writes them, lone
/// ops, pairs on other registers), SHA1, SHA256, SHA512, PMULL.
fn genCrypto(r: std.Random, out: []u32, pc: u64) usize {
    _ = pc;
    const vd = vreg(r);
    const vn = vreg(r);
    const vm = vreg(r);
    const rr = (vm << 16) | (vn << 5) | vd;
    switch (r.uintLessThan(u32, 12)) {
        0, 1, 2 => { // aese/aesd vd, vn; aesmc/aesimc vd, vd
            const d = r.boolean();
            out[0] = (if (d) @as(u32, 0x4e28_5800) else 0x4e28_4800) | (vn << 5) | vd;
            const second = (if (d) @as(u32, 0x4e28_7800) else 0x4e28_6800);
            out[1] = switch (r.uintLessThan(u32, 8)) {
                0 => second | (vd << 5) | vm, // into another register
                1 => second | (vm << 5) | vm, // of another register
                2 => (if (d) @as(u32, 0x4e28_6800) else 0x4e28_7800) | (vd << 5) | vd, // the other direction
                3 => 0x4e20_1c00 | (vn << 16) | (vd << 5) | vd, // something else first
                else => second | (vd << 5) | vd,
            };
            return 2;
        },
        3 => out[0] = 0x4e28_4800 | (r.uintLessThan(u32, 4) << 12) | (vn << 5) | vd, // any AES op
        4 => out[0] = 0x5e00_0000 | (r.uintLessThan(u32, 8) << 12) | rr, // SHA1C/P/M/SU0, SHA256H/H2/SU1
        5 => out[0] = 0x5e28_0800 | (r.uintLessThan(u32, 4) << 12) | (vn << 5) | vd, // SHA1H, SHA1SU1, SHA256SU0
        6 => out[0] = 0xce60_8000 | (r.uintLessThan(u32, 4) << 10) | rr, // SHA512H/H2/SU1, RAX1
        7 => out[0] = 0xcec0_8000 | (vn << 5) | vd, // SHA512SU0
        8, 9 => out[0] = 0x0ee0_e000 | (r.uintLessThan(u32, 2) << 30) | rr, // PMULL, PMULL2 .1q
        10 => out[0] = edgeV(r),
        else => out[0] = 0x0e20_0400 | (1 << 30) | (pick(r, u32, &.{ 0b10000, 0b10001 }) << 11) | (r.uintLessThan(u32, 4) << 22) | rr,
    }
    return 1;
}

/// LD1/ST1 (multiple structures, 1-4 registers, Q at random) at the data
/// pages (also across a page end: handed back), post-index by immediate or
/// register, the base SP or x20..x22; ST1 into the code page with rwx_code.
fn genLd1(r: std.Random, out: []u32, pc: u64) usize {
    _ = pc;
    const t = vreg(r);
    const q = r.uintLessThan(u32, 2) << 30;
    const l = r.uintLessThan(u32, 2) << 22;
    const opc = pick(r, u32, &.{ 0b0111, 0b1010, 0b0110, 0b0010, 0b1000 });
    const base = pick(r, u32, &.{ 20, 21, 22, 31, 19 });
    const post: u32 = switch (r.uintLessThan(u32, 3)) {
        0 => 0,
        1 => (1 << 23) | (31 << 16),
        else => (1 << 23) | (pick(r, u32, &.{ 9, 10 }) << 16),
    };
    // x9, x10: small positive or negative strides
    out[0] = 0xd280_0000 | (r.uintLessThan(u32, 128) << 5) | 9; // movz x9, #k
    out[1] = 0x9280_0000 | (r.uintLessThan(u32, 128) << 5) | 10; // movn x10, #k
    out[2] = if (r.uintLessThan(u32, 3) == 0)
        0x0c00_0000 | q | l | post | (opc << 12) | (r.uintLessThan(u32, 4) << 10) | (base << 5) | t
    else // single structure: LD1/ST1 lane, LD1R (and LD2-4 neighbours)
        0x0d00_0000 | q | l | post | (r.uintLessThan(u32, 8) << 13) | (r.uintLessThan(u32, 2) << 12) | (r.uintLessThan(u32, 4) << 10) | (if (r.uintLessThan(u32, 4) == 0) @as(u32, 1) << 21 else 0) | (base << 5) | t;
    if (out[2] & 0x0100_0000 != 0 and post == 0) out[2] &= ~@as(u32, 0x1f << 16);
    if (r.uintLessThan(u32, 4) == 0) // LDP/STP/LDNP/STNP of S, D, Q (opc 3: reserved)
        out[2] = 0x2c00_0000 | (r.uintLessThan(u32, 4) << 30) | (r.uintLessThan(u32, 4) << 23) | l | (r.uintLessThan(u32, 128) << 15) | (vreg(r) << 10) | (base << 5) | t;
    return 3;
}

fn genFpVec(r: std.Random, out: []u32, pc: u64) usize {
    _ = pc;
    const vd = vreg(r);
    const vn = vreg(r);
    const vm = vreg(r);
    const q = r.uintLessThan(u32, 2) << 30;
    const dbl = r.uintLessThan(u32, 2) << 22;
    const x9: u32 = 9;
    const rr = (vm << 16) | (vn << 5) | vd;
    switch (r.uintLessThan(u32, 18)) {
        0, 1 => { // edge f32 vector or lane insert
            const exps = [_]u32{ 0, 1, 2, 126, 127, 128, 158, 159, 253, 254, 255 };
            const mids = [_]u32{ 0x0000_0000, 0x8000_0000, 0x0000_0001, 0x007f_ffff, 0x0080_0000, 0x0080_0001, 0x0100_0000, 0x3380_0000, 0x3f80_0000, 0x3f80_0001, 0xbf00_0000, 0xbf80_0000, 0xbf80_0001, 0x4f00_0000, 0xcf00_0000, 0x4f80_0000, 0x7f7f_ffff, 0x7f80_0000, 0x7fc0_0042, 0x7f80_0042 };
            if (r.boolean()) {
                const w = mids[r.uintLessThan(usize, mids.len)] ^ (r.uintLessThan(u32, 2) << 31);
                out[0] = 0x5280_0000 | ((w & 0xffff) << 5) | x9; // movz w9, #lo
                out[1] = 0x72a0_0000 | (((w >> 16) & 0xffff) << 5) | x9; // movk w9, #hi, lsl 16
            } else {
                const top = (r.uintLessThan(u32, 2) << 15) | (exps[r.uintLessThan(usize, exps.len)] << 7) | pick(r, u32, &.{ 0, 1, 0x40, 0x7f });
                out[0] = 0x52a0_0000 | (top << 5) | x9; // movz w9, #top, lsl 16
                out[1] = 0x7280_0000 | (pick(r, u32, &.{ 0, 1, 0xffff }) << 5) | x9;
            }
            if (r.uintLessThan(u32, 3) == 0) {
                const lane = r.uintLessThan(u32, 4);
                out[2] = 0x4e04_1c00 | (lane << 19) | (x9 << 5) | vd; // ins vD.s[lane], w9
            } else {
                out[2] = 0x0e04_0c00 | q | (x9 << 5) | vd; // dup vD.{2,4}s, w9
            }
            return 3;
        },
        2 => { // edge f64 vector or lane insert
            const exps = [_]u32{ 0, 1, 2, 1022, 1023, 1024, 2045, 2046, 2047 };
            const top = (r.uintLessThan(u32, 2) << 15) | (exps[r.uintLessThan(usize, exps.len)] << 4) | pick(r, u32, &.{ 0, 1, 8, 15 });
            out[0] = 0xd2e0_0000 | (top << 5) | x9; // movz x9, #top, lsl 48
            out[1] = 0xf280_0000 | (pick(r, u32, &.{ 0, 1, 0xffff }) << 5) | x9;
            if (r.boolean()) {
                const lane = r.uintLessThan(u32, 2);
                out[2] = 0x4e08_1c00 | (lane << 20) | (x9 << 5) | vd; // ins vD.d[lane], x9
            } else {
                out[2] = 0x4e08_0c00 | (x9 << 5) | vd; // dup vD.2d, x9
            }
            return 3;
        },
        3 => { // FPCR: mostly 0, else RMode / FZ / DN
            const fc: u32 = if (r.uintLessThan(u32, 3) != 0) 0 else (r.uintLessThan(u32, 8) << 6) | (r.uintLessThan(u32, 2) << 9);
            out[0] = 0xd2a0_0000 | (fc << 5) | x9;
            out[1] = 0xd51b_4400 | x9; // msr fpcr, x9
            return 2;
        },
        4 => { // FPSR: 0, IXC, or all flags
            const fs = pick(r, u32, &.{ 0, 0x10, 0x9f });
            out[0] = 0xd280_0000 | (fs << 5) | x9;
            out[1] = 0xd51b_4420 | x9; // msr fpsr, x9
            return 2;
        },
        5, 6, 7, 8 => { // three same FP (FADD, FSUB, FMUL, FDIV, FMLA, FMLS, FMIN/FMAX/NM, FCMEQ/GE/GT + neighbours)
            const k = pick(r, u32, &.{
                0b0_0_11000, 0b0_0_11001, 0b0_0_11010, 0b0_0_11100, 0b0_0_11110,
                0b0_1_11000, 0b0_1_11001, 0b0_1_11010, 0b0_1_11110,
                0b1_0_11011, 0b1_0_11100, 0b1_0_11111, 0b1_1_11100,
                0b1_0_11010, // FADDP
            });
            const u = (k >> 6) & 1;
            const a = (k >> 5) & 1;
            const opc = k & 31;
            out[0] = 0x0e20_0400 | q | (u << 29) | (a << 23) | dbl | (opc << 11) | rr;
        },
        9, 10, 11 => { // two-register misc FP (FABS, FNEG, FSQRT, zero compares, SCVTF/UCVTF, FCVTZS/ZU)
            const k = pick(r, u32, &.{
                0b0_1_01100, 0b0_1_01101, 0b0_1_01110, 0b0_1_01111,
                0b1_1_01100, 0b1_1_01101, 0b1_1_01111, 0b1_1_11111,
                0b0_0_11101, 0b1_0_11101, 0b0_1_11011, 0b1_1_11011,
            });
            const u = (k >> 6) & 1;
            const a = (k >> 5) & 1;
            const opc = k & 31;
            out[0] = 0x0e20_0800 | q | (u << 29) | (a << 23) | dbl | (opc << 12) | (vn << 5) | vd;
        },
        12, 13 => { // by-element FMUL, FMLA, FMLS (+ .1d illegal neighbour when q==0, L==1)
            const opc = pick(r, u32, &.{ 0b1001, 0b0001, 0b0101 });
            const h = r.uintLessThan(u32, 2);
            const l = r.uintLessThan(u32, 2);
            const a: u32 = if (r.uintLessThan(u32, 8) == 0) 0 else 1;
            out[0] = 0x0f00_0000 | q | (a << 23) | dbl | (l << 21) | (opc << 12) | (h << 11) | rr;
        },
        14 => { // edge integer vectors for SCVTF/UCVTF
            out[0] = 0x5280_0000 | (pick(r, u32, &.{ 0, 1, 0xffff }) << 5) | x9;
            out[1] = 0x72a0_0000 | (pick(r, u32, &.{ 0, 0x4000, 0x7fff, 0x8000, 0xffff }) << 5) | x9;
            out[2] = 0x0e04_0c00 | q | (x9 << 5) | vd;
            return 3;
        },
        else => { // mix dirty high half before Q = 0
            out[0] = 0x4f01_e7e0 | vd; // movi vD.16b, #0x3f
            return 1;
        },
    }
    if (r.boolean()) {
        out[1] = 0xd53b_4420 | kt.dst(r); // mrs xD, fpsr
        out[2] = 0xd51b_4420 | 31; // msr fpsr, xzr
        return 3;
    }
    return 1;
}

test "a64 jit vec: arithmetic" {
    try kt.kind(.generic, genArith, .{ .rounds = 300, .words = 48 });
    try kt.kind(.generic, genArith, .{ .seed = 2, .rounds = 300, .words = 48 });
}
test "a64 jit vec: AdvSIMD FP" {
    try kt.kind(.generic, genFpVec, .{ .rounds = 300, .words = 48 });
    try kt.kind(.generic, genFpVec, .{ .seed = 2, .rounds = 300, .words = 48 });
}
test "a64 jit vec: moves and permutes" {
    try kt.kind(.generic, genMove, .{ .rounds = 300, .words = 48 });
}
test "a64 jit vec: crypto" {
    try kt.kind(.generic, genCrypto, .{ .rounds = 300, .words = 48 });
    try kt.kind(.generic, genCrypto, .{ .seed = 2, .rounds = 300, .words = 48 });
}
test "a64 jit vec: ld1, st1" {
    try kt.kind(.generic, genLd1, .{ .rounds = 200 });
    try kt.kind(.generic, genLd1, .{ .seed = 2, .warm_tlb = true, .rounds = 200 });
    try kt.kind(.generic, genLd1, .{ .seed = 3, .warm_tlb = true, .rwx_code = true, .rounds = 100 });
}
