// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! The decoded-op tier's op format and its decoder: one raw A64 word to one
//! 16-byte `Op` (a64/tier.md). The semantics are in
//! opexec.zig, the block cache and run loop in tier.zig.
//!
//! Rules:
//!  - one instruction is exactly one op, so an op index is a retired
//!    instruction count and pc is recomputed from the ops (`tier.pcAt`);
//!  - a dedicated op only for an encoding the reference interpreter accepts
//!    and whose every effect the op reproduces; anything else, every
//!    reserved or unknown encoding included, is `generic`, which runs the
//!    reference interpreter on the raw word, so its legality checks and
//!    semantics stay the only ones;
//!  - register 31 is resolved here: `sp` (slot 31) or `zr` (slot 32, which
//!    is regs.pc and reads 0 while ops run; see opexec.zig). An op never
//!    writes the zero slot, except flag-setting ops, which write 0 there
//!    (their write mask is 0 for Rd = XZR). Other destinations of XZR
//!    decode to `nop` (no side effect) or `generic` (loads, which may
//!    fault);
//!  - immediates are pre-decoded (bitmask immediates expanded, shifts
//!    folded, branch targets and pc-relative addresses absolute);
//!  - variants of one operation share a kind when a field selects the
//!    variant without a branch (masks, shift counts), because every kind
//!    costs dispatch code (kwemu.md, Interpreter machinery).

const std = @import("std");
const c = @import("common.zig");
const paravirt = @import("paravirt.zig");

/// Register-file slots, as u64 indexes from &regs.x (struct kw_regs).
pub const sp: u8 = 31;
/// regs.pc: reads as XZR (0) while ops run.
pub const zr: u8 = 32;
/// regs.nzcv (low 32 bits of this slot; fpcr is the high half).
pub const nzcv_slot: u8 = 33;
pub const tpidr_slot: u8 = 35;
/// V0: Vn's low half is slot 36 + 2n, its high half 37 + 2n.
pub const v0_slot: u8 = 36;

comptime {
    const R = c.Regs;
    std.debug.assert(@offsetOf(R, "sp") == @as(usize, sp) * 8);
    std.debug.assert(@offsetOf(R, "pc") == @as(usize, zr) * 8);
    std.debug.assert(@offsetOf(R, "nzcv") == @as(usize, nzcv_slot) * 8);
    std.debug.assert(@offsetOf(R, "tpidr_el0") == @as(usize, tpidr_slot) * 8);
    std.debug.assert(@offsetOf(R, "v") == @as(usize, v0_slot) * 8);
}

/// Op kinds. Field names: a, b, c (bytes 1..3, register slots unless
/// noted), d, e, f, g (bytes 4..7), imm (bytes 8..15). "W" = 32-bit form;
/// `w32` fields hold 32 for W and 0 for X (a result mask is ~0 >> w32).
pub const H = enum(u8) {
    /// no effect (hints, PRFM, non-flag ops whose Rd is XZR)
    nop,
    /// a = imm (MOVZ, MOVN, ADR, ADRP, MOV of a bitmask immediate)
    movi,
    /// a = (a & imm) | (e:f as u16 << d) (MOVK; imm keeps the other bits)
    movk,
    /// a = b & imm (MOV register, W and X; MRS/MSR TPIDR_EL0)
    mov,
    /// a = (b + imm) & (~0 >> d) (ADD/SUB immediate, SP forms; SUB negates imm)
    add_i,
    /// flags of (b << d) + imm + e; a = (result >> d) & -f (ADDS/SUBS
    /// immediate, CMP, CMN: imm = (imm12 << d), inverted for SUB, e = 1)
    adds_i,
    /// a = ((b & imm) & -e0 | (b ^ imm) & -e1) & (~0 >> d) (AND/ORR/EOR
    /// immediate: e = 1, 3, 2)
    logic_i,
    /// ANDS/TST immediate: N, Z of b & imm (at d = 0 or 32), C = V = 0; a = (b & imm) & -f
    ands_i,
    /// a = b + (op2(c) ^ inv) + cin (ADD/SUB shifted and extended register,
    /// NEG); op2(v) = ((v << d) >>a e) & imm; g: bit 0 inv/cin (SUB), bit 3 W
    addsub_r,
    /// ADDS/SUBS/CMP/CMN shifted and extended register: flags as adds_i;
    /// g: bit 0 SUB, bit 1 write a, bit 3 W
    addsubs_r,
    /// addsub_r and addsubs_r without a shift or extension (op2 = c): the
    /// common case, without the transform
    add_rr,
    adds_rr,
    /// a = (b & op2' & -m1 | (b ^ op2') & -m2) & mask, op2' = op2(c) ^ inv
    /// (AND, ORR, EOR, BIC, ORN, EON, MVN shifted register); g: bit 0 m1,
    /// bit 1 m2, bit 2 inv, bit 3 W
    logic_r,
    /// ANDS, BICS, TST shifted register; g: bit 1 write a, bit 2 inv, bit 3 W
    ands_r,
    /// logic_r with a rotated operand (ROR #d, 1..63): op2' = ror(c, d) ^ inv
    logic_ror,
    /// a = ((b << d) >> e) << f (UBFM: LSL, LSR, UBFX, UBFIZ, UXTB, UXTH)
    ubfm,
    /// a = (((b << d) >>a e) << f) & (~0 >> g) (SBFM: ASR, SBFX, SBFIZ, SXT*)
    sbfm,
    /// a = ((a & ~imm) | ((b << d) >> e) << f) & (~0 >> g) (BFM: BFI, BFXIL)
    bfm,
    /// a = EXTR(b:c, d) at 32 (e = 32) or 64 bits, d in 1..63 (EXTR, ROR imm)
    extr,
    /// a = shift(b, c) (LSLV, LSRV, ASRV, RORV), d = w32
    lslv,
    lsrv,
    asrv,
    rorv,
    /// a = cond ? b : (c ^ inv) + inc (CSEL, CSINC, CSINV, CSNEG and
    /// aliases); imm: bits 0..15 condition mask (`condMask`), 16 inc, 17
    /// inv, 18 W
    csel,
    /// CCMP/CCMN register and immediate: if cond, flags of b - (c + imm5)
    /// (or +), else the default; c = zr for the immediate form; d = w32;
    /// imm: 0..15 condition mask, 16..19 default NZCV, 20 SUB, 24..28 imm5
    ccmp,
    /// a = d + ((b * c) ^ -e) + e, & (~0 >> f) (MADD, MSUB, MUL, MNEG)
    madd,
    /// a = d + ((ext(b) * ext(c)) ^ -e) + e, ext = 32-bit sign (f = 1) or
    /// zero extension (SMADDL, SMSUBL, UMADDL, UMSUBL, SMULL, UMULL, ...)
    maddl,
    umulh,
    smulh,
    /// a = b / c, 0 for c = 0 (d = w32)
    udiv,
    sdiv,
    /// data-processing 1-source: d = 0 RBIT, 1 REV16, 2 REV32 / REV (W),
    /// 3 REV (X), 4 CLZ, 5 CLS; e = w32
    rev1,
    /// ADC, ADCS, SBC, SBCS: b + (c ^ -sub) + C; g: bit 0 SUB, bit 1 set
    /// flags, bit 2 write a (0 for ADCS/SBCS to XZR), bit 3 W
    adc,
    // ---- loads and stores (integer; also the B/H/S/D V forms of stores) ----
    // Width in the kind. `_i`: va = b + imm. `_w` (pre/post-index): va = b
    // + (imm & -f), then b = b + imm. `_r` (register offset): va = b +
    // op2(c), op2(v) = ((v << e) >>a f) & imm. Loads: a = ((v << d) >>a d)
    // & (~0 >> (g * 32)) (sign extension to 64 or to 32 bits; d = 0 for
    // zero extension). Stores: the data is slot a (zr for XZR).
    ldr8_i,
    ldr16_i,
    ldr32_i,
    ldr64_i,
    ldr8_w,
    ldr16_w,
    ldr32_w,
    ldr64_w,
    ldr8_r,
    ldr16_r,
    ldr32_r,
    ldr64_r,
    str8_i,
    str16_i,
    str32_i,
    str64_i,
    str8_w,
    str16_w,
    str32_w,
    str64_w,
    str8_r,
    str16_r,
    str32_r,
    str64_r,
    /// LDP/STP of W (32: d = 32 for LDPSW) and X registers (also STP of S
    /// and D registers): a, c = Rt, Rt2; va = b + imm (`_w`: b + (imm &
    /// -e), then b += imm)
    ldp32,
    ldp64,
    ldp32_w,
    ldp64_w,
    stp32,
    stp64,
    stp32_w,
    stp64_w,
    /// LDR B/H/S/D: a = V slot, d = log2 bytes, the rest of the register is
    /// zeroed; e = 0 immediate offset and literal (va = b + imm), 1 pre/post
    /// index (f = pre), 2 register offset (va = b + op2(c), f = s1, g = s2,
    /// imm = mask)
    ldr_v,
    /// LDR/STR Q: `_i`, `_w`, `_r` addressing as the integer forms
    ldr_q_i,
    ldr_q_w,
    ldr_q_r,
    str_q_i,
    str_q_w,
    str_q_r,
    /// LDP/STP Q (`_w`: e as for ldp*_w)
    ldp_q,
    ldp_q_w,
    stp_q,
    stp_q_w,
    // ---- branches ----
    /// side exits (superblocks): taken -> leave the block at imm, else go on
    bcond_x,
    cbz_x,
    tbz_x,
    /// direct branch inside the block's page, followed by the decoder (the
    /// next op is the target's); bl_f also sets x30 = (imm & ~0xfff) + d:e
    b_f,
    bl_f,
    // terminators: pc = imm (or the register) / end_pc
    /// B.cond: d:e = condition mask
    bcond,
    /// CBZ/CBNZ: taken = (a & (~0 >> e)) == 0, inverted by d
    cbz,
    /// TBZ/TBNZ: taken = bit d of a == e
    tbz,
    b,
    bl,
    /// BR, RET (pc = b)
    br,
    /// BLR (x30 = end_pc after reading b)
    blr,
    /// fall-through at the end of a block that does not end in a branch
    goto,
    /// anything else: the reference interpreter on the raw word (d..g as a
    /// u32), pc = imm
    generic,
};

/// One decoded instruction (16 bytes; ops.zig's fields above).
pub const Op = extern struct {
    h: H,
    a: u8 = 0,
    b: u8 = 0,
    c: u8 = 0,
    d: u8 = 0,
    e: u8 = 0,
    f: u8 = 0,
    g: u8 = 0,
    imm: u64 = 0,

    /// d..g as one u32 (the raw word of a `generic` op).
    pub inline fn aux(o: Op) u32 {
        return @as(u32, o.d) | (@as(u32, o.e) << 8) | (@as(u32, o.f) << 16) | (@as(u32, o.g) << 24);
    }
};

comptime {
    std.debug.assert(@sizeOf(Op) == 16 and @alignOf(Op) == 8);
    std.debug.assert(@typeInfo(H).@"enum".fields.len <= 192); // kwemu.max_ops
}

pub const Decoded = struct { op: Op, ends: bool = false };

/// Whether `h` is a side exit (taken: the block is left).
pub inline fn isSide(h: H) bool {
    return h == .bcond_x or h == .cbz_x or h == .tbz_x;
}
/// The side-exit form of a conditional-branch terminator, or null.
pub fn sideExit(h: H) ?H {
    return switch (h) {
        .bcond => .bcond_x,
        .cbz => .cbz_x,
        .tbz => .tbz_x,
        else => null,
    };
}
/// pc of the instruction after `o` (at pc) in block order.
pub inline fn nextPc(o: Op, pc: u64) u64 {
    return if (o.h == .b_f or o.h == .bl_f) o.imm else pc +% 4;
}

/// Bit i set = ConditionHolds(cond) for NZCV = i (N = bit 3 ... V = bit 0).
pub fn condMask(cond: u4) u16 {
    var m: u16 = 0;
    for (0..16) |i| {
        if (c.condHolds(@as(u32, @intCast(i)) << 28, cond)) m |= @as(u16, 1) << @intCast(i);
    }
    return m;
}

// ---- helpers -------------------------------------------------------------------

inline fn bits(inst: u32, comptime hi: u5, comptime lo: u5) u32 {
    return c.bits(inst, hi, lo);
}
inline fn bit(inst: u32, comptime n: u5) bool {
    return c.bit(inst, n);
}
/// Slot of register n in an XZR form.
inline fn xz(n: u32) u8 {
    return if (n == 31) zr else @intCast(n);
}
/// Slot of register n in an SP form.
inline fn xs(n: u32) u8 {
    return @intCast(n);
}
inline fn vs(n: u32) u8 {
    return @intCast(v0_slot + 2 * n);
}
inline fn w32(sf: bool) u8 {
    return if (sf) 0 else 32;
}

fn mk(h: H, a: u8, b: u8, cc: u8, d: u8, e: u8, f: u8, g: u8, imm: u64) Decoded {
    return .{ .op = .{ .h = h, .a = a, .b = b, .c = cc, .d = d, .e = e, .f = f, .g = g, .imm = imm } };
}
fn ends(d: Decoded) Decoded {
    var r = d;
    r.ends = true;
    return r;
}
pub fn generic(inst: u32, pc: u64, end: bool) Decoded {
    return .{ .op = .{
        .h = .generic,
        .d = @truncate(inst),
        .e = @truncate(inst >> 8),
        .f = @truncate(inst >> 16),
        .g = @truncate(inst >> 24),
        .imm = pc,
    }, .ends = end };
}
fn nop() Decoded {
    return .{ .op = .{ .h = .nop } };
}
fn movi(rd: u8, v: u64) Decoded {
    return mk(.movi, rd, 0, 0, 0, 0, 0, 0, v);
}
fn mov(rd: u8, rs: u8, mask: u64) Decoded {
    return mk(.mov, rd, rs, 0, 0, 0, 0, 0, mask);
}
inline fn dmask(sf: bool) u64 {
    return if (sf) std.math.maxInt(u64) else 0xffff_ffff;
}

/// Operand-2 transform op2(v) = ((v << s1) >>a s2) & mask for a shifted
/// register (LSL, LSR, ASR; ROR is not expressible) at 32 or 64 bits.
const Xform = struct { s1: u8, s2: u8, mask: u64 };
fn shiftXform(sf: bool, typ: u32, n: u32) ?Xform {
    const n8: u8 = @intCast(n);
    return switch (typ) {
        0 => .{ .s1 = n8, .s2 = 0, .mask = std.math.maxInt(u64) },
        1 => if (sf) .{ .s1 = 0, .s2 = n8, .mask = @as(u64, std.math.maxInt(u64)) >> @intCast(n) } else .{ .s1 = 32, .s2 = 32 + n8, .mask = @as(u64, 0xffff_ffff) >> @intCast(n) },
        2 => if (sf) .{ .s1 = 0, .s2 = n8, .mask = std.math.maxInt(u64) } else .{ .s1 = 32, .s2 = 32 + n8, .mask = std.math.maxInt(u64) },
        else => if (n == 0) .{ .s1 = 0, .s2 = 0, .mask = std.math.maxInt(u64) } else null,
    };
}
/// ExtendReg(v, option, shift) as an operand-2 transform (shift 0..4).
fn extendXform(option: u32, shift: u32) Xform {
    const k: u8 = @intCast(shift);
    if (option & 3 == 3) return .{ .s1 = k, .s2 = 0, .mask = std.math.maxInt(u64) }; // UXTX, SXTX
    const width: u8 = @as(u8, 8) << @intCast(option & 3);
    const signed = option & 4 != 0;
    return .{
        .s1 = 64 - width,
        .s2 = 64 - width - k,
        .mask = if (signed) std.math.maxInt(u64) else c.ones(@intCast(width)) << @intCast(k),
    };
}

// ---- decoder -------------------------------------------------------------------

/// Decode the word at `pc`. `ends`: the instruction ends a block (a branch,
/// or a generic op that cannot be followed: SVC, BRK, unallocated space,
/// ISB).
pub fn decode(inst: u32, pc: u64) Decoded {
    return switch (bits(inst, 28, 25)) {
        0b0000 => generic(inst, pc, !paravirt.isDefined(inst)), // UDF (paravirt stays in block), SME
        0b0001, 0b0010, 0b0011 => generic(inst, pc, true), // SVE, unallocated
        0b1000, 0b1001 => dpImm(inst, pc),
        0b1010, 0b1011 => branch(inst, pc),
        0b0100, 0b0110, 0b1100, 0b1110 => ldst(inst, pc),
        0b0101, 0b1101 => dpReg(inst, pc),
        else => generic(inst, pc, false), // FP and AdvSIMD data processing
    };
}

fn dpImm(inst: u32, pc: u64) Decoded {
    const sf = bit(inst, 31);
    const rd = c.rd(inst);
    const rn = c.rn(inst);
    switch (bits(inst, 25, 23)) {
        0b000, 0b001 => { // ADR, ADRP
            if (rd == 31) return nop();
            const imm = c.sextN((bits(inst, 23, 5) << 2) | bits(inst, 30, 29), 21);
            return movi(xz(rd), if (bit(inst, 31)) (pc & ~@as(u64, 0xfff)) +% (imm << 12) else pc +% imm);
        },
        0b010 => { // ADD, ADDS, SUB, SUBS (immediate)
            const sub = bit(inst, 30);
            var imm: u64 = bits(inst, 21, 10);
            if (bit(inst, 22)) imm <<= 12;
            if (!bit(inst, 29)) return mk(.add_i, xs(rd), xs(rn), 0, w32(sf), 0, 0, 0, if (sub) 0 -% imm else imm);
            const B = imm << @intCast(w32(sf));
            return mk(.adds_i, xz(rd), xs(rn), 0, w32(sf), @intFromBool(sub), @intFromBool(rd != 31), 0, if (sub) ~B else B);
        },
        0b100 => { // AND, ORR, EOR, ANDS (immediate)
            const n: u1 = @intFromBool(bit(inst, 22));
            if (!sf and n == 1) return generic(inst, pc, false);
            const m = c.decodeBitMasks(n, @intCast(bits(inst, 15, 10)), @intCast(bits(inst, 21, 16)), true, if (sf) 64 else 32) orelse
                return generic(inst, pc, false);
            const imm = m.wmask;
            const opc = bits(inst, 30, 29);
            if (opc == 3) return mk(.ands_i, xz(rd), xz(rn), 0, w32(sf), 0, @intFromBool(rd != 31), 0, imm);
            if (rn == 31) return movi(xs(rd), if (opc == 0) 0 else imm); // MOV (bitmask immediate)
            const e: u8 = switch (opc) {
                0 => 1,
                1 => 3,
                else => 2,
            };
            return mk(.logic_i, xs(rd), xz(rn), 0, w32(sf), e, 0, 0, imm);
        },
        0b101 => { // MOVN, MOVZ, MOVK
            const hw = bits(inst, 22, 21);
            if (!sf and hw >= 2) return generic(inst, pc, false);
            const opc = bits(inst, 30, 29);
            if (opc == 1) return generic(inst, pc, false);
            if (rd == 31) return nop();
            const sh: u6 = @intCast(hw * 16);
            const imm16: u64 = bits(inst, 20, 5);
            return switch (opc) {
                0 => movi(xz(rd), ~(imm16 << sh) & dmask(sf)),
                2 => movi(xz(rd), imm16 << sh),
                else => mk(.movk, xz(rd), 0, 0, sh, @truncate(imm16), @truncate(imm16 >> 8), 0, ~(@as(u64, 0xffff) << sh) & dmask(sf)),
            };
        },
        0b110 => { // SBFM, BFM, UBFM
            const n = bit(inst, 22);
            const opc = bits(inst, 30, 29);
            const immr: u32 = bits(inst, 21, 16);
            const imms: u32 = bits(inst, 15, 10);
            if (opc == 3 or sf != n) return generic(inst, pc, false);
            if (!sf and (immr >= 32 or imms >= 32)) return generic(inst, pc, false);
            _ = c.decodeBitMasks(@intFromBool(n), @intCast(imms), @intCast(immr), false, if (sf) 64 else 32) orelse return generic(inst, pc, false);
            if (rd == 31) return nop();
            const ds: u32 = if (sf) 64 else 32;
            const s1: u8 = @intCast(63 - imms);
            var s2: u8 = s1;
            var s3: u8 = 0;
            var field: u64 = undefined;
            if (imms >= immr) {
                s2 = @intCast(63 - imms + immr);
                field = c.ones(@intCast(imms - immr + 1));
            } else {
                s3 = @intCast(ds - immr);
                field = c.ones(@intCast(imms + 1)) << @intCast(s3);
            }
            return switch (opc) {
                0 => mk(.sbfm, xz(rd), xz(rn), 0, s1, s2, s3, w32(sf), 0),
                1 => mk(.bfm, xz(rd), xz(rn), 0, s1, s2, s3, w32(sf), field),
                else => mk(.ubfm, xz(rd), xz(rn), 0, s1, s2, s3, 0, 0),
            };
        },
        0b111 => { // EXTR
            const lsb = bits(inst, 15, 10);
            if (bits(inst, 30, 29) != 0 or bit(inst, 21) or bit(inst, 22) != sf) return generic(inst, pc, false);
            if (!sf and lsb >= 32) return generic(inst, pc, false);
            if (rd == 31) return nop();
            const rm = c.rm(inst);
            if (lsb == 0) return mov(xz(rd), xz(rm), dmask(sf));
            return mk(.extr, xz(rd), xz(rn), xz(rm), @intCast(lsb), w32(sf), 0, 0, 0);
        },
        else => return generic(inst, pc, false), // ADDG, SUBG (MTE)
    }
}

fn dpReg(inst: u32, pc: u64) Decoded {
    const sf = bit(inst, 31);
    const rd = c.rd(inst);
    const rn = c.rn(inst);
    const rm = c.rm(inst);
    const op2 = bits(inst, 24, 21);
    const wbit: u8 = if (sf) 0 else 8;
    if (!bit(inst, 28)) {
        if (op2 & 8 == 0) { // logical (shifted register)
            const amount = bits(inst, 15, 10);
            if (!sf and amount >= 32) return generic(inst, pc, false);
            const opc = bits(inst, 30, 29);
            const inv = bit(inst, 21);
            const typ = bits(inst, 23, 22);
            const m: u8 = switch (opc) {
                0 => 1,
                1 => 3,
                else => 2,
            };
            const x = shiftXform(sf, typ, amount) orelse { // ROR #amount
                if (opc == 3) return generic(inst, pc, false);
                if (rd == 31) return nop();
                return mk(.logic_ror, xz(rd), xz(rn), xz(rm), @intCast(amount), 0, 0, m | (if (inv) @as(u8, 4) else 0) | wbit, 0);
            };
            if (opc == 3) return mk(.ands_r, xz(rd), xz(rn), xz(rm), x.s1, x.s2, 0, (if (rd != 31) @as(u8, 2) else 0) | (if (inv) @as(u8, 4) else 0) | wbit, x.mask);
            if (rd == 31) return nop();
            if (opc == 1 and !inv and rn == 31 and amount == 0) return mov(xz(rd), xz(rm), dmask(sf)); // MOV
            return mk(.logic_r, xz(rd), xz(rn), xz(rm), x.s1, x.s2, 0, m | (if (inv) @as(u8, 4) else 0) | wbit, x.mask);
        }
        const sub: u8 = @intFromBool(bit(inst, 30));
        const setflags = bit(inst, 29);
        if (op2 & 1 == 0) { // add/subtract (shifted register)
            const typ = bits(inst, 23, 22);
            const amount = bits(inst, 15, 10);
            if (typ == 3 or (!sf and amount >= 32)) return generic(inst, pc, false);
            const x = shiftXform(sf, typ, amount).?;
            const plain = amount == 0 and typ != 2; // LSL/LSR #0: op2 = Rm
            if (setflags) return mk(if (plain) .adds_rr else .addsubs_r, xz(rd), xz(rn), xz(rm), x.s1, x.s2, 0, sub | (if (rd != 31) @as(u8, 2) else 0) | wbit, x.mask);
            if (rd == 31) return nop();
            return mk(if (plain) .add_rr else .addsub_r, xz(rd), xz(rn), xz(rm), x.s1, x.s2, 0, sub | wbit, x.mask);
        }
        // add/subtract (extended register)
        const shift = bits(inst, 12, 10);
        if (bits(inst, 23, 22) != 0 or shift > 4) return generic(inst, pc, false);
        const x = extendXform(bits(inst, 15, 13), shift);
        if (setflags) return mk(.addsubs_r, xz(rd), xs(rn), xz(rm), x.s1, x.s2, 0, sub | (if (rd != 31) @as(u8, 2) else 0) | wbit, x.mask);
        return mk(.addsub_r, xs(rd), xs(rn), xz(rm), x.s1, x.s2, 0, sub | wbit, x.mask);
    }
    if (op2 & 8 != 0) { // data-processing (3 source)
        if (bits(inst, 30, 29) != 0) return generic(inst, pc, false);
        const sub: u8 = @intFromBool(bit(inst, 15));
        const ra = xz(c.ra(inst));
        const d: Decoded = switch (bits(inst, 23, 21)) {
            0 => mk(.madd, xz(rd), xz(rn), xz(rm), ra, sub, w32(sf), 0, 0),
            1, 5 => if (!sf) return generic(inst, pc, false) else mk(.maddl, xz(rd), xz(rn), xz(rm), ra, sub, @intFromBool(bits(inst, 23, 21) == 1), 0, 0),
            2 => if (sf and sub == 0) mk(.smulh, xz(rd), xz(rn), xz(rm), 0, 0, 0, 0, 0) else return generic(inst, pc, false),
            6 => if (sf and sub == 0) mk(.umulh, xz(rd), xz(rn), xz(rm), 0, 0, 0, 0, 0) else return generic(inst, pc, false),
            else => return generic(inst, pc, false),
        };
        return if (rd == 31) nop() else d;
    }
    switch (op2) {
        2 => { // CCMP, CCMN (register and immediate)
            if (!bit(inst, 29) or bit(inst, 10) or bit(inst, 4)) return generic(inst, pc, false);
            const imm_form = bit(inst, 11);
            const imm5: u64 = if (imm_form) bits(inst, 20, 16) else 0;
            const v: u64 = condMask(@intCast(bits(inst, 15, 12))) | (@as(u64, bits(inst, 3, 0)) << 16) | (@as(u64, @intFromBool(bit(inst, 30))) << 20) | (imm5 << 24);
            return mk(.ccmp, 0, xz(rn), if (imm_form) zr else xz(rm), w32(sf), 0, 0, 0, v);
        },
        4 => { // CSEL, CSINC, CSINV, CSNEG
            if (bit(inst, 29) or bit(inst, 11)) return generic(inst, pc, false);
            if (rd == 31) return nop();
            const v: u64 = condMask(@intCast(bits(inst, 15, 12))) | (@as(u64, bits(inst, 10, 10)) << 16) | (@as(u64, @intFromBool(bit(inst, 30))) << 17) | (@as(u64, @intFromBool(!sf)) << 18);
            return mk(.csel, xz(rd), xz(rn), xz(rm), 0, 0, 0, 0, v);
        },
        0 => { // ADC, ADCS, SBC, SBCS
            if (bits(inst, 15, 10) != 0) return generic(inst, pc, false);
            const S = bit(inst, 29);
            if (rd == 31 and !S) return nop();
            return mk(.adc, xz(rd), xz(rn), xz(rm), 0, 0, 0, @as(u8, @intFromBool(bit(inst, 30))) | (if (S) @as(u8, 2) else 0) | (if (rd != 31) @as(u8, 4) else 0) | wbit, 0);
        },
        6 => {
            if (bit(inst, 30) and !bit(inst, 29) and bits(inst, 20, 16) == 0) { // 1-source
                const opc = bits(inst, 15, 10);
                if (opc > 5 or (opc == 3 and !sf)) return generic(inst, pc, false);
                if (rd == 31) return nop();
                return mk(.rev1, xz(rd), xz(rn), 0, @intCast(opc), w32(sf), 0, 0, 0);
            }
            if (bit(inst, 30) or bit(inst, 29)) return generic(inst, pc, false); // PAC, reserved
            const h: H = switch (bits(inst, 15, 10)) {
                2 => .udiv,
                3 => .sdiv,
                8 => .lslv,
                9 => .lsrv,
                10 => .asrv,
                11 => .rorv,
                else => return generic(inst, pc, false), // CRC32, MTE, PACGA
            };
            if (rd == 31) return nop();
            return mk(h, xz(rd), xz(rn), xz(rm), w32(sf), 0, 0, 0, 0);
        },
        else => return generic(inst, pc, false), // ADC/SBC, FlagM, unallocated
    }
}

fn branch(inst: u32, pc: u64) Decoded {
    if (inst & 0x7c00_0000 == 0x1400_0000) { // B, BL
        const target = pc +% c.sextN(@as(u64, bits(inst, 25, 0)) << 2, 28);
        return ends(mk(if (bit(inst, 31)) .bl else .b, 0, 0, 0, 0, 0, 0, 0, target));
    }
    if (inst & 0x7e00_0000 == 0x3400_0000) { // CBZ, CBNZ
        const target = pc +% c.sextN(@as(u64, bits(inst, 23, 5)) << 2, 21);
        return ends(mk(.cbz, xz(c.rd(inst)), 0, 0, @intFromBool(bit(inst, 24)), w32(bit(inst, 31)), 0, 0, target));
    }
    if (inst & 0x7e00_0000 == 0x3600_0000) { // TBZ, TBNZ
        const b: u8 = @intCast((bits(inst, 31, 31) << 5) | bits(inst, 23, 19));
        const target = pc +% c.sextN(@as(u64, bits(inst, 18, 5)) << 2, 16);
        return ends(mk(.tbz, xz(c.rd(inst)), 0, 0, b, @intFromBool(bit(inst, 24)), 0, 0, target));
    }
    if (inst & 0xff00_0010 == 0x5400_0000) { // B.cond
        if (bit(inst, 24)) return generic(inst, pc, true);
        const target = pc +% c.sextN(@as(u64, bits(inst, 23, 5)) << 2, 21);
        const m = condMask(@intCast(inst & 15));
        return ends(mk(.bcond, 0, 0, 0, @truncate(m), @truncate(m >> 8), 0, 0, target));
    }
    if (inst & 0xff00_0000 == 0xd400_0000) return generic(inst, pc, true); // SVC, BRK, HVC, ...
    if (inst & 0xffc0_0000 == 0xd500_0000) { // system
        const l = bit(inst, 21);
        const op0 = bits(inst, 20, 19);
        const op1 = bits(inst, 18, 16);
        const crn = bits(inst, 15, 12);
        const crm = bits(inst, 11, 8);
        const op2 = bits(inst, 7, 5);
        const rt = c.rd(inst);
        if (op0 == 0 and !l and op1 == 3 and rt == 31) {
            if (crn == 2) return nop(); // hints
            if (crn == 3 and op2 == 6) return generic(inst, pc, true); // ISB: a context synchronisation point
            if (crn == 3 and (op2 == 4 or op2 == 5) and !c.smp) return nop(); // DSB, DMB on one CPU
        }
        // MRS/MSR TPIDR_EL0
        if (op0 == 3 and op1 == 3 and crn == 13 and crm == 0 and op2 == 2) {
            if (l) return if (rt == 31) nop() else mov(xz(rt), tpidr_slot, std.math.maxInt(u64));
            return mov(tpidr_slot, xz(rt), std.math.maxInt(u64));
        }
        return generic(inst, pc, false);
    }
    if (inst & 0xfe00_0000 == 0xd600_0000) { // BR, BLR, RET
        if (inst & 0x001f_fc1f != 0x001f_0000) return generic(inst, pc, true);
        return switch (bits(inst, 24, 21)) {
            0, 2 => ends(mk(.br, 0, xz(c.rn(inst)), 0, 0, 0, 0, 0, 0)),
            1 => ends(mk(.blr, 0, xz(c.rn(inst)), 0, 0, 0, 0, 0, 0)),
            else => generic(inst, pc, true),
        };
    }
    return generic(inst, pc, true);
}

fn ldst(inst: u32, pc: u64) Decoded {
    if (inst & 0xbe00_0000 == 0x0c00_0000) return generic(inst, pc, false); // AdvSIMD structures
    if (inst & 0x3f00_0000 == 0x0800_0000) return generic(inst, pc, false); // exclusive, ordered
    if (inst & 0x3b00_0000 == 0x1800_0000) return literal(inst, pc);
    if (inst & 0x3800_0000 == 0x2800_0000) return pair(inst, pc);
    if (inst & 0x3800_0000 == 0x3800_0000) return single(inst, pc);
    return generic(inst, pc, false);
}

const ld_i = [4]H{ .ldr8_i, .ldr16_i, .ldr32_i, .ldr64_i };
const ld_w = [4]H{ .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w };
const ld_r = [4]H{ .ldr8_r, .ldr16_r, .ldr32_r, .ldr64_r };
const st_i = [4]H{ .str8_i, .str16_i, .str32_i, .str64_i };
const st_w = [4]H{ .str8_w, .str16_w, .str32_w, .str64_w };
const st_r = [4]H{ .str8_r, .str16_r, .str32_r, .str64_r };

/// The op of an integer load whose destination (Rt or a pair's Rt2) is XZR,
/// which the tier leaves `generic` (a load into slot 32 would clobber the
/// XZR that ops read there): the JIT translates it with the load kind,
/// whose write to XZR it drops (jit/emit.zig). V8 pops its padding register
/// this way (`ldp xzr, x30, [sp], #16`). Null for every other word.
pub fn xzrLoad(inst: u32, pc: u64) ?Op {
    if (bits(inst, 28, 25) & 0b0101 != 0b0100) return null; // not a load/store
    if (inst & 0xbe00_0000 == 0x0c00_0000 or inst & 0x3f00_0000 == 0x0800_0000) return null;
    const d = if (inst & 0x3b00_0000 == 0x1800_0000) literalZ(inst, pc, true) else if (inst & 0x3800_0000 == 0x2800_0000) pairZ(inst, pc, true) else if (inst & 0x3800_0000 == 0x3800_0000) singleZ(inst, pc, true) else return null;
    return switch (d.op.h) {
        .ldr8_i, .ldr16_i, .ldr32_i, .ldr64_i, .ldr8_w, .ldr16_w, .ldr32_w, .ldr64_w, .ldr8_r, .ldr16_r, .ldr32_r, .ldr64_r, .ldp32, .ldp64, .ldp32_w, .ldp64_w => if (d.op.a == zr or d.op.c == zr and (d.op.h == .ldp32 or d.op.h == .ldp64 or d.op.h == .ldp32_w or d.op.h == .ldp64_w)) d.op else null,
        else => null,
    };
}

fn literal(inst: u32, pc: u64) Decoded {
    return literalZ(inst, pc, false);
}
fn literalZ(inst: u32, pc: u64, comptime zr_ok: bool) Decoded {
    const va = pc +% c.sextN(@as(u64, bits(inst, 23, 5)) << 2, 21);
    const t = c.rd(inst);
    const opc = bits(inst, 31, 30);
    if (bit(inst, 26)) {
        return switch (opc) {
            0 => mk(.ldr_v, vs(t), zr, 0, 2, 0, 0, 0, va),
            1 => mk(.ldr_v, vs(t), zr, 0, 3, 0, 0, 0, va),
            2 => mk(.ldr_q_i, vs(t), zr, 0, 0, 0, 0, 0, va),
            else => generic(inst, pc, false),
        };
    }
    if (opc == 3) return nop(); // PRFM (literal)
    if (t == 31 and !zr_ok) return generic(inst, pc, false);
    return switch (opc) {
        0 => mk(.ldr32_i, xz(t), zr, 0, 0, 0, 0, 0, va),
        1 => mk(.ldr64_i, xz(t), zr, 0, 0, 0, 0, 0, va),
        else => mk(.ldr32_i, xz(t), zr, 0, 32, 0, 0, 0, va), // LDRSW
    };
}

fn pair(inst: u32, pc: u64) Decoded {
    return pairZ(inst, pc, false);
}
fn pairZ(inst: u32, pc: u64, comptime zr_ok: bool) Decoded {
    const idx = bits(inst, 25, 23); // 0 no-allocate, 1 post, 2 offset, 3 pre
    if (idx > 3) return generic(inst, pc, false);
    const opc = bits(inst, 31, 30);
    const simd = bit(inst, 26);
    const load = bit(inst, 22);
    if (opc == 3) return generic(inst, pc, false);
    const scale: u5 = if (simd) @intCast(2 + opc) else if (opc == 2) 3 else 2;
    const signed = !simd and opc == 1;
    if (signed and (!load or idx == 0)) return generic(inst, pc, false);
    const off = c.sextN(bits(inst, 21, 15), 7) << scale;
    const n = xs(c.rn(inst));
    const t = c.rd(inst);
    const t2 = c.rt2(inst);
    const wb = idx == 1 or idx == 3;
    const pre: u8 = @intFromBool(idx != 1);
    if (!simd) {
        if (load and (t == 31 or t2 == 31) and !zr_ok) return generic(inst, pc, false);
        const h: H = if (load)
            (if (scale == 3) (if (wb) H.ldp64_w else .ldp64) else (if (wb) H.ldp32_w else .ldp32))
        else if (scale == 3) (if (wb) H.stp64_w else .stp64) else (if (wb) H.stp32_w else .stp32);
        return mk(h, xz(t), n, xz(t2), if (signed) 32 else 0, pre, 0, 0, off);
    }
    if (scale == 4) {
        const h: H = if (load) (if (wb) H.ldp_q_w else .ldp_q) else if (wb) H.stp_q_w else .stp_q;
        return mk(h, vs(t), n, vs(t2), 0, pre, 0, 0, off);
    }
    if (load) return generic(inst, pc, false); // LDP S, D
    const h: H = if (scale == 3) (if (wb) H.stp64_w else .stp64) else (if (wb) H.stp32_w else .stp32);
    return mk(h, vs(t), n, vs(t2), 0, pre, 0, 0, off);
}

fn single(inst: u32, pc: u64) Decoded {
    return singleZ(inst, pc, false);
}
fn singleZ(inst: u32, pc: u64, comptime zr_ok: bool) Decoded {
    const Mode = enum { imm, post, pre, regoff };
    const mode: Mode = if (bit(inst, 24)) .imm else if (!bit(inst, 21)) switch (bits(inst, 11, 10)) {
        1 => .post,
        3 => .pre,
        else => .imm, // unscaled, unprivileged
    } else if (bits(inst, 11, 10) == 2) .regoff else return generic(inst, pc, false);
    if (bit(inst, 25)) return generic(inst, pc, false);
    const unpriv = !bit(inst, 24) and !bit(inst, 21) and bits(inst, 11, 10) == 2;
    const size = bits(inst, 31, 30);
    const opc = bits(inst, 23, 22);
    const simd = bit(inst, 26);
    const Kind = enum { store, zext, sext64, sext32, prfm };
    var scale: u5 = @intCast(size);
    var kind: Kind = undefined;
    if (simd) {
        if (opc >= 2) {
            if (size != 0) return generic(inst, pc, false);
            scale = 4;
        }
        if (unpriv) return generic(inst, pc, false);
        kind = if (opc & 1 == 0) .store else .zext;
    } else kind = switch (opc) {
        0 => .store,
        1 => .zext,
        2 => if (size == 3) .prfm else .sext64,
        3 => if (size <= 1) .sext32 else return generic(inst, pc, false),
        else => unreachable,
    };
    if (kind == .prfm and (mode == .post or mode == .pre or unpriv)) return generic(inst, pc, false);
    var x: Xform = undefined;
    var off: u64 = 0;
    switch (mode) {
        .imm => off = if (bit(inst, 24)) @as(u64, bits(inst, 21, 10)) << scale else c.sextN(bits(inst, 20, 12), 9),
        .regoff => {
            const option = bits(inst, 15, 13);
            if (option & 2 == 0) return generic(inst, pc, false);
            const k: u8 = if (bit(inst, 12)) scale else 0;
            x = if (option & 1 == 1) .{ .s1 = k, .s2 = 0, .mask = std.math.maxInt(u64) } // LSL, SXTX
            else .{ .s1 = 32, .s2 = 32 - k, .mask = if (option & 4 != 0) std.math.maxInt(u64) else @as(u64, 0xffff_ffff) << @intCast(k) }; // SXTW, UXTW
        },
        else => off = c.sextN(bits(inst, 20, 12), 9),
    }
    if (kind == .prfm) return nop();
    const t = c.rd(inst);
    const n = xs(c.rn(inst));
    const rm = xz(c.rm(inst));
    const pre: u8 = @intFromBool(mode == .pre);
    if (kind == .store) {
        const data = if (simd) vs(t) else xz(t);
        if (scale == 4) return switch (mode) {
            .imm => mk(.str_q_i, data, n, 0, 0, 0, 0, 0, off),
            .pre, .post => mk(.str_q_w, data, n, 0, 0, 0, pre, 0, off),
            .regoff => mk(.str_q_r, data, n, rm, 0, x.s1, x.s2, 0, x.mask),
        };
        return switch (mode) {
            .imm => mk(st_i[scale], data, n, 0, 0, 0, 0, 0, off),
            .pre, .post => mk(st_w[scale], data, n, 0, 0, 0, pre, 0, off),
            .regoff => mk(st_r[scale], data, n, rm, 0, x.s1, x.s2, 0, x.mask),
        };
    }
    if (simd) {
        const v = vs(t);
        if (scale == 4) return switch (mode) {
            .imm => mk(.ldr_q_i, v, n, 0, 0, 0, 0, 0, off),
            .pre, .post => mk(.ldr_q_w, v, n, 0, 0, 0, pre, 0, off),
            .regoff => mk(.ldr_q_r, v, n, rm, 0, x.s1, x.s2, 0, x.mask),
        };
        return switch (mode) {
            .imm => mk(.ldr_v, v, n, 0, scale, 0, 0, 0, off),
            .pre, .post => mk(.ldr_v, v, n, 0, scale, 1, pre, 0, off),
            .regoff => mk(.ldr_v, v, n, rm, scale, 2, x.s1, x.s2, x.mask),
        };
    }
    if (t == 31 and !zr_ok) return generic(inst, pc, false); // a load into XZR still accesses memory
    const width: u8 = @as(u8, 8) << @intCast(scale);
    const sx: u8 = if (kind == .zext) 0 else 64 - width;
    const sx32: u8 = @intFromBool(kind == .sext32);
    return switch (mode) {
        .imm => mk(ld_i[scale], xz(t), n, 0, sx, 0, 0, sx32, off),
        .pre, .post => mk(ld_w[scale], xz(t), n, 0, sx, 0, pre, sx32, off),
        .regoff => mk(ld_r[scale], xz(t), n, rm, sx, x.s1, x.s2, sx32, x.mask),
    };
}
