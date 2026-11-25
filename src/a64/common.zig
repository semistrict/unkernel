// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! Helpers every A64 group file uses: the per-instruction context (`St`),
//! traps, register access (SP vs XZR forms, V registers and lanes), field
//! extraction, NZCV flags and conditions, shifted and extended register
//! operands, bitmask immediates, memory accesses with precise faults, and
//! FP status (FPCR rounding, FPSR flags, NaN propagation).
//!
//! Rules for group files (see .agents/a64.md):
//!  - an instruction that traps or faults has no architectural effect:
//!    read all operands and do every access that can fault before writing
//!    any register (stores of several parts: `probeWrite` first);
//!  - an encoding a file does not implement returns `ill(s, inst)`; never
//!    treat an unknown encoding as a no-op;
//!  - no u128 shifts by a variable amount and no u128 multiplication or
//!    division (compiler-rt calls on wasm; the kernel does not link them):
//!    use `lanes`/`fromLanes` and 64-bit halves.

const std = @import("std");
const core = @import("../cpu/core.zig");
const kwemu = @import("../emu/kwemu.zig");
pub const fpu = core.fpu;
pub const state = @import("state.zig");

pub const Regs = state.Regs;
pub const Cpu = kwemu.Cpu;
pub const Hart = core.Hart;
pub const Error = core.Error;
/// `.next`: retired, the dispatcher advances pc by 4. `.jump`: retired, the
/// instruction set pc itself (taken branches only). `.trap`: did not retire
/// (except SVC), pc unchanged, Hart.trap/trap_val say why.
pub const Exit = core.Exit;
/// Several CPUs share the memory (wasm atomics build, core.smp): exclusives
/// and barriers must be real atomics then.
pub const smp = core.smp;
pub const sext = core.sext;

/// Context of one instruction: registers, hart (traps, exclusive monitor),
/// memory. r.pc is the address of the instruction being executed.
pub const St = struct {
    r: *Regs,
    h: *Hart,
    m: *Cpu,

    /// The A64-private part of struct kw_cpu (state.Priv).
    pub inline fn priv(s: St) *state.Priv {
        return &s.m.priv.a64;
    }
};

// ---- optional features ---------------------------------------------------------
//
// One bit per optional feature, the KW_ISA_* bits of include/kw_emu.h
// (checked in run.zig). To add one: implement its encodings behind
// `has(s, .x)` (else `ill`), test it, then add it to `implemented`; the
// kernel then advertises it (AT_HWCAP) through kw_set_isa.

pub const Feature = enum(u6) { aes = 0, pmull = 1, sha1 = 2, sha2 = 3, crc32 = 4, lse = 5, sha512 = 6, sha3 = 7 };
pub inline fn featureBit(f: Feature) u64 {
    return @as(u64, 1) << @intFromEnum(f);
}
/// The features this build implements (kw_isa_supported()).
pub const implemented: u64 = featureBit(.aes) | featureBit(.pmull) | featureBit(.sha1) | featureBit(.sha2) | featureBit(.crc32) | featureBit(.lse) | featureBit(.sha512) | featureBit(.sha3);
/// The features of `isa` that can be on together: SHA512 needs SHA2 (the
/// architecture has no SHA-512 without SHA-256, and ID_AA64ISAR0_EL1.SHA2
/// = 2 means both), so a set without SHA2 loses SHA512.
pub fn consistentIsa(isa: u64) u64 {
    if (isa & featureBit(.sha2) == 0) return isa & ~featureBit(.sha512);
    return isa;
}
/// Whether feature `f` is implemented and enabled (folds to false at
/// compile time while it is not implemented).
pub inline fn has(s: St, comptime f: Feature) bool {
    return implemented & featureBit(f) != 0 and s.priv().isa & featureBit(f) != 0;
}

// ---- traps ---------------------------------------------------------------------

/// Unallocated or unimplemented encoding: KW_EV_ILLEGAL with the raw word.
pub fn ill(s: St, inst: u32) Exit {
    @branchHint(.cold);
    s.h.trap = .illegal;
    s.h.trap_val = inst;
    return .trap;
}

/// Alignment fault at `addr` (exclusives, acquire/release, pc): KW_EV_MISALIGNED.
pub fn misaligned(s: St, addr: u64) Exit {
    @branchHint(.cold);
    s.h.trap = .misaligned;
    s.h.trap_val = addr;
    return .trap;
}

/// SVC: KW_EV_SYSCALL (the run loop retires it and moves pc past it).
pub fn syscall(s: St) Exit {
    s.h.trap = .syscall;
    return .trap;
}

/// BRK: KW_EV_BREAKPOINT, pc at it, ev.insn = the raw word.
pub fn breakpoint(s: St, inst: u32) Exit {
    s.h.trap = .breakpoint;
    s.h.trap_val = inst;
    return .trap;
}

/// A taken branch to `target`.
pub inline fn jump(s: St, target: u64) Exit {
    s.r.pc = target;
    return .jump;
}

// ---- instruction fields --------------------------------------------------------

/// inst<hi:lo>.
pub inline fn bits(inst: u32, comptime hi: u5, comptime lo: u5) u32 {
    comptime std.debug.assert(hi >= lo);
    return (inst >> lo) & ((@as(u32, 1) << (hi - lo + 1)) - 1);
}
pub inline fn bit(inst: u32, comptime n: u5) bool {
    return (inst >> n) & 1 != 0;
}
pub inline fn rd(inst: u32) u32 {
    return inst & 31;
}
pub inline fn rn(inst: u32) u32 {
    return (inst >> 5) & 31;
}
pub inline fn rm(inst: u32) u32 {
    return (inst >> 16) & 31;
}
pub inline fn ra(inst: u32) u32 {
    return (inst >> 10) & 31;
}
/// Second transfer register of pairs (inst<14:10>).
pub inline fn rt2(inst: u32) u32 {
    return (inst >> 10) & 31;
}
/// Sign-extend the low `n` bits of `v` (n in 1..64, run-time).
pub inline fn sextN(v: u64, n: u7) u64 {
    const sh: u6 = @intCast(64 - @as(u32, n));
    return @bitCast(@as(i64, @bitCast(v << sh)) >> sh);
}

// ---- general registers ---------------------------------------------------------

inline fn gpr(s: St) *[32]u64 {
    return @ptrCast(&s.r.x); // x[0..30] then sp
}
/// Xn, XZR for 31.
pub inline fn xr(s: St, n: u32) u64 {
    return if (n == 31) 0 else s.r.x[n];
}
/// Wn, WZR for 31.
pub inline fn wr(s: St, n: u32) u32 {
    return @truncate(xr(s, n));
}
/// Xn, SP for 31.
pub inline fn xsp(s: St, n: u32) u64 {
    return gpr(s)[n];
}
/// Xn or Wn (zero-extended) by `sf`, XZR for 31.
pub inline fn reg(s: St, sf: bool, n: u32) u64 {
    const v = xr(s, n);
    return if (sf) v else v & 0xffff_ffff;
}
/// Xn or Wn (zero-extended) by `sf`, SP for 31.
pub inline fn regSp(s: St, sf: bool, n: u32) u64 {
    const v = xsp(s, n);
    return if (sf) v else v & 0xffff_ffff;
}
/// Write Xn; 31 is XZR (discarded).
pub inline fn setX(s: St, n: u32, v: u64) void {
    if (n != 31) s.r.x[n] = v;
}
/// Write Xn; 31 is SP.
pub inline fn setXsp(s: St, n: u32, v: u64) void {
    gpr(s)[n] = v;
}
/// Write a `sf`-sized result (Wn writes zero the upper half); 31 is XZR.
pub inline fn setR(s: St, sf: bool, n: u32, v: u64) void {
    setX(s, n, if (sf) v else v & 0xffff_ffff);
}
/// Write a `sf`-sized result; 31 is SP.
pub inline fn setRsp(s: St, sf: bool, n: u32, v: u64) void {
    setXsp(s, n, if (sf) v else v & 0xffff_ffff);
}

// ---- V registers and lanes -----------------------------------------------------

pub inline fn vq(s: St, n: u32) u128 {
    return s.r.v[n];
}
pub inline fn vd(s: St, n: u32) u64 {
    return @truncate(s.r.v[n]);
}
pub inline fn vs(s: St, n: u32) u32 {
    return @truncate(s.r.v[n]);
}
pub inline fn vh(s: St, n: u32) u16 {
    return @truncate(s.r.v[n]);
}
pub inline fn vb(s: St, n: u32) u8 {
    return @truncate(s.r.v[n]);
}
/// Write all 128 bits.
pub inline fn setVq(s: St, n: u32, v: u128) void {
    s.r.v[n] = v;
}
/// Scalar writes (Bn, Hn, Sn, Dn) zero the rest of the register.
pub inline fn setVd(s: St, n: u32, v: u64) void {
    s.r.v[n] = v;
}
pub inline fn setVs(s: St, n: u32, v: u32) void {
    s.r.v[n] = v;
}
pub inline fn setVh(s: St, n: u32, v: u16) void {
    s.r.v[n] = v;
}
pub inline fn setVb(s: St, n: u32, v: u8) void {
    s.r.v[n] = v;
}
/// Vector result: a 64-bit (Q = 0) operation zeroes the upper half.
pub inline fn setVec(s: St, n: u32, q: bool, v: u128) void {
    s.r.v[n] = if (q) v else v & std.math.maxInt(u64);
}
/// The lanes of a 128-bit value (lane 0 = least significant).
pub inline fn lanes(comptime T: type, v: u128) [16 / @sizeOf(T)]T {
    return @bitCast(v);
}
pub inline fn fromLanes(comptime T: type, a: [16 / @sizeOf(T)]T) u128 {
    // Through a vector: from an array, LLVM rewrites equal lanes
    // ((x << 64) | x) into a multiplication, __multi3 on wasm.
    const v: @Vector(16 / @sizeOf(T), T) = a;
    return @bitCast(v);
}
/// The 128-bit value hi:lo. Build V values this way, not with
/// `@as(u128, hi) << 64 | lo`: LLVM turns `(x << 64) | x` into a
/// multiplication, which is __multi3 on wasm.
pub inline fn v128(lo: u64, hi: u64) u128 {
    return fromLanes(u64, .{ lo, hi });
}
/// Lane `i` of `v` (T one of u8, u16, u32, u64).
pub inline fn lane(comptime T: type, v: u128, i: usize) T {
    return lanes(T, v)[i];
}
pub inline fn setLane(comptime T: type, v: u128, i: usize, x: T) u128 {
    var a = lanes(T, v);
    a[i] = x;
    return fromLanes(T, a);
}
/// Unsigned integer type of an element of 8 << size bits (size 0..3).
pub fn Elem(comptime size: u2) type {
    return std.meta.Int(.unsigned, @as(u16, 8) << size);
}
/// Replicate the low `esize` bits of `v` over 64 bits.
pub fn replicate(v: u64, esize: u7) u64 {
    if (esize >= 64) return v;
    var r = v & ((@as(u64, 1) << @intCast(esize)) - 1);
    var e: u7 = esize;
    while (e < 64) : (e *= 2) r |= r << @intCast(e);
    return r;
}

// ---- NZCV and conditions -------------------------------------------------------

pub const N: u32 = 1 << 31;
pub const Z: u32 = 1 << 30;
pub const C: u32 = 1 << 29;
pub const V: u32 = 1 << 28;

/// Result and flags of AddWithCarry.
pub const Flagged = struct { r: u64, nzcv: u32 };

/// AddWithCarry(a, b, carry) at 32 (sf = false) or 64 bits. For a
/// subtraction pass ~b and carry 1 (SUBS, CMP) or C (SBCS). The result is
/// zero-extended.
pub fn addWithCarry(sf: bool, a: u64, b: u64, carry: u1) Flagged {
    if (sf) return awc(u64, a, b, carry);
    return awc(u32, @truncate(a), @truncate(b), carry);
}
fn awc(comptime T: type, a: T, b: T, carry: u1) Flagged {
    const s1 = @addWithOverflow(a, b);
    const s2 = @addWithOverflow(s1[0], @as(T, carry));
    const r = s2[0];
    const msb = @bitSizeOf(T) - 1;
    var f: u32 = 0;
    if (r >> msb != 0) f |= N;
    if (r == 0) f |= Z;
    if (s1[1] | s2[1] != 0) f |= C;
    if (((a ^ r) & (b ^ r)) >> msb != 0) f |= V;
    return .{ .r = r, .nzcv = f };
}
/// N and Z of a logical result (C = V = 0), at 32 or 64 bits.
pub fn nzLogic(sf: bool, r: u64) u32 {
    const v = if (sf) r else r & 0xffff_ffff;
    const msb: u6 = if (sf) 63 else 31;
    var f: u32 = 0;
    if ((v >> msb) & 1 != 0) f |= N;
    if (v == 0) f |= Z;
    return f;
}
/// ConditionHolds(cond) for the flags in `nzcv` (bits 31..28).
pub fn condHolds(nzcv: u32, cond: u4) bool {
    const n = nzcv & N != 0;
    const z = nzcv & Z != 0;
    const c = nzcv & C != 0;
    const v = nzcv & V != 0;
    const r = switch (cond >> 1) {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c and !z,
        5 => n == v,
        6 => n == v and !z,
        7 => true,
        else => unreachable,
    };
    return if (cond & 1 != 0 and cond != 15) !r else r;
}

// ---- operand decoding ----------------------------------------------------------

/// ShiftReg: shift type 0 LSL, 1 LSR, 2 ASR, 3 ROR by `amount` (< datasize)
/// at 32 (sf = false) or 64 bits; the result is zero-extended.
pub fn shiftReg(sf: bool, v: u64, typ: u2, amount: u6) u64 {
    if (sf) return shiftT(u64, v, typ, amount);
    return shiftT(u32, @truncate(v), typ, @truncate(amount));
}
fn shiftT(comptime T: type, v: T, typ: u2, amount: std.math.Log2Int(T)) T {
    const S = std.meta.Int(.signed, @bitSizeOf(T));
    return switch (typ) {
        0 => v << amount,
        1 => v >> amount,
        2 => @bitCast(@as(S, @bitCast(v)) >> amount),
        3 => std.math.rotr(T, v, amount),
    };
}
/// ExtendReg: option 0..7 = UXTB, UXTH, UXTW, UXTX, SXTB, SXTH, SXTW, SXTX
/// of `v`, then shifted left by `shift` (0..4). 64-bit result; a 32-bit
/// operation uses its low half.
pub fn extendReg(v: u64, option: u3, shift: u3) u64 {
    const e: u64 = switch (option) {
        0 => v & 0xff,
        1 => v & 0xffff,
        2 => v & 0xffff_ffff,
        3 => v,
        4 => sext(v, 8),
        5 => sext(v, 16),
        6 => sext(v, 32),
        7 => v,
    };
    return e << shift;
}

pub const BitMasks = struct { wmask: u64, tmask: u64 };

/// DecodeBitMasks(immN, imms, immr, immediate, datasize) of the Arm ARM.
/// null where the pseudocode is UNDEFINED (reserved encodings), including
/// an element size larger than `datasize` (32 or 64). Results are
/// zero-extended to 64 bits for datasize 32.
pub fn decodeBitMasks(n: u1, imms: u6, immr: u6, immediate: bool, datasize: u7) ?BitMasks {
    const combined: u7 = (@as(u7, n) << 6) | @as(u7, ~imms);
    if (combined <= 1) return null; // len < 1
    const len: u3 = @intCast(6 - @as(u32, @clz(combined)));
    const esize: u7 = @as(u7, 1) << len;
    if (esize > datasize) return null;
    const levels: u6 = @intCast(esize - 1);
    if (immediate and imms & levels == levels) return null;
    const S = imms & levels;
    const R = immr & levels;
    const d = (S -% R) & levels;
    const welem = ror(ones(@as(u7, S) + 1), R, esize);
    const telem = ones(@as(u7, d) + 1);
    const dmask: u64 = if (datasize == 64) std.math.maxInt(u64) else 0xffff_ffff;
    return .{ .wmask = replicate(welem, esize) & dmask, .tmask = replicate(telem, esize) & dmask };
}
/// The low `k` bits set (k in 0..64).
pub inline fn ones(k: u7) u64 {
    return if (k >= 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(k)) - 1;
}
/// Rotate the low `esize` bits of `v` right by `r` (< esize).
fn ror(v: u64, r: u6, esize: u7) u64 {
    if (r == 0) return v;
    const m = ones(esize);
    return ((v >> r) | (v << @intCast(esize - r))) & m;
}

// ---- memory --------------------------------------------------------------------
//
// Cpu.load/store take any unsigned integer type up to u64 and handle
// unaligned and page-straddling accesses; they record the fault address and
// access themselves and return error.PageFault. A single access either
// happens completely or not at all. Instructions with several accesses
// (pairs, 128-bit, structures, DC ZVA) must not leave memory half written:
// call `probeWrite` over the whole range before the first store.

pub inline fn load(s: St, comptime T: type, va: u64) Error!T {
    return s.m.load(T, va);
}
pub inline fn store(s: St, comptime T: type, va: u64, v: T) Error!void {
    return s.m.store(T, va, v);
}
/// 16 bytes at va (two 64-bit accesses; no single-copy atomicity).
pub fn load128(s: St, va: u64) Error!u128 {
    const lo = try s.m.load(u64, va);
    const hi = try s.m.load(u64, va +% 8);
    return v128(lo, hi);
}
/// 16 bytes at va, all or nothing.
pub fn store128(s: St, va: u64, v: u128) Error!void {
    try probeWrite(s, va, 16);
    const h = lanes(u64, v);
    try s.m.store(u64, va, h[0]);
    try s.m.store(u64, va +% 8, h[1]);
}
/// Fault now, before anything is written, unless all of [va, va + len) is
/// writable (len in 1..4096).
pub fn probeWrite(s: St, va: u64, len: u64) Error!void {
    try s.m.probe(va, .write);
    const last = va +% (len - 1);
    if (last >> 12 != va >> 12) try s.m.probe(last, .write);
}
/// Whether va is a multiple of `size` (a power of two); if not, records an
/// alignment fault (the caller returns .trap).
pub inline fn checkAlign(s: St, va: u64, size: u64) bool {
    if (va & (size - 1) == 0) return true;
    _ = misaligned(s, va);
    return false;
}
/// A full barrier when CPUs share the memory (DMB, DSB, acquire/release);
/// nothing for one CPU.
pub inline fn barrier(s: St) void {
    if (smp) _ = @atomicRmw(u32, &s.h.fence_word, .Add, 0, .seq_cst);
}

// ---- FP status -----------------------------------------------------------------
//
// FP arithmetic is fpu.zig (exact for all rounding modes and flags, its own
// defaults, see there). Mapping to Arm: FPCR.RMode via `fpRound`, fpu flags to FPSR
// via `fpsrBits` (or `fpRaise`). Differences the FP code must handle itself:
// fpu returns the default NaN for every NaN result, Arm propagates NaN
// operands unless FPCR.DN (use `processNaNs` first); fpu detects tininess
// after rounding, Arm before rounding (UFC differs for results that round
// up to the smallest normal); FPCR.FZ (flush denormals, IDC) and FPCR.AHP
// are not modelled by fpu.

/// Opt-in Wasm-FP / no-FPSR mode bit (reserved/RAZ in ARMv8.0): when set
/// with RMode = RNE (0) and FZ = 0, FP instructions skip FPSR updates and
/// compiled code executes bare Wasm FP instructions directly.
pub const FPCR_WASM_FP: u32 = 1 << 16;
pub const FPCR_FZ16: u32 = 1 << 19;
pub const FPCR_RMODE_SHIFT = 22;
pub const FPCR_FZ: u32 = 1 << 24;
pub const FPCR_DN: u32 = 1 << 25;
pub const FPCR_AHP: u32 = 1 << 26;
/// FPCR bits that exist in ARMv8.0 without trapping support (the trap
/// enables IOE..IXE, IDE read as zero: they are not implemented; FZ16 is
/// FEAT_FP16, Len and Stride AArch32 only: RES0), plus FPCR_WASM_FP.
pub const FPCR_MASK: u32 = FPCR_AHP | FPCR_DN | FPCR_FZ | (3 << FPCR_RMODE_SHIFT) | FPCR_WASM_FP;
pub const FPCR_WASM_MASK: u32 = FPCR_FZ | (3 << FPCR_RMODE_SHIFT) | FPCR_WASM_FP;

/// Whether FPCR selects opt-in Wasm-FP / no-FPSR mode (RNE, FZ = 0, bit 16 set).
pub inline fn wasmFpMode(fpcr: u32) bool {
    return fpcr & FPCR_WASM_MASK == FPCR_WASM_FP;
}

pub const FPSR_IOC: u32 = 1 << 0;
pub const FPSR_DZC: u32 = 1 << 1;
pub const FPSR_OFC: u32 = 1 << 2;
pub const FPSR_UFC: u32 = 1 << 3;
pub const FPSR_IXC: u32 = 1 << 4;
pub const FPSR_IDC: u32 = 1 << 7;
pub const FPSR_QC: u32 = 1 << 27;
/// FPSR bits that exist for AArch64 (the AArch32 NZCV bits read as zero).
pub const FPSR_MASK: u32 = FPSR_QC | FPSR_IDC | FPSR_IXC | FPSR_UFC | FPSR_OFC | FPSR_DZC | FPSR_IOC;

/// fpu rounding mode for FPCR.RMode (RN, RP, RM, RZ).
pub fn fpRound(fpcr: u32) u32 {
    return switch ((fpcr >> FPCR_RMODE_SHIFT) & 3) {
        0 => fpu.RNE,
        1 => fpu.RUP,
        2 => fpu.RDN,
        else => fpu.RTZ,
    };
}
/// FPSR cumulative bits for fpu flags (fpu.NV ... fpu.NX).
pub fn fpsrBits(fl: u32) u32 {
    var r: u32 = 0;
    if (fl & fpu.NV != 0) r |= FPSR_IOC;
    if (fl & fpu.DZ != 0) r |= FPSR_DZC;
    if (fl & fpu.OF != 0) r |= FPSR_OFC;
    if (fl & fpu.UF != 0) r |= FPSR_UFC;
    if (fl & fpu.NX != 0) r |= FPSR_IXC;
    return r;
}
/// Accumulate fpu flags into FPSR (call once per instruction, after the
/// last thing that can fault).
pub inline fn fpRaise(s: St, fl: u32) void {
    if (!wasmFpMode(s.r.fpcr)) s.r.fpsr |= fpsrBits(fl);
}

/// FPProcessNaN: the quieted NaN `a` (raises NV for a signalling one), or
/// the default NaN under FPCR.DN. B is u16, u32 or u64.
pub fn processNaN(comptime B: type, a: B, fpcr: u32, fl: *u32) B {
    const F = fpu.Fmt(B);
    if (F.isSnan(a)) fl.* |= fpu.NV;
    return if (fpcr & FPCR_DN != 0) F.qnan else a | (@as(B, 1) << (F.mb - 1));
}
/// FPProcessNaNs: the NaN result of a two-operand operation, or null if
/// neither operand is a NaN (then compute the result normally).
pub fn processNaNs(comptime B: type, a: B, b: B, fpcr: u32, fl: *u32) ?B {
    const F = fpu.Fmt(B);
    if (F.isSnan(a)) return processNaN(B, a, fpcr, fl);
    if (F.isSnan(b)) return processNaN(B, b, fpcr, fl);
    if (F.isNan(a)) return processNaN(B, a, fpcr, fl);
    if (F.isNan(b)) return processNaN(B, b, fpcr, fl);
    return null;
}
/// FPProcessNaNs3 (operand order as the pseudocode passes them; FPMulAdd
/// passes the addend first).
pub fn processNaNs3(comptime B: type, a: B, b: B, c: B, fpcr: u32, fl: *u32) ?B {
    const F = fpu.Fmt(B);
    if (F.isSnan(a)) return processNaN(B, a, fpcr, fl);
    if (F.isSnan(b)) return processNaN(B, b, fpcr, fl);
    if (F.isSnan(c)) return processNaN(B, c, fpcr, fl);
    if (F.isNan(a)) return processNaN(B, a, fpcr, fl);
    if (F.isNan(b)) return processNaN(B, b, fpcr, fl);
    if (F.isNan(c)) return processNaN(B, c, fpcr, fl);
    return null;
}
