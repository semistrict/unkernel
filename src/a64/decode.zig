// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! A64 top-level decode: routes a raw instruction to the group file that
//! owns it, following the Arm ARM's main encoding table (op0 = inst<28:25>)
//! and, for loads/stores and "Data Processing -- Scalar Floating-Point and
//! Advanced SIMD", its second-level tables down to the file boundaries.
//! Each group file exposes one `exec(s, inst) Error!Exit` that decodes the
//! rest and raises ILLEGAL (common.ill) for anything it does not implement.
//! This file is complete: adding instructions never needs a change here.

const c = @import("common.zig");
const fatwasm = @import("../emu/fatwasm.zig");
const dp_imm = @import("dp_imm.zig");
const branch = @import("branch.zig");
const ldst = @import("ldst.zig");
const simd_ldst = @import("simd_ldst.zig");
const dp_reg = @import("dp_reg.zig");
const fp = @import("fp.zig");
const simd_move = @import("simd_move.zig");
const simd_arith = @import("simd_arith.zig");
const crypto = @import("crypto.zig");

/// The file an encoding belongs to. `unallocated`: reserved, SME, SVE and
/// unallocated top-level space (always ILLEGAL here).
pub const Group = enum {
    unallocated,
    dp_imm,
    branch,
    ldst,
    simd_ldst,
    dp_reg,
    fp,
    simd_move,
    simd_arith,
    crypto,
};

pub fn route(inst: u32) Group {
    const op0 = c.bits(inst, 28, 25);
    return switch (op0) {
        // 0000 reserved (UDF) / SME, 0001 unallocated, 0010 SVE, 0011 unallocated
        0b0000, 0b0001, 0b0010, 0b0011 => .unallocated,
        0b1000, 0b1001 => .dp_imm,
        0b1010, 0b1011 => .branch,
        0b0100, 0b0110, 0b1100, 0b1110 => routeLdSt(inst),
        0b0101, 0b1101 => .dp_reg,
        0b0111, 0b1111 => routeSimdFp(inst),
        else => unreachable,
    };
}

/// Loads and stores: the AdvSIMD structure forms (LD1..LD4, ST1..ST4,
/// LD1R..LD4R, single and multiple, post-indexed or not) are
/// `0 Q 0011 0 x ...` with bit 26 (V) set; everything else (incl. the
/// scalar FP/SIMD register forms LDR/STR/LDP/STP of B..Q) is ldst.
fn routeLdSt(inst: u32) Group {
    if (inst & 0xbe00_0000 == 0x0c00_0000) return .simd_ldst;
    return .ldst;
}

/// "Data Processing -- Scalar Floating-Point and Advanced SIMD"
/// (op0 = inst<31:28>).
fn routeSimdFp(inst: u32) Group {
    const top = inst >> 28;
    // x0x1: scalar FP (conversions, 1/2/3-source, compare, select, imm).
    if (top & 0b0101 == 0b0001) return .fp;
    // 1100: SHA512, SHA3 (EOR3, BCAX, RAX1, XAR), SM3, SM4.
    if (top == 0b1100) return .crypto;
    if (top & 0b1000 != 0) return .unallocated;
    // AES, three-register SHA, two-register SHA.
    if (inst & 0xff3e_0c00 == 0x4e28_0800) return .crypto;
    if (inst & 0xff20_8c00 == 0x5e00_0000) return .crypto;
    if (inst & 0xff3e_0c00 == 0x5e28_0800) return .crypto;
    if (top & 1 != 0) {
        // AdvSIMD scalar (01x1): scalar copy (DUP element) is a move.
        if (inst & 0xdfe0_8400 == 0x5e00_0400) return .simd_move;
        return .simd_arith;
    }
    // AdvSIMD vector (0xx0).
    if (inst & 0x9fe0_8400 == 0x0e00_0400) return .simd_move; // copy: DUP, INS, UMOV, SMOV
    if (inst & 0xbf20_8c00 == 0x0e00_0000) return .simd_move; // table lookup: TBL, TBX
    if (inst & 0xbf20_8c00 == 0x0e00_0800) return .simd_move; // permute: UZP, TRN, ZIP
    if (inst & 0xbf20_8400 == 0x2e00_0000) return .simd_move; // extract: EXT
    if (inst & 0x9ff8_0400 == 0x0f00_0400) return .simd_move; // modified immediate: MOVI, MVNI, ORR, BIC, FMOV
    if (inst & 0x9f20_fc00 == 0x0e20_1c00) return .simd_move; // three same, logical: AND, BIC, ORR, ORN, EOR, BSL, BIT, BIF
    return .simd_arith;
}

/// Execute one fetched instruction at s.r.pc.
pub fn exec(s: c.St, inst: u32) c.Error!c.Exit {
    const e = switch (route(inst)) {
        .unallocated => if (inst & fatwasm.opcode_mask == fatwasm.opcode_base) try fatwasm.exec(s, inst) else return c.ill(s, inst),
        .dp_imm => try dp_imm.exec(s, inst),
        .branch => try branch.exec(s, inst),
        .ldst => try ldst.exec(s, inst),
        .simd_ldst => try simd_ldst.exec(s, inst),
        .dp_reg => try dp_reg.exec(s, inst),
        .fp => try fp.exec(s, inst),
        .simd_move => try simd_move.exec(s, inst),
        .simd_arith => try simd_arith.exec(s, inst),
        .crypto => try crypto.exec(s, inst),
    };
    if (e == .next) s.r.pc +%= 4;
    return e;
}

/// Fetch and execute one instruction (pc must be 4-byte aligned).
pub fn step(s: c.St) c.Error!c.Exit {
    return exec(s, try s.m.fetch(s.r.pc));
}
