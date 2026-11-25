// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! Fat ELF `.kw.wasm` support: precompiled Wasm modules embedded inside an
//! AArch64 ELF binary, installed into 16 dedicated `__indirect_function_table`
//! slots via the existing `kw_host_jit_compile` / `kw_host_jit_free` imports
//! and invoked by guest code through reserved UDF opcodes `0x0000_ff10..ff1f`
//! (`KW_WASM_CALL #idx`).
//!
//! Calling convention:
//!   * Guest places arguments in `x0..x3` and executes `UDF #(0xff10 + idx)`.
//!   * Entry `flags` specify which arguments are guest VA buffer pointers
//!     (bits 0..3), whether each pointer needs write access (bits 4..7, else
//!     read), and how buffer lengths are derived (bits 8..10):
//!       0: x2 bytes (default)
//!       1: x3 bytes
//!       2: x2 * 8 bytes
//!       3: x2 * 16 bytes
//!       4: x3 * x3 * 8 bytes (N x N double matrices)
//!       5: arg0 = 32 bytes, arg1.. = x2 bytes (e.g. SHA-256 state + data)
//!       6: x3 * 8 bytes
//!       7: x3 * 16 bytes
//!   * Each pointer buffer is walked page-by-page through the Sv48 MMU
//!     (`Cpu.translateContig`), demand-faulting cleanly on missing/A=0/D=0
//!     pages. If the slot is unloaded (native / `--no-jit`), or any buffer is
//!     not physically contiguous or writes into a page holding decoded code,
//!     `x0` receives `-1` so the guest falls back to standard AArch64 code.

const std = @import("std");
const c = @import("../a64/common.zig");
const jc = @import("jitcore.zig");

pub const enabled = jc.enabled;
pub const max_fns = 16;

/// Reserved UDF range `0x0000_ff10 .. 0x0000_ff1f` (`KW_WASM_CALL #idx`).
pub const opcode_base: u32 = 0x0000_ff10;
pub const opcode_mask: u32 = 0xffff_fff0;

pub const section_magic = "KWWASM\x00\x00";
pub const section_version: u32 = 1;

pub const flag_arg0_ptr: u32 = 1 << 0;
pub const flag_arg1_ptr: u32 = 1 << 1;
pub const flag_arg2_ptr: u32 = 1 << 2;
pub const flag_arg3_ptr: u32 = 1 << 3;
pub const flag_arg0_write: u32 = 1 << 4;
pub const flag_arg1_write: u32 = 1 << 5;
pub const flag_arg2_write: u32 = 1 << 6;
pub const flag_arg3_write: u32 = 1 << 7;
pub const flag_len_shift: u5 = 8;
pub const flag_len_mask: u32 = 0x7 << 8;
pub const flag_mem32: u32 = 1 << 16;

pub const len_x2: u32 = 0;
pub const len_x3: u32 = 1;
pub const len_x2_mul8: u32 = 2;
pub const len_x2_mul16: u32 = 3;
pub const len_x3_sq_mul8: u32 = 4;
pub const len_arg0_32_arg1_x2: u32 = 5;
pub const len_x3_mul8: u32 = 6;
pub const len_x3_mul16: u32 = 7;

const compile_warm: u32 = 1;
const sentinel: u64 = @bitCast(@as(i64, -1));

/// Host JIT imports, referenced only when `enabled` so non-JIT builds
/// (`kwemu.o`) keep zero `unkernel` imports.
const host = struct {
    extern "unkernel" fn kw_host_jit_caps() u32;
    extern "unkernel" fn kw_host_jit_compile(ptr: [*]const u8, len: usize, slot: u32, flags: u32) i32;
    extern "unkernel" fn kw_host_jit_free(slot: u32) void;
};

pub const FatFn = *const fn (a0: u64, a1: u64, a2: u64, a3: u64) callconv(.c) u64;

const SlotState = struct {
    loaded: bool = false,
    flags: u32 = 0,
};

var slots: [max_fns]SlotState = @splat(.{});
var last_stub: u32 = 0;

/// 16 distinct stub functions occupying dedicated slots in
/// `__indirect_function_table`. Each writes its own `id` to `last_stub` so
/// LLVM / wasm-ld ICF cannot merge them into a single table entry.
fn Stub(comptime id: u32) type {
    return struct {
        fn f(a0: u64, a1: u64, a2: u64, a3: u64) callconv(.c) u64 {
            _ = .{ a0, a1, a2, a3 };
            @as(*volatile u32, &last_stub).* = id;
            return sentinel;
        }
    };
}

const stubs: [max_fns]FatFn = blk: {
    var s: [max_fns]FatFn = undefined;
    for (0..max_fns) |i| s[i] = &Stub(i).f;
    break :blk s;
};

/// Unload all registered `.kw.wasm` modules, restoring their table stubs.
pub fn reset() void {
    if (!enabled) return;
    for (0..max_fns) |i| {
        if (slots[i].loaded) {
            host.kw_host_jit_free(@intCast(@intFromPtr(stubs[i])));
            slots[i] = .{};
        }
    }
}

/// Forget installed slots without calling `kw_host_jit_free` (used on
/// `kw_host_reset` after a snapshot restores into a fresh wasm instance).
pub fn forgetHost() void {
    for (0..max_fns) |i| slots[i] = .{};
}

/// Compile and install an embedded `.kw.wasm` module into slot `id` (0..15).
/// Returns 0 on success, or negative on failure / unsupported host.
pub fn register(id: u32, flags: u32, ptr: [*]const u8, len: usize) i32 {
    if (!enabled) return -1;
    if (id >= max_fns or len == 0) return -1;
    const caps = host.kw_host_jit_caps();
    if (caps & jc.cap_available == 0) return -1;
    const want_mem32 = (flags & flag_mem32) != 0;
    const have_mem32 = (caps & jc.cap_mem32) != 0;
    if (want_mem32 != have_mem32) return -1;
    const idx: usize = @intCast(id);
    const table_slot: u32 = @intCast(@intFromPtr(stubs[idx]));
    if (slots[idx].loaded) {
        host.kw_host_jit_free(table_slot);
        slots[idx] = .{};
    }
    const rc = host.kw_host_jit_compile(ptr, len, table_slot, compile_warm);
    if (rc != 0) return rc;
    slots[idx] = .{ .loaded = true, .flags = flags };
    return 0;
}

fn argLen(flags: u32, arg_idx: usize, x2: u64, x3: u64) ?u64 {
    const mode = (flags & flag_len_mask) >> flag_len_shift;
    return switch (mode) {
        len_x2 => x2,
        len_x3 => x3,
        len_x2_mul8 => if (x2 > (std.math.maxInt(u64) >> 3)) null else x2 << 3,
        len_x2_mul16 => if (x2 > (std.math.maxInt(u64) >> 4)) null else x2 << 4,
        len_x3_sq_mul8 => if (x3 > 0x5a82_7999) null else (x3 * x3) << 3,
        len_arg0_32_arg1_x2 => if (arg_idx == 0) 32 else x2,
        len_x3_mul8 => if (x3 > (std.math.maxInt(u64) >> 3)) null else x3 << 3,
        len_x3_mul16 => if (x3 > (std.math.maxInt(u64) >> 4)) null else x3 << 4,
        else => null,
    };
}

/// Execute `KW_WASM_CALL #idx` (`0x0000_ff10 .. 0x0000_ff1f`).
pub fn exec(s: c.St, inst: u32) c.Error!c.Exit {
    if (!enabled) {
        s.r.x[0] = sentinel;
        return .next;
    }
    const idx: usize = @intCast(inst & 0xf);
    const st = slots[idx];
    if (!st.loaded) {
        s.r.x[0] = sentinel;
        return .next;
    }
    const flags = st.flags;
    var a = [4]u64{ s.r.x[0], s.r.x[1], s.r.x[2], s.r.x[3] };
    inline for (0..4) |i| {
        if (flags & (@as(u32, 1) << @intCast(i)) != 0) {
            const len = argLen(flags, i, s.r.x[2], s.r.x[3]) orelse {
                s.r.x[0] = sentinel;
                return .next;
            };
            const write = flags & (@as(u32, 1) << @intCast(i + 4)) != 0;
            const host_addr = (try s.m.translateContig(a[i], len, write)) orelse {
                s.r.x[0] = sentinel;
                return .next;
            };
            a[i] = host_addr;
        }
    }
    // Load through volatile so LLVM always emits `call_indirect` into the
    // table slot patched by `kw_host_jit_compile` instead of devirtualizing.
    const fn_ptr = @as(*const volatile FatFn, &stubs[idx]).*;
    s.r.x[0] = fn_ptr(a[0], a[1], a[2], a[3]);
    return .next;
}

// ---- unit tests -----------------------------------------------------------------

test "a64 tier: KW_WASM_CALL unloaded returns -1 and neighbors stay illegal" {
    const t = std.testing;
    const mach = @import("../a64/machine.zig");
    var m = try mach.Machine.init(t.allocator);
    defer m.deinit();
    reset();

    // KW_WASM_CALL #0, KW_WASM_CALL #15, svc #0
    m.code(0x10000, &.{ opcode_base, opcode_base + 15, mach.svc0 });
    m.regs.x[0] = 123;
    try t.expectEqual(mach.EV_SYSCALL, m.run(100));
    try t.expectEqual(@as(u64, @bitCast(@as(i64, -1))), m.regs.x[0]);
    try t.expectEqual(@as(u64, 0x1000c), m.regs.pc);

    // Adjacent UDF encodings must still trap KW_EV_ILLEGAL.
    for ([_]u32{ opcode_base - 1, opcode_base + max_fns }) |bad| {
        m.code(0x20000, &.{bad});
        try t.expectEqual(mach.EV_ILLEGAL, m.run(10));
        try t.expectEqual(@as(u64, bad), m.cpu.ev.insn);
    }
}

test "a64 jit: fatwasm scalar, contiguous buffer, non-contiguous, SMC and fault" {
    if (!enabled) return error.SkipZigTest;
    const t = std.testing;
    const mach = @import("../a64/machine.zig");

    // Module 1: (import "env" "memory" (memory i64 1)) (export "f" (param i64 i64 i64 i64) (result i64) -> a0 + a1)
    const add_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x09, 0x01, 0x60, 0x04, 0x7e, 0x7e, 0x7e, 0x7e, 0x01, 0x7e,
        0x02, 0x0f, 0x01, 0x03, 'e',  'n',  'v',  0x06, 'm',  'e',  'm',  'o',  'r',  'y',  0x02, 0x04, 0x01,
        0x03, 0x02, 0x01, 0x00,
        0x07, 0x05, 0x01, 0x01, 'f',  0x00, 0x00,
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x7c, 0x0b,
    };

    // Module 2: if (a2 == 0) return 0; *(u64*)a0 = *(u64*)a1 + a3; return *(u64*)a0;
    const buf_wasm = [_]u8{
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x09, 0x01, 0x60, 0x04, 0x7e, 0x7e, 0x7e, 0x7e, 0x01, 0x7e,
        0x02, 0x0f, 0x01, 0x03, 'e',  'n',  'v',  0x06, 'm',  'e',  'm',  'o',  'r',  'y',  0x02, 0x04, 0x01,
        0x03, 0x02, 0x01, 0x00,
        0x07, 0x05, 0x01, 0x01, 'f',  0x00, 0x00,
        0x0a, 0x1f, 0x01, 0x1d, 0x00,
        0x20, 0x02, 0x50, 0x04, 0x40, 0x42, 0x00, 0x0f, 0x0b,
        0x20, 0x00, 0x20, 0x01, 0x29, 0x00, 0x00, 0x20, 0x03, 0x7c, 0x37, 0x00, 0x00,
        0x20, 0x00, 0x29, 0x00, 0x00, 0x0b,
    };

    var m = try mach.Machine.init(t.allocator);
    defer {
        reset();
        m.deinit();
    }
    reset();

    // 1. Scalar call via slot 0
    try t.expectEqual(@as(i32, 0), register(0, 0, &add_wasm, add_wasm.len));
    m.code(0x10000, &.{ opcode_base + 0, mach.svc0 });
    m.regs.x[0] = 40;
    m.regs.x[1] = 2;
    try t.expectEqual(mach.EV_SYSCALL, m.run(100));
    try t.expectEqual(@as(u64, 42), m.regs.x[0]);

    // 2. Buffer call via slot 3: contiguous 2-page dst and src
    const buf_flags = flag_arg0_ptr | flag_arg1_ptr | flag_arg0_write;
    try t.expectEqual(@as(i32, 0), register(3, buf_flags, &buf_wasm, buf_wasm.len));
    const dst_va: u64 = 0x40000;
    const src_va: u64 = 0x50000;
    m.mapData(dst_va, 2);
    m.mapData(src_va, 2);
    std.mem.writeInt(u64, m.at(src_va)[0..8], 1000, .little);

    m.code(0x11000, &.{ opcode_base + 3, mach.svc0 });
    m.regs.x[0] = dst_va;
    m.regs.x[1] = src_va;
    m.regs.x[2] = 8192; // spans 2 contiguous pages
    m.regs.x[3] = 234;
    try t.expectEqual(mach.EV_SYSCALL, m.run(100));
    try t.expectEqual(@as(u64, 1234), m.regs.x[0]);
    try t.expectEqual(@as(u64, 1234), std.mem.readInt(u64, m.at(dst_va)[0..8], .little));

    // 3. Non-contiguous physical pages -> returns -1 without writing dst
    const noncontig_va: u64 = 0x60000;
    const p0 = m.frame();
    _ = m.frame(); // gap frame
    const p2 = m.frame();
    m.map(noncontig_va, p0, 2 | 4); // R | W
    m.map(noncontig_va + 4096, p2, 2 | 4); // R | W, non-contiguous PA
    std.mem.writeInt(u64, m.at(noncontig_va)[0..8], 0xdeadbeef, .little);

    m.regs.pc = 0x11000;
    m.regs.x[0] = noncontig_va;
    m.regs.x[1] = src_va;
    m.regs.x[2] = 8192;
    m.regs.x[3] = 7;
    try t.expectEqual(mach.EV_SYSCALL, m.run(100));
    try t.expectEqual(sentinel, m.regs.x[0]);
    try t.expectEqual(@as(u64, 0xdeadbeef), std.mem.readInt(u64, m.at(noncontig_va)[0..8], .little));

    // 4. Write buffer hitting a code page (SMC tracking) -> returns -1 without writing
    const smc_va: u64 = 0x70000;
    const smc_pa = m.frame();
    const smc_code = [_]u32{ opcode_base + 3, mach.svc0 };
    @memcpy(m.bytes(smc_pa)[0 .. smc_code.len * 4], std.mem.sliceAsBytes(&smc_code));
    m.map(smc_va, smc_pa, 2 | 4 | 8); // R | W | X
    m.regs.pc = smc_va;
    m.regs.x[0] = smc_va + 64; // writable, same page as decoded code
    m.regs.x[1] = src_va;
    m.regs.x[2] = 8;
    m.regs.x[3] = 1;
    try t.expectEqual(mach.EV_SYSCALL, m.run(100));
    try t.expectEqual(sentinel, m.regs.x[0]);
    try t.expectEqual(@as(u64, 0), std.mem.readInt(u64, m.at(smc_va + 64)[0..8], .little));

    // 5. Read-only dst page -> raises KW_EV_FAULT with KW_ACCESS_WRITE and preserves x0/pc
    const ro_va: u64 = 0x80000;
    m.map(ro_va, m.frame(), 2); // R only
    m.regs.pc = 0x11000;
    m.regs.x[0] = ro_va;
    m.regs.x[1] = src_va;
    m.regs.x[2] = 8;
    m.regs.x[3] = 1;
    try t.expectEqual(mach.EV_FAULT, m.run(100));
    try t.expectEqual(@as(u32, 1), m.cpu.ev.access); // KW_ACCESS_WRITE
    try t.expectEqual(ro_va, m.cpu.ev.addr);
    try t.expectEqual(@as(u64, 0x11000), m.regs.pc);
    try t.expectEqual(ro_va, m.regs.x[0]);

    // 6. reset() frees slots and restores -1 fallback
    reset();
    m.regs.pc = 0x10000;
    m.regs.x[0] = 40;
    m.regs.x[1] = 2;
    try t.expectEqual(mach.EV_SYSCALL, m.run(100));
    try t.expectEqual(sentinel, m.regs.x[0]);
}

