// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! The C ABI of include/kw_emu.h (A64) and the reference run loop
//! (`runRef`): one instruction at a time through decode.zig, the semantic
//! oracle of the decoded-op tier (tier.zig), which is what kw_run runs
//! unless the library is built with -Dkwemu-a64-ref. struct kw_cpu, the
//! Sv48-format page-table walker, the TLBs and flushes are kwemu.Cpu
//! (src/emu/kwemu.zig); the exclusive monitor's address is Hart.resv, so kw_clear_reservation and
//! kw_set_pgd clear it.

const std = @import("std");
const core = @import("../cpu/core.zig");
const kwemu = @import("../emu/kwemu.zig");
const fatwasm = @import("../emu/fatwasm.zig");
const c = @import("common.zig");
const decode = @import("decode.zig");
const branch = @import("branch.zig");
const tier = @import("tier.zig");
const kw_h = @import("kw_h");
const options = @import("kwemu_options");

/// kw_run is the reference loop (`runRef`) instead of the decoded-op tier.
pub const use_ref = options.a64_ref;

pub const Regs = c.Regs;
const Cpu = kwemu.Cpu;

/// Poll *irq_pending at least this often (instructions) on straight-line code.
const poll_every = 64;

comptime {
    // Layout must match struct kw_regs in include/kw_emu.h exactly.
    const H = kw_h.struct_kw_regs;
    std.debug.assert(@sizeOf(Regs) == @sizeOf(H));
    std.debug.assert(@sizeOf(Regs) == 800);
    std.debug.assert(@alignOf(Regs) == 16 and @alignOf(H) == 16);
    for (.{ "x", "sp", "pc", "nzcv", "fpcr", "fpsr", "_pad", "tpidr_el0", "v" }, .{ 0, 248, 256, 264, 268, 272, 276, 280, 288 }) |f, off| {
        std.debug.assert(@offsetOf(Regs, f) == @offsetOf(H, f));
        std.debug.assert(@offsetOf(Regs, f) == off);
    }
    // common.zig's register views rely on sp following x30.
    std.debug.assert(@offsetOf(Regs, "sp") == 31 * 8);
    // Optional features: common.Feature matches the KW_ISA_* bits.
    for (.{ "aes", "pmull", "sha1", "sha2", "crc32", "lse", "sha512", "sha3" }, .{ kw_h.KW_ISA_AES, kw_h.KW_ISA_PMULL, kw_h.KW_ISA_SHA1, kw_h.KW_ISA_SHA2, kw_h.KW_ISA_CRC32, kw_h.KW_ISA_LSE, kw_h.KW_ISA_SHA512, kw_h.KW_ISA_SHA3 }) |name, v|
        std.debug.assert(c.featureBit(@field(c.Feature, name)) == v);
    std.debug.assert(@typeInfo(c.Feature).@"enum".fields.len == 8);
}

fn event(cpu: *Cpu, kind: core.Kind, access: u32, addr: u64, insn: u64) u32 {
    cpu.ev = .{ .kind = @intFromEnum(kind), .access = access, .addr = addr, .insn = insn };
    return @intFromEnum(kind);
}

/// kw_run: the decoded-op tier (tier.run), or the reference loop.
pub fn run(cpu: *Cpu, regs: *Regs) u32 {
    return if (use_ref) runRef(cpu, regs) else tier.run(cpu, regs);
}

/// The reference loop: decode and execute one instruction at a time. It
/// polls *irq_pending after every taken branch and every 64 instructions.
pub fn runRef(cpu: *Cpu, regs: *Regs) u32 {
    const start = cpu.budget;
    var left = start;
    defer {
        cpu.budget = left;
        cpu.icount +%= start - left;
    }
    const irq = cpu.irq_pending;
    if (irq) |p| if (p.* != 0) return tier.stopEvent(cpu, .interrupt, .entry, 0);
    const h = &cpu.priv.hart;
    const s: c.St = .{ .r = regs, .h = h, .m = cpu };
    var since_poll: u32 = 0;
    while (true) {
        if (left == 0) return tier.stopEvent(cpu, .budget, .budget, 0);
        if (regs.pc & 3 != 0) return event(cpu, .misaligned, 0, regs.pc, 0);
        const e = decode.step(s) catch {
            cpu.priv.smc = 0;
            return event(cpu, .fault, cpu.priv.fault_access, cpu.priv.fault_addr, 0);
        };
        // Set by stores into pages the tier's block cache tracks and by
        // IC IVAU; nothing here is decoded ahead, so just clear it.
        cpu.priv.smc = 0;
        switch (e) {
            .next => {
                left -= 1;
                since_poll += 1;
                if (since_poll < poll_every) continue;
            },
            .jump => left -= 1,
            .trap => if (h.trap == .counter) {
                // MRS CNTVCT_EL0: retires with the instructions retired before it.
                branch.finishCounter(s, cpu.icount +% (start - left));
                left -= 1;
                since_poll += 1;
                if (since_poll < poll_every) continue;
            } else return switch (h.trap) {
                .syscall => blk: {
                    left -= 1; // SVC retires
                    regs.pc +%= 4;
                    break :blk event(cpu, .syscall, 0, 0, 0);
                },
                .breakpoint => event(cpu, .breakpoint, 0, regs.pc, h.trap_val),
                .misaligned => event(cpu, .misaligned, 0, h.trap_val, 0),
                else => event(cpu, .illegal, 0, regs.pc, h.trap_val),
            },
        }
        since_poll = 0;
        if (irq) |p| if (p.* != 0) return tier.stopEvent(cpu, .interrupt, .reference, 0);
    }
}

pub fn cpuInit(cpu: *Cpu) void {
    kwemu.cpuInit(cpu);
    cpu.priv.a64.isa = c.implemented;
    tier.epochInit(cpu);
}

pub fn setIsa(cpu: *Cpu, isa: u64) u64 {
    cpu.priv.a64.isa = c.consistentIsa(isa & c.implemented);
    // Decoded blocks need not depend on it (feature instructions are
    // generic ops, which check it when they run), but drop them anyway.
    cpu.resetCache();
    return cpu.priv.a64.isa;
}

/// Export the C ABI (from the library root only, so that the unit tests can
/// link against the native archive without duplicate symbols).
pub fn exportAbi() void {
    @export(&kw_cpu_init, .{ .name = "kw_cpu_init" });
    @export(&kw_run, .{ .name = "kw_run" });
    @export(&kw_set_pgd, .{ .name = "kw_set_pgd" });
    @export(&kw_flush_tlb_all, .{ .name = "kw_flush_tlb_all" });
    @export(&kw_flush_tlb_range, .{ .name = "kw_flush_tlb_range" });
    @export(&kw_clear_reservation, .{ .name = "kw_clear_reservation" });
    @export(&kw_abi_version, .{ .name = "kw_abi_version" });
    @export(&kw_isa_supported, .{ .name = "kw_isa_supported" });
    @export(&kw_set_isa, .{ .name = "kw_set_isa" });
    @export(&kw_host_reset, .{ .name = "kw_host_reset" });
    // Not part of the ABI: the reference loop for differential tests (the
    // harness's --lockstep and --ref), where the last interrupt or budget
    // stop was seen (the kernel's profiler), and the counters of a -Dkwemu-stats
    // build, with the lookup from an emit buffer to its cpu that a host
    // (js/core/jit.ts) needs to read them.
    @export(&kw_debug_run_ref, .{ .name = "kw_debug_run_ref" });
    @export(&kw_debug_last_stop, .{ .name = "kw_debug_last_stop" });
    @export(&kw_fat_reset, .{ .name = "kw_fat_reset" });
    @export(&kw_fat_register, .{ .name = "kw_fat_register" });
    if (kwemu.stats_on) @export(&kw_debug_stats, .{ .name = "kw_debug_stats" });
    if (kwemu.stats_on and @import("../emu/jitcore.zig").enabled)
        @export(&kwemu.kw_debug_cpu_of_jit_buf, .{ .name = "kw_debug_cpu_of_jit_buf" });
}

fn kw_fat_reset() callconv(.c) void {
    fatwasm.reset();
}
fn kw_fat_register(id: u32, flags: u32, ptr: [*]const u8, len: usize) callconv(.c) i32 {
    return fatwasm.register(id, flags, ptr, len);
}

fn kw_debug_run_ref(cpu: *Cpu, regs: *Regs) callconv(.c) u32 {
    return runRef(cpu, regs);
}
/// How the last kw_run that returned KW_EV_INTERRUPT or KW_EV_BUDGET saw
/// it (tier.StopTier), and in `block` the start pc of the block that ran
/// before (0: unknown). For the kernel's profiler (arch/wasm/kernel/prof.c).
fn kw_debug_last_stop(cpu: *Cpu, block: *u64) callconv(.c) u32 {
    block.* = cpu.priv.a64.stop_block;
    return cpu.priv.a64.stop_tier;
}
/// Word by word through volatile pointers (a struct copy would call memmove).
fn kw_debug_stats(cpu: *Cpu, out: *kwemu.Stats) callconv(.c) void {
    const n = @sizeOf(kwemu.Stats) / 8;
    @import("tier.zig").axCore(cpu);
    const src: *volatile [n]u64 = @ptrCast(&cpu.priv.stats);
    const dst: *volatile [n]u64 = @ptrCast(out);
    for (0..n) |i| dst[i] = src[i];
}

fn kw_cpu_init(cpu: *Cpu) callconv(.c) void {
    cpuInit(cpu);
}
fn kw_run(cpu: *Cpu, regs: *Regs) callconv(.c) u32 {
    return run(cpu, regs);
}
fn kw_set_pgd(cpu: *Cpu, pgd: u64) callconv(.c) void {
    kwemu.setPgd(cpu, pgd);
}
fn kw_flush_tlb_all(cpu: *Cpu) callconv(.c) void {
    cpu.flushAll();
}
fn kw_flush_tlb_range(cpu: *Cpu, start: u64, end: u64) callconv(.c) void {
    cpu.flushRange(start, end);
}
fn kw_clear_reservation(cpu: *Cpu) callconv(.c) void {
    kwemu.clearReservation(cpu);
}
fn kw_abi_version() callconv(.c) u32 {
    return kwemu.abi_version;
}
fn kw_isa_supported() callconv(.c) u64 {
    return c.implemented;
}
fn kw_set_isa(cpu: *Cpu, isa: u64) callconv(.c) u64 {
    return setIsa(cpu, isa);
}
fn kw_host_reset(cpu: *Cpu) callconv(.c) void {
    fatwasm.forgetHost();
    kwemu.hostReset(cpu);
    tier.epochInit(cpu);
}
