// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! kwemu: the guest CPU emulator as a freestanding library implementing
//! include/kw_emu.h. No allocator and no global mutable state: everything
//! lives in the caller's `struct kw_cpu` (KW_CPU_SIZE bytes) and
//! `struct kw_regs`.
//!
//! MMU: walks Sv48-layout page tables straight out of linear memory
//! (`phys_base + pa`), like hardware, and caches successful translations in
//! three direct-mapped TLBs (read / write / exec) inside kw_cpu. The
//! emulator never writes PTEs: A=0 (or D=0 on a store) raises KW_EV_FAULT.
//! Only translations that were fully permitted are cached, so the kernel
//! must call the flush hooks when it downgrades or removes a PTE (as on real
//! hardware); upgrades need no flush. Linear address 0 is valid
//! (phys_base + pa may be 0): all guest-memory pointers are allowzero.
//!
//! Execution: a cache of decoded blocks ("blocks", see the comment above
//! `Block`), ISA-neutral; the ISA (src/a64/tier.zig) decodes, validates and
//! runs them, and implements the C ABI (src/a64/run.zig).

const std = @import("std");
const core = @import("../cpu/core.zig");
const c = @import("kw_h"); // include/kw_emu.h via translate-c

pub const abi_version: u32 = 4;
pub const cpu_size: usize = 9 << 20;

pub const Event = extern struct {
    kind: u32 = 0,
    access: u32 = 0,
    addr: u64 = 0,
    insn: u64 = 0,
};

const page_shift = 12;
const page_size: u64 = 1 << page_shift;
const page_mask: u64 = page_size - 1;
/// Exec TLB sets.
const tlb_bits = 8;
const tlb_n = 1 << tlb_bits;
/// Data TLB entries (read and write each). The JIT derives its index mask
/// from the array length.
const dtlb_bits = 10;
const dtlb_n = 1 << dtlb_bits;
const invalid_tag = std.math.maxInt(u64);
/// Write-TLB tag flag: the page holds decoded code, so stores must take the
/// slow path (which invalidates the affected blocks). Never matches a vpn.
const code_flag: u64 = 1 << 63;

/// Data TLB entry: host address = va +% add.
const Tlb = extern struct { tag: u64, add: u64 };
/// Exec TLB entry. `gen` is unique per fill (0 = invalid); a block is only
/// trusted on the fast path while the exec entry for its page still has
/// the generation the block was validated under. The exec TLB is 2-way set
/// associative (`tlb_n` sets, entries 2i and 2i+1, the most recently filled
/// first): hot code pages whose vpns collide would otherwise refill each
/// other, and every refill gets a new generation, so all their blocks would
/// be revalidated again and again.
const XTlb = extern struct { tag: u64, add: u64, gen: u64, _pad: u64 = 0 };

const pte_v: u64 = 1 << 0;
const pte_r: u64 = 1 << 1;
const pte_w: u64 = 1 << 2;
const pte_x: u64 = 1 << 3;
const pte_u: u64 = 1 << 4;
const pte_a: u64 = 1 << 6;
const pte_d: u64 = 1 << 7;
const ppn_mask: u64 = (1 << 44) - 1;

// ---- decoded-block cache ----------------------------------------------------------
//
// A block (a superblock: single entry, side exits) is a run of guest
// instructions inside one page in execution order, decoded by the ISA's
// hooks (`lookupIn`; A64: src/a64/tier.zig) into 16-byte op slots of the
// op arena, one op per instruction, so op index == retired-instruction
// count. Blocks are found through a 2-way set-associative table keyed by
// the virtual pc and are trusted without checks only while
//   block.tgen == exec-TLB[pc page].gen.
// Every exec-TLB fill gets a fresh generation, and kw_flush_tlb_all,
// kw_flush_tlb_range (for that page), kw_set_pgd and a guest's instruction
// cache invalidation (A64 IC IVAU, `icInvalidate`) all retire generations.
// A block whose generation is stale is revalidated before it runs: same
// virtual pc, same host (physical) address and the same raw instruction
// bits (the ISA keeps them with the ops); otherwise it is re-decoded. So
// remapping a VA, COW, page reuse and code rewritten by the kernel can
// never run stale code after the flush the kernel must issue anyway (or
// after the guest's invalidation).
//
// Guest stores are tracked exactly: pages holding decoded code are entered
// in `marks` (a hashed set of physical pages with a 64-byte-chunk bitmap),
// and write-TLB entries of such pages carry `code_flag` so stores take the
// slow path. A store hitting a marked chunk retires the generations of
// every exec mapping of that page and ends the current block right after
// the store, so self-modifying code is seen at the next instruction.
//
// The op arena (`nops`): blocks are decoded into the nursery; when it is
// full only the nursery is discarded (`resetNursery`): the blocks stay in
// the table with their compile countdown and are decoded again on their
// next use. A block's ops are copied to the tenured ring when it is
// compiled (`tenure`), so compiled code survives nursery resets; the ring
// makes room by dropping its oldest blocks and their code. The table's
// replacement keeps the way with more countdown progress (jitcore.zig,
// `victim`).

/// Block table: 2-way set associative (a pair of hot blocks whose pcs
/// collide must not evict each other forever).
const nsets = 16384;
const nblocks = 2 * nsets;
/// Op arena (16 bytes per op, 7 MiB). With 128K ops, node's startup
/// filled it every ~2M instructions (about 270 resets per `npx --version`,
/// each freeing every compiled block); 384K ops cut npx by ~7%.
/// Two parts: the nursery [0, Priv.nursery), where blocks are decoded, and
/// the tenured part after it, where a block's ops are copied when it is
/// compiled (`tenure`). A full nursery is reset without touching compiled
/// code (`resetNursery`); the tenured part is a ring whose oldest blocks
/// make room (their code is dropped).
const nops = 448 * 1024;
/// The tenured ring's size in ops (the nursery has the rest, 320K): it holds
/// the compiled set of one run of a Bun-based program, and a smaller
/// nursery made npm slower (a64/performance.md, Tenured ring).
pub const ten_default = 128 * 1024;
/// Tenured ops are numbered on from Priv.nursery for good (Block.op; the
/// ring position is that number modulo the ring's size, `opIndex`), so a
/// number names one copy and a module's member check (jitcore.zig,
/// checkedEntry) never mistakes a later copy for the one it was emitted
/// from. Past this number every tenured block is dropped and the numbering
/// starts again (`resetTenured`).
const ten_renumber: u32 = 0xf000_0000;
/// Block.op of a block whose ops were discarded by a nursery or tenured
/// reset: its pc, host and compile countdown stay (a re-decode of the same
/// pc and host takes the countdown over), and it is never revalidated.
pub const op_gone: u32 = std.math.maxInt(u32);
const nmarks = 4096;
const no_tgen = std.math.maxInt(u64);
const mark_wild = std.math.maxInt(u64);

const a64_state = @import("../a64/state.zig");
const a64_tier = @import("../a64/tier.zig");
/// One slot of the op arena; the ISA views the arena as its own 16-byte
/// op type (`opArena`).
pub const Slot = extern struct { lo: u64, hi: u64 };
/// The ISA-neutral JIT core (block state, slots, cache hooks).
const jc = @import("jitcore.zig");

pub const Block = extern struct {
    pc: u64,
    tgen: u64,
    /// Host address of the first instruction.
    host: u64,
    /// pc after the last instruction (terminator fall-through / link value).
    end_pc: u64,
    op: u32,
    /// Ops (incl. a trailing `goto` when the block does not end in a jump).
    n: u16,
    /// Guest instructions.
    ninsn: u16,
    /// 64-byte chunks of the page the block's instructions occupy.
    mask: u64,
    /// JIT tier (-Dkwemu-jit): compile countdown / compiled slot (see
    /// jitcore.zig); empty otherwise.
    jx: jc.BlockState = .{},
};

const Mark = extern struct { key: u64, mask: u64 };

pub const stats_on = @import("kwemu_options").stats;

/// Room for this many op kinds (the ISA's op enum, A64: ops.H) in the
/// per-op counters below.
pub const max_ops = 192;

pub const Stats = extern struct {
    lookups_slow: u64 = 0,
    revalidated: u64 = 0,
    decoded: u64 = 0,
    resets: u64 = 0,
    blocks: u64 = 0,
    block_insns: u64 = 0,
    single_steps: u64 = 0,
    walks_r: u64 = 0,
    walks_w: u64 = 0,
    walks_x: u64 = 0,
    generic_ops: u64 = 0,
    /// Executed ops per handler (ops.H); with the JIT tier, interpreted
    /// ops only. Room for `max_ops` kinds; A64 keeps further counters in
    /// the unused tail (tier.zig `Ar`). harness.zig reads this layout.
    op_hist: [max_ops]u64 = @splat(0),
    /// JIT tier counters; present only with -Dkwemu-stats, so the layout
    /// of struct kw_cpu without it is what it always was.
    jit: if (stats_on) JitStats else extern struct {} = .{},
};

pub const JitStats = extern struct {
    /// Instructions retired by compiled code, and hand-backs (compiled
    /// code returning "handed back at op k").
    insns: u64 = 0,
    handbacks: u64 = 0,
    /// Hand-backs per handler of the op handed back (ops.H).
    hb: [max_ops]u64 = @splat(0),
    /// A64: why instructions ran outside compiled code (tier.zig `Ax`).
    hb_gen: [40]u64 = @splat(0),
    /// How compiled code is entered and left, regions included (see jc.Jx).
    jx: [64]u64 = @splat(0),
    /// `generic` ops by word with the Rd and Rn fields cleared (A64,
    /// `countGenWord`): `gen_words_n` open-addressed entries of three words
    /// (the key | 1 << 32, executions, of those calls from compiled code);
    /// the last entry counts what finds no free entry.
    gen_words: [3 * gen_words_n]u64 = @splat(0),
};

pub const gen_words_n = 256;

/// -Dkwemu-stats: count one execution of the `generic` op `raw` in
/// JitStats.gen_words, or (`from_jit`) one call of it from compiled code.
pub fn countGenWord(cpu: *Cpu, raw: u32, from_jit: bool) void {
    if (!stats_on) return;
    const t = &cpu.priv.stats.jit.gen_words;
    const key = @as(u64, raw & 0xffff_fc00) | 1 << 32;
    var i: usize = @intCast(((raw >> 10) *% 0x9e37_79b1) >> 23);
    var probes: usize = 0;
    while (true) : (probes += 1) {
        if (probes == 16) {
            i = gen_words_n - 1;
            break;
        }
        if (i == gen_words_n - 1) i = 0;
        if (t[3 * i] == key) break;
        if (t[3 * i] == 0) {
            t[3 * i] = key;
            break;
        }
        i += 1;
    }
    t[3 * i + 1 + @intFromBool(from_jit)] += 1;
}

pub inline fn count(cpu: *Cpu, comptime field: []const u8, n: u64) void {
    if (stats_on) @field(cpu.priv.stats, field) += n;
}

/// Emulator-private part of struct kw_cpu.
const Priv = extern struct {
    hart: core.Hart,
    fault_addr: u64,
    fault_access: u32,
    /// Set by a store that hit decoded code / by IC IVAU: end the block.
    smc: u32,
    gen: u64,
    /// Next free op of the nursery / end of the nursery (= start of the
    /// tenured ring) / the ring's next free and oldest used op (numbers, see
    /// `ten_renumber`) / its size - 1 (a power of two; tests shrink both
    /// parts).
    ops_used: u64,
    nursery: u64,
    ten_head: u64,
    ten_tail: u64,
    ten_mask: u64,
    /// Counters, maintained only with -Dkwemu-stats (see `Stats`).
    stats: Stats,
    tlb_r: [dtlb_n]Tlb,
    tlb_w: [dtlb_n]Tlb,
    tlb_x: [2 * tlb_n]XTlb,
    marks: [nmarks]Mark,
    blocks: [nblocks]Block,
    /// Per set: the way the next new block replaces.
    victim: [nsets]u8,
    ops: [nops]Slot,
    /// JIT tier bookkeeping (-Dkwemu-jit); empty otherwise.
    jit: jc.State,
    /// A64 (src/a64): exclusive-monitor extras, enabled features.
    a64: a64_state.Priv,
};

inline fn hostLoad(comptime T: type, addr: u64) T {
    return std.mem.littleToNative(T, @as(*allowzero align(1) const T, @ptrFromInt(@as(usize, @intCast(addr)))).*);
}
inline fn hostStore(comptime T: type, addr: u64, v: T) void {
    @as(*allowzero align(1) T, @ptrFromInt(@as(usize, @intCast(addr)))).* = std.mem.nativeToLittle(T, v);
}
inline fn chunkMask(off: u64, len: u64) u64 {
    const lo: u6 = @intCast(off >> 6);
    const hi: u6 = @intCast((off + len - 1) >> 6);
    const top: u64 = if (hi == 63) std.math.maxInt(u64) else (@as(u64, 2) << hi) - 1;
    return top & ~((@as(u64, 1) << lo) - 1);
}
/// Index of way 0 of pc's set (way 1 follows it). Uses pc bits 1..28
/// (pc >> 16 instead of pc >> 15 left bit 15 out, so in every 64 KiB of
/// code each pc shared its set with pc ^ 0x8000, and a third hot block in
/// such a set made the three evict each other until the op arena reset).
pub inline fn blockSet(pc: u64) usize {
    return @intCast((((pc >> 1) ^ (pc >> 15)) & (nsets - 1)) * 2);
}

pub const Cpu = extern struct {
    phys_base: u64,
    pgd: u64,
    irq_pending: ?*const volatile u32,
    budget: u64,
    icount: u64,
    ev: Event,
    /// 1 while kwemu compiles (jitcore.zig windowOpen): another CPU may then set
    /// `flush_all` instead of waiting for this one's flush.
    jit_busy: u32,
    flush_all: u32,
    priv: Priv,

    // ---- TLB ------------------------------------------------------------------

    pub fn flushAll(self: *Cpu) void {
        // Volatile stores: keep LLVM from turning this into a memset call.
        inline for (.{ &self.priv.tlb_r, &self.priv.tlb_w }) |tlb| {
            const p: [*]volatile Tlb = tlb;
            for (0..dtlb_n) |i| p[i].tag = invalid_tag;
        }
        const x: [*]volatile XTlb = &self.priv.tlb_x;
        for (0..2 * tlb_n) |i| {
            x[i].tag = invalid_tag;
            x[i].gen = 0;
        }
    }

    pub fn flushRange(self: *Cpu, start: u64, end: u64) void {
        if (end <= start) return;
        const first = start >> page_shift;
        const last = (end - 1) >> page_shift;
        if (last - first >= tlb_n) return self.flushAll();
        var vpn = first;
        while (true) : (vpn += 1) {
            const i: usize = @intCast(vpn & (tlb_n - 1));
            const d: usize = @intCast(vpn & (dtlb_n - 1));
            if (self.priv.tlb_r[d].tag == vpn) self.priv.tlb_r[d].tag = invalid_tag;
            if (self.priv.tlb_w[d].tag & ~code_flag == vpn) self.priv.tlb_w[d].tag = invalid_tag;
            for (self.priv.tlb_x[2 * i ..][0..2]) |*x| {
                if (x.tag == vpn) {
                    x.tag = invalid_tag;
                    x.gen = 0;
                }
            }
            if (vpn == last) break;
        }
    }

    /// Retire every exec-TLB generation: each block is revalidated (same
    /// pc, host address and raw bits) before it runs again. Decoded code
    /// stays in the arena, so blocks that did not change cost one compare
    /// of their raw words.
    pub fn retireAll(self: *Cpu) void {
        for (&self.priv.tlb_x) |*e| {
            if (e.gen != 0) e.gen = self.newGen();
        }
    }

    /// A64 IC IVAU (by VA, to the point of unification; the I-cache is PIPT
    /// as CTR_EL0 says): retire the generations of every exec mapping of
    /// the physical page that holds `va`, and end the running block
    /// (`smc`). Cheap when that page holds no decoded code (the code-page
    /// filter says so): new code written for the first time costs only the
    /// two exec-TLB ways of va's own page. No permission check and no fault
    /// (an unmapped va has no decoded code that could still be trusted: its
    /// unmap was flushed).
    pub fn icInvalidate(self: *Cpu, va: u64) void {
        const vpn = va >> page_shift;
        for (self.priv.tlb_x[2 * @as(usize, @intCast(vpn & (tlb_n - 1))) ..][0..2]) |*e| {
            if (e.tag == vpn and e.gen != 0) e.gen = self.newGen();
        }
        if (self.hostPageOf(va)) |page| {
            if (self.isMarked(page)) {
                for (&self.priv.tlb_x) |*e| {
                    if (e.gen != 0 and e.tag != invalid_tag and (e.tag << page_shift) +% e.add == page) e.gen = self.newGen();
                }
            }
        }
        self.priv.smc = 1;
    }

    /// Host address of the page mapped at `va` (any permission), without a
    /// fault or a TLB fill; null if it is not mapped.
    fn hostPageOf(self: *Cpu, va: u64) ?u64 {
        const vpn = va >> page_shift;
        const x = self.priv.tlb_x[2 * @as(usize, @intCast(vpn & (tlb_n - 1))) ..][0..2];
        for (x) |e| if (e.tag == vpn) return (vpn << page_shift) +% e.add;
        const r = self.priv.tlb_r[@intCast(vpn & (dtlb_n - 1))];
        if (r.tag == vpn) return (vpn << page_shift) +% r.add;
        if (va >> 48 != 0) return null;
        var table = self.pgd;
        var level: u6 = 3;
        while (true) : (level -= 1) {
            const e = hostLoad(u64, self.phys_base +% table +% ((va >> (page_shift + 9 * @as(u6, level))) & 511) * 8);
            if (e & pte_v == 0) return null;
            if (e & (pte_r | pte_w | pte_x) != 0) {
                if (level != 0) return null;
                return self.phys_base +% (((e >> 10) & ppn_mask) << page_shift);
            }
            if (level == 0) return null;
            table = ((e >> 10) & ppn_mask) << page_shift;
        }
    }

    /// Host address of `va` if the read TLB maps all `len` bytes from it
    /// within one page (else null: take the per-access path).
    pub inline fn readFast(self: *Cpu, va: u64, comptime len: u64) ?u64 {
        const e = &self.priv.tlb_r[@intCast((va >> page_shift) & (dtlb_n - 1))];
        return if (e.tag == (va +% (len - 1)) >> page_shift) va +% e.add else null;
    }
    /// Host address of `va` if the write TLB maps all `len` bytes from it
    /// within one page and the page holds no decoded code (else null: take
    /// the per-access path, which tracks stores into code).
    pub inline fn writeFast(self: *Cpu, va: u64, comptime len: u64) ?u64 {
        const e = &self.priv.tlb_w[@intCast((va >> page_shift) & (dtlb_n - 1))];
        return if (e.tag == (va +% (len - 1)) >> page_shift) va +% e.add else null;
    }
    pub inline fn hostRead(comptime T: type, addr: u64) T {
        return hostLoad(T, addr);
    }
    pub inline fn hostWrite(comptime T: type, addr: u64, v: T) void {
        hostStore(T, addr, v);
    }

    inline fn newGen(self: *Cpu) u64 {
        self.priv.gen += 1;
        return self.priv.gen;
    }

    // ---- page-table walk ----------------------------------------------------

    fn fault(self: *Cpu, va: u64, access: core.Access) core.Error {
        @branchHint(.cold);
        self.priv.fault_addr = va;
        self.priv.fault_access = @intFromEnum(access);
        return error.PageFault;
    }

    /// Walk the page table for `va` and fill the TLB. Returns the host
    /// address of the page.
    noinline fn translate(self: *Cpu, va: u64, access: core.Access) core.Error!u64 {
        switch (access) {
            .read => count(self, "walks_r", 1),
            .write => count(self, "walks_w", 1),
            .exec => count(self, "walks_x", 1),
        }
        // User addresses are the whole 48 bits the four levels index (as
        // arm64's TTBR0 with 48-bit VAs): no sign-extended upper half.
        if (va >> 48 != 0) return self.fault(va, access);
        var table = self.pgd;
        var level: u6 = 3;
        const pte: u64 = while (true) : (level -= 1) {
            const idx = (va >> (page_shift + 9 * @as(u6, level))) & 511;
            const e = hostLoad(u64, self.phys_base +% table +% idx * 8);
            if (e & pte_v == 0) return self.fault(va, access);
            if (e & (pte_r | pte_w | pte_x) != 0) {
                if (level != 0) return self.fault(va, access); // no superpages
                break e;
            }
            if (level == 0) return self.fault(va, access);
            table = ((e >> 10) & ppn_mask) << page_shift;
        };
        const need: u64 = switch (access) {
            .read => pte_r | pte_u | pte_a,
            .write => pte_w | pte_u | pte_a | pte_d,
            .exec => pte_x | pte_u | pte_a,
        };
        if (pte & need != need) return self.fault(va, access);
        if (pte & (pte_w | pte_r) == pte_w) return self.fault(va, access); // reserved
        const host = self.phys_base +% (((pte >> 10) & ppn_mask) << page_shift);
        const vpn = va >> page_shift;
        const i: usize = @intCast(vpn & (tlb_n - 1));
        const add = host -% (vpn << page_shift);
        switch (access) {
            .read => self.priv.tlb_r[@intCast(vpn & (dtlb_n - 1))] = .{ .tag = vpn, .add = add },
            .write => self.priv.tlb_w[@intCast(vpn & (dtlb_n - 1))] = .{ .tag = if (self.isMarked(host)) vpn | code_flag else vpn, .add = add },
            .exec => {
                const x = self.priv.tlb_x[2 * i ..][0..2];
                if (x[0].tag != invalid_tag) x[1] = x[0];
                x[0] = .{ .tag = vpn, .add = add, .gen = self.newGen() };
            },
        }
        return host;
    }

    /// Host address of `va` for a read (no straddle handling).
    pub inline fn readHost(self: *Cpu, va: u64) core.Error!u64 {
        const vpn = va >> page_shift;
        const e = &self.priv.tlb_r[@intCast(vpn & (dtlb_n - 1))];
        if (e.tag == vpn) return va +% e.add;
        return (try self.translate(va, .read)) + (va & page_mask);
    }

    /// Host address of `va` for an exec fetch.
    inline fn execHost(self: *Cpu, va: u64) core.Error!u64 {
        const vpn = va >> page_shift;
        const x = self.priv.tlb_x[2 * @as(usize, @intCast(vpn & (tlb_n - 1))) ..][0..2];
        if (x[0].tag == vpn) return va +% x[0].add;
        if (x[1].tag == vpn) return va +% x[1].add;
        return (try self.translate(va, .exec)) + (va & page_mask);
    }

    /// Generation of pc's exec-TLB entry (0 if it has none).
    inline fn execGen(self: *Cpu, pc: u64) u64 {
        const vpn = pc >> page_shift;
        const x = self.priv.tlb_x[2 * @as(usize, @intCast(vpn & (tlb_n - 1))) ..][0..2];
        return if (x[0].tag == vpn) x[0].gen else if (x[1].tag == vpn) x[1].gen else 0;
    }

    /// Generation of the exec-TLB entry of pc's page (0 if it has none).
    pub fn execGenOf(self: *Cpu, pc: u64) u64 {
        return self.execGen(pc);
    }

    /// Host address of `va` for a write of `len` bytes that stays in the
    /// page; handles stores into pages with decoded code.
    pub fn writeHost(self: *Cpu, va: u64, len: u64) core.Error!u64 {
        const vpn = va >> page_shift;
        const i: usize = @intCast(vpn & (dtlb_n - 1));
        const e = &self.priv.tlb_w[i];
        if (e.tag == vpn) return va +% e.add;
        if (e.tag != vpn | code_flag) _ = try self.translate(va, .write);
        const host = va +% e.add;
        if (e.tag & code_flag != 0 and !self.codeStore(host, len)) e.tag = vpn; // page has no code left
        return host;
    }

    // ---- decoded-code tracking ------------------------------------------------

    inline fn markSlot(self: *Cpu, host_page: u64) *Mark {
        return &self.priv.marks[@intCast((host_page >> page_shift) & (nmarks - 1))];
    }

    fn isMarked(self: *Cpu, host: u64) bool {
        const page = host & ~page_mask;
        const m = self.markSlot(page);
        return m.mask != 0 and (m.key == page | 1 or m.key == mark_wild);
    }

    /// Record that `mask` chunks of `page` hold decoded code.
    fn markCode(self: *Cpu, page: u64, mask: u64) void {
        switch (self.markChunks(page, mask)) {
            .none => {},
            // Stores to this page must now take the slow path.
            .page => for (&self.priv.tlb_w) |*e| {
                if (e.tag != invalid_tag and e.tag & code_flag == 0 and (e.tag << page_shift) +% e.add == page)
                    e.tag |= code_flag;
            },
            // The slot became wild, or a wild slot got its first bits:
            // every page that hashes to it counts as marked now (isMarked),
            // so flag all their entries.
            .slot => {
                const idx = (page >> page_shift) & (nmarks - 1);
                for (&self.priv.tlb_w) |*e| {
                    if (e.tag != invalid_tag and e.tag & code_flag == 0 and ((((e.tag << page_shift) +% e.add) >> page_shift) & (nmarks - 1)) == idx)
                        e.tag |= code_flag;
                }
            },
        }
    }

    const Scan = enum { none, page, slot };

    /// The mark part of `markCode`: which write-TLB entries need the flag
    /// now. A slot keyed by one page flags that page's entries when its
    /// first bits are set (later fills check `isMarked`; a store clears an
    /// entry's flag only when the slot has no marked chunk left), so more
    /// chunks of a marked page need no scan. A wild slot (two pages or more
    /// share it) marks every page that hashes to it while its mask is not
    /// 0: when it becomes wild, or gets bits again after a store cleared
    /// them all, the entries of all those pages are flagged once; after
    /// that every fill and every entry of them is flagged already. (Before
    /// 2026-11-18 every mark of a wild slot scanned: about 300K scans of
    /// the 1024 entries per `npm --version`.)
    fn markChunks(self: *Cpu, page: u64, mask: u64) Scan {
        const m = self.markSlot(page);
        const key = page | 1;
        const was = m.mask;
        if (m.key == key) {
            m.mask |= mask;
            return if (was == 0) .page else .none;
        }
        if (m.key == mark_wild) {
            m.mask |= mask;
            return if (was == 0) .slot else .none;
        }
        if (m.key == 0 or was == 0) {
            m.key = key;
            m.mask = mask;
            return .page;
        }
        m.key = mark_wild;
        m.mask |= mask;
        return .slot;
    }

    /// A store of `len` bytes at host address `host` (in a code-flagged
    /// page). Invalidates the blocks it may hit. Returns whether the page
    /// still holds marked code.
    noinline fn codeStore(self: *Cpu, host: u64, len: u64) bool {
        const page = host & ~page_mask;
        const m = self.markSlot(page);
        if (m.key != page | 1 and m.key != mark_wild) return false;
        const bits = chunkMask(host & page_mask, len);
        if (m.mask & bits != 0) {
            m.mask &= ~bits;
            for (&self.priv.tlb_x) |*e| {
                if (e.gen != 0 and (m.key == mark_wild or (e.tag << page_shift) +% e.add == page))
                    e.gen = self.newGen();
            }
            self.priv.smc = 1;
        }
        return m.mask != 0;
    }

    /// Drop every decoded block and all compiled code (kw_cpu_init,
    /// kw_set_isa, kw_host_reset, tests).
    pub fn resetCache(self: *Cpu) void {
        count(self, "resets", 1);
        if (stats_on) for (&self.priv.blocks) |*b| a64_tier.axLost(self, b, true);
        if (jc.enabled) jc.releaseAll(self);
        // Volatile stores: no memset import in the wasm object.
        const bs: [*]volatile Block = &self.priv.blocks;
        for (0..nblocks) |i| {
            bs[i].pc = std.math.maxInt(u64);
            bs[i].tgen = no_tgen;
        }
        self.clearMarks();
        self.priv.ops_used = 0;
        self.priv.ten_head = self.priv.nursery;
        self.priv.ten_tail = self.priv.nursery;
    }

    fn clearMarks(self: *Cpu) void {
        const ms: [*]volatile Mark = &self.priv.marks;
        for (0..nmarks) |i| {
            ms[i].key = 0;
            ms[i].mask = 0;
        }
    }

    /// The nursery is full: discard the ops of every block decoded there.
    /// Compiled blocks keep running (their ops are in the tenured part); the
    /// other blocks stay in the table with their pc, host and compile
    /// countdown (`op_gone`), so a block that is decoded again goes on
    /// counting where it was instead of starting over.
    noinline fn resetNursery(self: *Cpu) void {
        count(self, "resets", 1);
        if (stats_on) a64_tier.axReset(self);
        // Code marks are rebuilt from the blocks whose ops stay: only
        // tenured blocks can still be valid.
        self.clearMarks();
        const bs: [*]volatile Block = &self.priv.blocks;
        for (0..nblocks) |i| {
            if (bs[i].pc == std.math.maxInt(u64) or bs[i].op == op_gone) continue;
            if (bs[i].op < self.priv.nursery) {
                if (jc.enabled) std.debug.assert(bs[i].jx.st < jc.compiled);
                bs[i].op = op_gone;
                bs[i].tgen = no_tgen;
            } else _ = self.markChunks(bs[i].host & ~page_mask, bs[i].mask);
        }
        // Stores into the marked pages take the slow path (markCode does
        // this per page).
        for (&self.priv.tlb_w) |*e| {
            if (e.tag != invalid_tag and e.tag & code_flag == 0 and self.isMarked((e.tag << page_shift) +% e.add))
                e.tag |= code_flag;
        }
        self.priv.ops_used = 0;
    }

    /// Free every compiled module and discard the ops of every block copied
    /// to the tenured ring (they keep pc, host and countdown, as in
    /// `resetNursery`); the numbering starts again. Their code marks stay
    /// until the next nursery reset (marks may cover more than decoded code).
    pub fn resetTenured(self: *Cpu) void {
        if (stats_on) a64_tier.axTenReset(self);
        if (jc.enabled) jc.releaseAll(self);
        const bs: [*]volatile Block = &self.priv.blocks;
        for (0..nblocks) |i| {
            if (bs[i].pc == std.math.maxInt(u64) or bs[i].op == op_gone) continue;
            if (bs[i].op >= self.priv.nursery) {
                bs[i].op = op_gone;
                bs[i].tgen = no_tgen;
            }
        }
        self.priv.ten_head = self.priv.nursery;
        self.priv.ten_tail = self.priv.nursery;
    }

    /// The tenured numbering is close to its end (`ten_renumber`).
    pub inline fn tenuredWorn(self: *const Cpu) bool {
        return self.priv.ten_head >= ten_renumber;
    }

    /// Arena index of the op numbered `op` (Block.op + k).
    pub inline fn opIndex(self: *const Cpu, op: u64) usize {
        const n = self.priv.nursery;
        return @intCast(if (op < n) op else n + ((op - n) & self.priv.ten_mask));
    }

    /// A record of the tenured ring: one slot before a block's ops.
    const TenRecord = extern struct {
        /// Block index (`no_block`: padding up to the ring's end).
        block: u32,
        /// Slots, this one included.
        size: u32,
        _: u64 = 0,
    };
    const no_block = std.math.maxInt(u32);

    fn tenRecord(self: *Cpu, at: u64) *volatile TenRecord {
        return @ptrCast(&self.priv.ops[self.opIndex(at)]);
    }

    /// Make room for `n` slots at the ring's head: the oldest records go.
    /// Their blocks' ops are discarded (`op_gone`, countdown and table entry
    /// kept, as in `resetNursery`) and their compiled code dropped.
    fn tenMakeRoom(self: *Cpu, n: u64) void {
        std.debug.assert(n <= self.priv.ten_mask + 1);
        while (self.priv.ten_head + n - self.priv.ten_tail > self.priv.ten_mask + 1) {
            const r = self.tenRecord(self.priv.ten_tail);
            if (r.block != no_block) {
                const b = &self.priv.blocks[r.block];
                if (b.op == self.priv.ten_tail + 1) {
                    if (stats_on) a64_tier.axTenEvict(self, b);
                    if (jc.enabled) jc.evict(self, b);
                    b.op = op_gone;
                    b.tgen = no_tgen;
                }
            }
            self.priv.ten_tail += r.size;
        }
    }

    /// b is about to be compiled: copy its `slots` arena slots (ops and
    /// whatever the ISA keeps after them) to the tenured ring, unless they
    /// are there already. The nursery copy stays as it is until the next
    /// nursery reset (`run` may still be running it). This may drop the
    /// ring's oldest blocks (`tenMakeRoom`): the caller checks the blocks it
    /// holds for `op_gone`.
    pub fn tenure(self: *Cpu, b: *Block, slots: u32) void {
        if (b.op >= self.priv.nursery) return;
        const need: u64 = slots + 1;
        const size = self.priv.ten_mask + 1;
        // A record does not wrap around the ring's end: pad up to it.
        const at = (self.priv.ten_head - self.priv.nursery) & self.priv.ten_mask;
        if (at + need > size) {
            self.tenMakeRoom(size - at);
            self.tenRecord(self.priv.ten_head).* = .{ .block = no_block, .size = @intCast(size - at) };
            self.priv.ten_head += size - at;
        }
        self.tenMakeRoom(need);
        const head = self.priv.ten_head;
        self.tenRecord(head).* = .{
            .block = @intCast((@intFromPtr(b) - @intFromPtr(&self.priv.blocks)) / @sizeOf(Block)),
            .size = @intCast(need),
        };
        const src: [*]const volatile Slot = @ptrCast(&self.priv.ops[b.op]);
        const dst: [*]volatile Slot = @ptrCast(&self.priv.ops[self.opIndex(head + 1)]);
        for (0..slots) |i| dst[i] = src[i];
        self.priv.ten_head = head + need;
        b.op = @intCast(head + 1);
    }

    // ---- memory interface used by the ISA's instructions ----------------------

    pub inline fn load(self: *Cpu, comptime T: type, va: u64) core.Error!T {
        const e = &self.priv.tlb_r[@intCast((va >> page_shift) & (dtlb_n - 1))];
        if (e.tag == (va +% (@sizeOf(T) - 1)) >> page_shift) return hostLoad(T, va +% e.add);
        return self.loadSlow(T, va);
    }

    noinline fn loadSlow(self: *Cpu, comptime T: type, va: u64) core.Error!T {
        if ((va & page_mask) + @sizeOf(T) <= page_size) return hostLoad(T, try self.readHost(va));
        const b0 = try self.readHost(va);
        const second = (va | page_mask) +% 1;
        const b1 = try self.readHost(second);
        const n0: usize = @intCast(second -% va); // wraps for an access that crosses 2^64
        var buf: [@sizeOf(T)]u8 = undefined;
        for (0..@sizeOf(T)) |k| buf[k] = if (k < n0) hostLoad(u8, b0 + k) else hostLoad(u8, b1 + (k - n0));
        return std.mem.readInt(T, &buf, .little);
    }

    /// Returns true if the store hit decoded code (the block must end).
    pub inline fn storeChk(self: *Cpu, comptime T: type, va: u64, v: T) core.Error!bool {
        const e = &self.priv.tlb_w[@intCast((va >> page_shift) & (dtlb_n - 1))];
        if (e.tag == (va +% (@sizeOf(T) - 1)) >> page_shift) {
            hostStore(T, va +% e.add, v);
            return false;
        }
        try self.storeSlow(T, va, v);
        return self.priv.smc != 0;
    }

    pub inline fn store(self: *Cpu, comptime T: type, va: u64, v: T) core.Error!void {
        _ = try self.storeChk(T, va, v);
    }

    noinline fn storeSlow(self: *Cpu, comptime T: type, va: u64, v: T) core.Error!void {
        const off = va & page_mask;
        if (off + @sizeOf(T) <= page_size) return hostStore(T, try self.writeHost(va, @sizeOf(T)), v);
        // Translate both pages first so a fault leaves memory untouched.
        const second = (va | page_mask) +% 1;
        const n0 = second -% va; // wraps for an access that crosses 2^64
        const b0 = try self.writeHost(va, n0);
        const b1 = try self.writeHost(second, @sizeOf(T) - n0);
        var buf: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &buf, v, .little);
        for (0..@sizeOf(T)) |k| {
            if (k < n0) hostStore(u8, b0 + k, buf[k]) else hostStore(u8, b1 + (k - n0), buf[k]);
        }
    }

    /// The instruction word at `va`, which is 4-byte aligned (the A64
    /// callers check pc first), so it stays in its page.
    pub inline fn fetch(self: *Cpu, va: u64) core.Error!u32 {
        return hostLoad(u32, try self.execHost(va));
    }

    /// For an atomic load (a load-exclusive when CPUs share the memory):
    /// read permission, in-page.
    pub inline fn readPtr(self: *Cpu, comptime T: type, va: u64) core.Error!*allowzero align(1) T {
        return @ptrFromInt(@as(usize, @intCast(try self.readHost(va))));
    }

    /// In-page read-modify-write (aligned atomics). W implies R in a valid PTE.
    pub inline fn rmwPtr(self: *Cpu, comptime T: type, va: u64) core.Error!*allowzero align(1) T {
        return @ptrFromInt(@as(usize, @intCast(try self.writeHost(va, @sizeOf(T)))));
    }

    /// Fault now unless the byte at `va` may be accessed as `access`; fills
    /// the TLB and has no other effect (a write probe does not count as a
    /// store into decoded code). Instructions with several accesses (A64
    /// pairs, structures) probe every page before their first store.
    pub fn probe(self: *Cpu, va: u64, access: core.Access) core.Error!void {
        switch (access) {
            .read => _ = try self.readHost(va),
            .write => {
                const vpn = va >> page_shift;
                if (self.priv.tlb_w[@intCast(vpn & (dtlb_n - 1))].tag & ~code_flag != vpn) _ = try self.translate(va, .write);
            },
            .exec => _ = try self.execHost(va),
        }
    }

    /// Fault now unless every 4 KiB page intersecting `[va, va + len)` may be
    /// accessed as `access`. Walks the upper Sv48 levels once per 2 MiB L0
    /// table and scans L0 PTEs directly (without thrashing the 1024-entry TLB),
    /// matching `translate` on every permission, A/D, reserved and VA-range
    /// check.
    pub fn probeRange(self: *Cpu, va: u64, len: u64, access: core.Access) core.Error!void {
        if (len == 0) return;
        const need: u64 = switch (access) {
            .read => pte_v | pte_r | pte_u | pte_a,
            .write => pte_v | pte_w | pte_u | pte_a | pte_d,
            .exec => pte_v | pte_x | pte_u | pte_a,
        };
        var cur = va;
        var rem = len;
        if (len <= 16 * page_size) {
            while (rem > 0) {
                try self.probe(cur, access);
                const step = @min(rem, page_size - (cur & page_mask));
                cur +%= step;
                rem -= step;
            }
            return;
        }
        while (rem > 0) {
            if (cur >> 48 != 0) return self.fault(cur, access);
            var table = self.pgd;
            inline for (.{ @as(u6, 3), @as(u6, 2), @as(u6, 1) }) |level| {
                const idx = (cur >> (page_shift + 9 * level)) & 511;
                const e = hostLoad(u64, self.phys_base +% table +% idx * 8);
                if (e & pte_v == 0 or e & (pte_r | pte_w | pte_x) != 0) {
                    return self.fault(cur, access);
                }
                table = ((e >> 10) & ppn_mask) << page_shift;
            }
            const l0_base = self.phys_base +% table;
            var idx0: u64 = (cur >> page_shift) & 511;
            while (idx0 < 512 and rem > 0) : (idx0 += 1) {
                const e = hostLoad(u64, l0_base +% idx0 * 8);
                if (e & need != need or (e & (pte_w | pte_r) == pte_w)) {
                    return self.fault(cur, access);
                }
                const step = @min(rem, page_size - (cur & page_mask));
                cur +%= step;
                rem -= step;
            }
        }
    }

    // ---- block lookup / decode --------------------------------------------------
    //
    // The cache machinery (block table, generations, revalidation, code-page
    // marks, arena resets) is ISA-neutral; what a block's ops are, how they
    // are decoded and how a block is checked against memory comes from a
    // comptime hook type `D` (A64: src/a64/tier.zig `Blocks`):
    //   D.max_slots: u32                 op-arena slots one block may use
    //   D.LookupError: error set         core.Error plus anything D.check returns
    //   D.check(pc) D.LookupError!void   before the exec fetch of a new block
    //   D.validate(cpu, b) bool          same raw instruction bits as decoded
    //   D.decode(cpu, b, pc, host)       fill b.{pc, host, end_pc, op, n,
    //                                    ninsn, mask} and its arena slots,
    //                                    starting at ops_used (advance it)

    /// The block at `pc`, decoded by `D` if no trusted copy is cached.
    pub inline fn lookupIn(self: *Cpu, comptime D: type, pc: u64) D.LookupError!*Block {
        const set = blockSet(pc);
        // A generation is unique to one fill, so a block whose tgen equals
        // either entry of pc's exec-TLB set was validated under that live
        // fill of its own page.
        const x = self.priv.tlb_x[2 * @as(usize, @intCast((pc >> page_shift) & (tlb_n - 1))) ..][0..2];
        const g0 = x[0].gen;
        const g1 = x[1].gen;
        const b0 = &self.priv.blocks[set];
        if (b0.pc == pc and (b0.tgen == g0 or b0.tgen == g1)) return b0;
        const b1 = &self.priv.blocks[set + 1];
        if (b1.pc == pc and (b1.tgen == g0 or b1.tgen == g1)) return b1;
        return self.lookupSlow(D, set, pc);
    }

    noinline fn lookupSlow(self: *Cpu, comptime D: type, set: usize, pc: u64) D.LookupError!*Block {
        count(self, "lookups_slow", 1);
        try D.check(pc);
        const host = try self.execHost(pc);
        const gen = self.execGen(pc);
        for (self.priv.blocks[set..][0..2]) |*b| {
            if (b.pc == pc and b.host == host and b.op != op_gone and D.validate(self, b)) {
                b.tgen = gen;
                self.markCode(host & ~page_mask, b.mask);
                count(self, "revalidated", 1);
                return b;
            }
        }
        count(self, "decoded", 1);
        const v = &self.priv.victim[set / 2];
        // Reuse a way that already holds this pc (stale copy), else the victim.
        const way: usize = if (self.priv.blocks[set].pc == pc) 0 else if (self.priv.blocks[set + 1].pc == pc) 1 else blk: {
            const w = if (jc.enabled) jc.victim(self, self.priv.blocks[set..][0..2], v.*) else v.*;
            v.* ^= 1;
            break :blk w;
        };
        const b = &self.priv.blocks[set + way];
        if (self.priv.ops_used + D.max_slots > self.priv.nursery) self.resetNursery();
        // The same pc and host decoded again (after a reset discarded its
        // ops, or with new contents): it goes on counting towards the
        // compile threshold.
        const heat: u32 = if (jc.enabled and b.pc == pc and b.host == host and b.jx.st < jc.compiled) b.jx.st else jc.never;
        if (stats_on) a64_tier.axDecode(self, b, pc, host);
        if (stats_on and heat == jc.never) a64_tier.axLost(self, b, false);
        if (jc.enabled) jc.drop(self, b); // its code belongs to the old contents
        D.decode(self, b, pc, host);
        b.tgen = gen;
        b.jx = if (jc.enabled) jc.newBlock(self) else .{};
        if (jc.enabled and heat != jc.never) b.jx.st = heat;
        self.markCode(host & ~page_mask, b.mask);
        return b;
    }

    /// The op arena as `T` (a 16-byte op type of some ISA).
    pub inline fn opArena(self: *Cpu, comptime T: type) [*]T {
        comptime std.debug.assert(@sizeOf(T) == @sizeOf(Slot) and @alignOf(T) <= @alignOf(Slot));
        return @ptrCast(&self.priv.ops);
    }

};

comptime {
    // Layout must match include/kw_emu.h exactly.
    std.debug.assert(@sizeOf(usize) == 8); // pointer field in kw_cpu
    std.debug.assert(@sizeOf(Cpu) <= cpu_size);
    // harness.zig reads this layout
    std.debug.assert(@sizeOf(Stats) == (11 + 192 + (if (stats_on) 2 + 192 + 40 + 64 + 3 * gen_words_n else 0)) * 8);
    if (stats_on) std.debug.assert(jc.Jx.end <= 64);
    std.debug.assert(cpu_size == c.KW_CPU_SIZE);
    std.debug.assert(abi_version == c.KW_ABI_VERSION);
    std.debug.assert(@alignOf(Cpu) == 8);
    std.debug.assert(@sizeOf(Slot) == 16);
    for (.{ "phys_base", "pgd", "irq_pending", "budget", "icount", "ev", "jit_busy", "flush_all" }) |f|
        std.debug.assert(@offsetOf(Cpu, f) == @offsetOf(c.struct_kw_cpu, f));
    std.debug.assert(@offsetOf(Cpu, "priv") >= @sizeOf(c.struct_kw_cpu));
    std.debug.assert(@sizeOf(Event) == @sizeOf(c.struct_kw_event));
    for (.{ "kind", "access", "addr", "insn" }) |f|
        std.debug.assert(@offsetOf(Event, f) == @offsetOf(c.struct_kw_event, f));
    // struct kw_regs and the KW_ISA_* bits are checked in src/a64/run.zig.
    for (.{ "budget", "syscall", "fault", "illegal", "breakpoint", "misaligned", "interrupt" }, .{
        c.KW_EV_BUDGET, c.KW_EV_SYSCALL, c.KW_EV_FAULT, c.KW_EV_ILLEGAL, c.KW_EV_BREAKPOINT, c.KW_EV_MISALIGNED, c.KW_EV_INTERRUPT,
    }) |name, v| std.debug.assert(@intFromEnum(@field(core.Kind, name)) == v);
    std.debug.assert(@intFromEnum(core.Access.read) == c.KW_ACCESS_READ);
    std.debug.assert(@intFromEnum(core.Access.write) == c.KW_ACCESS_WRITE);
    std.debug.assert(@intFromEnum(core.Access.exec) == c.KW_ACCESS_EXEC);
}

// ---- C ABI -----------------------------------------------------------------------

pub fn cpuInit(cpu: *Cpu) void {
    const words: [*]volatile u64 = @ptrCast(cpu);
    for (0..cpu_size / 8) |i| words[i] = 0;
    cpu.priv.hart = .{}; // no exclusive monitor open
    cpu.priv.gen = 1;
    cpu.priv.nursery = nops - ten_default;
    cpu.priv.ten_mask = ten_default - 1;
    cpu.flushAll();
    cpu.resetCache();
    if (jc.enabled) jc.init(cpu);
}

pub fn setPgd(cpu: *Cpu, pgd: u64) void {
    cpu.pgd = pgd;
    cpu.flushAll();
    cpu.priv.hart.resv = core.no_resv;
}

pub fn clearReservation(cpu: *Cpu) void {
    cpu.priv.hart.resv = core.no_resv;
}

/// kw_host_reset: the memory now belongs to a new module instance, whose
/// table holds the stubs again. Forget compiled code without freeing it,
/// ask the new host for its JIT, then drop every decoded block.
pub fn hostReset(cpu: *Cpu) void {
    if (jc.enabled) jc.forgetHost(cpu);
    cpu.resetCache();
}

/// -Dkwemu-stats with the JIT only (not part of the ABI): the struct kw_cpu
/// whose emit buffer is `buf` (the pointer kw_host_jit_compile receives),
/// so a host can find the cpu to pass to kw_debug_stats. Exported by
/// src/a64/run.zig.
pub fn kw_debug_cpu_of_jit_buf(buf: usize) callconv(.c) usize {
    return buf - (@offsetOf(Cpu, "priv") + @offsetOf(Priv, "jit") + @offsetOf(jc.State, "buf"));
}

// Named for the filter of the wasm64+jit unit tests (build.zig), which import it.
test "a64 jit: kw_debug_cpu_of_jit_buf finds the cpu of its emit buffer" {
    if (!jc.enabled) return error.SkipZigTest;
    const cpu = try std.testing.allocator.create(Cpu);
    defer std.testing.allocator.destroy(cpu);
    try std.testing.expectEqual(@intFromPtr(cpu), kw_debug_cpu_of_jit_buf(@intFromPtr(&cpu.priv.jit.buf)));
}
