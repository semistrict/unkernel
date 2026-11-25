// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! kwemu JIT tier, the ISA-neutral core (optional: `-Dkwemu-jit`, wasm64
//! builds only). The ISA's emitter (A64: src/a64/jit/emit.zig)
//! instantiates `Jit(I)` with its ISA hooks `I` (see `Jit`), as
//! kwemu.Cpu.lookupIn does for the block cache.
//!
//! Hot decoded blocks (see `Block` in kwemu.zig) are translated into small
//! wasm modules that the host compiles and installs into a slot of the
//! module's function table; `run` then enters a block through
//! `call_indirect` instead of interpreting its ops. Everything the JIT knows
//! comes from the decoded ops, so the existing block-cache rules (tgen
//! check, revalidation, SMC tracking) decide when compiled code may run.
//!
//! Host imports (wasm import module "unkernel"; only referenced when the
//! JIT is compiled in, so the default object has no such imports):
//!   kw_host_jit_caps() -> u32
//!       bit 0: JIT available; bit 1: the memory is 32-bit (memory64-lowered
//!       module: the generated code imports a 32-bit memory and wraps every
//!       address); bit 2: eager (compile on the first entry; tests);
//!       bit 3: the memory is shared (the generated code declares its
//!       import shared, with the largest maximum of its type); bit 4: the
//!       host also offers the kernel's function table as env.table, which a
//!       module may import to call kwemu functions (`Emit.callTable`).
//!       Called by kw_cpu_init (and kw_host_reset).
//!   kw_host_jit_compile(ptr, len, slot: u32, flags: u32) -> i32
//!       Compile the `len` bytes at `ptr` (a module importing env.memory,
//!       and env.table where bit 4 of the caps says so, and exporting
//!       function "f") and install "f" into table slot `slot`.
//!       0 = installed, < 0 = failed (the slot is left as it was). flags
//!       bit 0 (`compile_warm`): f(x, 0, irqp, jl) returns 0 at once, so the
//!       host may call it to make a lazy engine compile f inside the call.
//!   kw_host_jit_free(slot: u32)
//!       Put the slot's original function back and drop the module.
//!
//! Slots: kwemu owns `npool` stub functions; their table indices are the
//! slots. A stub returns "handed back at op 0", so a slot whose code is not
//! installed is harmless. The table never grows.
//!
//! Generated function: f(x, cpu, irqp, jl: i64) -> i64, where x = &regs.x
//! (x[32] is pc in struct kw_regs), irqp = cpu->irq_pending or a zero word, jl =
//! budget for extra self-loop iterations and region members (`run` has
//! already charged the first pass). It returns (jl_left << 8) | code:
//!   0..63    handed back before op k (not executed), registers written back;
//!   64..127  done: the terminator retired (or a stop), pc in x[32];
//!   128 + k  side exit taken at op k, pc in x[32].
//! Guest registers live in wasm locals (loaded on entry, written back at the
//! single exit). Loads and stores inline the data-TLB fast path (`tlbCheck`);
//! a miss (fault, straddle, store into a page holding code, ...) hands the op
//! back to the interpreter. The code depends only on pc and the raw
//! instruction bits (via the ops), never on host addresses or generations,
//! so a block that is revalidated keeps its slot.
//!
//! A module may hold a region of several blocks, also of several pages (see
//! "regions" below); the code then refers to the block it left from, which
//! it names in cpu.jit.cur (`lastBlock`), and it returns "done" instead of a
//! side exit (having refunded the rest of that block itself).

const std = @import("std");
const builtin = @import("builtin");
const options = @import("kwemu_options");
const kwemu = @import("kwemu.zig");
const core = @import("../cpu/core.zig");
const Cpu = kwemu.Cpu;
const Block = kwemu.Block;

pub const enabled = options.jit and builtin.cpu.arch == .wasm64;

/// Table slots (stub functions) owned by kwemu.
pub const npool = 1024;
/// Module emit buffer.
pub const buf_size = 32 * 1024;
/// Largest `jl` passed in (keeps `jl << 8` in 63 bits).
pub const max_jl: u64 = 1 << 55;
/// Result codes (low 8 bits of `enter`'s result): below code_done, op k
/// was handed back; code_done up to code_side, the pass is done and `run`
/// refunds nothing; from code_side, a side exit at op code - code_side.
/// The done codes say why (only statistics look at that): the terminator
/// left (code_done), a region's side exit left (code_done_side, the region
/// already refunded the rest of its member), or a budget, interrupt or
/// member check stopped the code at a block's start (code_done_stop).
pub const code_done = 64;
pub const code_done_side = code_done + 1;
pub const code_done_stop = code_done + 2;
pub const code_side = 128;

pub const cap_available: u32 = 1;
pub const cap_mem32: u32 = 2;
pub const cap_eager: u32 = 4;
pub const cap_shared: u32 = 8;
pub const cap_table: u32 = 16;
/// The engine compiles fixed-width SIMD (v128 locals and instructions):
/// the ISA may emit them (A64: V registers in v128 locals, AdvSIMD and
/// crypto translated inline). A host sets it only after validating a module
/// with SIMD instructions (V8 on x86-64 needs SSE4.1).
pub const cap_simd: u32 = 32;
/// The engine compiles Relaxed-SIMD FMA (`f32x4`/`f64x2.relaxed_{madd,nmadd}`)
/// and executes them as true single-rounded FMA (probed by the host).
pub const cap_relaxed_fma: u32 = 64;

/// The host offers env.table (cap_table), so the ISA may translate ops into
/// calls of kwemu functions (A64: `generic` ops). Set by every CPU's
/// `askHost`; all CPUs of a kernel have the same kind of host.
pub var table_ok: bool = false;
/// The host compiles SIMD (cap_simd); set like `table_ok`.
pub var simd_ok: bool = false;
/// The host compiles Relaxed-SIMD FMA with true single rounding
/// (cap_relaxed_fma); set like `table_ok`.
pub var relaxed_fma_ok: bool = false;
/// Most blocks in one compiled region (see "regions" below).
pub const max_members = 16;
/// A region compiled for a page also takes the blocks of that page that have
/// counted at least threshold >> warm_shift towards the compile threshold
/// (`pageWarm`).
const warm_shift = 2;

// Block JIT state (`BlockState.st`): one word, so the per-pass test is a
// single load.
/// Not compiled, and never will be (first op unsupported, compile failed,
/// JIT unavailable).
pub const never: u32 = 0;
/// 1 .. ref_bit-1: interpreted instructions left before the block is
/// compiled. With `compiled` set: member `(st >> member_shift) & 0xff` of
/// the region in pool entry `st & slot_mask`; `ref_bit` is the clock's
/// referenced bit.
pub const compiled: u32 = 1 << 31;
const ref_bit: u32 = 1 << 30;
const member_shift = 16;
const slot_mask: u32 = (1 << member_shift) - 1;

extern "unkernel" fn kw_host_jit_caps() u32;
extern "unkernel" fn kw_host_jit_compile(ptr: [*]const u8, len: usize, slot: u32, flags: u32) i32;
/// kw_host_jit_compile flags: every module returns 0 at once when called
/// with cpu = 0 (see `emitModule`).
const compile_warm: u32 = 1;
extern "unkernel" fn kw_host_jit_free(slot: u32) void;

/// Per-block fields. Empty without the JIT, so the default build keeps the
/// block layout it always had.
pub const BlockState = if (enabled) extern struct {
    /// never / countdown / compiled | ref_bit? | pool entry (see above).
    st: u32 = never,
    /// Times `run` entered this compiled block right after compiled code
    /// of another region left towards it (see `noteExit`).
    hits: u32 = 0,
} else extern struct {};

/// JIT bookkeeping inside struct kw_cpu (empty without the JIT).
pub const State = if (enabled) extern struct {
    caps: u32,
    /// `st` of a new block: the compile threshold (`never`: JIT unavailable).
    threshold: u32,
    /// *irq_pending stand-in when cpu->irq_pending is NULL.
    zero: u32,
    /// Written by the stubs (their distinct side effect).
    last_stub: u32,
    /// Clock hand over `owner`.
    hand: u32,
    /// Largest module emitted so far (bytes).
    max_bytes: u32,
    /// Most members per region (`max_members`; tests lower it).
    region_max: u32,
    /// Exits from a region towards a compiled block of another region, per
    /// KiB of the region's module, after which that block joins the region
    /// (`noteHot`; -Dkwemu-jit-grow, tests change it).
    grow_threshold: u32,
    /// Set by `enter`: the entered block; on return, the block the result
    /// code refers to (a region stores the member it left from).
    cur: u64,
    /// Regions recompiled with an added block (a block of their page that
    /// got hot / a block they often left towards), and member blocks
    /// compiled.
    grows: u64,
    exit_grows: u64,
    members_total: u64,
    compiles: u64,
    frees: u64,
    /// Calls into compiled code.
    entries: u64,
    /// Calls that ended "handed back at op k" (counted by `handedBack`).
    handbacks: u64,
    /// Host compile failures and oversized modules.
    fails: u64,
    /// Modules emitted again for a shorter prefix (did not fit the buffer).
    shortened: u64,
    /// Blocks whose first op is not supported.
    skips: u64,
    evictions: u64,
    /// Block index + 1 per pool entry, 0 = free.
    owner: [npool]u32,
    /// Per pool entry: block index + 1 of each region member (0: none).
    /// A member that was re-decoded since is dead: its `st` no longer
    /// names this entry, and the region's code never enters it again.
    members: [npool][max_members]u16,
    /// Per pool entry: the low 16 bits of each member's `op` when it was
    /// compiled (its code is that decode; see `noteHot`).
    member_op: [npool][max_members]u16,
    /// Per pool entry: virtual page of the region (of the block it was
    /// first compiled for); | page_closed when it takes no further members.
    page: [npool]u64,
    /// Per pool entry: module size in bytes.
    size: [npool]u16,
    /// AES tables the compiled AES ops read cpu-relative (`AesTab`;
    /// A64's AESE/AESD/AESMC/AESIMC).
    aes: AesTab,
    buf: [buf_size]u8,
} else extern struct {};

/// Tables for the compiled AES ops, filled by `init` from core.aes
/// (compiled code may only address cpu-relative data, never the library's
/// own constants). te[x]: MixColumns of the column (S[x], 0, 0, 0); a byte
/// in row r contributes rotl(te[x], 8 r). td[x]: InvMixColumns of (S^-1[x],
/// 0, 0, 0). im[x]: InvMixColumns of (x, 0, 0, 0). s, is: the S-box and its
/// inverse. mc[x]: MixColumns of (x, 0, 0, 0).
pub const AesTab = extern struct {
    te: [256]u32,
    td: [256]u32,
    im: [256]u32,
    s: [256]u8,
    is: [256]u8,
    mc: [256]u32,
};

// ---- slots -----------------------------------------------------------------------

pub const JitFn = *const fn (x: u64, cpu: u64, irqp: u64, jl: u64) callconv(.c) u64;

fn Stub(comptime id: u32) type {
    return struct {
        fn f(x: u64, cpu: u64, irqp: u64, jl: u64) callconv(.c) u64 {
            _ = x;
            _ = irqp;
            const off = @offsetOf(Cpu, "priv") + @offsetOf(@FieldType(Cpu, "priv"), "jit") + @offsetOf(State, "last_stub");
            const p: *volatile u32 = @ptrFromInt(cpu + off);
            p.* = id;
            return jl << 8; // handed back at op 0, nothing retired
        }
    };
}

/// The stubs; pool entry i lives in table slot @intFromPtr(pool[i]).
const pool: [npool]JitFn = blk: {
    @setEvalBranchQuota(20 * npool);
    var p: [npool]JitFn = undefined;
    for (0..npool) |i| p[i] = &Stub(i).f;
    break :blk p;
};

inline fn tableSlot(i: usize) u32 {
    return @intCast(@intFromPtr(pool[i]));
}

fn blockIndex(cpu: *Cpu, b: *const Block) u32 {
    return @intCast((@intFromPtr(b) - @intFromPtr(&cpu.priv.blocks)) / @sizeOf(Block));
}

/// Ask the host what it offers and set the compile threshold from it.
fn askHost(st: *State) void {
    st.caps = kw_host_jit_caps();
    table_ok = st.caps & cap_available != 0 and st.caps & cap_table != 0;
    simd_ok = st.caps & cap_available != 0 and st.caps & cap_simd != 0;
    relaxed_fma_ok = simd_ok and st.caps & cap_relaxed_fma != 0;
    st.threshold = if (st.caps & cap_available == 0)
        never
    else if (st.caps & cap_eager != 0)
        1
    else
        std.math.clamp(options.jit_threshold, 1, ref_bit - 1);
}

/// kw_host_reset: the host that compiled the slots is gone. Forget every
/// slot without kw_host_jit_free and ask the new host. The caller then
/// drops all blocks (resetCache), which finds no slot left to free.
pub fn forgetHost(cpu: *Cpu) void {
    const st = &cpu.priv.jit;
    // Volatile stores: no memset import in the wasm object.
    const owner: *volatile [npool]u32 = &st.owner;
    const members: *volatile [npool][max_members]u16 = &st.members;
    for (0..npool) |i| {
        owner[i] = 0;
        for (0..max_members) |j| members[i][j] = 0;
    }
    // No block may name a slot any more: re-decoding one would free it.
    const bs: [*]volatile Block = &cpu.priv.blocks;
    for (0..cpu.priv.blocks.len) |i| {
        bs[i].jx.st = never;
        bs[i].jx.hits = 0;
    }
    askHost(st);
}

/// Called by kw_cpu_init after the block has been zeroed.
pub fn init(cpu: *Cpu) void {
    const st = &cpu.priv.jit;
    askHost(st);
    st.region_max = max_members;
    // 0: never (no member counts that many exits)
    st.grow_threshold = if (options.jit_grow == 0) std.math.maxInt(u32) else options.jit_grow;
    // Volatile stores: no memcpy import in the wasm object.
    const t: *volatile AesTab = &st.aes;
    const A = core.aes;
    for (0..256) |i| {
        t.te[i] = A.mixColumn(A.sbox[i]);
        t.td[i] = A.invMixColumn(A.inv_sbox[i]);
        t.im[i] = A.invMixColumn(@intCast(i));
        t.s[i] = A.sbox[i];
        t.is[i] = A.inv_sbox[i];
        t.mc[i] = A.mixColumn(@intCast(i));
    }
}

/// Initial JIT state of a newly decoded block.
pub inline fn newBlock(cpu: *Cpu) BlockState {
    return .{ .st = cpu.priv.jit.threshold };
}

/// Block-table replacement (kwemu lookupSlow): of a set's two ways, the one
/// with less to lose: an empty way, else the one with less countdown
/// progress (compiled counts as more than any); `alt` decides a tie. The
/// way that stays loses half its progress, so a block that stopped running
/// does not keep its way for ever.
pub fn victim(cpu: *Cpu, ways: *[2]Block, alt: u8) usize {
    const h0 = heatOf(cpu, &ways[0]);
    const h1 = heatOf(cpu, &ways[1]);
    if (h0 == h1) return alt;
    const w: usize = @intFromBool(h1 < h0);
    const keep = &ways[1 - w];
    const s = keep.jx.st;
    if (s != never and s < compiled) keep.jx.st = s + (cpu.priv.jit.threshold -| s) / 2;
    return w;
}

fn heatOf(cpu: *Cpu, b: *const Block) u32 {
    if (b.pc == std.math.maxInt(u64)) return 0;
    const s = b.jx.st;
    if (s >= compiled) return std.math.maxInt(u32);
    if (s == never) return 0;
    return cpu.priv.jit.threshold -| s;
}

/// Pool entry + 1 of b's code, 0 = not compiled.
pub inline fn slotOf(b: *const Block) u32 {
    return if (b.jx.st >= compiled) (b.jx.st & slot_mask) + 1 else 0;
}

/// Member index of a compiled block in its region.
pub inline fn memberOf(b: *const Block) usize {
    return (b.jx.st >> member_shift) & 0xff;
}

/// Why a pool entry is freed (Jx.free_why): its region is compiled again
/// with an added block (a block of its page that got hot / a block it was
/// often left towards: `grows`, `exit_grows`), the
/// pool's clock needs the entry, the tenured ring dropped its last home
/// member's ops, or anything else (that block was decoded again or its
/// table entry replaced; a cache reset).
const Why = enum(u8) { grow, exit_grow, clock, ring, other };

/// Give pool entry `i` back to the host and detach it from its blocks.
fn freeSlot(cpu: *Cpu, i: usize, why: Why) void {
    const st = &cpu.priv.jit;
    jx(cpu, Jx.free_why + @intFromEnum(why), 1);
    for (&st.members[i]) |*m| {
        if (m.* == 0) continue;
        const b = &cpu.priv.blocks[m.* - 1];
        if (slotOf(b) == i + 1) b.jx.st = st.threshold; // recompiled when hot again
        m.* = 0;
    }
    st.owner[i] = 0;
    kw_host_jit_free(tableSlot(i));
    st.frees += 1;
}

/// A block is about to be overwritten: drop its code. A region with other
/// live members stays: its code checks a member's `st` before entering it,
/// and the overwrite resets that.
pub inline fn drop(cpu: *Cpu, b: *Block) void {
    if (b.jx.st >= compiled) dropMember(cpu, b, .other);
}

/// The tenured ring drops b's ops (kwemu.zig, `tenMakeRoom`): its code goes
/// as in `drop`, and it is compiled again once it has counted to the
/// threshold again.
pub fn evict(cpu: *Cpu, b: *Block) void {
    if (b.jx.st >= compiled) {
        dropMember(cpu, b, .ring);
        b.jx.st = cpu.priv.jit.threshold;
    }
}

noinline fn dropMember(cpu: *Cpu, b: *Block, why: Why) void {
    const i = b.jx.st & slot_mask;
    const me: u16 = @intCast(blockIndex(cpu, b) + 1);
    b.jx.st = never;
    for (cpu.priv.jit.members[i]) |m| {
        if (m != 0 and m != me and slotOf(&cpu.priv.blocks[m - 1]) == i + 1) return;
    }
    freeSlot(cpu, i, why);
}

/// Free every slot in use (cache reset).
pub fn releaseAll(cpu: *Cpu) void {
    for (0..npool) |i| {
        if (cpu.priv.jit.owner[i] != 0) freeSlot(cpu, i, .other);
    }
}

/// Clock: whether a live member of pool entry i was entered from `run`
/// since the last sweep; clears the referenced bits.
fn referenced(cpu: *Cpu, i: usize) bool {
    var any = false;
    for (cpu.priv.jit.members[i]) |m| {
        if (m == 0) continue;
        const b = &cpu.priv.blocks[m - 1];
        if (slotOf(b) == i + 1 and b.jx.st & ref_bit != 0) {
            b.jx.st &= ~ref_bit;
            any = true;
        }
    }
    return any;
}

/// A free pool entry, evicting one (clock, second chance) if none is free.
fn allocSlot(cpu: *Cpu) usize {
    const st = &cpu.priv.jit;
    var tries: usize = 0;
    while (true) : (tries += 1) {
        const i = st.hand;
        st.hand = (i + 1) % npool;
        const o = st.owner[i];
        if (o == 0) return i;
        if (referenced(cpu, i) and tries < 2 * npool) continue; // second chance
        freeSlot(cpu, i, .clock);
        st.evictions += 1;
        return i;
    }
}

/// A compile (gathering a region, emitting its module, the host's compile
/// and the slot bookkeeping) is a window in which other CPUs need not wait
/// for this one's TLB flushes (kw_emu.h, "jit_busy / flush_all"):
/// nothing in it runs guest code or reads guest memory or a page table
/// (only struct kw_cpu: decoded ops, exec-TLB generations, the emit
/// buffer). Without atomics (one CPU) nobody sets flush_all, and these
/// are plain stores and loads.
inline fn windowOpen(cpu: *Cpu) void {
    @atomicStore(u32, &cpu.jit_busy, 1, .seq_cst);
}

/// End the window and do the flush another CPU asked for meanwhile, if
/// any. True if it did: every block revalidates, and the one `run` was
/// about to run may be gone, so `run` must look it up again before it runs
/// anything. (An interrupt raised meanwhile is taken where the interpreter
/// takes it, after that block.)
fn windowClose(cpu: *Cpu) bool {
    @atomicStore(u32, &cpu.jit_busy, 0, .seq_cst);
    if (@atomicRmw(u32, &cpu.flush_all, .Xchg, 0, .seq_cst) == 0) return false;
    cpu.flushAll();
    return true;
}

/// What `ready` decided for a pass through a block: interpret it, run its
/// compiled code, or neither: a compile asked `run` to look the block up
/// again first (`windowClose`).
pub const Go = enum { interp, compiled, again };

/// The interpreter's self-loop over b continues while `left >= limit`
/// (instead of `left >= b.ninsn`), so the JIT adds no work to that loop:
/// a compiled block (the interpreter finished a handed-back pass) stops at
/// once and re-enters through `enter`; a block counting towards its
/// threshold stops when the countdown runs out (charged by `loopDone`), so
/// the next pass is compiled; anything else keeps b.ninsn.
pub inline fn loopLimit(b: *const Block, is_compiled: bool, left: u64) u64 {
    if (is_compiled) return std.math.maxInt(u64);
    const s = b.jx.st;
    if (s == never) return b.ninsn;
    return @max(@as(u64, b.ninsn) + 1, left -| s);
}

/// The interpreter leaves b after `n` more instructions in its self-loop.
pub inline fn loopDone(b: *Block, n: u64) void {
    const s = b.jx.st;
    if (s != never and s < compiled) b.jx.st = @intCast(@max(1, s -| n));
}

/// Run `b`'s compiled code (`ready` returned .compiled) with `jl` extra
/// self-loop instructions allowed; returns (jl_left << 8) | code. Not
/// inlined: keeps the indirect call out of the interpreter's hot loop.
/// `run` has charged b in full; a region charges every further member it
/// enters against `jl` the same way, and `code` refers to `lastBlock`.
pub noinline fn enter(cpu: *Cpu, b: *Block, x: [*]u64, jl: u64) u64 {
    const s = b.jx.st;
    b.jx.st = s | ref_bit; // referenced (clock)
    cpu.priv.jit.entries += 1;
    cpu.priv.jit.cur = @intFromPtr(b);
    const irqp: u64 = if (cpu.irq_pending) |p| @intFromPtr(p) else @intFromPtr(&cpu.priv.jit.zero);
    return pool[s & slot_mask](@intFromPtr(x), @intFromPtr(cpu), irqp, jl);
}

/// `enter` returned "handed back" (the interpreter finishes the pass).
pub inline fn handedBack(cpu: *Cpu) void {
    cpu.priv.jit.handbacks += 1;
}

/// The block whose code returned from `enter`: the entered block, or the
/// region member it left from. The result code refers to that block.
pub inline fn lastBlock(cpu: *Cpu) *Block {
    return @ptrFromInt(@as(usize, @intCast(cpu.priv.jit.cur)));
}

/// The pool entry `enter` is about to run (for `noteExit`).
pub inline fn slotIndex(b: *const Block) u32 {
    return b.jx.st & slot_mask;
}

// ---- statistics (-Dkwemu-stats) ----------------------------------------------------

/// Indices into Stats.jit.jx.
pub const Jx = struct {
    /// Calls from `run` into compiled code.
    pub const entries = 0;
    /// Compiled passes that handed an op back to the interpreter.
    pub const handbacks = 1;
    /// Compiled passes that left through a side exit (regions' included) /
    /// the terminator (conditional branch, jal, goto) / a jalr terminator.
    /// With exit_stop and handbacks these add up to entries.
    pub const exit_side = 2;
    pub const exit_direct = 3;
    pub const exit_jalr = 4;
    /// The block run next after a compiled pass that left through an exit
    /// (not a stop): 5 + 4 * (left through a jalr terminator) + 2 * (same
    /// page) + (compiled).
    pub const next = 5;
    /// ... of which the same block again / a block that is never compiled.
    pub const next_self = 13;
    pub const next_never = 14;
    /// Region code: branches from one member into another (or itself) /
    /// to a member that was not valid (stale tgen, re-decoded) / to a
    /// member when the budget or a pending interrupt stopped it / to a pc
    /// that is no member (these return to `run`).
    pub const region_hops = 15;
    pub const region_stale = 16;
    pub const region_stop = 17;
    pub const region_out = 18;
    /// The compiled block run next after a compiled pass is a member of the
    /// region that was entered / a self-loop's own module / in another
    /// region (... of which the entered region was closed or full).
    pub const next_same_region = 19;
    pub const next_selfloop = 20;
    pub const next_other_region = 21;
    pub const next_other_full = 22;
    /// Region code: branches from one member towards a member on another
    /// page (counted at the branch site, before the checked entry).
    pub const region_cross = 23;
    /// Compiled passes stopped at a block's start: the budget did not
    /// cover the next pass or member, an interrupt was pending, or a
    /// member's checked entry failed (code_done_stop).
    pub const exit_stop = 24;
    /// ... of exit_side: side exits of a region member (code_done_side).
    pub const exit_side_region = 25;
    /// How a compiled pass left (Xr) by the block `run` looked up next
    /// (Nx): next_by + n_nx * reason + class. Counted for every pass but
    /// hand-backs, unless `run` returned first (budget used up, interrupt
    /// pending), so a row adds up to at most its exit counter.
    pub const next_by = 26;
    pub const n_nx = 7;
    /// Lazy state (A64: NZCV) materialised by compiled code, by where
    /// (`Emit.site`): an exit to `run` or a hand-back, a self-loop's or
    /// member's own back edge, a branch to another member, right after a
    /// producer whose result reaches the member's end, before a call of the
    /// interpreter (and MRS NZCV).
    pub const sync_site = next_by + 4 * n_nx;
    pub const n_sites = 5;
    /// Pool entries freed (modules given back to the host), by `Why`.
    pub const free_why = sync_site + n_sites;
    pub const end = free_why + 5;
};

/// Exit reasons of Jx.next_by.
pub const Xr = enum(u32) { side, direct, jalr, stop };

/// Classes of the block run next, in Jx.next_by (the first that holds):
/// the block left, again / never compiled / not compiled yet / compiled:
/// a member of the region left / a self-loop's own module / in another
/// region, and the region left could still grow / ... and the region left
/// is closed or full.
pub const Nx = enum(u32) { self, never, interp, same_region, selfloop, other, other_full };

inline fn jx(cpu: *Cpu, i: usize, n: u64) void {
    if (kwemu.stats_on) cpu.priv.stats.jit.jx[i] += n;
}

// ---- wasm encoding -------------------------------------------------------------------

/// wasm opcodes (all MVP)
pub const W = struct {
    pub const block = 0x02;
    pub const loop = 0x03;
    pub const @"if" = 0x04;
    pub const @"else" = 0x05;
    pub const end = 0x0b;
    pub const br = 0x0c;
    pub const br_if = 0x0d;
    pub const br_table = 0x0e;
    pub const drop = 0x1a;
    pub const select = 0x1b;
    pub const local_get = 0x20;
    pub const local_set = 0x21;
    pub const local_tee = 0x22;
    pub const i32_load = 0x28;
    pub const i64_load = 0x29;
    pub const i64_store = 0x37;
    pub const i32_const = 0x41;
    pub const i64_const = 0x42;
    pub const i32_eqz = 0x45;
    pub const i32_eq = 0x46;
    pub const i64_eqz = 0x50;
    pub const i64_eq = 0x51;
    pub const i64_ne = 0x52;
    pub const i64_lt_s = 0x53;
    pub const i64_lt_u = 0x54;
    pub const i64_ge_s = 0x59;
    pub const i64_ge_u = 0x5a;
    pub const i32_and = 0x71;
    pub const i32_or = 0x72;
    pub const i32_shl = 0x74;
    pub const i32_shr_s = 0x75;
    pub const i32_shr_u = 0x76;
    pub const i32_div_s = 0x6d;
    pub const i32_div_u = 0x6e;
    pub const i32_rem_s = 0x6f;
    pub const i32_rem_u = 0x70;
    pub const i64_add = 0x7c;
    pub const i64_sub = 0x7d;
    pub const i64_mul = 0x7e;
    pub const i64_div_s = 0x7f;
    pub const i64_div_u = 0x80;
    pub const i64_rem_s = 0x81;
    pub const i64_rem_u = 0x82;
    pub const i64_and = 0x83;
    pub const i64_or = 0x84;
    pub const i64_xor = 0x85;
    pub const i64_shl = 0x86;
    pub const i64_shr_s = 0x87;
    pub const i64_shr_u = 0x88;
    pub const i32_wrap = 0xa7;
    pub const i64_ext_s = 0xac;
    pub const i64_ext_u = 0xad;
    pub const void_t = 0x40;
    pub const i64_t = 0x7e;
    pub const i32_t = 0x7f;
    pub const f64_t = 0x7c;
    pub const @"return" = 0x0f;
    pub const call = 0x10;
    pub const global_get = 0x23;
    pub const global_set = 0x24;
    pub const i32_store = 0x36;
    pub const i64_load8_s = 0x30;
    pub const i64_load8_u = 0x31;
    pub const i64_load16_s = 0x32;
    pub const i64_load16_u = 0x33;
    pub const i64_load32_s = 0x34;
    pub const i64_load32_u = 0x35;
    pub const i64_store8 = 0x3c;
    pub const i64_store16 = 0x3d;
    pub const i64_store32 = 0x3e;
    pub const f64_const = 0x44;
    pub const i32_ne = 0x47;
    pub const i64_gt_s = 0x55;
    pub const i64_gt_u = 0x56;
    pub const i64_le_s = 0x57;
    pub const i64_le_u = 0x58;
    pub const f32_eq = 0x5b;
    pub const f32_lt = 0x5d;
    pub const f32_le = 0x5f;
    pub const f64_eq = 0x61;
    pub const f64_ne = 0x62;
    pub const f64_lt = 0x63;
    pub const f64_gt = 0x64;
    pub const f64_le = 0x65;
    pub const f64_ge = 0x66;
    pub const i32_xor = 0x73;
    pub const i32_rotl = 0x77;
    pub const i32_rotr = 0x78;
    pub const i64_rotl = 0x89;
    pub const i64_rotr = 0x8a;
    pub const i64_clz = 0x79;
    pub const i64_ctz = 0x7a;
    pub const i64_popcnt = 0x7b;
    pub const i32_clz = 0x67;
    pub const i32_ctz = 0x68;
    pub const i32_popcnt = 0x69;
    pub const f32_sqrt = 0x91;
    pub const f64_ceil = 0x9b;
    pub const f64_floor = 0x9c;
    pub const f64_trunc = 0x9d;
    pub const f64_nearest = 0x9e;
    pub const f64_sqrt = 0x9f;
    pub const f64_add = 0xa0;
    pub const f64_sub = 0xa1;
    pub const f64_mul = 0xa2;
    pub const f64_div = 0xa3;
    pub const i64_trunc_f64_s = 0xb0;
    pub const i64_trunc_f64_u = 0xb1;
    pub const f32_convert_i32_s = 0xb2;
    pub const f32_convert_i32_u = 0xb3;
    pub const f32_convert_i64_s = 0xb4;
    pub const f32_convert_i64_u = 0xb5;
    pub const f32_demote = 0xb6;
    pub const f64_convert_i32_s = 0xb7;
    pub const f64_convert_i32_u = 0xb8;
    pub const f64_convert_i64_s = 0xb9;
    pub const f64_convert_i64_u = 0xba;
    pub const f64_promote = 0xbb;
    pub const i32_reinterpret = 0xbc;
    pub const i64_reinterpret = 0xbd;
    pub const f32_reinterpret = 0xbe;
    pub const f64_reinterpret = 0xbf;
};

/// The generated function's parameters (locals 0..3); every other local is
/// the ISA's (`I.locals`).
pub const l_x = 0;
pub const l_cpu = 1;
pub const l_irq = 2;
pub const l_jl = 3;

pub const priv_off = @offsetOf(Cpu, "priv");
pub const tlb_r_off = priv_off + @offsetOf(@FieldType(Cpu, "priv"), "tlb_r");
pub const tlb_w_off = priv_off + @offsetOf(@FieldType(Cpu, "priv"), "tlb_w");
/// Data-TLB index mask applied to va >> 8: (entries - 1) * 16-byte entries.
const dtlb_mask: u64 = (@typeInfo(@FieldType(@FieldType(Cpu, "priv"), "tlb_r")).array.len - 1) << 4;

const blocks_off = priv_off + @offsetOf(@FieldType(Cpu, "priv"), "blocks");
const tlb_x_off = priv_off + @offsetOf(@FieldType(Cpu, "priv"), "tlb_x");
const XTlb = @typeInfo(@FieldType(@FieldType(Cpu, "priv"), "tlb_x")).array.child;
/// Exec-TLB sets (2 ways each).
const xtlb_sets = @typeInfo(@FieldType(@FieldType(Cpu, "priv"), "tlb_x")).array.len / 2;
pub const jit_off = priv_off + @offsetOf(@FieldType(Cpu, "priv"), "jit");
const jst_off = @offsetOf(Block, "jx") + @offsetOf(BlockState, "st");

/// Region locals (`I.l_region` ..+3): `br_table` index of the code to run
/// next (2 m = member m's checked entry, 2 m + 1 = member m itself; see
/// `emitMembers`), the address of the member being run (stored to
/// cpu.jit.cur on exit), the tgen of a member while its checked entry tests
/// it, and the tgen of the member being run (the generation of the current
/// exec-TLB fill of its page).
pub const n_region_locals = 4;

pub const Error = error{ Unsupported, Overflow };

// ---- regions -------------------------------------------------------------------------
//
// A compiled module holds a region: up to `max_members` decoded blocks
// (members), on any number of pages, not just one block. Guest registers
// stay in wasm locals across the whole region: a member that leaves through
// its terminator (an indirect jump compares its target against every
// member's pc) or a side exit towards a member (itself included) continues
// there inside the module, through a `loop` over a `br_table`. The table has
// two entries per member: its checked entry (the checks below, then the
// member) and the member itself (only for the member `run` entered; an
// indirect jump to the start of the member being run continues in the
// member's own loop). A branch site only sets the index and branches to the
// loop, so a module carries the checks once per member, not once per branch
// site and target. Only exits to other pcs, hand-backs and stops return to
// `run`.
//
// Home and guest members: a block's `st` names one region, its home, and
// `run` enters the block only through that module. A region may also hold
// copies of blocks whose home is another region (guests). A member's code
// depends only on its decoded ops, so a copy is exact as long as the block
// still holds the same decode; the checks below do not look at `st`.
//
// Entering a member: `run` looked the entered block up, so it is valid
// under its page's current exec-TLB fill. A branch to a member m (from any
// member, m itself included) goes through m's checked entry:
//  - blocks[m].op is the op number m had when the module was emitted.
//    Members' ops are in the tenured ring (kwemu.zig, `tenure`), numbered
//    on for good, and a re-decode gives a block nursery ops (another
//    number), so this is "still the same decode".
//  - m is valid under the current fill of its own page: blocks[m].tgen
//    equals l_g, the tgen of the member being run (a generation is unique
//    to one fill of one page, and that fill is live), or else the
//    generation of one of the two exec-TLB entries of m's page's set,
//    exactly what `lookup` checks; l_g then becomes m's tgen. Nothing in
//    compiled code changes the exec TLB (stores into code hand back), so a
//    check made here holds until the module returns.
//  - like `run`: the budget covers all of m (jl >= its instructions) and
//    `I.poll` says nothing stops it (no interrupt pending; on A64 SMP, no
//    code changed by another CPU).
// Then it charges m in full and enters it; a side exit first refunds the
// rest of the member it leaves, so `jl` stays exact. Every member entry
// polls, so the irq word is read at least every 64 instructions. If a check
// fails, the module returns "done" with pc = the member's start, and `run`
// takes over (looks it up, revalidates or re-decodes it). `cpu.jit.cur`
// names the member a hand-back came from (`lastBlock`). Nothing in compiled
// code frees or reuses a slot or rewrites a block (only `run` does, between
// entries), so a region never outlives its members while it runs.
//
// The emitted code depends on pc, the raw instruction bits, cpu-relative
// layout offsets, the members' block-table positions and op indices, never
// on host addresses, generations or the pool entry: a revalidated member
// keeps its code. Regions are formed when a block gets hot (`compile`):
// that block, the members of an open region of its page that has room
// (which is then replaced, so a page tends to have one module), the blocks
// of its page that have counted a quarter of the threshold (`pageWarm`:
// otherwise each of them, when hot, compiles the region again), then
// blocks reachable from those through branch and call-return targets
// (`I.targets`) that are already decoded, valid under their page's current
// fill and not compiled elsewhere, on any page, in breadth-first order.
// They grow later along hot exits (`noteHot`): a block compiled in another
// region (a self-loop's own module included), on any page, that `run`
// often enters right after the region left joins it as a guest. A member that is re-decoded only loses its own
// entry (`dropMember`); copies of it in other regions fail the op check
// from then on. A region is freed when no home member is left (a member
// re-decoded, or its ops dropped from the tenured ring), when the clock
// evicts it, when it is grown (replaced) or on a cache reset; a nursery
// reset leaves it alone.

/// The blocks of a region being compiled (indices into cpu.priv.blocks;
/// [0] is the block that got hot) and its emission state.
pub const Region = struct {
    cpu: *Cpu,
    n: usize = 0,
    idx: [max_members]u32 = undefined,
    /// Member being emitted.
    cur: usize = 0,
    /// Nesting levels of the dispatch loop and of the current member's loop.
    top: u32 = 0,
    self: u32 = 0,

    pub fn block(rg: *const Region, m: usize) *const Block {
        return &rg.cpu.priv.blocks[rg.idx[m]];
    }
    /// Member whose block starts at pc.
    pub fn memberAt(rg: *const Region, pc: u64) ?usize {
        for (0..rg.n) |m| {
            if (rg.block(m).pc == pc) return m;
        }
        return null;
    }
};

/// State.page flag: the region takes no further members (it did not fit,
/// or it is a self-loop's module of its own).
const page_closed: u64 = 1 << 63;

/// `st` of member m of the region in pool entry i (without ref_bit).
inline fn memberSt(i: usize, m: usize) u32 {
    return compiled | (@as(u32, @intCast(m)) << member_shift) | @as(u32, @intCast(i));
}

/// Pool entry of an open region of b's page with room for another member
/// whose live (home) members on that page are on b's host page.
fn findRegion(cpu: *Cpu, b: *const Block, limit: usize) ?usize {
    const st = &cpu.priv.jit;
    for (0..npool) |i| {
        if (st.owner[i] == 0 or st.page[i] != b.pc >> 12) continue;
        var n: usize = 0;
        var live: usize = 0;
        var same = true;
        for (st.members[i]) |m| {
            if (m == 0) continue;
            n += 1;
            const o = &cpu.priv.blocks[m - 1];
            if (slotOf(o) != i + 1) continue;
            live += 1;
            if ((o.pc ^ b.pc) >> 12 == 0) same = same and o.host >> 12 == b.host >> 12;
        }
        if (same and live > 0 and n < limit) return i;
    }
    return null;
}

/// Whether block c is valid under the current exec-TLB fill of its page
/// (what `lookup` checks).
fn current(cpu: *Cpu, c: *const Block) bool {
    return c.tgen != 0 and c.tgen == cpu.execGenOf(c.pc);
}

fn addMember(rg: *Region, i: u32) void {
    for (rg.idx[0..rg.n]) |j| if (j == i) return;
    rg.idx[rg.n] = i;
    rg.n += 1;
}

/// The JIT for one ISA. `I` is a namespace of hooks; the core calls nothing
/// else of the ISA:
///
///   Op, Use                     the ISA's 16-byte op type; register usage,
///                               with `fn add(*Use, Use)`
///   EmitExt                     the ISA's per-module emission state
///                               (`Emit.x`), default-initialised
///   l_npc, l_code, l_va, l_e    i64 locals the core uses: next pc, result
///                               code, a data address and its TLB entry
///   l_region                    first of `n_region_locals` i64 locals,
///                               declared by `locals` when `region`
///   ops(cpu, b) []const Op      b's ops (b.n of them, terminator last)
///   slots(b) u32                op-arena slots b uses (ops and anything
///                               the ISA keeps after them; `tenure`)
///   supported(o) bool           translated (else handed back before it)
///   blockUse(ops) Use           usage of a translated prefix
///   beginMember(e, ops)         a block's or member's code starts (ops: its
///                               translated prefix): reset per-member state
///   helperCount(e, use) u32     helper functions the module needs (type
///                               1, after f; the core adds an i32 global)
///   locals(e, region)           the local declarations after the params
///   prologue(e, use), epilogue(e, use)
///                               load / write back the registers (the core
///                               then stores l_npc to x[32])
///   emitHelpers(e)              the helper function bodies
///   emitOp(e, o, k, pc, b, self_loop)
///   nextPc(o, pc) u64           pc of the next op in block order
///   selfLoop(last, b) bool      b's terminator `last` branches to b.pc
///   targets(o, b) [2]?u64       pcs op o of member b may continue at that
///                               may join its region (branch targets, call
///                               return sites)
///   memberTail(e, rg, b, last)  after member b's terminator: branch to the
///                               members it may continue at (`toMember`,
///                               `targetIf`); falls through to "leave"
///   indirect(last) bool         statistics: an indirect-jump terminator
///   poll(e)                     push an i32, nonzero: stop at this block
///                               or member start (interrupt pending, ...)
///   sync(e)                     before every branch that leaves the
///                               current straight-line code (an exit, a
///                               member transfer, a back edge): make
///                               ISA state that lives in emit-time
///                               knowledge (lazy flags) architectural in
///                               its locals, on that path only
///   syncHead(e) (optional)      instead of sync before a branch back to the
///                               start of the block or member being emitted
///                               (a loop head): may skip state that is dead
///                               there
pub fn Jit(comptime I: type) type {
    return struct {
        const Op = I.Op;
        const Use = I.Use;

        pub const Emit = struct {
            buf: *[buf_size]u8,
            pos: usize = 0,
            ok: bool = true,
            mem32: bool,
            /// The kernel's memory is shared (cap_shared): the import says so.
            shared: bool = false,
            /// The module imports the kernel's function table (cap_table).
            table: bool = false,
            /// The engine compiles SIMD (cap_simd).
            simd: bool = false,
            /// The engine compiles Relaxed-SIMD FMA with true single rounding (cap_relaxed_fma).
            relaxed_fma: bool = false,
            /// The module's register usage (what its prologue loads), from
            /// before the prologue on.
            use: ?*const Use = null,
            /// Open blocks/loops/ifs; the exit block is the outermost.
            nest: u32 = 0,
            /// Highest local index the code got, set or teed so far (an ISA
            /// may declare only the locals up to it).
            top: u32 = 0,
            /// Emitting the members of a region (see `emitMembers`).
            rg: ?*Region = null,
            /// The ISA's emission state.
            x: I.EmitExt = .{},
            /// Statistics: which kind of site the next `I.sync` serves
            /// (Jx.sync_site + site).
            site: u8 = 0,

            pub fn op(e: *Emit, b: u8) void {
                if (e.pos < buf_size) {
                    e.buf[e.pos] = b;
                    e.pos += 1;
                } else e.ok = false;
            }
            pub fn ops(e: *Emit, bs: []const u8) void {
                for (bs) |b| e.op(b);
            }
            pub fn uleb(e: *Emit, v0: u64) void {
                var v = v0;
                while (true) {
                    const b: u8 = @truncate(v & 0x7f);
                    v >>= 7;
                    if (v == 0) return e.op(b);
                    e.op(b | 0x80);
                }
            }
            pub fn sleb(e: *Emit, v0: i64) void {
                var v = v0;
                while (true) {
                    const b: u8 = @truncate(@as(u64, @bitCast(v)) & 0x7f);
                    v >>= 7;
                    if ((v == 0 and b & 0x40 == 0) or (v == -1 and b & 0x40 != 0)) return e.op(b);
                    e.op(b | 0x80);
                }
            }
            /// A 5-byte (padded) u32 LEB placeholder; returns its position.
            pub fn reserve(e: *Emit) usize {
                const at = e.pos;
                e.ops(&.{ 0x80, 0x80, 0x80, 0x80, 0x00 });
                return at;
            }
            /// Patch a placeholder with the byte count from after it to the end.
            pub fn patch(e: *Emit, at: usize) void {
                if (!e.ok) return;
                var v: u32 = @intCast(e.pos - at - 5);
                for (0..5) |i| {
                    e.buf[at + i] = @as(u8, @truncate(v & 0x7f)) | (if (i < 4) @as(u8, 0x80) else 0);
                    v >>= 7;
                }
            }
            /// The CPU whose block is being compiled (the owner of `buf`).
            pub fn cpu(e: *const Emit) *Cpu {
                const priv: *@FieldType(Cpu, "priv") = @alignCast(@fieldParentPtr("jit", @as(*State, @alignCast(@fieldParentPtr("buf", e.buf)))));
                return @alignCast(@fieldParentPtr("priv", priv));
            }
            pub fn k64(e: *Emit, v: u64) void {
                e.op(W.i64_const);
                e.sleb(@bitCast(v));
            }
            /// A SIMD instruction (prefix 0xfd, opcode as a u32 LEB).
            pub fn vop(e: *Emit, code: u32) void {
                e.op(0xfd);
                e.uleb(code);
            }
            /// A SIMD lane instruction (extract/replace_lane) with its lane.
            pub fn vlane(e: *Emit, code: u32, lane: u8) void {
                e.vop(code);
                e.op(lane);
            }
            pub fn k32(e: *Emit, v: i32) void {
                e.op(W.i32_const);
                e.sleb(v);
            }
            pub fn get(e: *Emit, l: u32) void {
                if (l > e.top) e.top = l;
                e.op(W.local_get);
                e.uleb(l);
            }
            pub fn set(e: *Emit, l: u32) void {
                if (l > e.top) e.top = l;
                e.op(W.local_set);
                e.uleb(l);
            }
            pub fn tee(e: *Emit, l: u32) void {
                if (l > e.top) e.top = l;
                e.op(W.local_tee);
                e.uleb(l);
            }
            pub fn kf(e: *Emit, v: f64) void {
                e.op(W.f64_const);
                const u: u64 = @bitCast(v);
                for (0..8) |i| e.op(@truncate(u >> @intCast(8 * i)));
            }
            pub fn call(e: *Emit, f: u32) void {
                e.op(W.call);
                e.uleb(f);
            }
            pub fn gget(e: *Emit) void {
                e.ops(&.{ W.global_get, 0 });
            }
            pub fn gset(e: *Emit) void {
                e.ops(&.{ W.global_set, 0 });
            }
            /// local l (i64) as f64 / an f32 held in the low 32 bits, widened.
            pub fn getD(e: *Emit, l: u32) void {
                e.get(l);
                e.op(W.f64_reinterpret);
            }
            pub fn getS(e: *Emit, l: u32) void {
                e.get(l);
                e.ops(&.{ W.i32_wrap, W.f32_reinterpret, W.f64_promote });
            }
            /// The i64 address on the stack becomes a memory address operand.
            pub fn addr(e: *Emit) void {
                if (e.mem32) e.op(W.i32_wrap);
            }
            pub fn mem(e: *Emit, opc: u8, alignment: u8, off: u64) void {
                e.op(opc);
                e.uleb(alignment);
                e.uleb(off);
            }
            /// Push *irqp (the CPU's word of pending interrupts). With a shared
            /// memory another CPU sets it (an IPI: a TLB shootdown waits for the
            /// answer), so the load is atomic (i32.atomic.load: a plain load in V8,
            /// a read-modify-write in JavaScriptCore, so each CPU's word needs a
            /// cache line of its own): an engine may hoist a plain load out of a loop
            /// that stores nothing, which would leave a compiled self-loop deaf.
            pub fn irqWord(e: *Emit) void {
                e.get(l_irq);
                e.addr();
                if (e.shared) {
                    e.ops(&.{ 0xfe, 0x10 }); // i32.atomic.load
                    e.uleb(2);
                    e.uleb(0);
                } else e.mem(W.i32_load, 2, 0);
            }
            pub fn open(e: *Emit, kind: u8, bt: u8) void {
                e.op(kind);
                e.op(bt);
                e.nest += 1;
            }
            pub fn close(e: *Emit) void {
                e.op(W.end);
                e.nest -= 1;
            }
            /// Branch to the end of the exit block (the epilogue).
            pub fn brExit(e: *Emit) void {
                e.op(W.br);
                e.uleb(e.nest - 1);
            }
            /// Call the kwemu function at table index `f` (type 0: four i64
            /// arguments on the stack, an i64 result) through env.table.
            pub fn callTable(e: *Emit, f: u64) void {
                std.debug.assert(e.table);
                if (e.mem32) e.k32(@intCast(f)) else e.k64(f);
                e.op(0x11); // call_indirect
                e.uleb(0); // type 0
                e.uleb(0); // table 0
            }
            /// Leave with pc = npc and result code `code` (branch to the exit
            /// with `brExit`, or fall through to it at the block's end).
            pub fn exit(e: *Emit, npc: u64, code: u64) void {
                e.site = 0;
                I.sync(e);
                e.k64(npc);
                e.set(I.l_npc);
                e.k64(code);
                e.set(I.l_code);
            }
            /// Leave the function: hand op k (at pc) back to the interpreter.
            pub fn handBack(e: *Emit, k: usize, pc: u64) void {
                e.exit(pc, k);
                e.brExit();
            }

            /// Inline data-TLB lookup for an access of `size` bytes at the
            /// address in local `I.l_va`: leaves the host address on the stack
            /// (as a memory operand), or hands op k (at pc) back on a miss
            /// (same test as Cpu.load/storeChk: tag == vpn of the last byte).
            /// `tlb_off`: tlb_r_off or tlb_w_off.
            pub fn tlbCheck(e: *Emit, k: usize, pc: u64, tlb_off: u64, size: u64) void {
                e.get(l_cpu);
                e.get(I.l_va);
                e.k64(8);
                e.op(W.i64_shr_u);
                e.k64(dtlb_mask);
                e.op(W.i64_and);
                e.op(W.i64_add);
                e.tee(I.l_e);
                e.addr();
                e.mem(W.i64_load, 3, tlb_off); // tag
                e.get(I.l_va);
                if (size > 1) {
                    e.k64(size - 1);
                    e.op(W.i64_add);
                }
                e.k64(12);
                e.op(W.i64_shr_u);
                e.op(W.i64_ne);
                e.open(W.@"if", W.void_t);
                e.handBack(k, pc);
                e.close();
                e.get(I.l_e);
                e.addr();
                e.mem(W.i64_load, 3, tlb_off + 8); // add
                e.get(I.l_va);
                e.op(W.i64_add);
                e.addr();
            }

            /// Self-loop back edge: stop (pc = block start, done) when `I.poll`
            /// says so or when the budget does not cover another pass, else
            /// charge the pass and branch to the loop head. The self-loop's
            /// `loop` is the innermost open block when this is emitted at the
            /// terminator's top level (`I.emitOp` with self_loop). In a region
            /// (a self-loop that joined one as a guest) it is the branch to the
            /// running member's own start (`toMember`), whose loop is `rg.self`.
            pub fn backEdge(e: *Emit, b: *const Block) void {
                if (e.rg) |rg| return e.toMember(rg, rg.cur);
                I.poll(e);
                e.get(l_jl);
                e.k64(b.ninsn);
                e.op(W.i64_lt_u);
                e.op(W.i32_or);
                e.open(W.@"if", W.void_t);
                e.exit(b.pc, code_done_stop);
                e.brExit();
                e.close();
                e.get(l_jl);
                e.k64(b.ninsn);
                e.op(W.i64_sub);
                e.set(l_jl);
                // the loop head: an ISA may know state is dead there
                e.site = 1;
                if (@hasDecl(I, "syncHead")) I.syncHead(e) else I.sync(e);
                e.op(W.br);
                e.uleb(e.nest - 2); // the loop
            }

            /// if (l_npc == member t's pc) toMember(t)
            pub fn targetIf(e: *Emit, rg: *Region, t: usize) void {
                e.get(I.l_npc);
                e.k64(rg.block(t).pc);
                e.op(W.i64_eq);
                e.open(W.@"if", W.void_t);
                e.toMember(rg, t);
                e.close();
            }

            /// Branch to member t's checked entry. A branch to the start of the
            /// member being run (an indirect jump) needs no op or generation
            /// test, and continues in the member's own loop.
            pub fn toMember(e: *Emit, rg: *Region, t: usize) void {
                if (t == rg.cur) {
                    const b = rg.block(t);
                    I.poll(e);
                    e.get(l_jl);
                    e.k64(b.ninsn);
                    e.op(W.i64_lt_u);
                    e.op(W.i32_or);
                    e.open(W.@"if", W.void_t);
                    e.statInc(Jx.region_stop);
                    e.exit(b.pc, code_done_stop);
                    e.brExit();
                    e.close();
                    e.get(l_jl);
                    e.k64(b.ninsn);
                    e.op(W.i64_sub);
                    e.set(l_jl);
                    e.statInc(Jx.region_hops);
                    e.site = 1;
                    if (@hasDecl(I, "syncHead")) I.syncHead(e) else I.sync(e);
                    e.op(W.br);
                    e.uleb(e.nest - rg.self);
                    return;
                }
                if ((rg.block(t).pc ^ rg.block(rg.cur).pc) >> 12 != 0) e.statInc(Jx.region_cross);
                e.k64(2 * t);
                e.set(l_next);
                e.site = 2;
                I.sync(e);
                e.op(W.br);
                e.uleb(e.nest - rg.top);
            }

            /// Side exit at op k of member b towards pc: refund the rest of b,
            /// then continue at a member or leave ("done": `run` refunds
            /// nothing).
            pub fn regionSide(e: *Emit, rg: *Region, pc: u64, k: usize, b: *const Block) void {
                const refund = b.ninsn - (k + 1);
                if (refund > 0) {
                    e.get(l_jl);
                    e.k64(refund);
                    e.op(W.i64_add);
                    e.set(l_jl);
                }
                if (rg.memberAt(pc)) |t| e.toMember(rg, t);
                e.statInc(Jx.region_out);
                e.exit(pc, code_done_side);
                e.brExit();
            }

            /// -Dkwemu-stats: count Stats.jx[i] in the generated code.
            pub fn statInc(e: *Emit, i: usize) void {
                if (!kwemu.stats_on) return;
                const off = priv_off + @offsetOf(@FieldType(Cpu, "priv"), "stats") + @offsetOf(kwemu.Stats, "jit") + @offsetOf(kwemu.JitStats, "jx") + 8 * i;
                e.get(l_cpu);
                e.addr();
                e.get(l_cpu);
                e.addr();
                e.mem(W.i64_load, 3, off);
                e.k64(1);
                e.op(W.i64_add);
                e.mem(W.i64_store, 3, off);
            }
        };

        const l_next = I.l_region;
        const l_cur = I.l_region + 1;
        const l_tg = I.l_region + 2;
        const l_g = I.l_region + 3;

        /// Compile b (it got hot); true: `run` must look b up again before it
        /// runs anything (`windowClose`).
        noinline fn compile(cpu: *Cpu, b: *Block) bool {
            @branchHint(.cold);
            const st = &cpu.priv.jit;
            if (!I.supported(I.ops(cpu, b)[0])) {
                b.jx.st = never;
                st.skips += 1;
                return false;
            }
            // Compiled blocks' ops live in the tenured ring (kwemu.zig, `nops`),
            // numbered on for good; when the numbers run out, everything
            // compiled goes (b's ops may go with it: look it up again).
            if (cpu.tenuredWorn()) {
                cpu.resetTenured();
                if (b.op == kwemu.op_gone) return true;
            }
            windowOpen(cpu);
            // An interrupt that is already pending (an IPI sent just before the
            // window opened, whose sender may now be waiting) is taken first: the
            // block runs interpreted this pass, and its next pass compiles it.
            if (cpu.irq_pending) |p| {
                if (p.* != 0) return windowClose(cpu);
            }
            var rg: Region = .{ .cpu = cpu };
            gatherRegion(cpu, b, &rg);
            install(cpu, &rg, b.pc >> 12, null);
            // Making room in the tenured ring may have dropped b's ops.
            return windowClose(cpu) or b.op == kwemu.op_gone;
        }

        /// Compile region `rg` (member 0 first) into a free pool entry for virtual
        /// page `page` and make it the home of every member that is not compiled
        /// elsewhere (the others are guests: their code is copied into this module,
        /// but `run` keeps entering them through their own region).
        /// `grow`: rg is the region in pool entry grow.? with a block added. If it
        /// does not fit the buffer, that region stays as it is (closed) instead of
        /// being cut: a cut would drop members, which then run interpreted until
        /// they get hot again. Otherwise member 0 is the block that got hot, which
        /// is never compiled again if this fails. Called inside the window
        /// (`windowOpen`).
        fn install(cpu: *Cpu, rg: *Region, page: u64, grow: ?u32) void {
            const st = &cpu.priv.jit;
            const b: *Block = &cpu.priv.blocks[rg.idx[0]];
            const hot = grow == null;
            // A module that does not fit the buffer is emitted again with less in
            // it: a region drops the members found last, down to b alone, and then
            // b is emitted for a shorter prefix (the rest is handed back), so a
            // block of many large ops is still compiled. Either way no other block
            // joins this region later (it would only push others out). Every retry
            // emits strictly less, so this ends; a module that did not fit is never
            // handed to the host. The code does not depend on the pool entry, so it
            // is emitted before one is taken.
            // Members' code is checked against their op number (`checkedEntry`),
            // which must outlive nursery resets: their ops move to the tenured
            // ring first. Making room there drops its oldest blocks, which may
            // be members (tenured before): those are left out, and nothing is
            // compiled if b is one.
            for (rg.idx[0..rg.n]) |x| {
                const m = &cpu.priv.blocks[x];
                cpu.tenure(m, I.slots(m));
            }
            if (b.op == kwemu.op_gone) return;
            var kept: usize = 0;
            for (rg.idx[0..rg.n]) |x| {
                if (cpu.priv.blocks[x].op == kwemu.op_gone) continue;
                rg.idx[kept] = x;
                kept += 1;
            }
            rg.n = kept;
            var closed = false;
            var limit: usize = b.n;
            const len = while (true) {
                if (emitModule(cpu, b, if (rg.n > 1) rg else null, &limit)) |n| break n else |e| switch (e) {
                    error.Unsupported => {
                        if (hot) b.jx.st = never;
                        st.skips += 1;
                        return;
                    },
                    error.Overflow => {
                        closed = true;
                        if (grow) |g| {
                            st.page[g] |= page_closed;
                            return;
                        }
                        if (rg.n > 1) {
                            rg.n = if (rg.n > 4) rg.n * 3 / 4 else rg.n - 1;
                            st.shortened += 1;
                        } else if (limit > 1) {
                            limit = limit * 3 / 4;
                            st.shortened += 1;
                        } else {
                            b.jx.st = never;
                            st.fails += 1;
                            return;
                        }
                    },
                }
            };
            if (grow) |g| {
                // (Making room in the tenured ring may have freed it already.)
                if (st.owner[g] != 0) freeSlot(cpu, g, .exit_grow);
                st.exit_grows += 1;
            }
            const i = allocSlot(cpu);
            // A guest-only module would never be entered (cut down to its guests).
            var home = false;
            for (rg.idx[0..rg.n]) |x| home = home or cpu.priv.blocks[x].jx.st < compiled;
            if (!home) return;
            st.max_bytes = @max(st.max_bytes, @as(u32, @intCast(len)));
            st.size[i] = @intCast(len);
            if (kw_host_jit_compile(&st.buf, len, tableSlot(i), compile_warm) < 0) {
                if (hot) b.jx.st = never;
                st.fails += 1;
                return;
            }
            st.owner[i] = rg.idx[0] + 1;
            // A self-loop's own module takes no members either: `findRegion` would
            // otherwise pull it into the next region of its page (kbench fp's dgemm
            // loop: 457 MIPS as a member, 540 on its own).
            if (rg.n == 1 and selfLoop(cpu, b)) closed = true;
            if (rg.n >= @min(st.region_max, max_members)) closed = true;
            st.page[i] = page | (if (closed) page_closed else 0);
            for (0..max_members) |m| {
                st.members[i][m] = if (m < rg.n) @intCast(rg.idx[m] + 1) else 0;
                if (m >= rg.n) continue;
                const x = &cpu.priv.blocks[rg.idx[m]];
                st.member_op[i][m] = @truncate(x.op);
                if (x.jx.st < compiled) x.jx.st = memberSt(i, m) | ref_bit;
            }
            st.members_total += rg.n;
            st.compiles += 1;
        }

        /// `run` is about to enter compiled block `b` right after compiled code of
        /// the region in pool entry `s` left (terminator or side exit) towards it.
        /// Cheap unless b is compiled in another region. True: a grow compiled, and
        /// `run` must look b up again (`windowClose`).
        pub inline fn noteExit(cpu: *Cpu, s: u32, b: *Block) bool {
            if (b.jx.st < compiled or b.jx.st & slot_mask == s) return false;
            // noteHot's first test, inline: most such exits leave closed regions.
            if (cpu.priv.jit.page[s] & page_closed != 0) return false;
            return noteHot(cpu, s, b);
        }

        /// Region growth along hot exits: a block compiled in another region
        /// (on any page, the region's own included) that `run` entered
        /// `grow_threshold` times per KiB of the region's module right after a
        /// region left towards it, joins the region that left last as a guest
        /// member (it keeps its own module, which `run` still enters). Calls,
        /// returns and jumps into other pages (libc, PLT stubs, functions that
        /// span pages), blocks of the page compiled into other modules, and
        /// self-loops entered from the region (their module stays their home)
        /// then stay inside one module. Closed or full regions are left alone
        /// (`noteExit` tests closed ones before the call).
        noinline fn noteHot(cpu: *Cpu, s: u32, b: *Block) bool {
            const st = &cpu.priv.jit;
            // Called on every such exit: only the cheap tests before counting.
            if (st.page[s] & page_closed != 0) return false;
            b.jx.hits += 1;
            // A grow recompiles the whole region: the bigger it is, the more exits
            // it takes (grow_threshold per KiB of module).
            if (b.jx.hits < st.grow_threshold *| @max(1, st.size[s] >> 10)) return false;
            b.jx.hits = 0;
            // Already a member with this decode: the region stopped before it
            // (budget, interrupt, a page revalidated since).
            const me = blockIndex(cpu, b) + 1;
            for (st.members[s], st.member_op[s]) |m, o| {
                if (m == me and o == @as(u16, @truncate(b.op))) return false;
            }
            const limit = @min(st.region_max, max_members);
            var rg: Region = .{ .cpu = cpu };
            for (st.members[s]) |m| {
                if (m != 0 and usable(cpu, &cpu.priv.blocks[m - 1])) addMember(&rg, m - 1);
            }
            if (rg.n == 0 or rg.n >= limit) {
                st.page[s] |= page_closed; // full: stop counting its exits
                return false;
            }
            addMember(&rg, blockIndex(cpu, b));
            windowOpen(cpu);
            install(cpu, &rg, st.page[s], s);
            // Making room in the tenured ring may have dropped b's ops.
            return windowClose(cpu) or b.op == kwemu.op_gone;
        }

        /// A decoded block that may be a region member: a self-loop only as a
        /// guest (compiled in its own module).
        fn usable(cpu: *Cpu, b: *const Block) bool {
            return b.pc != std.math.maxInt(u64) and b.op != kwemu.op_gone and I.supported(I.ops(cpu, b)[0]) and
                (b.jx.st >= compiled or !selfLoop(cpu, b));
        }

        /// A pass through `b` is about to run: whether it runs compiled. Counts the
        /// pass towards the threshold and compiles the block once it is hot. One
        /// load and no store for a block that is never compiled.
        pub inline fn ready(cpu: *Cpu, b: *Block) Go {
            const s = b.jx.st;
            if (s >= compiled) return .compiled;
            if (s == never) return .interp;
            if (s > b.ninsn) {
                b.jx.st = s - b.ninsn;
                return .interp;
            }
            if (compile(cpu, b)) return .again;
            return if (b.jx.st >= compiled) .compiled else .interp;
        }

        /// Statistics: how the compiled pass of `b` that returned `code` left it:
        /// 0 = handed back, else Jx.exit_*.
        pub fn countExit(cpu: *Cpu, b: *const Block, code: u32) u32 {
            jx(cpu, Jx.entries, 1);
            const kind: u32 = if (code < code_done)
                Jx.handbacks
            else if (code >= code_side or code == code_done_side)
                Jx.exit_side
            else if (code == code_done_stop)
                Jx.exit_stop
            else if (I.indirect(I.ops(cpu, b)[b.n - 1]))
                Jx.exit_jalr
            else
                Jx.exit_direct;
            jx(cpu, kind, 1);
            if (code == code_done_side) jx(cpu, Jx.exit_side_region, 1);
            return if (kind == Jx.handbacks) 0 else kind;
        }

        /// Statistics: block `b` runs after a compiled pass of `from` that left it
        /// the way `kind` (countExit) says.
        pub fn countNext(cpu: *Cpu, kind: u32, from: *const Block, b: *const Block, slot: u32) void {
            const xr: Xr = switch (kind) {
                Jx.exit_side => .side,
                Jx.exit_jalr => .jalr,
                Jx.exit_stop => .stop,
                else => .direct,
            };
            var nx: Nx = if (b == from) .self else if (b.jx.st == never) .never else .interp;
            if (kind != Jx.exit_stop) {
                const same_page = (from.pc ^ b.pc) >> 12 == 0;
                const i = Jx.next + 4 * @as(usize, @intFromBool(kind == Jx.exit_jalr)) + 2 * @as(usize, @intFromBool(same_page)) + @intFromBool(b.jx.st >= compiled);
                jx(cpu, i, 1);
                if (b == from) jx(cpu, Jx.next_self, 1);
                if (b.jx.st == never) jx(cpu, Jx.next_never, 1);
            }
            if (b.jx.st >= compiled) {
                const s = b.jx.st & slot_mask;
                const st = &cpu.priv.jit;
                const nc: Nx = if (s == slot)
                    .same_region
                else if (st.members[s][1] == 0 and selfLoop(cpu, b))
                    .selfloop
                else full: {
                    var n: usize = 0;
                    for (st.members[slot]) |m| n += @intFromBool(m != 0);
                    break :full if (st.page[slot] & page_closed != 0 or n >= st.region_max) .other_full else .other;
                };
                if (kind != Jx.exit_stop) switch (nc) {
                    .same_region => jx(cpu, Jx.next_same_region, 1),
                    .selfloop => jx(cpu, Jx.next_selfloop, 1),
                    .other => jx(cpu, Jx.next_other_region, 1),
                    .other_full => {
                        jx(cpu, Jx.next_other_region, 1);
                        jx(cpu, Jx.next_other_full, 1);
                    },
                    else => unreachable,
                };
                if (nx != .self) nx = nc;
            }
            jx(cpu, Jx.next_by + Jx.n_nx * @as(usize, @intFromEnum(xr)) + @intFromEnum(nx), 1);
        }

        /// Translate block `b`, at most its first `limit.*` ops, or region `rg`
        /// (whose member 0 is b; members are emitted in full and `limit` only
        /// reports b's translated prefix), into a module in cpu.priv.jit.buf;
        /// returns its length. On error.Overflow, `limit.*` is the number of ops of
        /// b that were emitted.
        pub fn emitModule(cpu: *Cpu, b: *const Block, rg: ?*Region, limit: *usize) Error!usize {
            const all = I.ops(cpu, b);
            var u: usize = 0;
            while (u < all.len and u < limit.* and I.supported(all[u])) u += 1;
            if (u == 0) return error.Unsupported;
            limit.* = u;
            const full = u == all.len;
            const self_loop = rg == null and full and I.selfLoop(all[all.len - 1], b);

            const caps = cpu.priv.jit.caps;
            var e: Emit = .{
                .buf = &cpu.priv.jit.buf,
                .mem32 = caps & cap_mem32 != 0,
                .shared = caps & cap_shared != 0,
                .table = caps & cap_table != 0,
                .simd = caps & cap_simd != 0,
                .relaxed_fma = caps & cap_simd != 0 and caps & cap_relaxed_fma != 0,
            };
            e.rg = rg;
            // b's per-member state (a region's members set their own).
            I.beginMember(&e, all[0..u]);
            // A region's prologue and epilogue cover the union over its members:
            // one member may enter another without returning. Every register the
            // code writes is loaded as well: a hand-back writes all of them back,
            // including those the executed part did not reach.
            var use = I.blockUse(all[0..u]);
            if (rg) |r| regionUse(cpu, r, &use);
            const nfn: u32 = 1 + I.helperCount(&e, use);

            e.ops(&.{ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00 });
            // type 0: (i64 i64 i64 i64) -> i64 (f); type 1: (i64 i64 i64) -> i64 (the helpers)
            e.ops(&.{ 0x01, 0x10, 0x02, 0x60, 0x04, W.i64_t, W.i64_t, W.i64_t, W.i64_t, 0x01, W.i64_t });
            e.ops(&.{ 0x60, 0x03, W.i64_t, W.i64_t, W.i64_t, 0x01, W.i64_t });
            // import env.memory: memory64 (flags 4) or memory32, min 0; a shared
            // memory (flags | 3) needs a maximum, which an import matches when the
            // memory's is not larger: 65536 pages (4 GiB) for memory32, 262144
            // (16 GiB, the kernel's --max-memory) for memory64.
            // With cap_table also env.table: funcref, min 0, table64 unless the
            // kernel was lowered (memory64 and table64 lowering go together).
            const table_len = 14; // the entry below
            const mem_len: u8 = if (e.shared) 0x12 else 0x0f;
            e.ops(&.{ 0x02, mem_len + (if (e.table) @as(u8, table_len) else 0), if (e.table) 0x02 else 0x01 });
            e.ops(&.{ 0x03, 'e', 'n', 'v', 0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02 });
            if (e.shared)
                e.ops(&.{ if (e.mem32) 0x03 else 0x07, 0x00, 0x80, 0x80, if (e.mem32) 0x04 else 0x10 })
            else
                e.ops(&.{ if (e.mem32) 0x00 else 0x04, 0x00 });
            if (e.table) {
                e.ops(&.{ 0x03, 'e', 'n', 'v', 0x05, 't', 'a', 'b', 'l', 'e', 0x01, 0x70 });
                e.ops(&.{ if (e.mem32) 0x00 else 0x04, 0x00 });
            }
            // functions: f (type 0), then the helpers (type 1)
            e.ops(&.{ 0x03, @intCast(1 + nfn), @intCast(nfn), 0x00 });
            for (1..nfn) |_| e.op(0x01);
            // the helpers' status word: global 0, mutable i32 (not exported)
            if (nfn > 1) e.ops(&.{ 0x06, 0x06, 0x01, 0x7f, 0x01, W.i32_const, 0x00, W.end });
            // export "f"
            e.ops(&.{ 0x07, 0x05, 0x01, 0x01, 'f', 0x00, 0x00 });
            e.op(0x0a);
            const code_sec = e.reserve();
            e.uleb(nfn);
            const body = e.reserve();
            I.locals(&e, rg != null);

            // cpu == 0 (never from `run`): return at once. The host calls f so
            // inside kw_host_jit_compile, so that an engine that compiles a
            // function only at its first call (V8) does it there, inside the
            // jit_busy window, instead of in `run`'s first call (kw_emu.h).
            e.get(l_cpu);
            e.op(W.i64_eqz);
            e.open(W.@"if", W.void_t);
            e.k64(0);
            e.op(W.@"return");
            e.close();

            e.use = &use;
            I.prologue(&e, use);
            e.open(W.block, W.void_t); // exit
            if (rg) |r| emitMembers(&e, cpu, r) else {
                if (self_loop) e.open(W.loop, W.void_t);
                var pc = b.pc;
                for (all[0..u], 0..) |o, k| {
                    I.emitOp(&e, o, k, pc, b, self_loop);
                    pc = I.nextPc(o, pc);
                }
                if (!full) e.handBack(u, pc);
                if (self_loop) e.close();
            }
            e.close();

            // epilogue: write back, pc, return (jl << 8) | code
            I.epilogue(&e, use);
            e.get(l_x);
            e.addr();
            e.get(I.l_npc);
            e.mem(W.i64_store, 3, 32 * 8);
            if (rg != null) regionExit(&e);
            e.get(l_jl);
            e.k64(8);
            e.op(W.i64_shl);
            e.get(I.l_code);
            e.op(W.i64_or);
            e.op(W.end);
            e.patch(body);
            I.emitHelpers(&e);
            e.patch(code_sec);
            if (!e.ok) return error.Overflow;
            return e.pos;
        }

        /// Index of the decoded block at pc t that may join b's region: valid under
        /// the current fill of its page (on b's page: also on b's host page), not
        /// compiled, compilable, no self-loop.
        fn candidate(cpu: *Cpu, b: *const Block, t: u64) ?u32 {
            const set = kwemu.blockSet(t);
            for (cpu.priv.blocks[set..][0..2], set..) |*c, i| {
                if (c.pc != t) continue;
                if ((t ^ b.pc) >> 12 == 0) {
                    if (c.tgen != b.tgen or c.host >> 12 != b.host >> 12) continue;
                } else if (!current(cpu, c)) continue;
                if (c.jx.st == never or c.jx.st >= compiled or !I.supported(I.ops(cpu, c)[0]) or selfLoop(cpu, c)) return null;
                return @intCast(i);
            }
            return null;
        }

        /// A block that branches or jumps back to its own start keeps a module of
        /// its own: as a plain wasm loop that loads only its own registers it is
        /// faster than inside a region, and much cheaper to leave and re-enter
        /// (data-TLB misses hand back to `run` every few passes in loops over large
        /// arrays).
        pub fn selfLoop(cpu: *Cpu, b: *const Block) bool {
            return I.selfLoop(I.ops(cpu, b)[b.n - 1], b);
        }

        /// Collect the region for b (see above).
        fn gatherRegion(cpu: *Cpu, b: *Block, rg: *Region) void {
            const st = &cpu.priv.jit;
            const limit = @min(st.region_max, max_members);
            rg.idx[0] = blockIndex(cpu, b);
            rg.n = 1;
            if (limit <= 1 or selfLoop(cpu, b)) return;
            if (findRegion(cpu, b, limit)) |i| {
                for (st.members[i]) |m| {
                    if (m != 0 and usable(cpu, &cpu.priv.blocks[m - 1])) addMember(rg, m - 1);
                }
                freeSlot(cpu, i, .grow);
                st.grows += 1;
            }
            pageWarm(cpu, b, rg, limit);
            var q: usize = 0;
            while (q < rg.n and rg.n < limit) : (q += 1) {
                const m = rg.block(q);
                for (I.ops(cpu, m)) |o| {
                    for (I.targets(o, m)) |t| {
                        if (rg.n == limit) return;
                        if (t) |pc| if (candidate(cpu, b, pc)) |i| addMember(rg, i);
                    }
                }
            }
        }

        /// Add the blocks of b's page that could be members (`candidate`) and have
        /// counted at least threshold >> warm_shift: most of them would get hot
        /// later, one at a time, and each would compile the region again to join
        /// it (`findRegion`). Probes every pc of the page (A64: instructions are
        /// 4-byte aligned); runs before the breadth-first search, so a module
        /// that does not fit the buffer drops cold members first.
        fn pageWarm(cpu: *Cpu, b: *const Block, rg: *Region, limit: usize) void {
            const min = cpu.priv.jit.threshold >> warm_shift;
            const base = b.pc & ~@as(u64, 0xfff);
            var pc = base;
            while (pc < base + 4096 and rg.n < limit) : (pc += 4) {
                const i = candidate(cpu, b, pc) orelse continue;
                if (heatOf(cpu, &cpu.priv.blocks[i]) >= min) addMember(rg, i);
            }
        }

        /// OR the register usage of members 1.. into `use` (member 0 is the block
        /// emitModule was called for).
        fn regionUse(cpu: *Cpu, rg: *const Region, use: *Use) void {
            for (1..rg.n) |m| {
                const all = I.ops(cpu, rg.block(m));
                var u: usize = 0;
                while (u < all.len and I.supported(all[u])) u += 1;
                use.add(I.blockUse(all[0..u]));
            }
        }

        /// The body of a region (inside the exit block): dispatch to the entered
        /// member, then every member, each after its checked entry. The `br_table`
        /// targets are, in order, checked entry 0, member 0, checked entry 1, ...:
        /// a checked entry falls through into its member, and a member never falls
        /// through (every path of it ends in a branch).
        fn emitMembers(e: *Emit, cpu: *Cpu, rg: *Region) void {
            // l_cur = cpu.jit.cur (the entered member), l_g = its tgen,
            // l_next = 2 * its index + 1
            e.get(l_cpu);
            e.addr();
            e.mem(W.i64_load, 3, jit_off + @offsetOf(State, "cur"));
            e.set(l_cur);
            e.get(l_cur);
            e.addr();
            e.mem(W.i64_load, 3, @offsetOf(Block, "tgen"));
            e.set(l_g);
            e.get(l_cur);
            e.addr();
            e.mem(W.i64_load32_u, 2, jst_off);
            e.k64(member_shift - 1);
            e.op(W.i64_shr_u);
            e.k64(0x1fe);
            e.op(W.i64_and);
            e.k64(1);
            e.op(W.i64_or);
            e.set(l_next);
            e.open(W.loop, W.void_t);
            rg.top = e.nest;
            for (0..2 * rg.n) |_| e.open(W.block, W.void_t);
            e.get(l_next);
            e.op(W.i32_wrap);
            e.op(W.br_table);
            e.uleb(2 * rg.n);
            for (0..2 * rg.n) |i| e.uleb(i);
            e.uleb(2 * rg.n - 1);
            for (0..rg.n) |m| {
                e.close();
                checkedEntry(e, rg, m);
                e.close();
                emitMember(e, cpu, rg, m);
            }
            e.close();
        }

        fn emitMember(e: *Emit, cpu: *Cpu, rg: *Region, m: usize) void {
            const b = rg.block(m);
            const all = I.ops(cpu, b);
            var u: usize = 0;
            while (u < all.len and I.supported(all[u])) u += 1;
            rg.cur = m;
            // A member may be entered first: its own state starts afresh.
            I.beginMember(e, all[0..u]);
            e.get(l_cpu);
            e.k64(blocks_off + @as(u64, rg.idx[m]) * @sizeOf(Block));
            e.op(W.i64_add);
            e.set(l_cur);
            e.open(W.loop, W.void_t);
            rg.self = e.nest;
            // A self-loop (a guest: its home is a module of its own) branches
            // back to its start through `backEdge`, as in that module.
            const self_loop = u == all.len and I.selfLoop(all[all.len - 1], b);
            var pc = b.pc;
            for (all[0..u], 0..) |o, k| {
                I.emitOp(e, o, k, pc, b, self_loop);
                pc = I.nextPc(o, pc);
            }
            if (u < all.len) {
                e.handBack(u, pc);
            } else {
                // The terminator set l_npc (and l_code = done): continue at a member.
                I.memberTail(e, rg, b, all[all.len - 1]);
                e.statInc(Jx.region_out);
                e.brExit();
            }
            e.close();
        }

        /// Member m's checked entry (see above): checks, charge, then fall through
        /// into m; or return to `run` with pc = m's start.
        fn checkedEntry(e: *Emit, rg: *Region, m: usize) void {
            const b = rg.block(m);
            const at = blocks_off + @as(u64, rg.idx[m]) * @sizeOf(Block);
            I.poll(e);
            e.get(l_jl);
            e.k64(b.ninsn);
            e.op(W.i64_lt_u);
            e.op(W.i32_or);
            // The same decoded block: its ops have the number they had when
            // this module was emitted (tenured op numbers are never reused).
            e.get(l_cpu);
            e.addr();
            e.mem(W.i32_load, 2, at + @offsetOf(Block, "op"));
            e.k32(@bitCast(b.op));
            e.op(W.i32_ne);
            e.op(W.i32_or);
            // Valid under the current fill of its page: the generation the member
            // being run was validated under (l_g), else the generation of one of
            // the two exec-TLB entries of its page's set (like `lookup`).
            const set: u64 = tlb_x_off + 2 * ((b.pc >> 12) % xtlb_sets) * @sizeOf(XTlb) + @offsetOf(XTlb, "gen");
            e.get(l_cpu);
            e.addr();
            e.mem(W.i64_load, 3, at + @offsetOf(Block, "tgen"));
            e.tee(l_tg);
            e.get(l_g);
            e.op(W.i64_ne);
            e.open(W.@"if", W.i32_t);
            e.get(l_tg);
            e.get(l_cpu);
            e.addr();
            e.mem(W.i64_load, 3, set);
            e.op(W.i64_ne);
            e.get(l_tg);
            e.get(l_cpu);
            e.addr();
            e.mem(W.i64_load, 3, set + @sizeOf(XTlb));
            e.op(W.i64_ne);
            e.op(W.i32_and);
            e.op(W.@"else");
            e.k32(0);
            e.close();
            e.op(W.i32_or);
            e.open(W.@"if", W.void_t);
            e.statInc(Jx.region_stop);
            e.exit(b.pc, code_done_stop);
            e.brExit();
            e.close();
            e.get(l_jl);
            e.k64(b.ninsn);
            e.op(W.i64_sub);
            e.set(l_jl);
            e.get(l_tg);
            e.set(l_g);
            e.statInc(Jx.region_hops);
        }

        /// Epilogue part: name the member the code leaves from.
        fn regionExit(e: *Emit) void {
            e.get(l_cpu);
            e.addr();
            e.get(l_cur);
            e.mem(W.i64_store, 3, jit_off + @offsetOf(State, "cur"));
        }
    };
}

comptime {
    // TLB index (va >> 8) & dtlb_mask assumes 16-byte entries, a power-of-two
    // count, and read and write TLBs of the same length.
    std.debug.assert(@sizeOf(@typeInfo(@FieldType(@FieldType(Cpu, "priv"), "tlb_r")).array.child) == 16);
    std.debug.assert(std.math.isPowerOfTwo(@typeInfo(@FieldType(@FieldType(Cpu, "priv"), "tlb_r")).array.len));
    std.debug.assert(@typeInfo(@FieldType(@FieldType(Cpu, "priv"), "tlb_r")).array.len ==
        @typeInfo(@FieldType(@FieldType(Cpu, "priv"), "tlb_w")).array.len);
}
