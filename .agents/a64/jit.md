# The A64 JIT (src/a64/jit)

Hot decoded blocks are compiled to wasm by the ISA-neutral JIT core,
`src/emu/jitcore.zig` (kwemu.md, JIT tier: slot pool, compile policy,
`install` and its emit retries, regions, compile window, host contract;
all of it holds for A64), whose hooks `jit/emit.zig` implements (`C =
jc.Jit(@This())`). `tier.run` enters the compiled code. Built with
`-Dkwemu-jit` (wasm64 only), which kernel/build.sh passes by default
(`KWEMU_JIT=0` leaves it out). The emitted code depends only on the
decoded ops, pc and cpu-relative offsets, uses MVP instructions plus
memory64 and, where the host compiles it (`kw_host_jit_caps` bit 5,
`jc.cap_simd`), fixed-width SIMD; every module starts with the `cpu == 0`
early return of the compile window.

What is translated:

| File | Translates |
| --- | --- |
| `alu.zig` | nop, movi, movk, mov, add/sub and logical in every form, the flag-setting forms, csel, ccmp, adc; owns the NZCV model |
| `bits.zig` | ubfm, sbfm, bfm, extr, variable shifts, madd, maddl, umulh, smulh, udiv, sdiv, rev1 |
| `mem.zig` | every load and store kind |
| `branch.zig` | side exits, terminators, b_f/bl_f, br/blr, goto; the region hooks `selfLoop`, `targets`, `memberTail`, `indirect` |
| `generic.zig` | frequent `generic` ops inline: FP/general moves, MOVI, DUP, UMOV, INS, vector logic, MRS of FPCR/FPSR/NZCV/DCZID, FCMP, scalar FP arithmetic and conversions with fast paths, loads into XZR |
| `atomic.zig` | exclusives, LDAR/STLR, LSE LD<op>/SWP/CAS, CLREX, barriers, DC ZVA |
| `vec.zig` | with SIMD: integer AdvSIMD, AdvSIMD FP fast paths, permutes, LD1/ST1 and V-register pairs, AES, PMULL, SHA1, SHA256, SHA512 |
| `emit.zig` | the core's hooks, family dispatch, `generic` calls |
| `common.zig` | locals, slot access, `Use` helpers |
| `kindtest.zig`, `tests.zig` | test helpers and the core's tests |

The other `generic` ops (LD2-4/ST2-4, the rest of AdvSIMD
(also all of it without SIMD), other system ops, exclusive pairs, CASP,
SHA3, FP off its fast path) are calls of the reference interpreter through the
kernel's function table (Generic ops, below); a host without the table
hands them back and a block is compiled up to its first one.

## How compiled code runs (`tier.run`)

- `jit.C.ready` counts each pass of a block towards the threshold (2^17
  retired instructions; 1 with an eager host) and compiles the block, with
  the region gathered around it, once hot. A block counting down runs at
  most its countdown's worth of self-loop passes inside `runOps` (`cap`),
  so its next pass compiles it.
- `jc.enter` calls the module with `jl` = the budget left; it returns
  `(jl_left << 8) | code`. Done (`code >= jc.code_done`): pc is in
  regs.pc; a side exit at op k (`code_side + k`) refunds the rest of
  `jc.lastBlock` (a region's own side exits refund themselves and return
  `code_done_side`); then the interrupt poll and epoch check of every
  block end, and `noteExit` (region growth).
- Hand-back (`code < 64`): op `code` of `lastBlock` and the rest of that
  pass run through `runOps` from op k of the same decode. `run` stores 0
  into slot 32 first (compiled code writes pc there at every exit, and
  ops read that slot as XZR). No further self-loop pass runs in `runOps`
  then (`cap = 0`): the next pass enters compiled code again.
- Compiled code polls (`emit.zig` `poll`) at every self-loop back edge and
  checked member entry: `*irqp` (`i32.atomic.load` on shared memory) and,
  where several CPUs exist (`tier.multi_cpu`: atomics builds and the unit
  tests), the code epoch: `*a64.epoch_ptr` (the address of tier.zig's
  `code_epoch`, stored by kw_cpu_init and kw_host_reset, since compiled
  code addresses memory only cpu-relative) against `a64.seen_epoch`.
  Either stops the code at the block's start (`code_done_stop`). So a
  compiled loop on one CPU sees another CPU's IC IVAU within one pass. A
  UP kernel's modules have no epoch poll.

## Register file and locals (`common.zig`)

i64 locals: 0..3 the parameters (x, cpu, irqp, jl), 4..7 the core's (next
pc, result code, data address, TLB entry), 8..11 region state, 12 NZCV,
13..20 `l_t` (scratch of one op, any family), 21..24 `l_alu`, 25..33
`l_mcache` (self-loop page cache), 34..36 `l_mem` (coalesced group host
addresses), 37..68 X0..X30 and SP (slot
s: 37 + s), 69 TPIDR_EL0, 70 `l_vtmp`, 71..78 `l_u` (vec.zig scratch).
Without SIMD the V halves follow as i64 (79..142, slot 36 + j: 79 + j);
with SIMD a v128 group (79..110 V0..V31, 111..118 `l_q`) and an i32 group
(119..134 `l_w`). A module declares locals only up to the highest index
its code uses (each group's count is a padded LEB patched after the body;
a group needed after another makes that one full), which is why the rare
ones are last.

The prologue loads every slot the module's members read or write (a
hand-back writes every written slot back, also those the executed part
did not reach); the single epilogue stores the written ones. V registers
go by whole register. NZCV is `i64.load32_u`/`i64.store32` at regs offset
264 (FPCR, the high half, is never touched). `cm.get(e, slot)` pushes a
slot's value (XZR: `i64.const 0`; a V half with SIMD:
`i64x2.extract_lane`), `cm.set(e, slot)` pops into it (XZR: dropped; a V
half with SIMD: `i64x2.replace_lane` through `l_vtmp`), `cm.setScalar`
writes a V register's low half and zeroes the high one, `cm.getV`/`setV`/
`setVq` move whole v128s, `cm.local(slot)` is an i64 local.

## The family API

Each op kind belongs to exactly one family file (`alu`, `bits`, `mem`,
`branch`; a comptime check in emit.zig fails the build otherwise). A
family file implements:

- `kinds`: the op kinds it owns.
- `supported(o: Op) bool`: whether this op is translated. The core
  compiles a block's ops up to the first unsupported one and hands that
  one back. Decide on fields if only some variants are done.
- `usage(o: Op) Use`: what it reads and writes, from `cm.rd(slot)` and
  `cm.wr(slot)` combined with `.with(...)`, plus `.flags_read` /
  `.flags_write`. Exact usage matters: the prologue and epilogue follow it.
- `emit(e: *Emit, o: Op, at: At)`: the op's code; `at.k` is its index in
  the block, `at.pc` its pc, `at.b` the block, `at.self_loop` whether the
  block is emitted as its own wasm loop.
- `Ext` (its part of the emission state, `e.x.alu`, ...), `beginMember(e,
  ops)` (a block's or member's code starts: reset per-member knowledge,
  since a member can be entered from another member or from `run`), and
  `sync(e)` (the NZCV contract, below).

What `emit` may do, with the core's `Emit` (jitcore.zig):

- Wasm: `e.op`, `e.ops`, `e.k64`, `e.k32`, `e.get`/`set`/`tee`,
  `e.mem(opcode, align, offset)` with `e.addr()` before it (wraps an
  address for a memory32 kernel), `e.open(kind, type)`/`e.close()`,
  opcodes in `jc.W`.
- Hand back: `e.handBack(at.k, at.pc)` leaves before the op (the
  interpreter runs it), so emit it before the op has written anything.
- Memory: address in `cm.l_va`, then `e.tlbCheck(at.k, at.pc,
  jc.tlb_r_off or jc.tlb_w_off, size)` leaves the host address on the
  stack or hands the op back on a miss (unmapped, straddling the page end,
  a page holding decoded code: `code_flag` in the write TLB), which keeps
  faults and SMC exact. One check covers a pair or Q access in one page.
- Leave: `e.exit(target, jc.code_done)` (a terminator; in a region it
  continues at a member through `memberTail`), side exits
  `e.exit(target, jc.code_side + at.k)` + `e.brExit()` or, in a region
  (`e.rg`), `e.regionSide(rg, target, at.k, at.b)`; a self-loop's
  terminator `e.backEdge(at.b)` (poll, budget, charge, loop; in a region,
  where a self-loop is a guest, the same through `toMember` to the
  member's own `loop`).
- Locals: `l_t` (scratch within one op), its own family range, the slots
  through `cm`. More locals or a module-level helper function
  (`helperCount`, `emitHelpers`: none yet) go through the core.

Shared touch points: `Use` and `EmitExt` (emit.zig) gain fields only
through the core; `syncHead` (emit.zig) is the one place the core gives a
family a different sync; kindtest.zig's filler generators (`movi`, `nop`,
`dst`, `anyReg`, `base`, `sf`) are shared, a family keeps its own
generators; op field layouts and semantics change only with the tier.

### Worked example: `add_i`

`a = (b + imm) & (~0 >> d)` (ops.zig). In alu.zig `supported` has
`.add_i => true`, `usage` gives `cm.rd(o.b).with(cm.wr(o.a))`, and `emit`:

```zig
.add_i => {
    cm.get(e, o.b);
    e.k64(o.imm);
    e.op(W.i64_add);
    if (o.d != 0) {
        e.k64(~@as(u64, 0) >> @intCast(o.d));
        e.op(W.i64_and);
    }
    cm.set(e, o.a);
},
```

Its test, at the bottom of alu.zig:

```zig
fn genAddI(r: std.Random) u32 { // ADD/SUB (immediate), any Rn/Rd incl. SP
    return 0x1100_0000 | kt.sf(r) | (r.uintLessThan(u32, 2) << 30) | (r.uintLessThan(u32, 2) << 22) |
        (r.uintLessThan(u32, 4096) << 10) | (kt.anyReg(r) << 5) | kt.anyReg(r);
}
test "a64 jit alu: add_i" {
    try kt.kind(.add_i, kt.one(genAddI), .{});
}
```

`kt.kind` runs 200 programs of 24 such words (then SVC; `Opt` changes
counts, budget, seed, `warm_tlb`, `rwx_code`) from edge and random
register values on a kw_run machine and on the reference loop, and
compares event, icount, every register, the monitor fields and all
memory. In the wasm64 unit tests the kw_run machine compiles every block
on first entry, and the test fails unless some op of that kind was
compiled (it cannot pass vacuously); natively it checks the tier. A
generator may write several words (a setup sequence) and gets the pc of
the first.

## NZCV: the contract between alu.zig and its consumers

- `cm.l_nzcv` holds NZCV in bits 31..28 on every path that leaves
  straight-line code: the core calls `sync(e)` (each family's) before
  every exit, member transfer and self-loop back edge, and the epilogue
  stores l_nzcv when any op has `flags_write`. Every exit and hand-back
  thus leaves regs.nzcv exact.
- A consumer asks `alu.cond(e, mask)` (pushes an i32: the condition holds)
  or `alu.nzcv(e)` (the whole word as i64), never l_nzcv itself. Both push
  an expression and change no state and no local but their own, so they
  may be called on any path, also while the caller holds values in `l_t`.
- `sync` emits code only for the path it is called on and must not change
  the emit-time state of the path that continues.
- `alu.beginMember` starts every block and member with nothing known but
  l_nzcv.
- Before a branch back to the start of the block or member being emitted
  the core calls `syncHead` instead (vec's sync, alu's `syncHead`, the
  others' `sync`). alu.zig skips NZCV there when the block writes it
  before anything reads it or can leave (`head_dead`).

**Lazy flags.** A flag-setting op leaves A in `l_alu + 0`, B in `l_alu +
1` (or as a constant in `e.x.alu.bk`) and the result R in `l_alu + 2` (W
forms shifted left by 32), and records what the flags are: NZCV of A - B
(SUBS, CMP), of A + B (ADDS, CMN), or N and Z of a logical R (ANDS, TST);
CCMP/CCMN add their condition in `l_alu + 3` and their default NZCV.
`cond` computes from that: CMP + B.cond or CSEL is one i64 compare (EQ
`a == b`, HS `ge_u`, GT `gt_s`, ...), MI/PL the sign of R, VS/VC and the
signed conditions after CMN from the overflow term, a CCMP chain a
`select`; ADC's carry is `cond(HS)`. NZCV is packed into l_nzcv (about 20
wasm ops) only by `sync` on the path that leaves, or once right after a
producer whose flags reach the end of its member (the "eager" site; not in
a self-loop block, whose back edge may skip it). A backward scan in
`beginMember` marks producers whose flags are overwritten before any read
and before anything that can leave (an op of another family counts as
both): they emit only their register result, or nothing. ADCS/SBCS pack
NZCV exactly. A `generic` call leaves NZCV exact (`alu.exactNow`).

## Memory (`mem.zig`)

Each op computes its address into `l_va` (literal: the absolute address in
the op), then one `tlbCheck` over the whole access (a W pair 8 bytes, an X
pair or Q 16, LDP/STP Q 32), then plain wasm loads and stores at the host
address (further elements by offset immediates; with SIMD Q forms are
`v128.load`/`v128.store`). The host address is kept in `l_mem`, widened
back to i64 on a memory32 kernel, where `tlbCheck` leaves an i32. Shared
memory changes nothing: the tier's fast path uses plain accesses too.

- Handed back before anything is written: every data-TLB miss (unmapped,
  no permission, not in the TLB yet, a straddle, one that wraps past
  2^64), an LDP/STP Q whose 32 bytes cross a page end, and stores into a
  page holding decoded code (the interpreter stores and tracks them).
- Register writes keep the reference's order (writeback, Rt, Rt2). Past
  the check nothing can fail, so only coinciding registers matter: a
  writeback whose base is also a loaded register is left out, Rt2 is
  written after Rt (LDP with Rt == Rt2), store data is read before the
  writeback (`STR Xn, [Xn], #8` stores the old base).
- About 70 module bytes per single access (address 8, `tlbCheck` 55 with
  its hand-back, the access 5), about 12 more per further element.

## Generic ops in compiled code (emit.zig, `generic`)

Compiled code calls `opexec.jitGeneric(x, cpu, raw, pc)` with
`call_indirect` through the kernel's own function table, which the module
imports as `env.table` (funcref, table64 unless lowered). The host offers
it with caps bit 4 (`jc.cap_table`; `jc.table_ok` per CPU). jitGeneric
has the type of a compiled function (four i64, an i64 result), so no
other import or export is needed.

- Around the call: lazy NZCV is materialised and every register slot the
  module writes is stored (`Emit.use`); after it the register locals are
  stale and are loaded again before the next op of a family or on any
  exit (`sync`), so a run of generic ops spills and reloads once
  (`EmitExt.stale`). About 6 bytes per slot plus 30 for the call.
- Results: 0 retired; 1 not executed (a fault or trap, CNTVCT): handed
  back, the interpreter runs it again; 2 retired and the block must end (a
  store into decoded code, IC IVAU): a side exit to pc + 4 (in a region
  `regionSide`, through the next member's checked entry).
- The emission state is reached through a pointer in `Emit`: state larger
  than about 64 bytes initialised by value became a `memcpy` call, which
  the kernel does not link (`a64-test.mjs` checks `kwemu-jit.o`'s imports).

**Inline forms** (`generic.zig`; still `generic` in the tier, no table
needed), chosen from a stats kernel's counts: FMOV between general and FP
registers (W/S, X/D, X/V.D[1]), UMOV, INS (general) of a 64-bit lane, DUP
(general), MOVI/MVNI (and FMOV vector immediate), AND/BIC/ORR/ORN/EOR
(vector), MRS of FPCR, FPSR, NZCV and DCZID_EL0, FCMP/FCMPE (hands back on
a NaN operand or with FPCR.FZ). musl's soft long double in printf and its
memset are the main users.

**Scalar FP fast paths** (`fpFast`, `cvtz`): FADD, FSUB, FMUL, FDIV, FSQRT,
FMAX, FMIN, FMAXNM, FMINNM, FMADD, FMSUB, FNMADD, FNMSUB (S, D), SCVTF, UCVTF
(W/X to S/D), FCVTZS, FCVTZU (S/D to W/X), FCVT (S to D, D to S) inline,
bit-exact with fparith.zig, with the interpreter call on the rest; FMOV,
FABS, FNEG (register), FMOV (immediate) and FCSEL need no slow path. The
arithmetic fast path needs FPCR.RMode = nearest and FZ = 0. FADD, FSUB,
FMUL, FDIV and FSQRT check only the result: a normal number away from both
ends of the range is no NaN, infinity or tiny value, and with FZ = 0 a
denormal operand raises nothing, so IXC is the only flag that can arise
and wasm's result is Arm's. The hot path is one test and branch: such a
result with FPSR.IXC already set (FPCR and FPSR in one load) is written,
and FPSR is not. Behind it, ±0.0 results take the fast path when exact (a
zero sum always; FMUL, FDIV and FSQRT only from a zero first operand,
either for FMUL), and with FPSR.IXC clear IXC is computed exactly: TwoSum
for FADD/FSUB, the exact product in double for single FMUL/FDIV/FSQRT,
significant bits for SCVTF/UCVTF, the round trip for FCVT to single;
double FMUL/FDIV/FSQRT call the interpreter then. Scalar
FMAX/FMIN/FMAXNM/FMINNM run inline via wasm min/max when FPCR.FZ = 0 and
the result is not NaN. Scalar FMADD/FMSUB/FNMADD/FNMSUB run inline while
FPCR allows fast path and FPSR.IXC is already set, with operands in range
or ±0.0, all in one test (a branch on IXC alone first made the IXC-set
path about 12% slower, 2026-11-24): single promotes to f64 and rejects
exact ties (`0x1000_0000`); double uses Dekker TwoProd + Knuth TwoSum over
bounded exponents (`580..1460`) and rejects exact powers of two or errors
near 0.5 ULP. FCVTZS/FCVTZU do all of toFixed inline (NaN to 0, saturation, IOC);
only FZ calls. The slow path stores only the operands the interpreter reads
and loads only what it writes.

**Loads into XZR.** The tier leaves an integer load whose Rt (or Rt2) is
XZR `generic` (slot 32 is XZR for the ops). Compiled code runs it as its
load kind (`ops.xzrLoad`; mem.zig drops the write to XZR). V8 pops its
padding register so (`ldp xzr, x30, [sp], #16`).

## Atomics in compiled code (`atomic.zig`)

Inline: LDAR, STLR, LDXR, LDAXR, STXR, STLXR (B, H, W, X), LD<op> and SWP
(every size and operation), CAS (every size), CLREX, DMB and DSB (SMP:
`c.barrier`'s seq_cst RMW of Hart.fence_word; on one CPU the tier makes
them `nop`), ISB (nothing: the block ends after it and `run` checks the
epoch) and DC ZVA. Pairs (LDXP, STXP, LDAXP, STLXP) and CASP stay calls
(no measured workload runs them). LSE forms are translated only while the
compiling CPU has LSE enabled.

- **Semantics** are excl.zig's and lse.zig's field for field: LDXR opens
  the monitor in `struct kw_cpu` (Hart.resv, Hart.resv_val,
  Priv.excl_size, Priv.excl_val2 = 0), STXR stores only when resv == va
  and excl_size == bytes, closes the monitor either way and writes Ws
  last. The kernel's clears write the same fields, which compiled code
  reads from memory each time.
- **Faults**: a misaligned address and every data-TLB miss hand back
  before anything is written. Everything that may store takes the write
  TLB (so stores into decoded code go to the interpreter): STLR, STXR (also
  when its monitor does not hold), LD<op> into XZR and a failing CAS
  (RMWs need write permission), DC ZVA over its aligned 64 bytes.
- **One CPU**: plain wasm loads and stores (a module never stops inside an
  op).
- **SMP** (`c.smp`: the build decides, not whether the memory is shared):
  every access is a wasm atomic as the interpreter's builtins compile:
  LDAR/LDXR `i64.atomic.load*`, STLR `i64.atomic.store*`, STXR an
  `i64.atomic.rmw*.cmpxchg` against resv_val, CAS one cmpxchg,
  LDADD/LDCLR (AND with ~Rs)/LDEOR/LDSET/SWP one `i64.atomic.rmw*`, min
  and max a CAS loop. All seq_cst, so every acquire/release variant is
  covered and interpreter and compiled code on different CPUs mix safely.
  V8 accepts atomics on a non-shared memory (the wasm64 unit tests of an
  atomics build); host addresses are naturally aligned. DC ZVA stays eight
  plain stores.

## AdvSIMD and crypto (`vec.zig`)

With SIMD (`jc.simd_ok`; js/core/jit.ts sets caps bit 5 after
`WebAssembly.validate` of a small v128 module, which V8 on an x86-64
without SSE4.1 refuses) the V registers are v128 locals and vec.zig
translates the AdvSIMD and crypto ops OpenSSL, musl, V8 and zlib run.
Without SIMD, or while an op's feature is off for the compiling CPU
(`vec.enabled` reads `a64.isa` of `Emit.cpu()`), they are calls. The
decode routes by `decode.route` and then takes each group file's own masks
and legality rules, so it accepts only words the reference runs with
exactly that meaning.

- **Lane ops.** Three same (ADD, SUB, CMEQ, CMTST, CMGT, CMGE, CMHI, CMHS
  (64-bit unsigned compares with the sign bits flipped: wasm has no
  unsigned i64x2 compare), [SU]MAX, [SU]MIN, MUL, MLA, MLS, URHADD,
  USHL/SSHL on S and D lanes (lane by lane in i64; B and H lanes are
  calls); scalar D forms), pairwise ADDP/[SU]MAXP/[SU]MINP, ADDP (scalar),
  across lanes (ADDV, [SU]MAXV, [SU]MINV by halving shuffles), two-reg
  misc (CNT, NOT, ABS, NEG, compares with zero), shift by immediate
  ([SU]SHR, [SU]SRA, SHL, SLI, SRI, [SU]SHLL(2), SHRN(2); a right shift by
  the element size, which wasm takes modulo, is folded at decode), three
  different ([SU]MULL, [SU]MLAL, [SU]MLSL by extmul; [SU]ADDL/SUBL/ADDW/
  SUBW), by element (MUL, MLA, MLS, [SU]MULL, [SU]MLAL, [SU]MLSL, FMUL,
  FMLA, FMLS), BSL/BIT/BIF, ORR/BIC (immediate), SMOV, INS (general) of
  B/H/S lanes.
- **AdvSIMD FP fast paths.** Vector FADD, FSUB, FMUL, FDIV, FMAX, FMIN,
  FMAXNM, FMINNM, FSQRT, FCMEQ, FCMGE, FCMGT, compares with zero (2S, 4S,
  2D), FABS, FNEG (all bits, no fast-path guard), single-precision FMLA/FMLS
  (vector and by-element, computed in f64x2 without double rounding), and
  2S/4S SCVTF/UCVTF (FPCR and FPSR.IXC checks only) and FCVTZS/FCVTZU
  (per-lane range checks); the arithmetic checks each lane for a normal
  exponent or an exact zero, and Q = 0 neutralizes inactive high lanes. Off the fast path, `vecFpSlow`
  stores only the V registers read and calls `opexec.jitGeneric` without
  spilling unrelated locals.
- **Permutations.** UZP, ZIP, TRN, EXT, REV16/32/64, DUP (element), INS
  (element), XTN(2) and SHRN(2)'s packing are one `i8x16.shuffle` each;
  UZP/ZIP/TRN and EXT indices come from `simd_move.permuteOp`/`ext` run at
  decode on numbered bytes, so they are the reference's by construction.
  TBL/TBX: `i8x16.swizzle` per table register on the index minus 16 j
  (swizzle gives 0 from 16 on, TBL's rule), TBX a bitselect.
- **Loads and stores.** LD1/ST1 (1-4 registers, one lane), LD1R,
  LDP/STP/LDNP/STNP of S, D and Q: one TLB check over the whole access
  (else handed back), then v128 loads and stores, writeback last. LD2-4/
  ST2-4 are calls.
- **AES.** Table lookups in `jc.AesTab` (cpu-relative, filled from core.aes
  by `jitcore.init`). AESE/AESD do AddRoundKey and leave the state pending
  in a v128 scratch (`vec.Ext`); if the next op is AESMC (AESIMC) of that
  register into itself, the pair is one T-table round (per column four
  i32 loads of `te`/`td` rotated by row and XORed); any other op first
  writes the pending register (S-box bytes), and so does every path out
  (`vec.sync`, first in emit.zig's `sync` and `syncHead`). Nothing can
  leave between the two ops of a pair, so the fusion is exact. Not
  constant-time on the host (a64/isa.md, Crypto).
- **PMULL/PMULL2 .1Q**: Karatsuba over the 32-bit halves, each 32x32
  carry-less product core.bitmanip's multiply-and-mask sequence.
- **SHA**: SHA1C/P/M and SHA256H/H2 run their four rounds on i32 locals
  (`l_w`), renaming state words instead of moving them; SHA512H/H2 on i64
  locals; SHA1H and the schedule updates are lane operations. RAX1, EOR3,
  BCAX, XAR stay calls (OpenSSL does not use them here).

## Tests

- `jit/tests.zig` (native and wasm64): a hand-back resumes at op k with
  XZR = 0; a compiled self-loop retires exactly its budget and polls
  `*irq_pending` (raised from the compile hook) and the code epoch at its
  back edge; two blocks on two pages that branch to each other grow into
  one region; a flush asked for during a compile. The compile hook
  (`Hook`, called by the JIT host after every compile, inside the compile
  window) stands in for another CPU; `Hook.call` runs a family test's own
  action there.
- **Kind tests** (kindtest.zig, one line per kind in each family file).
  ALU generators (`Rel`) put a `relate` word before a third of the ops (the
  second source a relative of the first); flag producers mix in `soup`
  (CSEL, CCMP, ADC, MRS/MSR NZCV, an FP op, B.cond, a faulting load); lazy
  flags have their own tests (consumed after an unrelated op, overwritten
  unread, live across a hand-back, exit or side exit, a fault between
  producer and consumer, self-loops with `head_dead`). Memory kinds run
  from cold and warm TLBs (`warm_tlb`) and with writable code pages
  (`rwx_code`), with addresses at page ends, straddling, unmapped,
  read-only, wrapping past 2^64, and one group in eight closing a loop so
  later passes hit the TLB. vec.zig draws every class with random fields
  (so reserved and untranslated neighbours too) between MOVI edge vectors,
  AES pairs as OpenSSL writes them plus every broken pairing, and the
  translated AdvSIMD FP forms (and FADDP) over FP edge lanes with random
  Q and sz.
  generic.zig's forms and the FP fast paths run over FP edge values with
  random FPCR, reading FPSR after half the ops (the end state alone hid a
  missing IXC). atomic.zig mixes monitor sequences, CAS made to pass, and
  aliased registers.
- branch.zig's own tests: every B.cond against every NZCV value, CBZ/CBNZ
  and TBZ/TBNZ on edge values and every bit, each as side exit, self-loop
  terminator, terminator as a block's last instruction (`max_block`) and
  inside a region; every budget from 1 to 99 over a self-loop, a two-page
  call loop, a BR loop and a self-loop run as a region's guest (also with
  every poll stopping on a changed code epoch); an interrupt during a compile stops loops at
  the exact count; a second `kw_cpu` storing SVC over a compiled loop and
  executing IC IVAU makes the loop leave at its next back edge or checked
  entry.
- **The JIT fuzz**: the block fuzz (a64/tier.md) in the wasm64 unit tests,
  with a random threshold (eager or 1..200), region size and growth
  threshold, and a flush during every other compile. 2026-11-18: 3000
  programs with every branch and memory kind compiled and 10000 with
  every ALU kind (before those streams merged), no difference.
- The guests in JIT mode (a64.md, Tests), with and without SIMD and
  lowered to memory32.
- A second `kw_cpu` of the test machine (`newCpu`) never compiles: the
  slot pool belongs to the wasm instance, which both share here.

Mutation checks (2026-11-17/18): dozens of single-point mutations in every
family (wrong condition codes, flags, shift counts, register order,
widths, writeback order, missing guards, monitor fields, sync and poll
points, SIMD lane and shuffle indices) are each caught by `zig build
test`. Known gaps: a back-edge or checked-entry budget test off by one
(`jl < ninsn - 1`) lets jl wrap, so it shows as a hanging test, not a
failed assertion; CAS or STXR made plain on SMP is caught only by kwtest
on 4 CPUs; equivalent mutants: a narrow cmpxchg's expected value left
unmasked (wasm wraps it), a dropped sign extension of an XZR load value.
