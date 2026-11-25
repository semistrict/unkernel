// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
// Host side of kwemu's JIT tier (src/emu/jitcore.zig; contract in
// include/kw_emu.h and .agents/kwemu.md).
//
// A kernel whose kwemu is built with the tier (kernel/build.sh does so by
// default) imports three functions from module "unkernel":
//
//   kw_host_jit_caps() -> u32        bit 0 available, bit 1 32-bit memory
//                                    (the lowered vmlinux32.wasm), bit 2 eager
//                                    (tests only; never set here), bit 3
//                                    shared memory, bit 4 the kernel's table
//                                    is offered as env.table
//                                    (always, here), bit 5 the engine
//                                    compiles fixed-width SIMD (v128), bit 6
//                                    the engine compiles Relaxed-SIMD FMA and
//                                    executes true single-rounded FMA
//   kw_host_jit_compile(ptr, len, slot, flags) -> i32
//                                    compile the module at ptr (it imports
//                                    env.memory and maybe env.table, exports
//                                    "f") and install "f"
//                                    in table slot `slot`; 0 ok, -1 failed;
//                                    flags bit 0: f(cpu = 0) returns at once
//   kw_host_jit_free(slot)           put the slot's original function back
//
// kwemu owns the slots: a pool of stub functions that are already in the
// kernel's exported __indirect_function_table, so the table never grows.
// Compiling is synchronous (new WebAssembly.Module / Instance); the modules
// are a few KiB. V8 compiles a wasm function only at its first call, which
// would otherwise come from kwemu after the import returned; with flags bit
// 0 the host calls f(0, 0, 0, 0) once (it returns at once), so that this
// compile happens inside the import as well, where other CPUs need not wait
// for this one's TLB flushes (kw_emu.h, "Flushes during a JIT compile").
// A failure returns -1 (kwemu then interprets that block for good) and
// never throws into the kernel. The only reference to a compiled
// module is its table slot, so a free (or the kernel being dropped)
// releases it. A kernel built without the tier never calls these imports.
//
// Released is not collected, though. A compiled module costs the engine
// more memory than the garbage collector sees (JavaScriptCore: about 24 KiB
// per real kwemu module, V8: about 12 KiB), and JavaScriptCore schedules its
// next collection from a heap size that includes the kernel's wasm memory:
// with 512 MiB of RAM it waited for about 3 GB of further allocation, which
// compiling never reaches, so freed modules were never collected. The host
// therefore counts what the engine has collected (a FinalizationRegistry on
// every module) and, while RETAINED_NUDGE or more freed modules are still
// uncollected, allocates a 64 MiB ArrayBuffer on every compile (not on a
// free, which is outside that window) and detaches it at once (transfer(0);
// it is never touched, so it costs about 10-30 us). JavaScriptCore counts
// the allocation towards its next collection, which then reclaims the
// modules. V8 (Chromium) starts a collection only for an allocation about
// as large as the external memory it already has, which includes the
// kernel's wasm memory: with 512 MiB of
// RAM a quiet page reached the refusal limit with 64 and 256 MiB nudges and
// collected at once with 512 MiB. So while nudges see no collection for
// NUDGE_PATIENCE_MS, each one is twice as large (up to NUDGE_MAX; an
// allocation that fails halves it). If an engine still does not collect,
// compiles are refused (-1, kwemu interprets those blocks) while
// RETAINED_LIMIT freed modules are uncollected.

/** JIT counters of one kernel (Kernel.jit.stats). */
export interface JitStats {
	/** The host offers the JIT (KernelConfig.jit, and the engine can compile synchronously). */
	enabled: boolean;
	/** The kernel's kwemu asked for it (kw_host_jit_caps was called): false for a kernel without the tier. */
	requested: boolean;
	/** Blocks compiled and installed. */
	compiled: number;
	/** Compiled blocks currently installed (compiled - freed). */
	live: number;
	/** Most blocks installed at once (kwemu's pool has 1024 slots). */
	maxLive: number;
	/** Slots kwemu released (block overwritten, evicted or cache reset). */
	freed: number;
	/** Compiles that failed (kwemu interprets those blocks), plus rejected calls. */
	failed: number;
	/** Total time spent compiling and instantiating (with V8, also the first call's compile), in ms. */
	compileMs: number;
	/** Longest single compile, instantiation and first call, in ms (the longest the CPU could not answer an interrupt for one). */
	maxCompileMs: number;
	/** Total size of the compiled modules, in bytes. */
	bytes: number;
	/** Compiled modules the engine has garbage-collected (-1: no FinalizationRegistry, unknown). */
	collected: number;
	/** Freed modules not collected yet (freed - collected; -1 when unknown). */
	retained: number;
	/** Most freed modules uncollected at once (-1 when unknown). */
	maxRetained: number;
	/** ArrayBuffers allocated to prompt a collection (retained above RETAINED_NUDGE). */
	gcNudges: number;
	/** Compiles refused because retained reached RETAINED_LIMIT (kwemu interprets those blocks). */
	refused: number;
}

/**
 * kwemu's own counters, from a kernel built with -Dkwemu-stats
 * (`kernel/build.sh stats`) that has compiled at least one block.
 */
export interface KwemuStats {
	/** Guest instructions retired (struct kw_cpu icount). */
	icount: number;
	/**
	 * kwemu.Stats as 64-bit words, in the layout of src/emu/kwemu.zig: 11
	 * counters, op_hist[192], JIT insns, hand-backs, hb[192], hb_gen[40],
	 * jx[64] (indices in src/emu/jitcore.zig Jx), gen_words[3 * 256] (A64:
	 * `generic` ops by word, kwemu.zig countGenWord).
	 */
	words: number[];
}

/** JitStats fields in the order of a counter mirror (JitHost.mirror, a secondary CPU's). */
const MIRROR_KEYS = ["compiled", "live", "maxLive", "freed", "failed", "compileMs", "bytes", "collected", "retained",
	"maxRetained", "gcNudges", "refused", "maxCompileMs"] as const satisfies readonly (keyof JitStats)[];
/** Float64 slots of one mirror: enabled, requested, the counters, the emit buffer (0: none yet). */
export const JIT_MIRROR_WORDS = 2 + MIRROR_KEYS.length + 1;

/** The counters a secondary CPU's host mirrored (JitHostOptions.mirror). */
export function jitStatsFromMirror(m: Float64Array): JitStats {
	const st = { enabled: m[0] === 1, requested: m[1] === 1 } as JitStats;
	MIRROR_KEYS.forEach((k, i) => (st[k] = m[2 + i] ?? 0));
	return st;
}

/** The emit buffer address a mirror recorded (undefined before the first compile). */
export function jitBufFromMirror(m: Float64Array): number | undefined {
	return m[JIT_MIRROR_WORDS - 1] || undefined;
}

/** Sum of several CPUs' counters (maxima summed too: they are per CPU). */
export function sumJitStats(first: JitStats, rest: JitStats[]): JitStats {
	const st = { ...first };
	for (const o of rest) {
		st.enabled ||= o.enabled;
		st.requested ||= o.requested;
		for (const k of MIRROR_KEYS) st[k] = k === "maxCompileMs" ? Math.max(st[k], o[k]) : (st[k] < 0 || o[k] < 0 ? -1 : st[k] + o[k]);
	}
	return st;
}

export interface JitHostOptions {
	/** Offer the JIT (default true); false: kw_host_jit_caps() returns 0. */
	enabled?: boolean;
	/** The kernel's memory, read at call time (boot may switch memory types). */
	memory: () => WebAssembly.Memory;
	/** Whether that memory is a memory64 one, read at call time. */
	memory64: () => boolean;
	log?: (msg: string) => void;
	/** Uncollected freed modules from which the host prompts a collection (default RETAINED_NUDGE; Infinity: never). */
	retainedNudge?: number;
	/** Uncollected freed modules at which compiles are refused (default RETAINED_LIMIT; Infinity: never). */
	retainedLimit?: number;
	/**
	 * A secondary CPU's host (cpu.ts): copy the counters here after every
	 * compile and free (JIT_MIRROR_WORDS, on a SharedArrayBuffer), where
	 * CPU 0's host reads them (jitStatsFromMirror).
	 */
	mirror?: Float64Array;
}

const CAP_AVAILABLE = 1;
const CAP_MEM32 = 2;
const CAP_SHARED = 8;
/** The kernel's function table is offered to modules as env.table. */
const CAP_TABLE = 16;
/** The engine compiles fixed-width SIMD: kwemu may emit v128 code. */
const CAP_SIMD = 32;
/** The engine compiles Relaxed-SIMD FMA with true single rounding. */
const CAP_RELAXED_FMA = 64;
/**
 * A module whose function returns i8x16.popcnt(i8x16.splat(0)): engines
 * without SIMD (V8 on an x86-64 without SSE4.1) do not validate it.
 */
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
/**
 * A module exporting f64x2/f32x4 relaxed_madd and relaxed_nmadd: engines
 * without Relaxed-SIMD do not validate it, and engines without hardware FMA
 * fail the exact single-rounded FMA check.
 */
const RELAXED_FMA_PROBE = new Uint8Array([
	0, 97, 115, 109, 1, 0, 0, 0,
	1, 15, 2, 96, 3, 124, 124, 124, 1, 124, 96, 3, 125, 125, 125, 1, 125,
	3, 5, 4, 0, 0, 1, 1,
	7, 19, 4, 1, 100, 0, 0, 2, 100, 110, 0, 1, 1, 115, 0, 2, 2, 115, 110, 0, 3,
	10, 85, 4,
	20, 0, 32, 0, 253, 20, 32, 1, 253, 20, 32, 2, 253, 20, 253, 135, 2, 253, 33, 0, 11,
	20, 0, 32, 0, 253, 20, 32, 1, 253, 20, 32, 2, 253, 20, 253, 136, 2, 253, 33, 0, 11,
	20, 0, 32, 0, 253, 19, 32, 1, 253, 19, 32, 2, 253, 19, 253, 133, 2, 253, 31, 0, 11,
	20, 0, 32, 0, 253, 19, 32, 1, 253, 19, 32, 2, 253, 19, 253, 134, 2, 253, 31, 0, 11,
]);
/** Whether this engine compiles fixed-width SIMD (cap bit 5). */
export function jitSimd(): boolean {
	try {
		return WebAssembly.validate(SIMD_PROBE);
	} catch {
		return false;
	}
}
/** Whether this engine compiles Relaxed-SIMD FMA and executes true single-rounded FMA (cap bit 6). */
export function jitRelaxedFma(): boolean {
	try {
		if (!WebAssembly.validate(RELAXED_FMA_PROBE)) return false;
		const ex = new WebAssembly.Instance(new WebAssembly.Module(RELAXED_FMA_PROBE)).exports as {
			d: (a: number, b: number, c: number) => number;
			dn: (a: number, b: number, c: number) => number;
			s: (a: number, b: number, c: number) => number;
			sn: (a: number, b: number, c: number) => number;
		};
		const a64 = 1 + 2 ** -52, c64 = -1 - 2 ** -51, e64 = 2 ** -104;
		const a32 = Math.fround(1 + 2 ** -23), c32 = Math.fround(-1 - 2 ** -22), e32 = Math.fround(2 ** -46);
		return ex.d(a64, a64, c64) === e64 &&
			ex.dn(a64, a64, -c64) === -e64 &&
			ex.s(a32, a32, c32) === e32 &&
			ex.sn(a32, a32, -c32) === -e32;
	} catch {
		return false;
	}
}
/** kw_host_jit_compile flags bit 0: f(x, 0, irqp, jl) returns 0 at once. */
const WARM = 1;
/** kwemu's emit buffer is 32 KiB; anything much larger is not from kwemu. */
const MAX_MODULE = 1 << 20;
/** Failure messages logged before going quiet. */
const MAX_LOGGED = 5;
/** Words in kwemu.Stats with -Dkwemu-stats (src/emu/kwemu.zig), and the offset of icount in struct kw_cpu. */
const KWEMU_STATS_WORDS = 11 + 192 + 2 + 192 + 40 + 64 + 3 * 256;
const KW_CPU_ICOUNT = 32;
/** Uncollected freed modules from which every compile and free prompts a collection. */
const RETAINED_NUDGE = 256;
/** Uncollected freed modules at which compiles are refused until the engine catches up. */
const RETAINED_LIMIT = 2048;
/** Size of one prompt. Untouched and detached at once, so it costs address space only briefly. */
const NUDGE_BYTES = 64 << 20;
/** Largest prompt (the size doubles while prompts do not lead to a collection). */
const NUDGE_MAX = 2 ** 32;
/** Prompts without a collection for this long double the prompt size. */
const NUDGE_PATIENCE_MS = 250;

/** Table index API shared by table64 (BigInt indices) and 32-bit tables. */
interface AnyTable {
	readonly length: number | bigint;
	get(index: number | bigint): unknown;
	set(index: number | bigint, value: unknown): void;
}

type Detachable = ArrayBuffer & { transfer?: (len: number) => ArrayBuffer };

export class JitHost {
	readonly stats: JitStats;
	#opts: JitHostOptions;
	#table: AnyTable | undefined;
	#big = false;
	/** slot -> the function it held before its first compile (kwemu's stub). */
	#saved = new Map<number, unknown>();
	#logged = 0;
	/** Collected-module counter, or undefined when the engine has no FinalizationRegistry. */
	#fin: FinalizationRegistry<number> | undefined;
	#nudgeAt: number;
	#limit: number;
	#nudgeBytes = NUDGE_BYTES;
	/** When, and at which `collected` count, the prompt size was last reconsidered. */
	#nudgeCheckAt = 0;
	#nudgeCheckCollected = 0;
	/** -Dkwemu-stats kernels: the stats-only exports, and the emit buffer of the first compile. */
	#debugStats: ((cpu: bigint, out: bigint) => void) | undefined;
	#cpuOfBuf: ((buf: bigint) => bigint) | undefined;
	/** The engine compiles SIMD (asked once, at the first kw_host_jit_caps). */
	#simd: boolean | undefined;
	/** The engine executes true single-rounded Relaxed-SIMD FMA (asked once). */
	#relaxedFma: boolean | undefined;
	#buf: number | undefined;

	constructor(opts: JitHostOptions) {
		this.#opts = opts;
		const enabled = opts.enabled !== false && typeof WebAssembly === "object" &&
			typeof WebAssembly.Module === "function" && typeof WebAssembly.Instance === "function";
		const tracked = typeof FinalizationRegistry === "function";
		this.stats = {
			enabled, requested: false, compiled: 0, live: 0, maxLive: 0, freed: 0, failed: 0, compileMs: 0, maxCompileMs: 0, bytes: 0,
			collected: tracked ? 0 : -1, retained: tracked ? 0 : -1, maxRetained: tracked ? 0 : -1, gcNudges: 0, refused: 0,
		};
		if (tracked) this.#fin = new FinalizationRegistry(() => this.#collected());
		this.#nudgeAt = opts.retainedNudge ?? RETAINED_NUDGE;
		this.#limit = opts.retainedLimit ?? RETAINED_LIMIT;
		this.#publish();
	}

	/** Copy the counters to the mirror, if any. */
	#publish(): void {
		const m = this.#opts.mirror;
		if (!m) return;
		const st = this.stats;
		m[0] = st.enabled ? 1 : 0;
		m[1] = st.requested ? 1 : 0;
		MIRROR_KEYS.forEach((k, i) => (m[2 + i] = st[k]));
		m[JIT_MIRROR_WORDS - 1] = this.#buf ?? 0;
	}

	/** Bind to the kernel instance's function table; call after instantiation, before _start. */
	attach(exports: WebAssembly.Exports): void {
		const t = exports.__indirect_function_table;
		this.#table = t instanceof WebAssembly.Table ? (t as unknown as AnyTable) : undefined;
		this.#big = typeof this.#table?.length === "bigint";
		const ds = exports.kw_debug_stats, cb = exports.kw_debug_cpu_of_jit_buf;
		this.#debugStats = typeof ds === "function" ? (ds as (cpu: bigint, out: bigint) => void) : undefined;
		this.#cpuOfBuf = typeof cb === "function" ? (cb as (buf: bigint) => bigint) : undefined;
		this.#buf = undefined;
	}

	/**
	 * kwemu's counters, if the kernel exports them (-Dkwemu-stats) and has
	 * compiled a block (that is how the cpu is found). It writes the emit
	 * buffer, which is free outside kw_host_jit_compile, so any call from
	 * JavaScript is safe.
	 */
	kwemuStats(of: number | undefined = this.#buf): KwemuStats | undefined {
		if (!this.#debugStats || !this.#cpuOfBuf || this.#buf === undefined || of === undefined) return undefined;
		// `of`: the emit buffer of another CPU (SMP, jitBufFromMirror): its
		// counters are copied through this CPU's buffer, which is free
		// while this host's JavaScript runs.
		const cpu = this.#cpuOfBuf(BigInt(of));
		// The emit buffer (32 KiB) is free between compiles: copy the
		// counters (10152 bytes) into it, then read them.
		this.#debugStats(cpu, BigInt(this.#buf));
		const v = new DataView(this.#opts.memory().buffer);
		const words: number[] = [];
		for (let i = 0; i < KWEMU_STATS_WORDS; i++) words.push(Number(v.getBigUint64(this.#buf + 8 * i, true)));
		return { icount: Number(v.getBigUint64(Number(cpu) + KW_CPU_ICOUNT, true)), words };
	}

	/** The three `unkernel` imports. */
	imports(): Record<string, unknown> {
		return {
			kw_host_jit_caps: (): number => this.caps(),
			kw_host_jit_compile: (ptr: bigint | number, len: bigint | number, slot: number, flags?: number): number =>
				this.compile(ptr, len, slot, flags ?? 0),
			kw_host_jit_free: (slot: number): void => this.free(slot),
		};
	}

	caps(): number {
		this.stats.requested = true;
		this.#publish();
		if (!this.stats.enabled) return 0;
		if (!this.#table) {
			this.#warn("the kernel exports no __indirect_function_table; JIT off");
			return 0;
		}
		const shared = typeof SharedArrayBuffer !== "undefined" && this.#opts.memory().buffer instanceof SharedArrayBuffer;
		this.#simd ??= jitSimd();
		this.#relaxedFma ??= this.#simd && jitRelaxedFma();
		return CAP_AVAILABLE | (this.#opts.memory64() ? 0 : CAP_MEM32) | (shared ? CAP_SHARED : 0) | CAP_TABLE | (this.#simd ? CAP_SIMD : 0) | (this.#relaxedFma ? CAP_RELAXED_FMA : 0);
	}

	compile(ptr: bigint | number, len: bigint | number, slot: number, flags = 0): number {
		const st = this.stats;
		if (st.retained >= this.#nudgeAt) this.#nudge();
		if (st.retained >= this.#limit) {
			// Not a failure: the engine has not caught up with collecting yet.
			if (st.refused++ === 0) this.#opts.log?.(`[jit] ${st.retained} freed modules not collected by the engine yet; refusing compiles until it catches up`);
			this.#publish();
			return -1;
		}
		const t0 = performance.now();
		try {
			const table = this.#table;
			if (!st.enabled || !table) throw new Error("JIT not enabled");
			if (!this.#validSlot(slot)) throw new Error(`bad slot ${slot}`);
			const p = Number(ptr), n = Number(len);
			this.#buf ??= p;
			const mem = this.#opts.memory();
			if (!(n > 0 && n <= MAX_MODULE && p >= 0 && p + n <= mem.buffer.byteLength)) throw new Error(`bad module range ${p}+${n}`);
			// The Module constructor copies the bytes, so a view into the
			// kernel's memory is enough, unless the memory is shared (the JS
			// API takes no shared views; V8 accepts them, others need not).
			const view = new Uint8Array(mem.buffer, p, n);
			const module = new WebAssembly.Module(mem.buffer instanceof ArrayBuffer ? view : view.slice());
			const f = new WebAssembly.Instance(module, { env: { memory: mem, table: table as unknown as WebAssembly.Table } }).exports.f;
			if (typeof f !== "function") throw new Error("the module exports no function f");
			// cpu = 0: f returns at once, after V8 has compiled it (flags bit 0).
			if (flags & WARM) (f as (x: bigint, cpu: bigint, irqp: bigint, jl: bigint) => bigint)(0n, 0n, 0n, 0n);
			const idx = this.#big ? BigInt(slot) : slot;
			const prev = table.get(idx);
			table.set(idx, f);
			this.#fin?.register(module, 0);
			// A compile into an installed slot (after a second kw_cpu_init,
			// which forgets slots) replaces it and keeps the original stub.
			if (this.#saved.has(slot)) this.#freed();
			else this.#saved.set(slot, prev);
			st.compiled++;
			st.bytes += n;
			return 0;
		} catch (e) {
			st.failed++;
			this.#warn(`compile into slot ${slot} failed: ${(e as Error)?.message ?? e}`);
			return -1;
		} finally {
			const ms = performance.now() - t0;
			st.compileMs += ms;
			if (ms > st.maxCompileMs) st.maxCompileMs = ms;
			st.live = this.#saved.size;
			if (st.live > st.maxLive) st.maxLive = st.live;
			this.#publish();
		}
	}

	free(slot: number): void {
		try {
			const orig = this.#saved.get(slot);
			if (!this.#saved.has(slot) || !this.#table) throw new Error(`free of slot ${slot}, which holds no compiled block`);
			this.#table.set(this.#big ? BigInt(slot) : slot, orig);
			this.#saved.delete(slot);
			this.#freed();
		} catch (e) {
			this.stats.failed++;
			this.#warn((e as Error)?.message ?? String(e));
		} finally {
			this.stats.live = this.#saved.size;
			this.#publish();
		}
	}

	#freed(): void {
		const st = this.stats;
		st.freed++;
		if (!this.#fin) return;
		st.retained = st.freed - st.collected;
		if (st.retained > st.maxRetained) st.maxRetained = st.retained;
	}

	#collected(): void {
		const st = this.stats;
		st.collected++;
		// A module replaced without a free is counted as freed, so this
		// never goes below 0; max() guards against the order of callbacks.
		st.retained = Math.max(0, st.freed - st.collected);
		this.#publish();
	}

	/** Make the engine count a large allocation towards its next garbage collection. */
	#nudge(): void {
		// Collections are reported asynchronously (finalizers), so judge a
		// size only after a while, never within one burst of compiles.
		const now = performance.now();
		if (this.stats.collected !== this.#nudgeCheckCollected) {
			this.#nudgeCheckCollected = this.stats.collected;
			this.#nudgeCheckAt = now;
		} else if (now - this.#nudgeCheckAt >= NUDGE_PATIENCE_MS) {
			if (this.#nudgeCheckAt > 0 && this.#nudgeBytes < NUDGE_MAX) this.#nudgeBytes *= 2;
			this.#nudgeCheckAt = now;
		}
		try {
			const ab = new ArrayBuffer(this.#nudgeBytes) as Detachable;
			// Detaching frees the backing store now (no mapping waits for
			// the collection); the engine has already counted it.
			ab.transfer?.(0);
			this.stats.gcNudges++;
		} catch {
			// Out of address space (phones): try smaller next time.
			if (this.#nudgeBytes > 1 << 20) this.#nudgeBytes >>= 1;
		}
	}

	/** An existing, non-null table entry that is not slot 0 (the null function). */
	#validSlot(slot: number): boolean {
		const table = this.#table!;
		if (!Number.isInteger(slot) || slot <= 0 || slot >= Number(table.length)) return false;
		return this.#saved.has(slot) || typeof table.get(this.#big ? BigInt(slot) : slot) === "function";
	}

	#warn(msg: string): void {
		if (this.#logged >= MAX_LOGGED) return;
		this.#logged++;
		this.#opts.log?.(`[jit] ${msg}${this.#logged === MAX_LOGGED ? " (further JIT messages suppressed)" : ""}`);
	}
}
