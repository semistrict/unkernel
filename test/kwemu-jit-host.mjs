// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
// Host side of kwemu's optional JIT imports (src/emu/jitcore.zig), for the
// tests and benchmarks: compiles each module synchronously against the
// instance's memory and installs its "f" into the requested slot of the
// exported __indirect_function_table (table64: BigInt indices), saving the
// stub that was there; kw_host_jit_free puts the stub back.
//
//   const jit = createJitHost({ eager, mem32, strict });
//   instantiate(..., { unkernel: jit.imports, ... }); jit.attach(instance.exports);
//
// available: caps bit 0 (false: kwemu never compiles, the interpreter alone
// runs); eager: caps bit 2 (compile on first entry); mem32: caps bit 1 (the module
// was lowered with wasm-opt --memory64-lowering); table: caps bit 4 (each
// module gets the instance's table as env.table; default true); simd: caps bit 5
// (kwemu may emit v128 code; default true, `KWEMU_JIT_SIMD=0` in the
// environment turns it off); strict: a compile into a
// slot that is already installed is an error (it is legal after a second
// kw_cpu_init, which forgets its slots). Errors are collected in stats.errors.
// Every compile also calls the new f with cpu = 0, which must return 0 at
// once (what js/core/jit.ts does to have V8 compile it inside the call), and
// then the export kwemu_test_compile_hook, if the instance has one (the unit
// tests' stand-in for another CPU acting during a compile).
// Probe module for Wasm Relaxed-SIMD FMA: exports d (f64x2.relaxed_madd, 0x107),
// dn (f64x2.relaxed_nmadd, 0x108), s (f32x4.relaxed_madd, 0x105) and sn
// (f32x4.relaxed_nmadd, 0x106). Only sets cap_relaxed_fma (bit 6) if the module
// validates AND all four perform true single-rounded FMA (not double-rounded mul+add).
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
let relaxedFmaCached;
export function jitRelaxedFma() {
  if (relaxedFmaCached !== undefined) return relaxedFmaCached;
  try {
    if (!WebAssembly.validate(RELAXED_FMA_PROBE)) return (relaxedFmaCached = false);
    const ex = new WebAssembly.Instance(new WebAssembly.Module(RELAXED_FMA_PROBE)).exports;
    const a64 = 1 + 2 ** -52, c64 = -1 - 2 ** -51, e64 = 2 ** -104;
    const a32 = Math.fround(1 + 2 ** -23), c32 = Math.fround(-1 - 2 ** -22), e32 = Math.fround(2 ** -46);
    return (relaxedFmaCached =
      ex.d(a64, a64, c64) === e64 &&
      ex.dn(a64, a64, -c64) === -e64 &&
      ex.s(a32, a32, c32) === e32 &&
      ex.sn(a32, a32, -c32) === -e32);
  } catch {
    return (relaxedFmaCached = false);
  }
}

export function createJitHost({ available = true, eager = false, mem32 = false, strict = false, table: table_ = true, simd = process.env.KWEMU_JIT_SIMD !== "0", relaxedFma = process.env.KWEMU_JIT_RELAXED_FMA !== "0" } = {}) {
  let mem, table, hook, big = true;
  const saved = new Map();
  const stats = { compiles: 0, frees: 0, failures: 0, bytes: 0, compileNs: 0n, errors: [] };
  const idx = (slot) => (big ? BigInt(slot) : slot);
  const rfma = simd && relaxedFma && jitRelaxedFma();
  const imports = {
    kw_host_jit_caps: () => (available ? 1 : 0) | (mem32 ? 2 : 0) | (eager ? 4 : 0) | (table_ ? 16 : 0) | (simd ? 32 : 0) | (rfma ? 64 : 0),
    kw_host_jit_compile(ptr, len, slot, flags) {
      const t0 = process.hrtime.bigint();
      try {
        if (!table) throw new Error("the module does not export __indirect_function_table");
        const bytes = new Uint8Array(mem.buffer, Number(ptr), Number(len)).slice();
        const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory: mem, table } });
        if (saved.has(slot)) {
          if (strict) stats.errors.push(`compile into installed slot ${slot}`);
        } else saved.set(slot, table.get(idx(slot)));
        table.set(idx(slot), inst.exports.f);
        // flags bit 0: f(cpu = 0) returns 0 at once (kw_emu.h), which a
        // kernel host calls to make a lazy engine compile f here. Every
        // module must have it.
        if (!(flags & 1)) stats.errors.push(`compile without flags bit 0 (slot ${slot})`);
        else if (inst.exports.f(0n, 0n, 0n, 0n) !== 0n) stats.errors.push(`f(0, 0, 0, 0) did not return 0 (slot ${slot})`);
        stats.compiles++;
        stats.bytes += Number(len);
        // The unit tests' stand-in for another CPU acting during the compile.
        hook?.();
        return 0;
      } catch (e) {
        stats.failures++;
        if (stats.errors.length < 20) stats.errors.push(`compile failed (slot ${slot}, ${Number(len)} bytes): ${e.message}`);
        return -1;
      } finally {
        stats.compileNs += process.hrtime.bigint() - t0;
      }
    },
    kw_host_jit_free(slot) {
      if (!saved.has(slot)) {
        stats.errors.push(`free of slot ${slot} that is not installed`);
        return;
      }
      table.set(idx(slot), saved.get(slot));
      saved.delete(slot);
      stats.frees++;
    },
  };
  return {
    imports,
    stats,
    installed: () => saved.size,
    attach(exports) {
      mem = exports.memory;
      table = exports.__indirect_function_table; // needed once kwemu compiles
      big = typeof table?.length === "bigint";
      hook = exports.kwemu_test_compile_hook;
    },
  };
}
