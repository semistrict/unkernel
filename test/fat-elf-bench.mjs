// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//
// End-to-end Fat ELF `.kw.wasm` demo, correctness verifier & benchmark:
//   - Compiles `test/guest/fat_bench.c` to a static `aarch64-linux-musl` ELF
//     (and to a native binary on ARM64 hosts).
//   - Compiles the 4 Wasm64 SIMD kernels (`sha256`, `fp_daxpy_2d`, `fp_matmul`,
//     `complex_ld2`) and packs them into `.kw.wasm` inside a SINGLE Fat ELF.
//   - Runs that exact single Fat ELF on:
//       * Native ARM64 (reference, if host is arm64)
//       * Direct Wasm64 (the exact same `.wasm` kernels run directly in Node)
//       * kwemu-harness-jit.wasm with `--fallback` (standard AArch64 JIT)
//       * kwemu-harness-jit.wasm (Fat-ELF `.kw.wasm` fast path)
//       * kwemu-harness.wasm (no-JIT: automatic AArch64 fallback via -1 sentinel)
//       * kwemu-harness native (automatic AArch64 fallback via -1 sentinel)
//   - Verifies 100% identical checksums across every single mode & kernel.
//
// Usage: node test/fat-elf-bench.mjs [zig-out prefix] [--reps N]

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createJitHost } from "./kwemu-jit-host.mjs";
import { compileKernels, buildKwWasmSection, appendElfSection, runZig } from "./fat-elf-pack.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const argv = process.argv.slice(2);
const prefix = resolve(argv.find((a, i) => !a.startsWith("--") && argv[i - 1] !== "--reps") ?? join(root, "zig-out"));
const repsIdx = argv.indexOf("--reps");
const reps = repsIdx >= 0 ? Number(argv[repsIdx + 1]) : 3;
const workDir = join(prefix, "fat-elf");
const dec = new TextDecoder();

await mkdir(workDir, { recursive: true });

// 1. Build the AArch64 static ELF and (on arm64) native binary.
const benchSrc = join(root, "test", "guest", "fat_bench.c");
const plainElf = join(workDir, "fat_bench_plain.elf");
const fatElf = join(workDir, "fat_bench_fat.elf");
const nativeBin = join(workDir, "fat_bench_native");

{
  const r = runZig([
    "cc", "-target", "aarch64-linux-musl", "-mcpu=generic",
    "-O3", "-ffp-contract=off", "-static",
    benchSrc, "-o", plainElf,
  ]);
  if (r.status !== 0) throw new Error(`aarch64 compile failed:\n${r.stderr}`);
}

const isNativeArm64 = process.arch === "arm64";
if (isNativeArm64) {
  const r = runZig(["cc", "-O3", "-ffp-contract=off", benchSrc, "-o", nativeBin]);
  if (r.status !== 0) throw new Error(`native compile failed:\n${r.stderr}`);
}

// 2. Compile the 4 Wasm64 kernels & pack them into `.kw.wasm` inside the Fat ELF.
const kernels = await compileKernels(workDir);
const kwWasmSec = buildKwWasmSection(kernels);
const fatBytes = appendElfSection(await readFile(plainElf), ".kw.wasm", kwWasmSec);
await writeFile(fatElf, fatBytes);

console.log(`Fat ELF built: ${fatElf} (${fatBytes.length} bytes; .kw.wasm section: ${kwWasmSec.length} bytes)`);
for (const k of kernels) {
  console.log(`  slot ${k.fnId}: ${k.name.padEnd(12)} ${String(k.wasmBytes.length).padStart(5)} bytes  flags=0x${k.flags.toString(16)}`);
}
console.log("");

function parseBenchOut(stdout) {
  const map = new Map();
  for (const line of stdout.trim().split("\n")) {
    const m = /^BENCH (\S+) (\S+) ([\d.]+) ([0-9a-f]{16})$/.exec(line.trim());
    if (m) map.set(m[1], { mode: m[2], ms: Number(m[3]), sum: m[4] });
  }
  return map;
}

const wasmNoJitMod = await WebAssembly.compile(await readFile(join(prefix, "bin", "kwemu-harness.wasm")));
const wasmJitMod = await WebAssembly.compile(await readFile(join(prefix, "bin", "kwemu-harness-jit.wasm")));

async function runHarnessWasm(elfPath, args, module, jit) {
  let out = "", err = "", mem;
  const imports = {
    env: {
      host_write: (fd, ptr, len) => {
        const s = dec.decode(new Uint8Array(mem.buffer, Number(ptr), Number(len)));
        if (fd === 2) err += s; else out += s;
      },
      host_now_ns: () => Number(process.hrtime.bigint()),
    },
  };
  if (jit) imports.unkernel = jit.imports;
  const inst = await WebAssembly.instantiate(module, imports);
  const ex = inst.exports;
  mem = ex.memory;
  jit?.attach(ex);

  const image = await readFile(elfPath);
  const argBytes = new TextEncoder().encode([elfPath, ...args].join("\0") + "\0");
  const total = image.length + argBytes.length;
  const ptr = Number(ex.kwh_buf(BigInt(total)));
  if (!ptr) throw new Error("image too large for harness scratch buffer");
  new Uint8Array(mem.buffer, ptr, total).set(image);
  new Uint8Array(mem.buffer, ptr + image.length, argBytes.length).set(argBytes);
  const status = ex.kwh_run(BigInt(image.length), BigInt(argBytes.length), 1);
  if (status !== 0 || (jit && jit.stats.errors.length)) {
    throw new Error(`harness failed (status=${status}):\n${err}\n${jit ? jit.stats.errors.join("\n") : ""}`);
  }
  return { out, err, jitStats: jit?.stats };
}

function runNativeBin(bin, args = []) {
  const r = spawnSync(bin, args, { encoding: "utf8" });
  if (r.status !== 0) throw new Error(`${bin} failed (${r.status}): ${r.stderr}`);
  return r.stdout;
}

// 3. Direct Wasm64 runner (invokes the exact same 4 `.wasm` kernels directly in Node).
function hashDoublesBigInt(f64View) {
  const u64 = new BigUint64Array(f64View.buffer, f64View.byteOffset, f64View.length);
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < u64.length; i++) {
    h = ((h ^ u64[i]) * prime) & mask;
  }
  return h.toString(16).padStart(16, "0");
}

function runDirectWasm() {
  const mem = new WebAssembly.Memory({ initial: 512n, maximum: 1024n, address: "i64" });
  const fns = {};
  for (const k of kernels) {
    const inst = new WebAssembly.Instance(new WebAssembly.Module(k.wasmBytes), { env: { memory: mem } });
    fns[k.name] = inst.exports.f;
    // Warm up V8 Liftoff/TurboFan on f(0,0,0,0) === 0
    if (fns[k.name](0n, 0n, 0n, 0n) !== 0n) throw new Error(`direct wasm warm-up failed for ${k.name}`);
  }
  const res = new Map();

  // sha256
  {
    const len = 1 << 20;
    const hOff = 65536;
    const bufOff = hOff + 4096;
    const buf = new Uint8Array(mem.buffer, bufOff, len);
    for (let i = 0; i < len; i++) buf[i] = (i * 131 + (i >> 9)) & 0xff;
    const h = new Uint32Array(mem.buffer, hOff, 8);
    h.set([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    const t0 = process.hrtime.bigint();
    for (let r = 0; r < 12; r++) {
      fns.sha256(BigInt(hOff), BigInt(bufOff), BigInt(len), 0n);
      buf[r % len] ^= h[0] & 0xff;
    }
    const dt = Number(process.hrtime.bigint() - t0) / 1e6;
    const sum = ((BigInt(h[0]) << 32n) | BigInt(h[7])).toString(16).padStart(16, "0");
    res.set("sha256", { mode: "wasm", ms: dt, sum });
  }

  // fp_daxpy_2d
  {
    const n = 512 * 1024;
    const yOff = 2 << 20;
    const xOff = yOff + n * 8;
    const y = new Float64Array(mem.buffer, yOff, n);
    const x = new Float64Array(mem.buffer, xOff, n);
    for (let i = 0; i < n; i++) {
      x[i] = ((i % 97) - 48) * 0.125;
      y[i] = ((i % 53) - 26) * 0.25;
    }
    const dv = new DataView(new ArrayBuffer(8));
    dv.setFloat64(0, 1.0078125, true);
    const alphaBits = dv.getBigUint64(0, true);
    const t0 = process.hrtime.bigint();
    for (let r = 0; r < 24; r++) {
      fns.fp_daxpy_2d(BigInt(yOff), BigInt(xOff), BigInt(n), alphaBits);
    }
    const dt = Number(process.hrtime.bigint() - t0) / 1e6;
    res.set("fp_daxpy_2d", { mode: "wasm", ms: dt, sum: hashDoublesBigInt(y) });
  }

  // fp_matmul
  {
    const N = 96;
    const aOff = 2 << 20;
    const bOff = aOff + N * N * 8;
    const cOff = bOff + N * N * 8;
    const A = new Float64Array(mem.buffer, aOff, N * N);
    const B = new Float64Array(mem.buffer, bOff, N * N);
    const C = new Float64Array(mem.buffer, cOff, N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        A[i * N + j] = ((i * 7 + j * 3) % 17) * 0.25;
        B[i * N + j] = ((i + 2 * j) % 13) * 0.5 - 1.0;
        C[i * N + j] = 0;
      }
    }
    const t0 = process.hrtime.bigint();
    for (let r = 0; r < 12; r++) {
      fns.fp_matmul(BigInt(cOff), BigInt(aOff), BigInt(bOff), BigInt(N));
      A[r % (N * N)] += 0.0625;
    }
    const dt = Number(process.hrtime.bigint() - t0) / 1e6;
    res.set("fp_matmul", { mode: "wasm", ms: dt, sum: hashDoublesBigInt(C) });
  }

  // complex_ld2
  {
    const nPairs = 256 * 1024;
    const inOff = 2 << 20;
    const outOff = inOff + nPairs * 16;
    const inp = new Float64Array(mem.buffer, inOff, nPairs * 2);
    const out = new Float64Array(mem.buffer, outOff, nPairs * 2);
    for (let i = 0; i < nPairs; i++) {
      inp[2 * i] = ((i % 61) - 30) * 0.125;
      inp[2 * i + 1] = ((i % 47) - 23) * 0.25;
      out[2 * i] = 0;
      out[2 * i + 1] = 0;
    }
    const t0 = process.hrtime.bigint();
    for (let r = 0; r < 24; r++) {
      const src = (r & 1) ? outOff : inOff;
      const dst = (r & 1) ? inOff : outOff;
      fns.complex_ld2(BigInt(dst), BigInt(src), BigInt(nPairs), 0n);
    }
    const dt = Number(process.hrtime.bigint() - t0) / 1e6;
    res.set("complex_ld2", { mode: "wasm", ms: dt, sum: hashDoublesBigInt(inp) });
  }

  return res;
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[s.length >> 1];
}

// 4. Verify automatic fallback on native harness & wasm64 no-jit harness using the EXACT SAME Fat ELF!
{
  const natFall = parseBenchOut(runNativeBin(join(prefix, "bin", "kwemu-harness"), [fatElf]));
  const noJitFall = parseBenchOut((await runHarnessWasm(fatElf, [], wasmNoJitMod, null)).out);
  for (const k of kernels) {
    if (natFall.get(k.name)?.mode !== "a64") throw new Error(`native harness did not take fallback for ${k.name}`);
    if (noJitFall.get(k.name)?.mode !== "a64") throw new Error(`wasm64 no-jit did not take fallback for ${k.name}`);
    if (natFall.get(k.name).sum !== noJitFall.get(k.name).sum) {
      throw new Error(`fallback checksum mismatch for ${k.name}`);
    }
  }
  console.log("Verified automatic AArch64 fallback (sentinel -1) on native harness & wasm64 no-jit with the same Fat ELF.");
}

// 5. Benchmark across all 4 modes (median of `reps` runs).
const names = kernels.map((k) => k.name);
const results = {
  native: new Map(names.map((n) => [n, []])),
  directWasm: new Map(names.map((n) => [n, []])),
  a64Jit: new Map(names.map((n) => [n, []])),
  fatWasm: new Map(names.map((n) => [n, []])),
};
const checksums = new Map();

function checkSums(label, map, expectMode) {
  for (const name of names) {
    const e = map.get(name);
    if (!e) throw new Error(`${label}: missing result for ${name}`);
    if (expectMode && e.mode !== expectMode) {
      throw new Error(`${label} ${name}: expected mode ${expectMode}, got ${e.mode}`);
    }
    const prev = checksums.get(name);
    if (prev && prev !== e.sum) {
      throw new Error(`CHECKSUM MISMATCH on ${name} (${label}): got ${e.sum}, expected ${prev}`);
    }
    checksums.set(name, e.sum);
  }
}

for (let i = 0; i < reps; i++) {
  if (isNativeArm64) {
    const m = parseBenchOut(runNativeBin(nativeBin));
    checkSums("native-arm64", m, "a64");
    for (const n of names) results.native.get(n).push(m.get(n).ms);
  }
  {
    const m = runDirectWasm();
    checkSums("direct-wasm64", m, "wasm");
    for (const n of names) results.directWasm.get(n).push(m.get(n).ms);
  }
  {
    const r = await runHarnessWasm(fatElf, ["--fallback"], wasmJitMod, createJitHost());
    const m = parseBenchOut(r.out);
    checkSums("a64-jit-fallback", m, "a64");
    for (const n of names) results.a64Jit.get(n).push(m.get(n).ms);
  }
  {
    const r = await runHarnessWasm(fatElf, [], wasmJitMod, createJitHost());
    const m = parseBenchOut(r.out);
    checkSums("fat-elf-kw-wasm", m, "fat");
    for (const n of names) results.fatWasm.get(n).push(m.get(n).ms);
  }
}

console.log(`\nBenchmark results (median of ${reps} runs, all checksums bit-exact identical):\n`);
if (isNativeArm64) {
  console.log("| Benchmark | Native ARM64 (ms) | Direct Wasm64 (ms) | AArch64 JIT (ms) | Fat-ELF `.kw.wasm` (ms) | Speedup vs A64 JIT | Checksum |");
  console.log("| --- | ---: | ---: | ---: | ---: | ---: | --- |");
  for (const n of names) {
    const nat = median(results.native.get(n));
    const dw = median(results.directWasm.get(n));
    const aj = median(results.a64Jit.get(n));
    const fw = median(results.fatWasm.get(n));
    console.log(`| ${n} | ${nat.toFixed(2)} | ${dw.toFixed(2)} | ${aj.toFixed(2)} | ${fw.toFixed(2)} | ${(aj / fw).toFixed(2)}x | \`${checksums.get(n)}\` |`);
  }
} else {
  console.log("| Benchmark | Direct Wasm64 (ms) | AArch64 JIT (ms) | Fat-ELF `.kw.wasm` (ms) | Speedup vs A64 JIT | Checksum |");
  console.log("| --- | ---: | ---: | ---: | ---: | --- |");
  for (const n of names) {
    const dw = median(results.directWasm.get(n));
    const aj = median(results.a64Jit.get(n));
    const fw = median(results.fatWasm.get(n));
    console.log(`| ${n} | ${dw.toFixed(2)} | ${aj.toFixed(2)} | ${fw.toFixed(2)} | ${(aj / fw).toFixed(2)}x | \`${checksums.get(n)}\` |`);
  }
}
