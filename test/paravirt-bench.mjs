// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
// Benchmark comparing baseline AArch64 vs kwemu paravirt bulk ops
// (KW_MEMCPY, KW_MEMSET, KW_MEMCMP, KW_DAXPY, KW_DDOT) against
// Direct Wasm and Native ARM64 (warm steady-state timing).
//
// Usage: node test/paravirt-bench.mjs [zig-out prefix] [--reps N]
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { createJitHost } from "./kwemu-jit-host.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const argv = process.argv.slice(2);
const prefix = resolve(argv.find((a, i) => !a.startsWith("--") && argv[i - 1] !== "--reps") ?? join(root, "zig-out"));
const repsArg = argv.indexOf("--reps");
const runs = repsArg >= 0 ? Number(argv[repsArg + 1]) : 5;

const elf = join(prefix, "guest", "a64", "paravirt_bench.elf");
const srcC = join(root, "test", "guest", "a64", "paravirt_bench.c");
const jitPath = join(prefix, "bin", "kwemu-harness-jit.wasm");
const dec = new TextDecoder();

const benchmarks = [
  { id: 0, name: "mem_seq", reps: 8, desc: "8 MiB memset+memcpy+memcmp x8" },
  { id: 1, name: "mem_copy", reps: 16, desc: "8 MiB memcpy x16 (128 MiB)" },
  { id: 2, name: "mem_set", reps: 16, desc: "8 MiB memset x16 (128 MiB)" },
  { id: 3, name: "mem_cmp", reps: 16, desc: "8 MiB memcmp x16 (128 MiB)" },
  { id: 4, name: "fp_daxpy_2d", reps: 64, desc: "256x256 daxpy x64 (4.19M ops)" },
  { id: 5, name: "fp_matmul", reps: 20, desc: "96x96 matmul (ddot) x20" },
];

const jitModule = await WebAssembly.compile(await readFile(jitPath));

const tmp = await mkdtemp(join(tmpdir(), "kw-pv-bench-"));
const directWasmPath = join(tmp, "paravirt_bench_direct.wasm");
const nativeBinPath = join(tmp, "paravirt_bench_native");

try {
  // Compile Direct Wasm using zig cc (wasm32-freestanding + bulk-memory)
  const wBuild = spawnSync("mise", [
    "exec", "zig@0.16.0", "--", "zig", "cc",
    "-target", "wasm32-freestanding", "-O3",
    "-mbulk-memory", "-fno-vectorize", "-fno-slp-vectorize",
    "-DPARAVIRT_WASM_EXPORT",
    "-Wl,--no-entry", "-Wl,--export=init_bench", "-Wl,--export=run_bench", "-Wl,--initial-memory=33554432",
    srcC, "-o", directWasmPath,
  ], { encoding: "utf8" });
  if (wBuild.status !== 0) {
    throw new Error(`Direct Wasm build failed:\n${wBuild.stderr}`);
  }
  const directWasmMod = await WebAssembly.compile(await readFile(directWasmPath));

  // Compile Native binary
  const nBuild = spawnSync("clang", [
    "-O3", "-fno-vectorize", "-fno-slp-vectorize",
    "-DPARAVIRT_NO_PV", srcC, "-o", nativeBinPath,
  ], { encoding: "utf8" });
  if (nBuild.status !== 0) {
    throw new Error(`Native build failed:\n${nBuild.stderr}`);
  }

  function median(arr) {
    const s = [...arr].sort((a, b) => a - b);
    return s[(s.length - 1) >> 1];
  }

  function parseOut(stdout) {
    const m = /([0-9a-f]{16}) ms=([\d.]+)/.exec(stdout);
    if (!m) throw new Error(`unexpected benchmark output: ${JSON.stringify(stdout)}`);
    return { sum: m[1], ms: Number(m[2]) };
  }

  async function runKwemuJit(benchName, mode, reps) {
    let out = "", err = "", mem;
    const jit = createJitHost(); // default threshold
    const imports = {
      env: {
        host_write: (fd, ptr, len) => {
          const s = dec.decode(new Uint8Array(mem.buffer, Number(ptr), Number(len)));
          if (fd === 2) err += s; else out += s;
        },
        host_now_ns: () => Number(process.hrtime.bigint()),
      },
      unkernel: jit.imports,
    };
    const inst = await WebAssembly.instantiate(jitModule, imports);
    const ex = inst.exports;
    mem = ex.memory;
    jit.attach(ex);
    const image = await readFile(elf);
    const args = ["paravirt_bench", benchName, mode, String(reps)];
    const argBytes = new TextEncoder().encode(args.join("\0") + "\0");
    const total = image.length + argBytes.length;
    const ptr = Number(ex.kwh_buf(BigInt(total)));
    new Uint8Array(mem.buffer, ptr, total).set(image);
    new Uint8Array(mem.buffer, ptr + image.length, argBytes.length).set(argBytes);
    const status = ex.kwh_run(BigInt(image.length), BigInt(argBytes.length), 1);
    if (status !== 0) throw new Error(`kwemu ${benchName}/${mode} failed (${status}): ${err}`);
    const { sum, ms } = parseOut(out);
    const insn = /instret=(\d+)/.exec(err)?.[1] ?? "0";
    return { ms, sum, insns: Number(insn) };
  }

  async function runDirectWasm(benchId, reps) {
    const inst = await WebAssembly.instantiate(directWasmMod, {});
    const { init_bench, run_bench } = inst.exports;
    // Warm up once for V8 TurboFan
    init_bench(benchId);
    run_bench(benchId, 1);
    const times = [];
    let sum = "";
    for (let i = 0; i < runs; i++) {
      init_bench(benchId);
      const t0 = process.hrtime.bigint();
      const r = run_bench(benchId, reps);
      const dtMs = Number(process.hrtime.bigint() - t0) / 1e6;
      times.push(dtMs);
      sum = BigInt.asUintN(64, r).toString(16).padStart(16, "0");
    }
    return { ms: median(times), sum };
  }

  function runNative(benchName, reps) {
    const times = [];
    let sum = "";
    for (let i = 0; i < runs; i++) {
      const p = spawnSync(nativeBinPath, [benchName, "base", String(reps)], { encoding: "utf8" });
      if (p.status !== 0) throw new Error(`native ${benchName} failed: ${p.stderr}`);
      const r = parseOut(p.stdout);
      times.push(r.ms);
      sum = r.sum;
    }
    return { ms: median(times), sum };
  }

  console.log(`Paravirt bulk-op benchmark (median of ${runs} runs, steady-state loop time, wasm64+jit default threshold)`);
  console.log("");
  console.log("| benchmark | workload | kwemu base (ms) | kwemu pv (ms) | speedup | Direct Wasm (ms) | Native ARM64 (ms) | pv / Wasm | base Minsns | pv Minsns | checksum |");
  console.log("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");

  for (const b of benchmarks) {
    const baseTimes = [], pvTimes = [];
    let baseSum = "", pvSum = "", baseInsns = 0, pvInsns = 0;
    for (let i = 0; i < runs; i++) {
      const rb = await runKwemuJit(b.name, "base", b.reps);
      baseTimes.push(rb.ms);
      baseSum = rb.sum;
      baseInsns = rb.insns;
      const rp = await runKwemuJit(b.name, "pv", b.reps);
      pvTimes.push(rp.ms);
      pvSum = rp.sum;
      pvInsns = rp.insns;
    }
    const dw = await runDirectWasm(b.id, b.reps);
    const nat = runNative(b.name, b.reps);

    if (baseSum !== pvSum || baseSum !== dw.sum || baseSum !== nat.sum) {
      throw new Error(`Checksum mismatch on ${b.name}: base=${baseSum} pv=${pvSum} wasm=${dw.sum} native=${nat.sum}`);
    }
    const bMs = median(baseTimes);
    const pMs = median(pvTimes);
    const speedup = bMs / pMs;
    const vsWasm = pMs / dw.ms;
    console.log(
      `| ${b.name} | ${b.desc} | ${bMs.toFixed(2)} | ${pMs.toFixed(2)} | ${speedup.toFixed(2)}x | ${dw.ms.toFixed(2)} | ${nat.ms.toFixed(2)} | ${vsWasm.toFixed(2)}x | ${(baseInsns / 1e6).toFixed(2)} | ${(pvInsns / 1e6).toFixed(3)} | ${baseSum} |`
    );
  }
} finally {
  await rm(tmp, { recursive: true, force: true });
}
