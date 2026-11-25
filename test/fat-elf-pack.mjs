// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//
// Fat ELF `.kw.wasm` packer:
//   1. Compiles Wasm64 kernel modules from C (`zig cc -target wasm64-freestanding -O3 -msimd128`).
//   2. Validates each module (imports only `env.memory`, exports `f`, zero data segments).
//   3. Builds the `.kw.wasm` section (`"KWWASM\0\0"`, version 1, entry table, raw `.wasm` payloads).
//   4. Appends `.kw.wasm` as a proper ELF64 section to an existing static aarch64 ELF binary.
//
// Usage:
//   node test/fat-elf-pack.mjs <in.elf> <out.elf> [--kernels-dir DIR]

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

export const FAT_FLAGS = {
  ARG0_PTR: 1 << 0,
  ARG1_PTR: 1 << 1,
  ARG2_PTR: 1 << 2,
  ARG3_PTR: 1 << 3,
  ARG0_WRITE: 1 << 4,
  ARG1_WRITE: 1 << 5,
  ARG2_WRITE: 1 << 6,
  ARG3_WRITE: 1 << 7,
  LEN_X2: 0 << 8,
  LEN_X3: 1 << 8,
  LEN_X2_MUL8: 2 << 8,
  LEN_X2_MUL16: 3 << 8,
  LEN_X3_SQ_MUL8: 4 << 8,
  LEN_ARG0_32_ARG1_X2: 5 << 8,
};

export const KERNEL_SPECS = [
  {
    name: "sha256",
    define: "KERNEL_SHA256",
    fnId: 0,
    flags: FAT_FLAGS.ARG0_PTR | FAT_FLAGS.ARG1_PTR | FAT_FLAGS.ARG0_WRITE | FAT_FLAGS.LEN_ARG0_32_ARG1_X2,
  },
  {
    name: "fp_daxpy_2d",
    define: "KERNEL_DAXPY",
    fnId: 1,
    flags: FAT_FLAGS.ARG0_PTR | FAT_FLAGS.ARG1_PTR | FAT_FLAGS.ARG0_WRITE | FAT_FLAGS.LEN_X2_MUL8,
  },
  {
    name: "fp_matmul",
    define: "KERNEL_MATMUL",
    fnId: 2,
    flags: FAT_FLAGS.ARG0_PTR | FAT_FLAGS.ARG1_PTR | FAT_FLAGS.ARG2_PTR | FAT_FLAGS.ARG0_WRITE | FAT_FLAGS.LEN_X3_SQ_MUL8,
  },
  {
    name: "complex_ld2",
    define: "KERNEL_COMPLEX",
    fnId: 3,
    flags: FAT_FLAGS.ARG0_PTR | FAT_FLAGS.ARG1_PTR | FAT_FLAGS.ARG0_WRITE | FAT_FLAGS.LEN_X2_MUL16,
  },
];

function readUleb(bytes, pos) {
  let val = 0;
  let shift = 0;
  while (pos < bytes.length) {
    const b = bytes[pos++];
    val |= (b & 0x7f) << shift;
    if ((b & 0x80) === 0) return [val >>> 0, pos];
    shift += 7;
  }
  throw new Error("truncated LEB128 in wasm module");
}

/** Verify a compiled `.wasm` kernel is safe to instantiate against kwemu's linear memory. */
export function validateWasmKernel(name, bytes) {
  if (bytes.length < 8 || bytes[0] !== 0x00 || bytes[1] !== 0x61 || bytes[2] !== 0x73 || bytes[3] !== 0x6d) {
    throw new Error(`${name}: invalid wasm magic`);
  }
  let pos = 8;
  while (pos < bytes.length) {
    const id = bytes[pos++];
    const [len, next] = readUleb(bytes, pos);
    pos = next;
    if (id === 11) {
      throw new Error(`${name}: contains a data section (${len} bytes), which would overwrite kwemu linear memory`);
    }
    pos += len;
  }
  const mod = new WebAssembly.Module(bytes);
  const imports = WebAssembly.Module.imports(mod);
  for (const imp of imports) {
    if (imp.module !== "env" || imp.name !== "memory" || imp.kind !== "memory") {
      throw new Error(`${name}: unexpected import ${imp.module}.${imp.name} (${imp.kind})`);
    }
  }
  if (imports.length !== 1) throw new Error(`${name}: expected 1 import (env.memory), got ${imports.length}`);
  const exports = WebAssembly.Module.exports(mod);
  if (!exports.some((e) => e.name === "f" && e.kind === "function")) {
    throw new Error(`${name}: missing exported function f`);
  }
}

export function runZig(args) {
  if (process.env.ZIG) {
    return spawnSync(process.env.ZIG, args, { encoding: "utf8" });
  }
  const r = spawnSync("mise", ["exec", "zig@0.16.0", "--", "zig", ...args], { encoding: "utf8" });
  if (r.error) return spawnSync("zig", args, { encoding: "utf8" });
  return r;
}

/** Compile all 4 Wasm64 kernels and return `{ name, fnId, flags, wasmBytes, path }[]`. */
export async function compileKernels(outDir) {
  await mkdir(outDir, { recursive: true });
  const src = join(root, "test", "guest", "fat_kernels.c");
  const results = [];
  for (const spec of KERNEL_SPECS) {
    const outPath = join(outDir, `${spec.name}.wasm`);
    const r = runZig([
      "cc",
      "-target", "wasm64-freestanding",
      "-O3",
      "-msimd128",
      "-ffp-contract=off",
      "-nostdlib",
      "-Wl,--no-entry",
      "-Wl,--import-memory",
      "-Wl,--export=f",
      "-Wl,--strip-all",
      `-D${spec.define}`,
      src,
      "-o", outPath,
    ]);
    if (r.status !== 0) {
      throw new Error(`failed compiling ${spec.name}: ${r.stderr || r.stdout}`);
    }
    const wasmBytes = new Uint8Array(await readFile(outPath));
    validateWasmKernel(spec.name, wasmBytes);
    results.push({ ...spec, wasmBytes, path: outPath });
  }
  return results;
}

/** Build the binary `.kw.wasm` section payload. */
export function buildKwWasmSection(entries) {
  if (entries.length > 16) throw new Error(`too many entries: ${entries.length} > 16`);
  const headerSize = 16 + entries.length * 16;
  let total = headerSize;
  const offsets = [];
  for (const e of entries) {
    total = (total + 7) & ~7;
    offsets.push(total);
    total += e.wasmBytes.length;
  }
  const buf = new Uint8Array(total);
  const dv = new DataView(buf.buffer);
  buf.set(new TextEncoder().encode("KWWASM"), 0);
  buf[6] = 0;
  buf[7] = 0;
  dv.setUint32(8, 1, true); // version = 1
  dv.setUint32(12, entries.length, true); // count
  for (let i = 0; i < entries.length; i++) {
    const eo = 16 + i * 16;
    dv.setUint32(eo + 0, entries[i].fnId, true);
    dv.setUint32(eo + 4, entries[i].flags, true);
    dv.setUint32(eo + 8, offsets[i], true);
    dv.setUint32(eo + 12, entries[i].wasmBytes.length, true);
    buf.set(entries[i].wasmBytes, offsets[i]);
  }
  return buf;
}

/** Append a non-ALLOC section `sectionName` cleanly to an ELF64 little-endian binary. */
export function appendElfSection(elfBytes, sectionName, sectionData) {
  const src = new Uint8Array(elfBytes);
  const dv = new DataView(src.buffer, src.byteOffset, src.byteLength);
  if (src.length < 64 || src[0] !== 0x7f || src[1] !== 0x45 || src[2] !== 0x4c || src[3] !== 0x46 || src[4] !== 2 || src[5] !== 1) {
    throw new Error("input is not an ELF64 little-endian binary");
  }
  const e_shoff = Number(dv.getBigUint64(40, true));
  const e_shentsize = dv.getUint16(58, true);
  const e_shnum = dv.getUint16(60, true);
  const e_shstrndx = dv.getUint16(62, true);
  if (!e_shoff || !e_shnum || e_shentsize < 64 || e_shstrndx >= e_shnum) {
    throw new Error("ELF has no valid section header table");
  }

  const strHdrOff = e_shoff + e_shstrndx * e_shentsize;
  const oldStrOff = Number(dv.getBigUint64(strHdrOff + 24, true));
  const oldStrSize = Number(dv.getBigUint64(strHdrOff + 32, true));
  const oldShstrtab = src.subarray(oldStrOff, oldStrOff + oldStrSize);

  const nameBytes = new TextEncoder().encode(sectionName + "\0");
  const newNameOffset = oldShstrtab.length;
  const newShstrtab = new Uint8Array(oldShstrtab.length + nameBytes.length);
  newShstrtab.set(oldShstrtab, 0);
  newShstrtab.set(nameBytes, oldShstrtab.length);

  // Append: [aligned sectionData] [aligned newShstrtab] [aligned new section header table]
  const align8 = (n) => (n + 7) & ~7;
  const secOff = align8(src.length);
  const newStrOff = align8(secOff + sectionData.length);
  const newShoff = align8(newStrOff + newShstrtab.length);
  const newShnum = e_shnum + 1;
  const totalSize = newShoff + newShnum * e_shentsize;

  const out = new Uint8Array(totalSize);
  const odw = new DataView(out.buffer);
  out.set(src, 0);
  out.set(sectionData, secOff);
  out.set(newShstrtab, newStrOff);

  // Copy existing section headers to the new table at newShoff.
  out.set(src.subarray(e_shoff, e_shoff + e_shnum * e_shentsize), newShoff);

  // Point .shstrtab's header to the expanded string table.
  const newStrHdrOff = newShoff + e_shstrndx * e_shentsize;
  odw.setBigUint64(newStrHdrOff + 24, BigInt(newStrOff), true);
  odw.setBigUint64(newStrHdrOff + 32, BigInt(newShstrtab.length), true);

  // Write the new `.kw.wasm` section header (SHT_PROGBITS = 1, flags = 0).
  const newSecHdrOff = newShoff + e_shnum * e_shentsize;
  odw.setUint32(newSecHdrOff + 0, newNameOffset, true); // sh_name
  odw.setUint32(newSecHdrOff + 4, 1, true); // sh_type = SHT_PROGBITS
  odw.setBigUint64(newSecHdrOff + 8, 0n, true); // sh_flags
  odw.setBigUint64(newSecHdrOff + 16, 0n, true); // sh_addr
  odw.setBigUint64(newSecHdrOff + 24, BigInt(secOff), true); // sh_offset
  odw.setBigUint64(newSecHdrOff + 32, BigInt(sectionData.length), true); // sh_size
  odw.setUint32(newSecHdrOff + 40, 0, true); // sh_link
  odw.setUint32(newSecHdrOff + 44, 0, true); // sh_info
  odw.setBigUint64(newSecHdrOff + 48, 8n, true); // sh_addralign
  odw.setBigUint64(newSecHdrOff + 56, 0n, true); // sh_entsize

  // Update ELF64 header e_shoff and e_shnum.
  odw.setBigUint64(40, BigInt(newShoff), true);
  odw.setUint16(60, newShnum, true);
  return out;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  let kernelsDir = join(root, "zig-out", "fat-elf");
  const pos = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--kernels-dir" && i + 1 < args.length) {
      kernelsDir = resolve(args[++i]);
    } else {
      pos.push(args[i]);
    }
  }
  if (pos.length !== 2) {
    console.error("usage: node test/fat-elf-pack.mjs <in.elf> <out.elf> [--kernels-dir DIR]");
    process.exit(2);
  }
  const [inElf, outElf] = pos.map((p) => resolve(p));
  const kernels = await compileKernels(kernelsDir);
  const section = buildKwWasmSection(kernels);
  const elfBytes = await readFile(inElf);
  const fatElf = appendElfSection(elfBytes, ".kw.wasm", section);
  await writeFile(outElf, fatElf);
  console.log(`packed ${kernels.length} kernels (${section.length} bytes .kw.wasm) -> ${outElf} (${fatElf.length} bytes)`);
}
