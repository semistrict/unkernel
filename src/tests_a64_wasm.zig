// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! wasm64 unit-test root of the A64 CPU with the JIT tier (run by
//! test/kwemu-unit-wasm.mjs through src/test_runner_wasm.zig against an
//! eager JIT host): the tests that run code through kw_run.

const jit_tests = @import("a64/jit/tests.zig");

/// Called by the JIT host after every compile (test/kwemu-jit-host.mjs).
export fn kwemu_test_compile_hook() void {
    jit_tests.compileHook();
}

test {
    _ = @import("a64/tier_tests.zig");
    _ = @import("a64/jit/emit.zig");
    _ = jit_tests;
    _ = @import("emu/kwemu.zig"); // kw_debug_cpu_of_jit_buf
    _ = @import("emu/fatwasm.zig");
}
