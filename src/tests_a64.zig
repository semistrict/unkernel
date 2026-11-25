// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
//! Native unit-test root of `zig build test`: the A64 CPU and the
//! softfloat it builds on.
test {
    _ = @import("cpu/fpu.zig");
    _ = @import("a64/tests.zig");
    _ = @import("a64/crypto_tests.zig");
    _ = @import("a64/tests_excl.zig");
    _ = @import("a64/tests_lse.zig");
    _ = @import("a64/tier_tests.zig");
    _ = @import("a64/jit/emit.zig");
    _ = @import("a64/jit/tests.zig");
    _ = @import("emu/fatwasm.zig");
}
