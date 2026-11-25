// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
/*
 * Precompiled Wasm64 kernels for the Fat ELF `.kw.wasm` demo & benchmark.
 * Compiled individually with:
 *   zig cc -target wasm64-freestanding -O3 -msimd128 -ffp-contract=off \
 *          -nostdlib -Wl,--no-entry -Wl,--import-memory -Wl,--export=f \
 *          -DKERNEL_<NAME> test/guest/fat_kernels.c -o <name>.wasm
 *
 * Constraints enforced by the packer:
 *   - Imports only `(import "env" "memory" (memory i64 ...))`.
 *   - Exports `f(i64, i64, i64, i64) -> i64`.
 *   - `f(0, 0, 0, 0)` returns `0` immediately (warm-up contract).
 *   - Zero active data segments and zero shadow-stack usage so instantiating
 *     against kwemu's linear memory never overwrites host memory.
 */

#include <stdint.h>
#include <stddef.h>
#include <wasm_simd128.h>

#ifdef KERNEL_SHA256

static inline uint32_t ror32(uint32_t x, uint32_t n)
{
	return (x >> n) | (x << (32u - n));
}

static inline uint32_t load_be32(const uint8_t *p)
{
	return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
	       ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

#define SHA_ROUND(a, b, c, d, e, f, g, h, wv, kv) do { \
	uint32_t _t1 = (h) + (ror32((e), 6) ^ ror32((e), 11) ^ ror32((e), 25)) \
	               + (((e) & (f)) ^ (~(e) & (g))) + (kv) + (wv); \
	uint32_t _t2 = (ror32((a), 2) ^ ror32((a), 13) ^ ror32((a), 22)) \
	               + (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c))); \
	(d) += _t1; \
	(h) = _t1 + _t2; \
} while (0)

#define SHA_NEXT(w0, w1, w9, w14) do { \
	uint32_t _s0 = ror32((w1), 7) ^ ror32((w1), 18) ^ ((w1) >> 3); \
	uint32_t _s1 = ror32((w14), 17) ^ ror32((w14), 19) ^ ((w14) >> 10); \
	(w0) = (w0) + _s0 + (w9) + _s1; \
} while (0)

int64_t f(int64_t h_ptr, int64_t data_ptr, int64_t len, int64_t unused)
{
	(void)unused;
	if (h_ptr == 0 || data_ptr == 0 || len <= 0)
		return 0;

	uint32_t *state = (uint32_t *)(uintptr_t)h_ptr;
	const uint8_t *p = (const uint8_t *)(uintptr_t)data_ptr;
	uint64_t blocks = (uint64_t)len >> 6;

	uint32_t h0 = state[0], h1 = state[1], h2 = state[2], h3 = state[3];
	uint32_t h4 = state[4], h5 = state[5], h6 = state[6], h7 = state[7];

	for (uint64_t blk = 0; blk < blocks; blk++, p += 64) {
		uint32_t w0 = load_be32(p + 0);
		uint32_t w1 = load_be32(p + 4);
		uint32_t w2 = load_be32(p + 8);
		uint32_t w3 = load_be32(p + 12);
		uint32_t w4 = load_be32(p + 16);
		uint32_t w5 = load_be32(p + 20);
		uint32_t w6 = load_be32(p + 24);
		uint32_t w7 = load_be32(p + 28);
		uint32_t w8 = load_be32(p + 32);
		uint32_t w9 = load_be32(p + 36);
		uint32_t w10 = load_be32(p + 40);
		uint32_t w11 = load_be32(p + 44);
		uint32_t w12 = load_be32(p + 48);
		uint32_t w13 = load_be32(p + 52);
		uint32_t w14 = load_be32(p + 56);
		uint32_t w15 = load_be32(p + 60);

		uint32_t a = h0, b = h1, c = h2, d = h3;
		uint32_t e = h4, ff = h5, g = h6, hh = h7;

		SHA_ROUND(a, b, c, d, e, ff, g, hh, w0, 0x428a2f98u);
		SHA_ROUND(hh, a, b, c, d, e, ff, g, w1, 0x71374491u);
		SHA_ROUND(g, hh, a, b, c, d, e, ff, w2, 0xb5c0fbcfu);
		SHA_ROUND(ff, g, hh, a, b, c, d, e, w3, 0xe9b5dba5u);
		SHA_ROUND(e, ff, g, hh, a, b, c, d, w4, 0x3956c25bu);
		SHA_ROUND(d, e, ff, g, hh, a, b, c, w5, 0x59f111f1u);
		SHA_ROUND(c, d, e, ff, g, hh, a, b, w6, 0x923f82a4u);
		SHA_ROUND(b, c, d, e, ff, g, hh, a, w7, 0xab1c5ed5u);

		SHA_ROUND(a, b, c, d, e, ff, g, hh, w8, 0xd807aa98u);
		SHA_ROUND(hh, a, b, c, d, e, ff, g, w9, 0x12835b01u);
		SHA_ROUND(g, hh, a, b, c, d, e, ff, w10, 0x243185beu);
		SHA_ROUND(ff, g, hh, a, b, c, d, e, w11, 0x550c7dc3u);
		SHA_ROUND(e, ff, g, hh, a, b, c, d, w12, 0x72be5d74u);
		SHA_ROUND(d, e, ff, g, hh, a, b, c, w13, 0x80deb1feu);
		SHA_ROUND(c, d, e, ff, g, hh, a, b, w14, 0x9bdc06a7u);
		SHA_ROUND(b, c, d, e, ff, g, hh, a, w15, 0xc19bf174u);

		SHA_NEXT(w0, w1, w9, w14); SHA_ROUND(a, b, c, d, e, ff, g, hh, w0, 0xe49b69c1u);
		SHA_NEXT(w1, w2, w10, w15); SHA_ROUND(hh, a, b, c, d, e, ff, g, w1, 0xefbe4786u);
		SHA_NEXT(w2, w3, w11, w0); SHA_ROUND(g, hh, a, b, c, d, e, ff, w2, 0x0fc19dc6u);
		SHA_NEXT(w3, w4, w12, w1); SHA_ROUND(ff, g, hh, a, b, c, d, e, w3, 0x240ca1ccu);
		SHA_NEXT(w4, w5, w13, w2); SHA_ROUND(e, ff, g, hh, a, b, c, d, w4, 0x2de92c6fu);
		SHA_NEXT(w5, w6, w14, w3); SHA_ROUND(d, e, ff, g, hh, a, b, c, w5, 0x4a7484aau);
		SHA_NEXT(w6, w7, w15, w4); SHA_ROUND(c, d, e, ff, g, hh, a, b, w6, 0x5cb0a9dcu);
		SHA_NEXT(w7, w8, w0, w5); SHA_ROUND(b, c, d, e, ff, g, hh, a, w7, 0x76f988dau);

		SHA_NEXT(w8, w9, w1, w6); SHA_ROUND(a, b, c, d, e, ff, g, hh, w8, 0x983e5152u);
		SHA_NEXT(w9, w10, w2, w7); SHA_ROUND(hh, a, b, c, d, e, ff, g, w9, 0xa831c66du);
		SHA_NEXT(w10, w11, w3, w8); SHA_ROUND(g, hh, a, b, c, d, e, ff, w10, 0xb00327c8u);
		SHA_NEXT(w11, w12, w4, w9); SHA_ROUND(ff, g, hh, a, b, c, d, e, w11, 0xbf597fc7u);
		SHA_NEXT(w12, w13, w5, w10); SHA_ROUND(e, ff, g, hh, a, b, c, d, w12, 0xc6e00bf3u);
		SHA_NEXT(w13, w14, w6, w11); SHA_ROUND(d, e, ff, g, hh, a, b, c, w13, 0xd5a79147u);
		SHA_NEXT(w14, w15, w7, w12); SHA_ROUND(c, d, e, ff, g, hh, a, b, w14, 0x06ca6351u);
		SHA_NEXT(w15, w0, w8, w13); SHA_ROUND(b, c, d, e, ff, g, hh, a, w15, 0x14292967u);

		SHA_NEXT(w0, w1, w9, w14); SHA_ROUND(a, b, c, d, e, ff, g, hh, w0, 0x27b70a85u);
		SHA_NEXT(w1, w2, w10, w15); SHA_ROUND(hh, a, b, c, d, e, ff, g, w1, 0x2e1b2138u);
		SHA_NEXT(w2, w3, w11, w0); SHA_ROUND(g, hh, a, b, c, d, e, ff, w2, 0x4d2c6dfcu);
		SHA_NEXT(w3, w4, w12, w1); SHA_ROUND(ff, g, hh, a, b, c, d, e, w3, 0x53380d13u);
		SHA_NEXT(w4, w5, w13, w2); SHA_ROUND(e, ff, g, hh, a, b, c, d, w4, 0x650a7354u);
		SHA_NEXT(w5, w6, w14, w3); SHA_ROUND(d, e, ff, g, hh, a, b, c, w5, 0x766a0abbu);
		SHA_NEXT(w6, w7, w15, w4); SHA_ROUND(c, d, e, ff, g, hh, a, b, w6, 0x81c2c92eu);
		SHA_NEXT(w7, w8, w0, w5); SHA_ROUND(b, c, d, e, ff, g, hh, a, w7, 0x92722c85u);

		SHA_NEXT(w8, w9, w1, w6); SHA_ROUND(a, b, c, d, e, ff, g, hh, w8, 0xa2bfe8a1u);
		SHA_NEXT(w9, w10, w2, w7); SHA_ROUND(hh, a, b, c, d, e, ff, g, w9, 0xa81a664bu);
		SHA_NEXT(w10, w11, w3, w8); SHA_ROUND(g, hh, a, b, c, d, e, ff, w10, 0xc24b8b70u);
		SHA_NEXT(w11, w12, w4, w9); SHA_ROUND(ff, g, hh, a, b, c, d, e, w11, 0xc76c51a3u);
		SHA_NEXT(w12, w13, w5, w10); SHA_ROUND(e, ff, g, hh, a, b, c, d, w12, 0xd192e819u);
		SHA_NEXT(w13, w14, w6, w11); SHA_ROUND(d, e, ff, g, hh, a, b, c, w13, 0xd6990624u);
		SHA_NEXT(w14, w15, w7, w12); SHA_ROUND(c, d, e, ff, g, hh, a, b, w14, 0xf40e3585u);
		SHA_NEXT(w15, w0, w8, w13); SHA_ROUND(b, c, d, e, ff, g, hh, a, w15, 0x106aa070u);

		SHA_NEXT(w0, w1, w9, w14); SHA_ROUND(a, b, c, d, e, ff, g, hh, w0, 0x19a4c116u);
		SHA_NEXT(w1, w2, w10, w15); SHA_ROUND(hh, a, b, c, d, e, ff, g, w1, 0x1e376c08u);
		SHA_NEXT(w2, w3, w11, w0); SHA_ROUND(g, hh, a, b, c, d, e, ff, w2, 0x2748774cu);
		SHA_NEXT(w3, w4, w12, w1); SHA_ROUND(ff, g, hh, a, b, c, d, e, w3, 0x34b0bcb5u);
		SHA_NEXT(w4, w5, w13, w2); SHA_ROUND(e, ff, g, hh, a, b, c, d, w4, 0x391c0cb3u);
		SHA_NEXT(w5, w6, w14, w3); SHA_ROUND(d, e, ff, g, hh, a, b, c, w5, 0x4ed8aa4au);
		SHA_NEXT(w6, w7, w15, w4); SHA_ROUND(c, d, e, ff, g, hh, a, b, w6, 0x5b9cca4fu);
		SHA_NEXT(w7, w8, w0, w5); SHA_ROUND(b, c, d, e, ff, g, hh, a, w7, 0x682e6ff3u);

		SHA_NEXT(w8, w9, w1, w6); SHA_ROUND(a, b, c, d, e, ff, g, hh, w8, 0x748f82eeu);
		SHA_NEXT(w9, w10, w2, w7); SHA_ROUND(hh, a, b, c, d, e, ff, g, w9, 0x78a5636fu);
		SHA_NEXT(w10, w11, w3, w8); SHA_ROUND(g, hh, a, b, c, d, e, ff, w10, 0x84c87814u);
		SHA_NEXT(w11, w12, w4, w9); SHA_ROUND(ff, g, hh, a, b, c, d, e, w11, 0x8cc70208u);
		SHA_NEXT(w12, w13, w5, w10); SHA_ROUND(e, ff, g, hh, a, b, c, d, w12, 0x90befffau);
		SHA_NEXT(w13, w14, w6, w11); SHA_ROUND(d, e, ff, g, hh, a, b, c, w13, 0xa4506cebu);
		SHA_NEXT(w14, w15, w7, w12); SHA_ROUND(c, d, e, ff, g, hh, a, b, w14, 0xbef9a3f7u);
		SHA_NEXT(w15, w0, w8, w13); SHA_ROUND(b, c, d, e, ff, g, hh, a, w15, 0xc67178f2u);

		h0 += a; h1 += b; h2 += c; h3 += d;
		h4 += e; h5 += ff; h6 += g; h7 += hh;
	}

	state[0] = h0; state[1] = h1; state[2] = h2; state[3] = h3;
	state[4] = h4; state[5] = h5; state[6] = h6; state[7] = h7;
	return 0;
}

#elif defined(KERNEL_DAXPY)

int64_t f(int64_t y_ptr, int64_t x_ptr, int64_t n, int64_t alpha_bits)
{
	if (y_ptr == 0 || x_ptr == 0 || n <= 0)
		return 0;

	double *y = (double *)(uintptr_t)y_ptr;
	const double *x = (const double *)(uintptr_t)x_ptr;
	union { uint64_t u; double d; } conv = { .u = (uint64_t)alpha_bits };
	double alpha = conv.d;
	v128_t va = wasm_f64x2_splat(alpha);
	size_t count = (size_t)n;
	size_t i = 0;

	for (; i + 4 <= count; i += 4) {
		v128_t y0 = wasm_v128_load(y + i);
		v128_t x0 = wasm_v128_load(x + i);
		v128_t y1 = wasm_v128_load(y + i + 2);
		v128_t x1 = wasm_v128_load(x + i + 2);
		wasm_v128_store(y + i, wasm_f64x2_add(y0, wasm_f64x2_mul(va, x0)));
		wasm_v128_store(y + i + 2, wasm_f64x2_add(y1, wasm_f64x2_mul(va, x1)));
	}
	for (; i < count; i++)
		y[i] = y[i] + alpha * x[i];
	return 0;
}

#elif defined(KERNEL_MATMUL)

int64_t f(int64_t c_ptr, int64_t a_ptr, int64_t b_ptr, int64_t n_arg)
{
	if (c_ptr == 0 || a_ptr == 0 || b_ptr == 0 || n_arg <= 0)
		return 0;

	double *C = (double *)(uintptr_t)c_ptr;
	const double *A = (const double *)(uintptr_t)a_ptr;
	const double *B = (const double *)(uintptr_t)b_ptr;
	size_t N = (size_t)n_arg;

	for (size_t i = 0; i < N; i++) {
		double *ci = C + i * N;
		for (size_t j = 0; j < N; j++)
			ci[j] = 0.0;
		const double *ai = A + i * N;
		for (size_t k = 0; k < N; k++) {
			double aik = ai[k];
			v128_t va = wasm_f64x2_splat(aik);
			const double *bk = B + k * N;
			size_t j = 0;
			for (; j + 4 <= N; j += 4) {
				v128_t c0 = wasm_v128_load(ci + j);
				v128_t b0 = wasm_v128_load(bk + j);
				v128_t c1 = wasm_v128_load(ci + j + 2);
				v128_t b1 = wasm_v128_load(bk + j + 2);
				wasm_v128_store(ci + j, wasm_f64x2_add(c0, wasm_f64x2_mul(va, b0)));
				wasm_v128_store(ci + j + 2, wasm_f64x2_add(c1, wasm_f64x2_mul(va, b1)));
			}
			for (; j < N; j++)
				ci[j] = ci[j] + aik * bk[j];
		}
	}
	return 0;
}

#elif defined(KERNEL_COMPLEX)

int64_t f(int64_t out_ptr, int64_t in_ptr, int64_t n_pairs, int64_t unused)
{
	(void)unused;
	if (out_ptr == 0 || in_ptr == 0 || n_pairs <= 0)
		return 0;

	double *out = (double *)(uintptr_t)out_ptr;
	const double *in = (const double *)(uintptr_t)in_ptr;
	const double cr = 0.8660254037844386;
	const double ci = 0.5;
	v128_t vcr = wasm_f64x2_splat(cr);
	v128_t vci = wasm_f64x2_splat(ci);
	size_t count = (size_t)n_pairs;
	size_t i = 0;

	/* Deinterleave 2 complex (re, im) pairs at a time (LD2/ST2 equivalent). */
	for (; i + 2 <= count; i += 2) {
		v128_t z0 = wasm_v128_load(in + 2 * i);
		v128_t z1 = wasm_v128_load(in + 2 * i + 2);
		v128_t re = wasm_i64x2_shuffle(z0, z1, 0, 2);
		v128_t im = wasm_i64x2_shuffle(z0, z1, 1, 3);
		v128_t ore = wasm_f64x2_sub(wasm_f64x2_mul(re, vcr), wasm_f64x2_mul(im, vci));
		v128_t oim = wasm_f64x2_add(wasm_f64x2_mul(re, vci), wasm_f64x2_mul(im, vcr));
		wasm_v128_store(out + 2 * i, wasm_i64x2_shuffle(ore, oim, 0, 2));
		wasm_v128_store(out + 2 * i + 2, wasm_i64x2_shuffle(ore, oim, 1, 3));
	}
	for (; i < count; i++) {
		double re = in[2 * i];
		double im = in[2 * i + 1];
		out[2 * i] = re * cr - im * ci;
		out[2 * i + 1] = re * ci + im * cr;
	}
	return 0;
}

#else
#error "Define one of KERNEL_SHA256, KERNEL_DAXPY, KERNEL_MATMUL, KERNEL_COMPLEX"
#endif
