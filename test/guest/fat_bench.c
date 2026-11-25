// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
/*
 * Fat ELF `.kw.wasm` guest demo & benchmark:
 *   sha256       SHA-256 over a 1 MiB buffer (12 reps)
 *   fp_daxpy_2d  double-precision y += alpha * x over 512K doubles (24 reps)
 *   fp_matmul    96x96 double-precision matrix multiply (12 reps)
 *   complex_ld2  interleaved complex rotation over 256K complex doubles (24 reps)
 *
 * Each benchmark attempts `KW_WASM_CALL #idx` (UDF `0x0000_ff10 + idx`) unless
 * `--fallback` is passed; when the `.kw.wasm` function is not loaded (native
 * harness, `--no-jit`, or non-contiguous buffer), `KW_WASM_CALL` returns -1
 * and the guest seamlessly runs its standard AArch64 C implementation.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_force_fallback = 0;

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#if defined(__aarch64__) && defined(__linux__)
#define DEFINE_KW_CALL(idx, opc) \
static inline int64_t kw_wasm_call_##idx(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) \
{ \
	if (g_force_fallback) \
		return -1; \
	register uint64_t x0 __asm__("x0") = a0; \
	register uint64_t x1 __asm__("x1") = a1; \
	register uint64_t x2 __asm__("x2") = a2; \
	register uint64_t x3 __asm__("x3") = a3; \
	__asm__ volatile( \
		".inst " #opc \
		: "+r"(x0) \
		: "r"(x1), "r"(x2), "r"(x3) \
		: "memory"); \
	return (int64_t)x0; \
}
DEFINE_KW_CALL(0, 0x0000ff10)
DEFINE_KW_CALL(1, 0x0000ff11)
DEFINE_KW_CALL(2, 0x0000ff12)
DEFINE_KW_CALL(3, 0x0000ff13)
#else
static inline int64_t kw_wasm_call_0(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) { (void)a0; (void)a1; (void)a2; (void)a3; return -1; }
static inline int64_t kw_wasm_call_1(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) { (void)a0; (void)a1; (void)a2; (void)a3; return -1; }
static inline int64_t kw_wasm_call_2(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) { (void)a0; (void)a1; (void)a2; (void)a3; return -1; }
static inline int64_t kw_wasm_call_3(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) { (void)a0; (void)a1; (void)a2; (void)a3; return -1; }
#endif

static inline uint64_t d2u(double d)
{
	uint64_t u;
	memcpy(&u, &d, 8);
	return u;
}

static uint64_t hash_doubles(const double *p, size_t n)
{
	uint64_t h = 0xcbf29ce484222325ULL;
	for (size_t i = 0; i < n; i++) {
		h ^= d2u(p[i]);
		h *= 0x100000001b3ULL;
	}
	return h;
}

/* ---- 0: sha256 ----------------------------------------------------------- */

static const uint32_t SHA_K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32u - (n))))

static void sha256_fallback(uint32_t h[8], const uint8_t *buf, size_t len)
{
	for (size_t off = 0; off + 64 <= len; off += 64) {
		const uint8_t *p = buf + off;
		uint32_t w[64];
		for (int i = 0; i < 16; i++)
			w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) |
			       ((uint32_t)p[4 * i + 2] << 8) | p[4 * i + 3];
		for (int i = 16; i < 64; i++) {
			uint32_t s0 = ROR32(w[i - 15], 7) ^ ROR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
			uint32_t s1 = ROR32(w[i - 2], 17) ^ ROR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
			w[i] = w[i - 16] + s0 + w[i - 7] + s1;
		}
		uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
		uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
		for (int i = 0; i < 64; i++) {
			uint32_t t1 = hh + (ROR32(e, 6) ^ ROR32(e, 11) ^ ROR32(e, 25)) + ((e & f) ^ (~e & g)) + SHA_K[i] + w[i];
			uint32_t t2 = (ROR32(a, 2) ^ ROR32(a, 13) ^ ROR32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
			hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
		}
		h[0] += a; h[1] += b; h[2] += c; h[3] += d;
		h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
	}
}

static void run_sha256(void)
{
	const size_t len = 1u << 20;
	const int reps = 12;
	uint8_t *buf = malloc(len);
	for (size_t i = 0; i < len; i++)
		buf[i] = (uint8_t)(i * 131u + (i >> 9));
	uint32_t h[8] = {
		0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
		0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
	};
	int used_fat = 0;
	uint64_t t0 = now_ns();
	for (int r = 0; r < reps; r++) {
		if (kw_wasm_call_0((uint64_t)(uintptr_t)h, (uint64_t)(uintptr_t)buf, len, 0) == 0)
			used_fat = 1;
		else
			sha256_fallback(h, buf, len);
		buf[r % len] ^= (uint8_t)h[0];
	}
	uint64_t dt = now_ns() - t0;
	uint64_t sum = ((uint64_t)h[0] << 32) | h[7];
	printf("BENCH sha256 %s %.3f %016llx\n", used_fat ? "fat" : "a64", (double)dt / 1e6, (unsigned long long)sum);
	free(buf);
}

/* ---- 1: fp_daxpy_2d ------------------------------------------------------ */

static void daxpy_fallback(double *y, const double *x, size_t n, double alpha)
{
	for (size_t i = 0; i < n; i++)
		y[i] = y[i] + alpha * x[i];
}

static void run_daxpy(void)
{
	const size_t n = 512u * 1024u;
	const int reps = 24;
	double *y = malloc(n * sizeof(double));
	for (size_t i = 0; i < n; i++)
		y[i] = (double)((int)(i % 53) - 26) * 0.25;
	double *x = malloc(n * sizeof(double));
	for (size_t i = 0; i < n; i++)
		x[i] = (double)((int)(i % 97) - 48) * 0.125;
	const double alpha = 1.0078125;
	const uint64_t alpha_bits = d2u(alpha);
	int used_fat = 0;
	uint64_t t0 = now_ns();
	for (int r = 0; r < reps; r++) {
		if (kw_wasm_call_1((uint64_t)(uintptr_t)y, (uint64_t)(uintptr_t)x, n, alpha_bits) == 0)
			used_fat = 1;
		else
			daxpy_fallback(y, x, n, alpha);
	}
	uint64_t dt = now_ns() - t0;
	uint64_t sum = hash_doubles(y, n);
	printf("BENCH fp_daxpy_2d %s %.3f %016llx\n", used_fat ? "fat" : "a64", (double)dt / 1e6, (unsigned long long)sum);
	free(x);
	free(y);
}

/* ---- 2: fp_matmul -------------------------------------------------------- */

static void matmul_fallback(double *C, const double *A, const double *B, size_t N)
{
	for (size_t i = 0; i < N; i++) {
		double *ci = C + i * N;
		for (size_t j = 0; j < N; j++)
			ci[j] = 0.0;
		const double *ai = A + i * N;
		for (size_t k = 0; k < N; k++) {
			double aik = ai[k];
			const double *bk = B + k * N;
			for (size_t j = 0; j < N; j++)
				ci[j] = ci[j] + aik * bk[j];
		}
	}
}

static void run_matmul(void)
{
	const size_t N = 96;
	const int reps = 12;
	double *A = malloc(N * N * sizeof(double));
	for (size_t i = 0; i < N; i++)
		for (size_t j = 0; j < N; j++)
			A[i * N + j] = (double)((i * 7 + j * 3) % 17) * 0.25;
	double *B = malloc(N * N * sizeof(double));
	for (size_t i = 0; i < N; i++)
		for (size_t j = 0; j < N; j++)
			B[i * N + j] = (double)((i + 2 * j) % 13) * 0.5 - 1.0;
	double *C = malloc(N * N * sizeof(double));
	for (size_t i = 0; i < N * N; i++)
		C[i] = 0.0;
	int used_fat = 0;
	uint64_t t0 = now_ns();
	for (int r = 0; r < reps; r++) {
		if (kw_wasm_call_2((uint64_t)(uintptr_t)C, (uint64_t)(uintptr_t)A, (uint64_t)(uintptr_t)B, N) == 0)
			used_fat = 1;
		else
			matmul_fallback(C, A, B, N);
		A[(size_t)r % (N * N)] += 0.0625;
	}
	uint64_t dt = now_ns() - t0;
	uint64_t sum = hash_doubles(C, N * N);
	printf("BENCH fp_matmul %s %.3f %016llx\n", used_fat ? "fat" : "a64", (double)dt / 1e6, (unsigned long long)sum);
	free(C);
	free(B);
	free(A);
}

/* ---- 3: complex_ld2 ------------------------------------------------------ */

static void complex_fallback(double *out, const double *in, size_t n_pairs)
{
	const double cr = 0.8660254037844386;
	const double ci = 0.5;
	for (size_t i = 0; i < n_pairs; i++) {
		double re = in[2 * i];
		double im = in[2 * i + 1];
		out[2 * i] = re * cr - im * ci;
		out[2 * i + 1] = re * ci + im * cr;
	}
}

static void run_complex(void)
{
	const size_t n_pairs = 256u * 1024u;
	const int reps = 24;
	double *in = malloc(n_pairs * 2 * sizeof(double));
	for (size_t i = 0; i < n_pairs; i++) {
		in[2 * i] = (double)((int)(i % 61) - 30) * 0.125;
		in[2 * i + 1] = (double)((int)(i % 47) - 23) * 0.25;
	}
	double *out = malloc(n_pairs * 2 * sizeof(double));
	for (size_t i = 0; i < n_pairs * 2; i++)
		out[i] = 0.0;
	int used_fat = 0;
	uint64_t t0 = now_ns();
	for (int r = 0; r < reps; r++) {
		double *src = (r & 1) ? out : in;
		double *dst = (r & 1) ? in : out;
		if (kw_wasm_call_3((uint64_t)(uintptr_t)dst, (uint64_t)(uintptr_t)src, n_pairs, 0) == 0)
			used_fat = 1;
		else
			complex_fallback(dst, src, n_pairs);
	}
	uint64_t dt = now_ns() - t0;
	uint64_t sum = hash_doubles(in, n_pairs * 2);
	printf("BENCH complex_ld2 %s %.3f %016llx\n", used_fat ? "fat" : "a64", (double)dt / 1e6, (unsigned long long)sum);
	free(out);
	free(in);
}

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--fallback") == 0)
			g_force_fallback = 1;
	}
	run_sha256();
	run_daxpy();
	run_matmul();
	run_complex();
	return 0;
}
