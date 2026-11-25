// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
/*
 * Paravirt bulk-op benchmark: compares baseline C/libc vs kwemu paravirt
 * bulk instructions (KW_MEMCPY, KW_MEMSET, KW_MEMCMP, KW_DAXPY, KW_DDOT).
 *
 * Portable across AArch64 Linux guest (kwemu), Native ARM64, and Direct Wasm
 * (with -DPARAVIRT_WASM_EXPORT for freestanding wasm32/wasm64).
 *
 * Usage (CLI):
 *   paravirt_bench <bench> <mode> [reps]
 *     bench: mem_seq | mem_copy | mem_set | mem_cmp | fp_daxpy_2d | fp_matmul
 *     mode:  base | pv
 */
#include <stdint.h>
#include <stddef.h>
#pragma STDC FP_CONTRACT OFF

#if defined(__aarch64__) && !defined(PARAVIRT_NO_PV)
#include "kw_paravirt.h"
#define HAVE_KW_PV 1
#else
#define HAVE_KW_PV 0
#endif

#ifndef PARAVIRT_WASM_EXPORT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

#define MEM_BYTES (8u << 20)
static uint8_t g_buf_a[MEM_BYTES] __attribute__((aligned(4096)));
static uint8_t g_buf_b[MEM_BYTES] __attribute__((aligned(4096)));

#ifdef PARAVIRT_WASM_EXPORT
static void *base_memcpy(void *dst, const void *src, size_t n)
{
	return __builtin_memcpy(dst, src, n);
}
static void *base_memset(void *dst, int c, size_t n)
{
	return __builtin_memset(dst, c, n);
}
static int base_memcmp(const void *a, const void *b, size_t n)
{
	const uint8_t *pa = (const uint8_t *)a;
	const uint8_t *pb = (const uint8_t *)b;
	for (size_t i = 0; i < n; i++) {
		if (pa[i] != pb[i])
			return (int)pa[i] - (int)pb[i];
	}
	return 0;
}
#else
static void *base_memcpy(void *dst, const void *src, size_t n)
{
	return memcpy(dst, src, n);
}
static void *base_memset(void *dst, int c, size_t n)
{
	return memset(dst, c, n);
}
static int base_memcmp(const void *a, const void *b, size_t n)
{
	return memcmp(a, b, n);
}
#endif

__attribute__((noinline))
static void base_daxpy(double *y, const double *x, size_t n, double a)
{
	for (size_t i = 0; i < n; i++)
		y[i] += a * x[i];
}

__attribute__((noinline))
static double base_ddot(const double *a, const double *b, size_t n)
{
	double s = 0.0;
	for (size_t i = 0; i < n; i++)
		s += a[i] * b[i];
	return s;
}

static uint64_t f64_bits(double v)
{
	uint64_t u;
	__builtin_memcpy(&u, &v, sizeof(u));
	return u;
}

/* Touch every page of g_buf_a and g_buf_b so timing measures warm steady-state. */
static void init_mem(void)
{
	for (size_t i = 0; i < MEM_BYTES; i += 4096) {
		uint8_t v = (uint8_t)((i >> 12) * 7 + 3);
		g_buf_a[i] = v;
		g_buf_b[i] = v;
	}
}

/* 1. mem_seq: 8 MiB memset + memcpy + memcmp (reps default 8). */
static uint64_t bench_mem_seq(int use_pv, int reps)
{
	uint64_t acc = 0;
	for (int r = 0; r < reps; r++) {
		uint8_t val = (uint8_t)((r + 1) * 37 + 11);
#if HAVE_KW_PV
		if (use_pv) {
			kw_memset(g_buf_a, val, MEM_BYTES);
			g_buf_a[r * 997] = (uint8_t)(val ^ 0x5a);
			kw_memcpy(g_buf_b, g_buf_a, MEM_BYTES);
			int eq = kw_memcmp(g_buf_a, g_buf_b, MEM_BYTES);
			g_buf_b[MEM_BYTES - 1] ^= 1;
			int neq = kw_memcmp(g_buf_a, g_buf_b, MEM_BYTES);
			acc = acc * 131 + (uint64_t)(uint32_t)eq + (uint64_t)(uint32_t)(neq != 0) + g_buf_b[r * 997];
			continue;
		}
#endif
		(void)use_pv;
		base_memset(g_buf_a, val, MEM_BYTES);
		g_buf_a[r * 997] = (uint8_t)(val ^ 0x5a);
		base_memcpy(g_buf_b, g_buf_a, MEM_BYTES);
		int eq = base_memcmp(g_buf_a, g_buf_b, MEM_BYTES);
		g_buf_b[MEM_BYTES - 1] ^= 1;
		int neq = base_memcmp(g_buf_a, g_buf_b, MEM_BYTES);
		acc = acc * 131 + (uint64_t)(uint32_t)eq + (uint64_t)(uint32_t)(neq != 0) + g_buf_b[r * 997];
	}
	return acc;
}

/* Pure 8 MiB memcpy (reps default 16 = 128 MiB copied). */
static uint64_t bench_mem_copy(int use_pv, int reps)
{
	uint64_t acc = 0;
	for (int r = 0; r < reps; r++) {
		g_buf_a[(size_t)r * 4096 % MEM_BYTES] = (uint8_t)(r + 7);
#if HAVE_KW_PV
		if (use_pv) {
			kw_memcpy(g_buf_b, g_buf_a, MEM_BYTES);
		} else
#endif
		{
			(void)use_pv;
			base_memcpy(g_buf_b, g_buf_a, MEM_BYTES);
		}
		acc = acc * 31 + g_buf_b[(size_t)r * 4096 % MEM_BYTES];
	}
	return acc;
}

/* Pure 8 MiB memset (reps default 16 = 128 MiB filled). */
static uint64_t bench_mem_set(int use_pv, int reps)
{
	uint64_t acc = 0;
	for (int r = 0; r < reps; r++) {
		int v = (r * 29 + 5) & 0xff;
#if HAVE_KW_PV
		if (use_pv) {
			kw_memset(g_buf_a, v, MEM_BYTES);
		} else
#endif
		{
			(void)use_pv;
			base_memset(g_buf_a, v, MEM_BYTES);
		}
		acc = acc * 31 + g_buf_a[MEM_BYTES - 1 - (size_t)r];
	}
	return acc;
}

/* Pure 8 MiB memcmp (reps default 16 = 128 MiB compared). */
static uint64_t bench_mem_cmp(int use_pv, int reps)
{
	uint64_t acc = 0;
	for (int r = 0; r < reps; r++) {
		int d;
#if HAVE_KW_PV
		if (use_pv) {
			d = kw_memcmp(g_buf_a, g_buf_b, MEM_BYTES);
		} else
#endif
		{
			(void)use_pv;
			d = base_memcmp(g_buf_a, g_buf_b, MEM_BYTES);
		}
		acc = acc * 31 + (uint64_t)(uint32_t)(d + r + 1);
	}
	return acc;
}

/* 2. fp_daxpy_2d: 256 x 256 grid (65,536 doubles = 512 KiB per matrix). */
enum { DAXPY_ROWS = 256, DAXPY_COLS = 256, DAXPY_N = DAXPY_ROWS * DAXPY_COLS };
static double g_y[DAXPY_N] __attribute__((aligned(4096)));
static double g_x[DAXPY_N] __attribute__((aligned(4096)));

static void init_daxpy(void)
{
	for (size_t i = 0; i < DAXPY_N; i++) {
		g_y[i] = (double)((i * 7 + 3) % 251) * 0.125 - 10.0;
		g_x[i] = (double)((i * 11 + 5) % 199) * 0.0625 + 0.5;
	}
}

static uint64_t bench_fp_daxpy_2d(int use_pv, int reps)
{
	for (int r = 0; r < reps; r++) {
		double alpha = (r & 1) ? -0.375 : 0.625;
		for (size_t row = 0; row < DAXPY_ROWS; row++) {
			double *yr = g_y + row * DAXPY_COLS;
			const double *xr = g_x + row * DAXPY_COLS;
#if HAVE_KW_PV
			if (use_pv) {
				kw_daxpy(yr, xr, DAXPY_COLS, alpha);
				continue;
			}
#endif
			(void)use_pv;
			base_daxpy(yr, xr, DAXPY_COLS, alpha);
		}
	}
	double sum = 0.0;
	for (size_t i = 0; i < DAXPY_N; i += 17)
		sum += g_y[i];
	return f64_bits(sum);
}

/* 3. fp_matmul: 96 x 96 double matmul using contiguous dot product (KW_DDOT). */
enum { MM_N = 96 };
static double g_A[MM_N][MM_N] __attribute__((aligned(4096)));
static double g_B[MM_N][MM_N] __attribute__((aligned(4096)));
static double g_Bt[MM_N][MM_N] __attribute__((aligned(4096)));
static double g_C[MM_N][MM_N] __attribute__((aligned(4096)));

static void init_matmul(void)
{
	for (int i = 0; i < MM_N; i++)
		for (int j = 0; j < MM_N; j++) {
			g_A[i][j] = ((i * 7 + j * 3) % 17) * 0.25;
			g_B[i][j] = ((i + 2 * j) % 13) * 0.5 - 1.0;
		}
	for (int i = 0; i < MM_N; i++)
		for (int j = 0; j < MM_N; j++)
			g_Bt[j][i] = g_B[i][j];
}

static uint64_t bench_fp_matmul(int use_pv, int reps)
{
	double acc = 0.0;
	for (int r = 0; r < reps; r++) {
		for (int i = 0; i < MM_N; i++) {
			for (int j = 0; j < MM_N; j++) {
#if HAVE_KW_PV
				if (use_pv) {
					g_C[i][j] = kw_ddot(g_A[i], g_Bt[j], MM_N);
					continue;
				}
#endif
				(void)use_pv;
				g_C[i][j] = base_ddot(g_A[i], g_Bt[j], MM_N);
			}
		}
		acc += g_C[r % MM_N][(r * 5) % MM_N];
	}
	return f64_bits(acc);
}

#ifdef PARAVIRT_WASM_EXPORT
__attribute__((export_name("init_bench")))
void init_bench(int bench_id)
{
	switch (bench_id) {
	case 0: case 1: case 2: case 3: init_mem(); break;
	case 4: init_daxpy(); break;
	case 5: init_matmul(); break;
	default: break;
	}
}

__attribute__((export_name("run_bench")))
uint64_t run_bench(int bench_id, int reps)
{
	switch (bench_id) {
	case 0: return bench_mem_seq(0, reps ? reps : 8);
	case 1: return bench_mem_copy(0, reps ? reps : 16);
	case 2: return bench_mem_set(0, reps ? reps : 16);
	case 3: return bench_mem_cmp(0, reps ? reps : 16);
	case 4: return bench_fp_daxpy_2d(0, reps ? reps : 64);
	case 5: return bench_fp_matmul(0, reps ? reps : 20);
	default: return 0;
	}
}
#else
int main(int argc, char **argv)
{
	const char *what = argc > 1 ? argv[1] : "mem_seq";
	const char *mode = argc > 2 ? argv[2] : "base";
	int reps = argc > 3 ? atoi(argv[3]) : 0;
	int use_pv = !strcmp(mode, "pv");
	uint64_t r = 0, t0, t1;

	if (!strcmp(what, "mem_seq")) {
		init_mem();
		t0 = now_ns();
		r = bench_mem_seq(use_pv, reps ? reps : 8);
		t1 = now_ns();
	} else if (!strcmp(what, "mem_copy")) {
		init_mem();
		t0 = now_ns();
		r = bench_mem_copy(use_pv, reps ? reps : 16);
		t1 = now_ns();
	} else if (!strcmp(what, "mem_set")) {
		init_mem();
		t0 = now_ns();
		r = bench_mem_set(use_pv, reps ? reps : 16);
		t1 = now_ns();
	} else if (!strcmp(what, "mem_cmp")) {
		init_mem();
		t0 = now_ns();
		r = bench_mem_cmp(use_pv, reps ? reps : 16);
		t1 = now_ns();
	} else if (!strcmp(what, "fp_daxpy_2d")) {
		init_daxpy();
		t0 = now_ns();
		r = bench_fp_daxpy_2d(use_pv, reps ? reps : 64);
		t1 = now_ns();
	} else if (!strcmp(what, "fp_matmul")) {
		init_matmul();
		t0 = now_ns();
		r = bench_fp_matmul(use_pv, reps ? reps : 20);
		t1 = now_ns();
	} else {
		fprintf(stderr, "unknown bench: %s\n", what);
		return 2;
	}
	printf("%s/%s: %016llx ms=%.3f\n", what, mode, (unsigned long long)r, (double)(t1 - t0) / 1e6);
	return 0;
}
#endif
