/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
 *
 * Paravirt bulk-op helpers for AArch64 guests running under kwemu.
 *
 * Reserved permanently-undefined AArch64 UDF opcodes 0x0000_ff00..0x0000_ff0f:
 *   0x0000_ff00  KW_MEMCPY  x0=dst, x1=src, x2=len       -> x0=dst (overlap-safe)
 *   0x0000_ff01  KW_MEMSET  x0=dst, x1=byte(u8), x2=len  -> x0=dst
 *   0x0000_ff02  KW_MEMCMP  x0=a,   x1=b,   x2=len       -> x0=signed diff
 *   0x0000_ff03  KW_DAXPY   x0=y,   x1=x,   x2=n, d0=a   -> y[i] += a * x[i]
 *   0x0000_ff04  KW_DDOT    x0=a,   x1=b,   x2=n         -> d0=sum(a[i] * b[i])
 *
 * Example LD_PRELOAD / IFUNC usage:
 *   void *memcpy(void *dst, const void *src, size_t n) {
 *       return kw_memcpy(dst, src, n);
 *   }
 *   void *memset(void *dst, int c, size_t n) {
 *       return kw_memset(dst, c, n);
 *   }
 *   int memcmp(const void *a, const void *b, size_t n) {
 *       return kw_memcmp(a, b, n);
 *   }
 */
#ifndef KW_PARAVIRT_H
#define KW_PARAVIRT_H

#include <stddef.h>
#include <stdint.h>

#define KW_PV_MEMCPY 0x0000ff00u
#define KW_PV_MEMSET 0x0000ff01u
#define KW_PV_MEMCMP 0x0000ff02u
#define KW_PV_DAXPY  0x0000ff03u
#define KW_PV_DDOT   0x0000ff04u

#if defined(__aarch64__)

static inline void *kw_memcpy(void *dst, const void *src, size_t len)
{
	register void *x0 __asm__("x0") = dst;
	register const void *x1 __asm__("x1") = src;
	register size_t x2 __asm__("x2") = len;
	__asm__ volatile(".inst 0x0000ff00"
			 : "+r"(x0)
			 : "r"(x1), "r"(x2)
			 : "memory");
	return x0;
}

static inline void *kw_memset(void *dst, int byte, size_t len)
{
	register void *x0 __asm__("x0") = dst;
	register uint64_t x1 __asm__("x1") = (uint8_t)byte;
	register size_t x2 __asm__("x2") = len;
	__asm__ volatile(".inst 0x0000ff01"
			 : "+r"(x0)
			 : "r"(x1), "r"(x2)
			 : "memory");
	return x0;
}

static inline int kw_memcmp(const void *a, const void *b, size_t len)
{
	register intptr_t x0 __asm__("x0") = (intptr_t)a;
	register const void *x1 __asm__("x1") = b;
	register size_t x2 __asm__("x2") = len;
	__asm__ volatile(".inst 0x0000ff02"
			 : "+r"(x0)
			 : "r"(x1), "r"(x2)
			 : "memory");
	return (int)x0;
}

static inline void kw_daxpy(double *y, const double *x, size_t n, double a)
{
	register double *x0 __asm__("x0") = y;
	register const double *x1 __asm__("x1") = x;
	register size_t x2 __asm__("x2") = n;
	register double d0 __asm__("d0") = a;
	__asm__ volatile(".inst 0x0000ff03"
			 :
			 : "r"(x0), "r"(x1), "r"(x2), "w"(d0)
			 : "memory");
}

static inline double kw_ddot(const double *a, const double *b, size_t n)
{
	register const double *x0 __asm__("x0") = a;
	register const double *x1 __asm__("x1") = b;
	register size_t x2 __asm__("x2") = n;
	register double d0 __asm__("d0");
	__asm__ volatile(".inst 0x0000ff04"
			 : "=w"(d0)
			 : "r"(x0), "r"(x1), "r"(x2)
			 : "memory");
	return d0;
}

#endif /* __aarch64__ */

#endif /* KW_PARAVIRT_H */
