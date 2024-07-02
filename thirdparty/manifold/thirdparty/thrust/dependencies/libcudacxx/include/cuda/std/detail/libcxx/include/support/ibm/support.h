// -*- C++ -*-
//===----------------------- support/ibm/support.h ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SUPPORT_IBM_SUPPORT_H
#define _LIBCUDACXX_SUPPORT_IBM_SUPPORT_H

extern "builtin" int __popcnt4(unsigned int);
extern "builtin" int __popcnt8(unsigned long long);
extern "builtin" unsigned int __cnttz4(unsigned int);
extern "builtin" unsigned int __cnttz8(unsigned long long);
extern "builtin" unsigned int __cntlz4(unsigned int);
extern "builtin" unsigned int __cntlz8(unsigned long long);

// Builtin functions for counting population
#define __builtin_popcount(x) __popcnt4(x)
#define __builtin_popcountll(x) __popcnt8(x)
#if defined(__64BIT__)
#define __builtin_popcountl(x) __builtin_popcountll(x)
#else
#define __builtin_popcountl(x) __builtin_popcount(x)
#endif

// Builtin functions for counting trailing zeros
#define __builtin_ctz(x) __cnttz4(x)
#define __builtin_ctzll(x) __cnttz8(x)
#if defined(__64BIT__)
#define __builtin_ctzl(x) __builtin_ctzll(x)
#else
#define __builtin_ctzl(x) __builtin_ctz(x)
#endif

// Builtin functions for counting leading zeros
#define __builtin_clz(x) __cntlz4(x)
#define __builtin_clzll(x) __cntlz8(x)
#if defined(__64BIT__)
#define __builtin_clzl(x) __builtin_clzll(x)
#else
#define __builtin_clzl(x) __builtin_clz(x)
#endif

#if defined(__64BIT__)
#define __SIZE_WIDTH__ 64
#else
#define __SIZE_WIDTH__ 32
#endif

#endif // _LIBCUDACXX_SUPPORT_IBM_SUPPORT_H
