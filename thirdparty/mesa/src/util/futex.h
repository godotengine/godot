/*
 * Copyright © 2015 Intel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UTIL_FUTEX_H
#define UTIL_FUTEX_H

#if defined(HAVE_LINUX_FUTEX_H)
#define UTIL_FUTEX_SUPPORTED 1
#elif defined(__FreeBSD__)
#define UTIL_FUTEX_SUPPORTED 1
#elif defined(__OpenBSD__)
#define UTIL_FUTEX_SUPPORTED 1
#elif defined(_WIN32) && !defined(WINDOWS_NO_FUTEX)
#define UTIL_FUTEX_SUPPORTED 1
#else
#define UTIL_FUTEX_SUPPORTED 0
#endif

#if UTIL_FUTEX_SUPPORTED
#include <stdint.h>
#include <c11/time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if UTIL_FUTEX_SUPPORTED
int futex_wake(uint32_t *addr, int count);
int futex_wait(uint32_t *addr, int32_t value, const struct timespec *timeout);
#endif

#ifdef __cplusplus
}
#endif

#endif /* UTIL_FUTEX_H */
