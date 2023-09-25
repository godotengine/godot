/**
 * \file common.h
 *
 * \brief Utility macros for internal use in the library
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef MBEDTLS_LIBRARY_COMMON_H
#define MBEDTLS_LIBRARY_COMMON_H

#include "mbedtls/build_info.h"
#include "alignment.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stddef.h>

/** Helper to define a function as static except when building invasive tests.
 *
 * If a function is only used inside its own source file and should be
 * declared `static` to allow the compiler to optimize for code size,
 * but that function has unit tests, define it with
 * ```
 * MBEDTLS_STATIC_TESTABLE int mbedtls_foo(...) { ... }
 * ```
 * and declare it in a header in the `library/` directory with
 * ```
 * #if defined(MBEDTLS_TEST_HOOKS)
 * int mbedtls_foo(...);
 * #endif
 * ```
 */
#if defined(MBEDTLS_TEST_HOOKS)
#define MBEDTLS_STATIC_TESTABLE
#else
#define MBEDTLS_STATIC_TESTABLE static
#endif

#if defined(MBEDTLS_TEST_HOOKS)
extern void (*mbedtls_test_hook_test_fail)(const char *test, int line, const char *file);
#define MBEDTLS_TEST_HOOK_TEST_ASSERT(TEST) \
    do { \
        if ((!(TEST)) && ((*mbedtls_test_hook_test_fail) != NULL)) \
        { \
            (*mbedtls_test_hook_test_fail)( #TEST, __LINE__, __FILE__); \
        } \
    } while (0)
#else
#define MBEDTLS_TEST_HOOK_TEST_ASSERT(TEST)
#endif /* defined(MBEDTLS_TEST_HOOKS) */

/** Allow library to access its structs' private members.
 *
 * Although structs defined in header files are publicly available,
 * their members are private and should not be accessed by the user.
 */
#define MBEDTLS_ALLOW_PRIVATE_ACCESS

/** Return an offset into a buffer.
 *
 * This is just the addition of an offset to a pointer, except that this
 * function also accepts an offset of 0 into a buffer whose pointer is null.
 * (`p + n` has undefined behavior when `p` is null, even when `n == 0`.
 * A null pointer is a valid buffer pointer when the size is 0, for example
 * as the result of `malloc(0)` on some platforms.)
 *
 * \param p     Pointer to a buffer of at least n bytes.
 *              This may be \p NULL if \p n is zero.
 * \param n     An offset in bytes.
 * \return      Pointer to offset \p n in the buffer \p p.
 *              Note that this is only a valid pointer if the size of the
 *              buffer is at least \p n + 1.
 */
static inline unsigned char *mbedtls_buffer_offset(
    unsigned char *p, size_t n)
{
    return p == NULL ? NULL : p + n;
}

/** Return an offset into a read-only buffer.
 *
 * Similar to mbedtls_buffer_offset(), but for const pointers.
 *
 * \param p     Pointer to a buffer of at least n bytes.
 *              This may be \p NULL if \p n is zero.
 * \param n     An offset in bytes.
 * \return      Pointer to offset \p n in the buffer \p p.
 *              Note that this is only a valid pointer if the size of the
 *              buffer is at least \p n + 1.
 */
static inline const unsigned char *mbedtls_buffer_offset_const(
    const unsigned char *p, size_t n)
{
    return p == NULL ? NULL : p + n;
}

/**
 * Perform a fast block XOR operation, such that
 * r[i] = a[i] ^ b[i] where 0 <= i < n
 *
 * \param   r Pointer to result (buffer of at least \p n bytes). \p r
 *            may be equal to either \p a or \p b, but behaviour when
 *            it overlaps in other ways is undefined.
 * \param   a Pointer to input (buffer of at least \p n bytes)
 * \param   b Pointer to input (buffer of at least \p n bytes)
 * \param   n Number of bytes to process.
 */
inline void mbedtls_xor(unsigned char *r, const unsigned char *a, const unsigned char *b, size_t n)
{
    size_t i = 0;
#if defined(MBEDTLS_EFFICIENT_UNALIGNED_ACCESS)
    for (; (i + 4) <= n; i += 4) {
        uint32_t x = mbedtls_get_unaligned_uint32(a + i) ^ mbedtls_get_unaligned_uint32(b + i);
        mbedtls_put_unaligned_uint32(r + i, x);
    }
#endif
    for (; i < n; i++) {
        r[i] = a[i] ^ b[i];
    }
}

/* Fix MSVC C99 compatible issue
 *      MSVC support __func__ from visual studio 2015( 1900 )
 *      Use MSVC predefine macro to avoid name check fail.
 */
#if (defined(_MSC_VER) && (_MSC_VER <= 1900))
#define /*no-check-names*/ __func__ __FUNCTION__
#endif

/* Define `asm` for compilers which don't define it. */
/* *INDENT-OFF* */
#ifndef asm
#define asm __asm__
#endif
/* *INDENT-ON* */

/* Always provide a static assert macro, so it can be used unconditionally.
 * It will expand to nothing on some systems.
 * Can be used outside functions (but don't add a trailing ';' in that case:
 * the semicolon is included here to avoid triggering -Wextra-semi when
 * MBEDTLS_STATIC_ASSERT() expands to nothing).
 * Can't use the C11-style `defined(static_assert)` on FreeBSD, since it
 * defines static_assert even with -std=c99, but then complains about it.
 */
#if defined(static_assert) && !defined(__FreeBSD__)
#define MBEDTLS_STATIC_ASSERT(expr, msg)    static_assert(expr, msg);
#else
#define MBEDTLS_STATIC_ASSERT(expr, msg)
#endif

#endif /* MBEDTLS_LIBRARY_COMMON_H */
