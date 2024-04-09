/**
 * \file common.h
 *
 * \brief Utility macros for internal use in the library
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_LIBRARY_COMMON_H
#define MBEDTLS_LIBRARY_COMMON_H

#if defined(MBEDTLS_CONFIG_FILE)
#include MBEDTLS_CONFIG_FILE
#else
#include "mbedtls/config.h"
#endif

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

/* Define `inline` on some non-C99-compliant compilers. */
#if (defined(__ARMCC_VERSION) || defined(_MSC_VER)) && \
    !defined(inline) && !defined(__cplusplus)
#define inline __inline
#endif

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

/** Byte Reading Macros
 *
 * Given a multi-byte integer \p x, MBEDTLS_BYTE_n retrieves the n-th
 * byte from x, where byte 0 is the least significant byte.
 */
#define MBEDTLS_BYTE_0(x) ((uint8_t) ((x)         & 0xff))
#define MBEDTLS_BYTE_1(x) ((uint8_t) (((x) >> 8) & 0xff))
#define MBEDTLS_BYTE_2(x) ((uint8_t) (((x) >> 16) & 0xff))
#define MBEDTLS_BYTE_3(x) ((uint8_t) (((x) >> 24) & 0xff))
#define MBEDTLS_BYTE_4(x) ((uint8_t) (((x) >> 32) & 0xff))
#define MBEDTLS_BYTE_5(x) ((uint8_t) (((x) >> 40) & 0xff))
#define MBEDTLS_BYTE_6(x) ((uint8_t) (((x) >> 48) & 0xff))
#define MBEDTLS_BYTE_7(x) ((uint8_t) (((x) >> 56) & 0xff))

/**
 * Get the unsigned 32 bits integer corresponding to four bytes in
 * big-endian order (MSB first).
 *
 * \param   data    Base address of the memory to get the four bytes from.
 * \param   offset  Offset from \p base of the first and most significant
 *                  byte of the four bytes to build the 32 bits unsigned
 *                  integer from.
 */
#ifndef MBEDTLS_GET_UINT32_BE
#define MBEDTLS_GET_UINT32_BE(data, offset)                  \
    (                                                           \
        ((uint32_t) (data)[(offset)] << 24)         \
        | ((uint32_t) (data)[(offset) + 1] << 16)         \
        | ((uint32_t) (data)[(offset) + 2] <<  8)         \
        | ((uint32_t) (data)[(offset) + 3])         \
    )
#endif

/**
 * Put in memory a 32 bits unsigned integer in big-endian order.
 *
 * \param   n       32 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 32
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p base where to put the most significant
 *                  byte of the 32 bits unsigned integer \p n.
 */
#ifndef MBEDTLS_PUT_UINT32_BE
#define MBEDTLS_PUT_UINT32_BE(n, data, offset)                \
    {                                                               \
        (data)[(offset)] = MBEDTLS_BYTE_3(n);             \
        (data)[(offset) + 1] = MBEDTLS_BYTE_2(n);             \
        (data)[(offset) + 2] = MBEDTLS_BYTE_1(n);             \
        (data)[(offset) + 3] = MBEDTLS_BYTE_0(n);             \
    }
#endif

/**
 * Get the unsigned 32 bits integer corresponding to four bytes in
 * little-endian order (LSB first).
 *
 * \param   data    Base address of the memory to get the four bytes from.
 * \param   offset  Offset from \p base of the first and least significant
 *                  byte of the four bytes to build the 32 bits unsigned
 *                  integer from.
 */
#ifndef MBEDTLS_GET_UINT32_LE
#define MBEDTLS_GET_UINT32_LE(data, offset)                   \
    (                                                           \
        ((uint32_t) (data)[(offset)])         \
        | ((uint32_t) (data)[(offset) + 1] <<  8)         \
        | ((uint32_t) (data)[(offset) + 2] << 16)         \
        | ((uint32_t) (data)[(offset) + 3] << 24)         \
    )
#endif

/**
 * Put in memory a 32 bits unsigned integer in little-endian order.
 *
 * \param   n       32 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 32
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p base where to put the least significant
 *                  byte of the 32 bits unsigned integer \p n.
 */
#ifndef MBEDTLS_PUT_UINT32_LE
#define MBEDTLS_PUT_UINT32_LE(n, data, offset)                \
    {                                                               \
        (data)[(offset)] = MBEDTLS_BYTE_0(n);             \
        (data)[(offset) + 1] = MBEDTLS_BYTE_1(n);             \
        (data)[(offset) + 2] = MBEDTLS_BYTE_2(n);             \
        (data)[(offset) + 3] = MBEDTLS_BYTE_3(n);             \
    }
#endif

/**
 * Get the unsigned 16 bits integer corresponding to two bytes in
 * little-endian order (LSB first).
 *
 * \param   data    Base address of the memory to get the two bytes from.
 * \param   offset  Offset from \p base of the first and least significant
 *                  byte of the two bytes to build the 16 bits unsigned
 *                  integer from.
 */
#ifndef MBEDTLS_GET_UINT16_LE
#define MBEDTLS_GET_UINT16_LE(data, offset)                   \
    (                                                           \
        ((uint16_t) (data)[(offset)])         \
        | ((uint16_t) (data)[(offset) + 1] <<  8)         \
    )
#endif

/**
 * Put in memory a 16 bits unsigned integer in little-endian order.
 *
 * \param   n       16 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 16
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p base where to put the least significant
 *                  byte of the 16 bits unsigned integer \p n.
 */
#ifndef MBEDTLS_PUT_UINT16_LE
#define MBEDTLS_PUT_UINT16_LE(n, data, offset)                \
    {                                                               \
        (data)[(offset)] = MBEDTLS_BYTE_0(n);             \
        (data)[(offset) + 1] = MBEDTLS_BYTE_1(n);             \
    }
#endif

/**
 * Get the unsigned 16 bits integer corresponding to two bytes in
 * big-endian order (MSB first).
 *
 * \param   data    Base address of the memory to get the two bytes from.
 * \param   offset  Offset from \p base of the first and most significant
 *                  byte of the two bytes to build the 16 bits unsigned
 *                  integer from.
 */
#ifndef MBEDTLS_GET_UINT16_BE
#define MBEDTLS_GET_UINT16_BE(data, offset)                   \
    (                                                           \
        ((uint16_t) (data)[(offset)] << 8)          \
        | ((uint16_t) (data)[(offset) + 1])          \
    )
#endif

/**
 * Put in memory a 16 bits unsigned integer in big-endian order.
 *
 * \param   n       16 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 16
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p base where to put the most significant
 *                  byte of the 16 bits unsigned integer \p n.
 */
#ifndef MBEDTLS_PUT_UINT16_BE
#define MBEDTLS_PUT_UINT16_BE(n, data, offset)                \
    {                                                               \
        (data)[(offset)] = MBEDTLS_BYTE_1(n);             \
        (data)[(offset) + 1] = MBEDTLS_BYTE_0(n);             \
    }
#endif

/**
 * Get the unsigned 64 bits integer corresponding to eight bytes in
 * big-endian order (MSB first).
 *
 * \param   data    Base address of the memory to get the eight bytes from.
 * \param   offset  Offset from \p base of the first and most significant
 *                  byte of the eight bytes to build the 64 bits unsigned
 *                  integer from.
 */
#ifndef MBEDTLS_GET_UINT64_BE
#define MBEDTLS_GET_UINT64_BE(data, offset)                   \
    (                                                           \
        ((uint64_t) (data)[(offset)] << 56)         \
        | ((uint64_t) (data)[(offset) + 1] << 48)         \
        | ((uint64_t) (data)[(offset) + 2] << 40)         \
        | ((uint64_t) (data)[(offset) + 3] << 32)         \
        | ((uint64_t) (data)[(offset) + 4] << 24)         \
        | ((uint64_t) (data)[(offset) + 5] << 16)         \
        | ((uint64_t) (data)[(offset) + 6] <<  8)         \
        | ((uint64_t) (data)[(offset) + 7])         \
    )
#endif

/**
 * Put in memory a 64 bits unsigned integer in big-endian order.
 *
 * \param   n       64 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 64
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p base where to put the most significant
 *                  byte of the 64 bits unsigned integer \p n.
 */
#ifndef MBEDTLS_PUT_UINT64_BE
#define MBEDTLS_PUT_UINT64_BE(n, data, offset)                \
    {                                                               \
        (data)[(offset)] = MBEDTLS_BYTE_7(n);             \
        (data)[(offset) + 1] = MBEDTLS_BYTE_6(n);             \
        (data)[(offset) + 2] = MBEDTLS_BYTE_5(n);             \
        (data)[(offset) + 3] = MBEDTLS_BYTE_4(n);             \
        (data)[(offset) + 4] = MBEDTLS_BYTE_3(n);             \
        (data)[(offset) + 5] = MBEDTLS_BYTE_2(n);             \
        (data)[(offset) + 6] = MBEDTLS_BYTE_1(n);             \
        (data)[(offset) + 7] = MBEDTLS_BYTE_0(n);             \
    }
#endif

/**
 * Get the unsigned 64 bits integer corresponding to eight bytes in
 * little-endian order (LSB first).
 *
 * \param   data    Base address of the memory to get the eight bytes from.
 * \param   offset  Offset from \p base of the first and least significant
 *                  byte of the eight bytes to build the 64 bits unsigned
 *                  integer from.
 */
#ifndef MBEDTLS_GET_UINT64_LE
#define MBEDTLS_GET_UINT64_LE(data, offset)                   \
    (                                                           \
        ((uint64_t) (data)[(offset) + 7] << 56)         \
        | ((uint64_t) (data)[(offset) + 6] << 48)         \
        | ((uint64_t) (data)[(offset) + 5] << 40)         \
        | ((uint64_t) (data)[(offset) + 4] << 32)         \
        | ((uint64_t) (data)[(offset) + 3] << 24)         \
        | ((uint64_t) (data)[(offset) + 2] << 16)         \
        | ((uint64_t) (data)[(offset) + 1] <<  8)         \
        | ((uint64_t) (data)[(offset)])         \
    )
#endif

/**
 * Put in memory a 64 bits unsigned integer in little-endian order.
 *
 * \param   n       64 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 64
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p base where to put the least significant
 *                  byte of the 64 bits unsigned integer \p n.
 */
#ifndef MBEDTLS_PUT_UINT64_LE
#define MBEDTLS_PUT_UINT64_LE(n, data, offset)                \
    {                                                               \
        (data)[(offset)] = MBEDTLS_BYTE_0(n);             \
        (data)[(offset) + 1] = MBEDTLS_BYTE_1(n);             \
        (data)[(offset) + 2] = MBEDTLS_BYTE_2(n);             \
        (data)[(offset) + 3] = MBEDTLS_BYTE_3(n);             \
        (data)[(offset) + 4] = MBEDTLS_BYTE_4(n);             \
        (data)[(offset) + 5] = MBEDTLS_BYTE_5(n);             \
        (data)[(offset) + 6] = MBEDTLS_BYTE_6(n);             \
        (data)[(offset) + 7] = MBEDTLS_BYTE_7(n);             \
    }
#endif

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

/* Suppress compiler warnings for unused functions and variables. */
#if !defined(MBEDTLS_MAYBE_UNUSED) && defined(__has_attribute)
#    if __has_attribute(unused)
#        define MBEDTLS_MAYBE_UNUSED __attribute__((unused))
#    endif
#endif
#if !defined(MBEDTLS_MAYBE_UNUSED) && defined(__GNUC__)
#    define MBEDTLS_MAYBE_UNUSED __attribute__((unused))
#endif
#if !defined(MBEDTLS_MAYBE_UNUSED) && defined(__IAR_SYSTEMS_ICC__) && defined(__VER__)
/* IAR does support __attribute__((unused)), but only if the -e flag (extended language support)
 * is given; the pragma always works.
 * Unfortunately the pragma affects the rest of the file where it is used, but this is harmless.
 * Check for version 5.2 or later - this pragma may be supported by earlier versions, but I wasn't
 * able to find documentation).
 */
#    if (__VER__ >= 5020000)
#        define MBEDTLS_MAYBE_UNUSED _Pragma("diag_suppress=Pe177")
#    endif
#endif
#if !defined(MBEDTLS_MAYBE_UNUSED) && defined(_MSC_VER)
#    define MBEDTLS_MAYBE_UNUSED __pragma(warning(suppress:4189))
#endif
#if !defined(MBEDTLS_MAYBE_UNUSED)
#    define MBEDTLS_MAYBE_UNUSED
#endif

#endif /* MBEDTLS_LIBRARY_COMMON_H */
