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

#include "mbedtls/build_info.h"
#include "alignment.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stddef.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif /* __ARM_NEON */

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

/** \def ARRAY_LENGTH
 * Return the number of elements of a static or stack array.
 *
 * \param array         A value of array (not pointer) type.
 *
 * \return The number of elements of the array.
 */
/* A correct implementation of ARRAY_LENGTH, but which silently gives
 * a nonsensical result if called with a pointer rather than an array. */
#define ARRAY_LENGTH_UNSAFE(array)            \
    (sizeof(array) / sizeof(*(array)))

#if defined(__GNUC__)
/* Test if arg and &(arg)[0] have the same type. This is true if arg is
 * an array but not if it's a pointer. */
#define IS_ARRAY_NOT_POINTER(arg)                                     \
    (!__builtin_types_compatible_p(__typeof__(arg),                \
                                   __typeof__(&(arg)[0])))
/* A compile-time constant with the value 0. If `const_expr` is not a
 * compile-time constant with a nonzero value, cause a compile-time error. */
#define STATIC_ASSERT_EXPR(const_expr)                                \
    (0 && sizeof(struct { unsigned int STATIC_ASSERT : 1 - 2 * !(const_expr); }))

/* Return the scalar value `value` (possibly promoted). This is a compile-time
 * constant if `value` is. `condition` must be a compile-time constant.
 * If `condition` is false, arrange to cause a compile-time error. */
#define STATIC_ASSERT_THEN_RETURN(condition, value)   \
    (STATIC_ASSERT_EXPR(condition) ? 0 : (value))

#define ARRAY_LENGTH(array)                                           \
    (STATIC_ASSERT_THEN_RETURN(IS_ARRAY_NOT_POINTER(array),         \
                               ARRAY_LENGTH_UNSAFE(array)))

#else
/* If we aren't sure the compiler supports our non-standard tricks,
 * fall back to the unsafe implementation. */
#define ARRAY_LENGTH(array) ARRAY_LENGTH_UNSAFE(array)
#endif
/** Allow library to access its structs' private members.
 *
 * Although structs defined in header files are publicly available,
 * their members are private and should not be accessed by the user.
 */
#define MBEDTLS_ALLOW_PRIVATE_ACCESS

/**
 * \brief       Securely zeroize a buffer then free it.
 *
 *              Similar to making consecutive calls to
 *              \c mbedtls_platform_zeroize() and \c mbedtls_free(), but has
 *              code size savings, and potential for optimisation in the future.
 *
 *              Guaranteed to be a no-op if \p buf is \c NULL and \p len is 0.
 *
 * \param buf   Buffer to be zeroized then freed.
 * \param len   Length of the buffer in bytes
 */
void mbedtls_zeroize_and_free(void *buf, size_t len);

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
#if defined(__ARM_NEON)
    for (; (i + 16) <= n; i += 16) {
        uint8x16_t v1 = vld1q_u8(a + i);
        uint8x16_t v2 = vld1q_u8(b + i);
        uint8x16_t x = veorq_u8(v1, v2);
        vst1q_u8(r + i, x);
    }
#elif defined(__amd64__) || defined(__x86_64__) || defined(__aarch64__)
    /* This codepath probably only makes sense on architectures with 64-bit registers */
    for (; (i + 8) <= n; i += 8) {
        uint64_t x = mbedtls_get_unaligned_uint64(a + i) ^ mbedtls_get_unaligned_uint64(b + i);
        mbedtls_put_unaligned_uint64(r + i, x);
    }
#else
    for (; (i + 4) <= n; i += 4) {
        uint32_t x = mbedtls_get_unaligned_uint32(a + i) ^ mbedtls_get_unaligned_uint32(b + i);
        mbedtls_put_unaligned_uint32(r + i, x);
    }
#endif
#endif
    for (; i < n; i++) {
        r[i] = a[i] ^ b[i];
    }
}

/**
 * Perform a fast block XOR operation, such that
 * r[i] = a[i] ^ b[i] where 0 <= i < n
 *
 * In some situations, this can perform better than mbedtls_xor (e.g., it's about 5%
 * better in AES-CBC).
 *
 * \param   r Pointer to result (buffer of at least \p n bytes). \p r
 *            may be equal to either \p a or \p b, but behaviour when
 *            it overlaps in other ways is undefined.
 * \param   a Pointer to input (buffer of at least \p n bytes)
 * \param   b Pointer to input (buffer of at least \p n bytes)
 * \param   n Number of bytes to process.
 */
static inline void mbedtls_xor_no_simd(unsigned char *r,
                                       const unsigned char *a,
                                       const unsigned char *b,
                                       size_t n)
{
    size_t i = 0;
#if defined(MBEDTLS_EFFICIENT_UNALIGNED_ACCESS)
#if defined(__amd64__) || defined(__x86_64__) || defined(__aarch64__)
    /* This codepath probably only makes sense on architectures with 64-bit registers */
    for (; (i + 8) <= n; i += 8) {
        uint64_t x = mbedtls_get_unaligned_uint64(a + i) ^ mbedtls_get_unaligned_uint64(b + i);
        mbedtls_put_unaligned_uint64(r + i, x);
    }
#else
    for (; (i + 4) <= n; i += 4) {
        uint32_t x = mbedtls_get_unaligned_uint32(a + i) ^ mbedtls_get_unaligned_uint32(b + i);
        mbedtls_put_unaligned_uint32(r + i, x);
    }
#endif
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
#if defined(__IAR_SYSTEMS_ICC__)
#define asm __asm
#else
#define asm __asm__
#endif
#endif
/* *INDENT-ON* */

/*
 * Define the constraint used for read-only pointer operands to aarch64 asm.
 *
 * This is normally the usual "r", but for aarch64_32 (aka ILP32,
 * as found in watchos), "p" is required to avoid warnings from clang.
 *
 * Note that clang does not recognise '+p' or '=p', and armclang
 * does not recognise 'p' at all. Therefore, to update a pointer from
 * aarch64 assembly, it is necessary to use something like:
 *
 * uintptr_t uptr = (uintptr_t) ptr;
 * asm( "ldr x4, [%x0], #8" ... : "+r" (uptr) : : )
 * ptr = (void*) uptr;
 *
 * Note that the "x" in "%x0" is neccessary; writing "%0" will cause warnings.
 */
#if defined(__aarch64__) && defined(MBEDTLS_HAVE_ASM)
#if UINTPTR_MAX == 0xfffffffful
/* ILP32: Specify the pointer operand slightly differently, as per #7787. */
#define MBEDTLS_ASM_AARCH64_PTR_CONSTRAINT "p"
#elif UINTPTR_MAX == 0xfffffffffffffffful
/* Normal case (64-bit pointers): use "r" as the constraint for pointer operands to asm */
#define MBEDTLS_ASM_AARCH64_PTR_CONSTRAINT "r"
#else
#error "Unrecognised pointer size for aarch64"
#endif
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

/* Define compiler branch hints */
#if defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#define MBEDTLS_LIKELY(x)       __builtin_expect(!!(x), 1)
#define MBEDTLS_UNLIKELY(x)     __builtin_expect(!!(x), 0)
#endif
#endif
#if !defined(MBEDTLS_LIKELY)
#define MBEDTLS_LIKELY(x)       x
#define MBEDTLS_UNLIKELY(x)     x
#endif

#if defined(__GNUC__) && !defined(__ARMCC_VERSION) && !defined(__clang__) \
    && !defined(__llvm__) && !defined(__INTEL_COMPILER)
/* Defined if the compiler really is gcc and not clang, etc */
#define MBEDTLS_COMPILER_IS_GCC
#endif

/* For gcc -Os, override with -O2 for a given function.
 *
 * This will not affect behaviour for other optimisation settings, e.g. -O0.
 */
#if defined(MBEDTLS_COMPILER_IS_GCC) && defined(__OPTIMIZE_SIZE__)
#define MBEDTLS_OPTIMIZE_FOR_PERFORMANCE __attribute__((optimize("-O2")))
#else
#define MBEDTLS_OPTIMIZE_FOR_PERFORMANCE
#endif

#endif /* MBEDTLS_LIBRARY_COMMON_H */
