/**
 * \file alignment.h
 *
 * \brief Utility code for dealing with unaligned memory accesses
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

#ifndef MBEDTLS_LIBRARY_ALIGNMENT_H
#define MBEDTLS_LIBRARY_ALIGNMENT_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "mbedtls/build_info.h"

/*
 * Define MBEDTLS_EFFICIENT_UNALIGNED_ACCESS for architectures where unaligned memory
 * accesses are known to be efficient.
 *
 * All functions defined here will behave correctly regardless, but might be less
 * efficient when this is not defined.
 */
#if defined(__ARM_FEATURE_UNALIGNED) \
    || defined(__i386__) || defined(__amd64__) || defined(__x86_64__)
/*
 * __ARM_FEATURE_UNALIGNED is defined where appropriate by armcc, gcc 7, clang 9
 * (and later versions) for Arm v7 and later; all x86 platforms should have
 * efficient unaligned access.
 */
#define MBEDTLS_EFFICIENT_UNALIGNED_ACCESS
#endif

/**
 * Read the unsigned 16 bits integer from the given address, which need not
 * be aligned.
 *
 * \param   p pointer to 2 bytes of data
 * \return  Data at the given address
 */
inline uint16_t mbedtls_get_unaligned_uint16(const void *p)
{
    uint16_t r;
    memcpy(&r, p, sizeof(r));
    return r;
}

/**
 * Write the unsigned 16 bits integer to the given address, which need not
 * be aligned.
 *
 * \param   p pointer to 2 bytes of data
 * \param   x data to write
 */
inline void mbedtls_put_unaligned_uint16(void *p, uint16_t x)
{
    memcpy(p, &x, sizeof(x));
}

/**
 * Read the unsigned 32 bits integer from the given address, which need not
 * be aligned.
 *
 * \param   p pointer to 4 bytes of data
 * \return  Data at the given address
 */
inline uint32_t mbedtls_get_unaligned_uint32(const void *p)
{
    uint32_t r;
    memcpy(&r, p, sizeof(r));
    return r;
}

/**
 * Write the unsigned 32 bits integer to the given address, which need not
 * be aligned.
 *
 * \param   p pointer to 4 bytes of data
 * \param   x data to write
 */
inline void mbedtls_put_unaligned_uint32(void *p, uint32_t x)
{
    memcpy(p, &x, sizeof(x));
}

/**
 * Read the unsigned 64 bits integer from the given address, which need not
 * be aligned.
 *
 * \param   p pointer to 8 bytes of data
 * \return  Data at the given address
 */
inline uint64_t mbedtls_get_unaligned_uint64(const void *p)
{
    uint64_t r;
    memcpy(&r, p, sizeof(r));
    return r;
}

/**
 * Write the unsigned 64 bits integer to the given address, which need not
 * be aligned.
 *
 * \param   p pointer to 8 bytes of data
 * \param   x data to write
 */
inline void mbedtls_put_unaligned_uint64(void *p, uint64_t x)
{
    memcpy(p, &x, sizeof(x));
}

/** Byte Reading Macros
 *
 * Given a multi-byte integer \p x, MBEDTLS_BYTE_n retrieves the n-th
 * byte from x, where byte 0 is the least significant byte.
 */
#define MBEDTLS_BYTE_0(x) ((uint8_t) ((x)         & 0xff))
#define MBEDTLS_BYTE_1(x) ((uint8_t) (((x) >>  8) & 0xff))
#define MBEDTLS_BYTE_2(x) ((uint8_t) (((x) >> 16) & 0xff))
#define MBEDTLS_BYTE_3(x) ((uint8_t) (((x) >> 24) & 0xff))
#define MBEDTLS_BYTE_4(x) ((uint8_t) (((x) >> 32) & 0xff))
#define MBEDTLS_BYTE_5(x) ((uint8_t) (((x) >> 40) & 0xff))
#define MBEDTLS_BYTE_6(x) ((uint8_t) (((x) >> 48) & 0xff))
#define MBEDTLS_BYTE_7(x) ((uint8_t) (((x) >> 56) & 0xff))

/*
 * Detect GCC built-in byteswap routines
 */
#if defined(__GNUC__) && defined(__GNUC_PREREQ)
#if __GNUC_PREREQ(4, 8)
#define MBEDTLS_BSWAP16 __builtin_bswap16
#endif /* __GNUC_PREREQ(4,8) */
#if __GNUC_PREREQ(4, 3)
#define MBEDTLS_BSWAP32 __builtin_bswap32
#define MBEDTLS_BSWAP64 __builtin_bswap64
#endif /* __GNUC_PREREQ(4,3) */
#endif /* defined(__GNUC__) && defined(__GNUC_PREREQ) */

/*
 * Detect Clang built-in byteswap routines
 */
#if defined(__clang__) && defined(__has_builtin)
#if __has_builtin(__builtin_bswap16) && !defined(MBEDTLS_BSWAP16)
#define MBEDTLS_BSWAP16 __builtin_bswap16
#endif /* __has_builtin(__builtin_bswap16) */
#if __has_builtin(__builtin_bswap32) && !defined(MBEDTLS_BSWAP32)
#define MBEDTLS_BSWAP32 __builtin_bswap32
#endif /* __has_builtin(__builtin_bswap32) */
#if __has_builtin(__builtin_bswap64) && !defined(MBEDTLS_BSWAP64)
#define MBEDTLS_BSWAP64 __builtin_bswap64
#endif /* __has_builtin(__builtin_bswap64) */
#endif /* defined(__clang__) && defined(__has_builtin) */

/*
 * Detect MSVC built-in byteswap routines
 */
#if defined(_MSC_VER)
#if !defined(MBEDTLS_BSWAP16)
#define MBEDTLS_BSWAP16 _byteswap_ushort
#endif
#if !defined(MBEDTLS_BSWAP32)
#define MBEDTLS_BSWAP32 _byteswap_ulong
#endif
#if !defined(MBEDTLS_BSWAP64)
#define MBEDTLS_BSWAP64 _byteswap_uint64
#endif
#endif /* defined(_MSC_VER) */

/* Detect armcc built-in byteswap routine */
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 410000) && !defined(MBEDTLS_BSWAP32)
#define MBEDTLS_BSWAP32 __rev
#endif

/*
 * Where compiler built-ins are not present, fall back to C code that the
 * compiler may be able to detect and transform into the relevant bswap or
 * similar instruction.
 */
#if !defined(MBEDTLS_BSWAP16)
static inline uint16_t mbedtls_bswap16(uint16_t x)
{
    return
        (x & 0x00ff) << 8 |
        (x & 0xff00) >> 8;
}
#define MBEDTLS_BSWAP16 mbedtls_bswap16
#endif /* !defined(MBEDTLS_BSWAP16) */

#if !defined(MBEDTLS_BSWAP32)
static inline uint32_t mbedtls_bswap32(uint32_t x)
{
    return
        (x & 0x000000ff) << 24 |
        (x & 0x0000ff00) <<  8 |
        (x & 0x00ff0000) >>  8 |
        (x & 0xff000000) >> 24;
}
#define MBEDTLS_BSWAP32 mbedtls_bswap32
#endif /* !defined(MBEDTLS_BSWAP32) */

#if !defined(MBEDTLS_BSWAP64)
static inline uint64_t mbedtls_bswap64(uint64_t x)
{
    return
        (x & 0x00000000000000ffULL) << 56 |
        (x & 0x000000000000ff00ULL) << 40 |
        (x & 0x0000000000ff0000ULL) << 24 |
        (x & 0x00000000ff000000ULL) <<  8 |
        (x & 0x000000ff00000000ULL) >>  8 |
        (x & 0x0000ff0000000000ULL) >> 24 |
        (x & 0x00ff000000000000ULL) >> 40 |
        (x & 0xff00000000000000ULL) >> 56;
}
#define MBEDTLS_BSWAP64 mbedtls_bswap64
#endif /* !defined(MBEDTLS_BSWAP64) */

#if !defined(__BYTE_ORDER__)
static const uint16_t mbedtls_byte_order_detector = { 0x100 };
#define MBEDTLS_IS_BIG_ENDIAN (*((unsigned char *) (&mbedtls_byte_order_detector)) == 0x01)
#else
#define MBEDTLS_IS_BIG_ENDIAN ((__BYTE_ORDER__) == (__ORDER_BIG_ENDIAN__))
#endif /* !defined(__BYTE_ORDER__) */

/**
 * Get the unsigned 32 bits integer corresponding to four bytes in
 * big-endian order (MSB first).
 *
 * \param   data    Base address of the memory to get the four bytes from.
 * \param   offset  Offset from \p data of the first and most significant
 *                  byte of the four bytes to build the 32 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT32_BE(data, offset)                                \
    ((MBEDTLS_IS_BIG_ENDIAN)                                               \
        ? mbedtls_get_unaligned_uint32((data) + (offset))                  \
        : MBEDTLS_BSWAP32(mbedtls_get_unaligned_uint32((data) + (offset))) \
    )

/**
 * Put in memory a 32 bits unsigned integer in big-endian order.
 *
 * \param   n       32 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 32
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the most significant
 *                  byte of the 32 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT32_BE(n, data, offset)                                   \
    {                                                                            \
        if (MBEDTLS_IS_BIG_ENDIAN)                                               \
        {                                                                        \
            mbedtls_put_unaligned_uint32((data) + (offset), (uint32_t) (n));     \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            mbedtls_put_unaligned_uint32((data) + (offset), MBEDTLS_BSWAP32((uint32_t) (n))); \
        }                                                                        \
    }

/**
 * Get the unsigned 32 bits integer corresponding to four bytes in
 * little-endian order (LSB first).
 *
 * \param   data    Base address of the memory to get the four bytes from.
 * \param   offset  Offset from \p data of the first and least significant
 *                  byte of the four bytes to build the 32 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT32_LE(data, offset)                                \
    ((MBEDTLS_IS_BIG_ENDIAN)                                               \
        ? MBEDTLS_BSWAP32(mbedtls_get_unaligned_uint32((data) + (offset))) \
        : mbedtls_get_unaligned_uint32((data) + (offset))                  \
    )


/**
 * Put in memory a 32 bits unsigned integer in little-endian order.
 *
 * \param   n       32 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 32
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the least significant
 *                  byte of the 32 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT32_LE(n, data, offset)                                   \
    {                                                                            \
        if (MBEDTLS_IS_BIG_ENDIAN)                                               \
        {                                                                        \
            mbedtls_put_unaligned_uint32((data) + (offset), MBEDTLS_BSWAP32((uint32_t) (n))); \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            mbedtls_put_unaligned_uint32((data) + (offset), ((uint32_t) (n)));   \
        }                                                                        \
    }

/**
 * Get the unsigned 16 bits integer corresponding to two bytes in
 * little-endian order (LSB first).
 *
 * \param   data    Base address of the memory to get the two bytes from.
 * \param   offset  Offset from \p data of the first and least significant
 *                  byte of the two bytes to build the 16 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT16_LE(data, offset)                                \
    ((MBEDTLS_IS_BIG_ENDIAN)                                               \
        ? MBEDTLS_BSWAP16(mbedtls_get_unaligned_uint16((data) + (offset))) \
        : mbedtls_get_unaligned_uint16((data) + (offset))                  \
    )

/**
 * Put in memory a 16 bits unsigned integer in little-endian order.
 *
 * \param   n       16 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 16
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the least significant
 *                  byte of the 16 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT16_LE(n, data, offset)                                   \
    {                                                                            \
        if (MBEDTLS_IS_BIG_ENDIAN)                                               \
        {                                                                        \
            mbedtls_put_unaligned_uint16((data) + (offset), MBEDTLS_BSWAP16((uint16_t) (n))); \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            mbedtls_put_unaligned_uint16((data) + (offset), (uint16_t) (n));     \
        }                                                                        \
    }

/**
 * Get the unsigned 16 bits integer corresponding to two bytes in
 * big-endian order (MSB first).
 *
 * \param   data    Base address of the memory to get the two bytes from.
 * \param   offset  Offset from \p data of the first and most significant
 *                  byte of the two bytes to build the 16 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT16_BE(data, offset)                                \
    ((MBEDTLS_IS_BIG_ENDIAN)                                               \
        ? mbedtls_get_unaligned_uint16((data) + (offset))                  \
        : MBEDTLS_BSWAP16(mbedtls_get_unaligned_uint16((data) + (offset))) \
    )

/**
 * Put in memory a 16 bits unsigned integer in big-endian order.
 *
 * \param   n       16 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 16
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the most significant
 *                  byte of the 16 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT16_BE(n, data, offset)                                   \
    {                                                                            \
        if (MBEDTLS_IS_BIG_ENDIAN)                                               \
        {                                                                        \
            mbedtls_put_unaligned_uint16((data) + (offset), (uint16_t) (n));     \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            mbedtls_put_unaligned_uint16((data) + (offset), MBEDTLS_BSWAP16((uint16_t) (n))); \
        }                                                                        \
    }

/**
 * Get the unsigned 24 bits integer corresponding to three bytes in
 * big-endian order (MSB first).
 *
 * \param   data    Base address of the memory to get the three bytes from.
 * \param   offset  Offset from \p data of the first and most significant
 *                  byte of the three bytes to build the 24 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT24_BE(data, offset)        \
    (                                              \
        ((uint32_t) (data)[(offset)] << 16)        \
        | ((uint32_t) (data)[(offset) + 1] << 8)   \
        | ((uint32_t) (data)[(offset) + 2])        \
    )

/**
 * Put in memory a 24 bits unsigned integer in big-endian order.
 *
 * \param   n       24 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 24
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the most significant
 *                  byte of the 24 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT24_BE(n, data, offset)                \
    {                                                         \
        (data)[(offset)] = MBEDTLS_BYTE_2(n);                 \
        (data)[(offset) + 1] = MBEDTLS_BYTE_1(n);             \
        (data)[(offset) + 2] = MBEDTLS_BYTE_0(n);             \
    }

/**
 * Get the unsigned 24 bits integer corresponding to three bytes in
 * little-endian order (LSB first).
 *
 * \param   data    Base address of the memory to get the three bytes from.
 * \param   offset  Offset from \p data of the first and least significant
 *                  byte of the three bytes to build the 24 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT24_LE(data, offset)               \
    (                                                     \
        ((uint32_t) (data)[(offset)])                     \
        | ((uint32_t) (data)[(offset) + 1] <<  8)         \
        | ((uint32_t) (data)[(offset) + 2] << 16)         \
    )

/**
 * Put in memory a 24 bits unsigned integer in little-endian order.
 *
 * \param   n       24 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 24
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the least significant
 *                  byte of the 24 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT24_LE(n, data, offset)                \
    {                                                         \
        (data)[(offset)] = MBEDTLS_BYTE_0(n);                 \
        (data)[(offset) + 1] = MBEDTLS_BYTE_1(n);             \
        (data)[(offset) + 2] = MBEDTLS_BYTE_2(n);             \
    }

/**
 * Get the unsigned 64 bits integer corresponding to eight bytes in
 * big-endian order (MSB first).
 *
 * \param   data    Base address of the memory to get the eight bytes from.
 * \param   offset  Offset from \p data of the first and most significant
 *                  byte of the eight bytes to build the 64 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT64_BE(data, offset)                                \
    ((MBEDTLS_IS_BIG_ENDIAN)                                               \
        ? mbedtls_get_unaligned_uint64((data) + (offset))                  \
        : MBEDTLS_BSWAP64(mbedtls_get_unaligned_uint64((data) + (offset))) \
    )

/**
 * Put in memory a 64 bits unsigned integer in big-endian order.
 *
 * \param   n       64 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 64
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the most significant
 *                  byte of the 64 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT64_BE(n, data, offset)                                   \
    {                                                                            \
        if (MBEDTLS_IS_BIG_ENDIAN)                                               \
        {                                                                        \
            mbedtls_put_unaligned_uint64((data) + (offset), (uint64_t) (n));     \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            mbedtls_put_unaligned_uint64((data) + (offset), MBEDTLS_BSWAP64((uint64_t) (n))); \
        }                                                                        \
    }

/**
 * Get the unsigned 64 bits integer corresponding to eight bytes in
 * little-endian order (LSB first).
 *
 * \param   data    Base address of the memory to get the eight bytes from.
 * \param   offset  Offset from \p data of the first and least significant
 *                  byte of the eight bytes to build the 64 bits unsigned
 *                  integer from.
 */
#define MBEDTLS_GET_UINT64_LE(data, offset)                                \
    ((MBEDTLS_IS_BIG_ENDIAN)                                               \
        ? MBEDTLS_BSWAP64(mbedtls_get_unaligned_uint64((data) + (offset))) \
        : mbedtls_get_unaligned_uint64((data) + (offset))                  \
    )

/**
 * Put in memory a 64 bits unsigned integer in little-endian order.
 *
 * \param   n       64 bits unsigned integer to put in memory.
 * \param   data    Base address of the memory where to put the 64
 *                  bits unsigned integer in.
 * \param   offset  Offset from \p data where to put the least significant
 *                  byte of the 64 bits unsigned integer \p n.
 */
#define MBEDTLS_PUT_UINT64_LE(n, data, offset)                                   \
    {                                                                            \
        if (MBEDTLS_IS_BIG_ENDIAN)                                               \
        {                                                                        \
            mbedtls_put_unaligned_uint64((data) + (offset), MBEDTLS_BSWAP64((uint64_t) (n))); \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            mbedtls_put_unaligned_uint64((data) + (offset), (uint64_t) (n));     \
        }                                                                        \
    }

#endif /* MBEDTLS_LIBRARY_ALIGNMENT_H */
