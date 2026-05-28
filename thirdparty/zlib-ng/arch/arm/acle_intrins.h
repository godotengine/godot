#ifndef ARM_ACLE_INTRINS_H
#define ARM_ACLE_INTRINS_H

#include <stdint.h>
#ifdef _MSC_VER
#  include <intrin.h>
#elif defined(HAVE_ARM_ACLE_H)
#  include <arm_acle.h>
#endif

#ifdef ARM_CRC32
#if defined(__aarch64__)
#  define Z_TARGET_CRC Z_TARGET("+crc")
#else
#  define Z_TARGET_CRC
#endif

#if !defined(ARM_CRC32_INTRIN) && !defined(_MSC_VER)
#ifdef __aarch64__
static inline uint32_t __crc32b(uint32_t __a, uint8_t __b) {
    uint32_t __c;
    __asm__("crc32b %w0, %w1, %w2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}

static inline uint32_t __crc32h(uint32_t __a, uint16_t __b) {
    uint32_t __c;
    __asm__("crc32h %w0, %w1, %w2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}

static inline uint32_t __crc32w(uint32_t __a, uint32_t __b) {
    uint32_t __c;
    __asm__("crc32w %w0, %w1, %w2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}

static inline uint32_t __crc32d(uint32_t __a, uint64_t __b) {
    uint32_t __c;
    __asm__("crc32x %w0, %w1, %x2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}
#else
static inline uint32_t __crc32b(uint32_t __a, uint8_t __b) {
    uint32_t __c;
    __asm__("crc32b %0, %1, %2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}

static inline uint32_t __crc32h(uint32_t __a, uint16_t __b) {
    uint32_t __c;
    __asm__("crc32h %0, %1, %2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}

static inline uint32_t __crc32w(uint32_t __a, uint32_t __b) {
    uint32_t __c;
    __asm__("crc32w %0, %1, %2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}

static inline uint32_t __crc32d(uint32_t __a, uint64_t __b) {
    return __crc32w (__crc32w (__a, __b & 0xffffffffULL), __b >> 32);
}
#endif
#endif
#endif

#ifdef ARM_SIMD
#ifdef _MSC_VER
typedef uint32_t uint16x2_t;

#define __uqsub16 _arm_uqsub16
#elif !defined(ARM_SIMD_INTRIN)
typedef uint32_t uint16x2_t;

static inline uint16x2_t __uqsub16(uint16x2_t __a, uint16x2_t __b) {
    uint16x2_t __c;
    __asm__("uqsub16 %0, %1, %2" : "=r" (__c) : "r"(__a), "r"(__b));
    return __c;
}
#endif
#endif

#endif // include guard ARM_ACLE_INTRINS_H
