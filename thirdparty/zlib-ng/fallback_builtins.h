#ifndef FALLBACK_BUILTINS_H
#define FALLBACK_BUILTINS_H

#if defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) || defined(_M_AMD64) || defined(_M_IA64) ||  defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC)

#include <intrin.h>

/* This is not a general purpose replacement for __builtin_ctz. The function expects that value is != 0.
 * Because of that assumption trailing_zero is not initialized and the return value is not checked.
 * Tzcnt and bsf give identical results except when input value is 0, therefore this can not be allowed.
 * If tzcnt instruction is not supported, the cpu will itself execute bsf instead.
 * Performance tzcnt/bsf is identical on Intel cpu, tzcnt is faster than bsf on AMD cpu.
 */
static __forceinline int __builtin_ctz(unsigned int value) {
    Assert(value != 0, "Invalid input value: 0");
# if defined(X86_FEATURES) && !(_MSC_VER < 1700)
    return (int)_tzcnt_u32(value);
# else
    unsigned long trailing_zero;
    _BitScanForward(&trailing_zero, value);
    return (int)trailing_zero;
# endif
}
#define HAVE_BUILTIN_CTZ

#ifdef _M_AMD64
/* This is not a general purpose replacement for __builtin_ctzll. The function expects that value is != 0.
 * Because of that assumption trailing_zero is not initialized and the return value is not checked.
 */
static __forceinline int __builtin_ctzll(unsigned long long value) {
    Assert(value != 0, "Invalid input value: 0");
# if defined(X86_FEATURES) && !(_MSC_VER < 1700)
    return (int)_tzcnt_u64(value);
# else
    unsigned long trailing_zero;
    _BitScanForward64(&trailing_zero, value);
    return (int)trailing_zero;
# endif
}
#define HAVE_BUILTIN_CTZLL
#endif // Microsoft AMD64

#endif // Microsoft AMD64/IA64/x86/ARM/ARM64 test
#endif // _MSC_VER & !clang

#endif // include guard FALLBACK_BUILTINS_H
