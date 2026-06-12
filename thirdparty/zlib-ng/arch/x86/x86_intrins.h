#ifndef X86_INTRINS_H
#define X86_INTRINS_H

/* Unfortunately GCC didn't support these things until version 10.
 * Similarly, AppleClang didn't support them in Xcode 9.2 but did in 9.3.
 */
#ifdef __AVX2__
#include <immintrin.h>

#if (!defined(__clang__) && !defined(__NVCOMPILER) && defined(__GNUC__) && __GNUC__ < 10) \
    || (defined(__apple_build_version__) && __apple_build_version__ < 9020039)
static inline __m256i _mm256_zextsi128_si256(__m128i a) {
    __m128i r;
    __asm__ volatile ("vmovdqa %1,%0" : "=x" (r) : "x" (a));
    return _mm256_castsi128_si256(r);
}

#ifdef __AVX512F__
static inline __m512i _mm512_zextsi128_si512(__m128i a) {
    __m128i r;
    __asm__ volatile ("vmovdqa %1,%0" : "=x" (r) : "x" (a));
    return _mm512_castsi128_si512(r);
}
#endif // __AVX512F__
#endif // gcc/AppleClang version test

#endif // __AVX2__

/* GCC <9 is missing some AVX512 intrinsics.
 */
#ifdef __AVX512F__
#if (!defined(__clang__) && !defined(__NVCOMPILER) && defined(__GNUC__) && __GNUC__ < 9)
#include <immintrin.h>

#define PACK(c0, c1, c2, c3) (((int)(unsigned char)(c0) << 24) | ((int)(unsigned char)(c1) << 16) | \
                              ((int)(unsigned char)(c2) << 8) | ((int)(unsigned char)(c3)))

static inline __m512i _mm512_set_epi8(char __q63, char __q62, char __q61, char __q60,
                                      char __q59, char __q58, char __q57, char __q56,
                                      char __q55, char __q54, char __q53, char __q52,
                                      char __q51, char __q50, char __q49, char __q48,
                                      char __q47, char __q46, char __q45, char __q44,
                                      char __q43, char __q42, char __q41, char __q40,
                                      char __q39, char __q38, char __q37, char __q36,
                                      char __q35, char __q34, char __q33, char __q32,
                                      char __q31, char __q30, char __q29, char __q28,
                                      char __q27, char __q26, char __q25, char __q24,
                                      char __q23, char __q22, char __q21, char __q20,
                                      char __q19, char __q18, char __q17, char __q16,
                                      char __q15, char __q14, char __q13, char __q12,
                                      char __q11, char __q10, char __q09, char __q08,
                                      char __q07, char __q06, char __q05, char __q04,
                                      char __q03, char __q02, char __q01, char __q00) {
    return _mm512_set_epi32(PACK(__q63, __q62, __q61, __q60), PACK(__q59, __q58, __q57, __q56),
                            PACK(__q55, __q54, __q53, __q52), PACK(__q51, __q50, __q49, __q48),
                            PACK(__q47, __q46, __q45, __q44), PACK(__q43, __q42, __q41, __q40),
                            PACK(__q39, __q38, __q37, __q36), PACK(__q35, __q34, __q33, __q32),
                            PACK(__q31, __q30, __q29, __q28), PACK(__q27, __q26, __q25, __q24),
                            PACK(__q23, __q22, __q21, __q20), PACK(__q19, __q18, __q17, __q16),
                            PACK(__q15, __q14, __q13, __q12), PACK(__q11, __q10, __q09, __q08),
                            PACK(__q07, __q06, __q05, __q04), PACK(__q03, __q02, __q01, __q00));
}

#undef PACK

#endif // gcc version test
#endif // __AVX512F__

/* Missing zero-extension AVX and AVX512 intrinsics.
 * Fixed in Microsoft Visual Studio 2017 version 15.7
 * https://developercommunity.visualstudio.com/t/missing-zero-extension-avx-and-avx512-intrinsics/175737
 */
#if defined(_MSC_VER) && _MSC_VER < 1914
#ifdef __AVX2__
static inline __m256i _mm256_zextsi128_si256(__m128i a) {
    return _mm256_inserti128_si256(_mm256_setzero_si256(), a, 0);
}
#endif // __AVX2__

#ifdef __AVX512F__
static inline __m512i _mm512_zextsi128_si512(__m128i a) {
    return _mm512_inserti32x4(_mm512_setzero_si512(), a, 0);
}
#endif // __AVX512F__
#endif // defined(_MSC_VER) && _MSC_VER < 1914

/* Visual C++ toolchains before v142 have constant overflow in AVX512 intrinsics */
#if defined(_MSC_VER) && defined(__AVX512F__) && !defined(_MM_K0_REG8)
#  undef _mm512_extracti32x4_epi32
#  define _mm512_extracti32x4_epi32(v1, e1) _mm512_maskz_extracti32x4_epi32(UINT8_MAX, v1, e1)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
/* For whatever reason this intrinsic is 64 bit only with MSVC?
 * While we don't have 64 bit GPRs, it should at least be able to move it to stack
 * or shuffle it over 2 registers */
#if !defined(_M_AMD64)
/* So, while we can't move directly to a GPR, hopefully this move to
 * a stack resident variable doesn't equate to something awful */
static inline int64_t _mm_cvtsi128_si64(__m128i a) {
    union { __m128i v; int64_t i; } u;
    u.v = a;
    return u.i;
}

static inline __m128i _mm_cvtsi64_si128(int64_t a) {
   return _mm_set_epi64x(0, a);
}
#endif
#endif

#if defined(__GNUC__) && defined(__i386__) && !defined(__clang__)
static inline int64_t _mm_cvtsi128_si64(__m128i a) {
    union { __m128i v; int64_t i; } u;
    u.v = a;
    return u.i;
}
#define _mm_cvtsi64_si128(a) _mm_set_epi64x(0, a)
#endif

#endif // include guard X86_INTRINS_H
