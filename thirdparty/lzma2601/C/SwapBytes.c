/* SwapBytes.c -- Byte Swap conversion filter
2024-03-01 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include "Compiler.h"
#include "CpuArch.h"
#include "RotateDefs.h"
#include "SwapBytes.h"

typedef UInt16 CSwapUInt16;
typedef UInt32 CSwapUInt32;

// #define k_SwapBytes_Mode_BASE   0

#ifdef MY_CPU_X86_OR_AMD64

#define k_SwapBytes_Mode_SSE2   1
#define k_SwapBytes_Mode_SSSE3  2
#define k_SwapBytes_Mode_AVX2   3

  // #if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1900)
  #if defined(__clang__) && (__clang_major__ >= 4) \
      || defined(Z7_GCC_VERSION) && (Z7_GCC_VERSION >= 40701)
      #define k_SwapBytes_Mode_MAX  k_SwapBytes_Mode_AVX2
      #define SWAP_ATTRIB_SSE2  __attribute__((__target__("sse2")))
      #define SWAP_ATTRIB_SSSE3 __attribute__((__target__("ssse3")))
      #define SWAP_ATTRIB_AVX2  __attribute__((__target__("avx2")))
  #elif defined(_MSC_VER)
    #if (_MSC_VER == 1900)
      #pragma warning(disable : 4752) // found Intel(R) Advanced Vector Extensions; consider using /arch:AVX
    #endif
    #if (_MSC_VER >= 1900)
      #define k_SwapBytes_Mode_MAX  k_SwapBytes_Mode_AVX2
    #elif (_MSC_VER >= 1500)  // (VS2008)
      #define k_SwapBytes_Mode_MAX  k_SwapBytes_Mode_SSSE3
    #elif (_MSC_VER >= 1310)  // (VS2003)
      #define k_SwapBytes_Mode_MAX  k_SwapBytes_Mode_SSE2
    #endif
  #endif // _MSC_VER

/*
// for debug
#ifdef k_SwapBytes_Mode_MAX
#undef k_SwapBytes_Mode_MAX
#endif
*/

#ifndef k_SwapBytes_Mode_MAX
#define k_SwapBytes_Mode_MAX 0
#endif

#if (k_SwapBytes_Mode_MAX != 0) && defined(MY_CPU_AMD64)
  #define k_SwapBytes_Mode_MIN  k_SwapBytes_Mode_SSE2
#else
  #define k_SwapBytes_Mode_MIN  0
#endif

#if (k_SwapBytes_Mode_MAX >= k_SwapBytes_Mode_AVX2)
  #define USE_SWAP_AVX2
#endif
#if (k_SwapBytes_Mode_MAX >= k_SwapBytes_Mode_SSSE3)
  #define USE_SWAP_SSSE3
#endif
#if (k_SwapBytes_Mode_MAX >= k_SwapBytes_Mode_SSE2)
  #define USE_SWAP_128
#endif

#if k_SwapBytes_Mode_MAX <= k_SwapBytes_Mode_MIN || !defined(USE_SWAP_128)
#define FORCE_SWAP_MODE
#endif


#ifdef USE_SWAP_128
/*
 <mmintrin.h> MMX
<xmmintrin.h> SSE
<emmintrin.h> SSE2
<pmmintrin.h> SSE3
<tmmintrin.h> SSSE3
<smmintrin.h> SSE4.1
<nmmintrin.h> SSE4.2
<ammintrin.h> SSE4A
<wmmintrin.h> AES
<immintrin.h> AVX, AVX2, FMA
*/

#include <emmintrin.h> // sse2
// typedef __m128i v128;

#define SWAP2_128(i) { \
  const __m128i v = *(const __m128i *)(const void *)(items + (i) * 8); \
                    *(      __m128i *)(      void *)(items + (i) * 8) = \
    _mm_or_si128( \
      _mm_slli_epi16(v, 8), \
      _mm_srli_epi16(v, 8)); }
// _mm_or_si128() has more ports to execute than _mm_add_epi16().

static
#ifdef SWAP_ATTRIB_SSE2
SWAP_ATTRIB_SSE2
#endif
void
Z7_FASTCALL
SwapBytes2_128(CSwapUInt16 *items, const CSwapUInt16 *lim)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SWAP2_128(0)  SWAP2_128(1)  items += 2 * 8;
    SWAP2_128(0)  SWAP2_128(1)  items += 2 * 8;
  }
  while (items != lim);
}

/*
// sse2
#define SWAP4_128_pack(i) { \
  __m128i v = *(const __m128i *)(const void *)(items + (i) * 4); \
  __m128i v0 = _mm_unpacklo_epi8(v, mask); \
  __m128i v1 = _mm_unpackhi_epi8(v, mask); \
  v0 = _mm_shufflelo_epi16(v0, 0x1b); \
  v1 = _mm_shufflelo_epi16(v1, 0x1b); \
  v0 = _mm_shufflehi_epi16(v0, 0x1b); \
  v1 = _mm_shufflehi_epi16(v1, 0x1b); \
  *(__m128i *)(void *)(items + (i) * 4) = _mm_packus_epi16(v0, v1); }

static
#ifdef SWAP_ATTRIB_SSE2
SWAP_ATTRIB_SSE2
#endif
void
Z7_FASTCALL
SwapBytes4_128_pack(CSwapUInt32 *items, const CSwapUInt32 *lim)
{
  const __m128i mask = _mm_setzero_si128();
  // const __m128i mask = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SWAP4_128_pack(0); items += 1 * 4;
    // SWAP4_128_pack(0); SWAP4_128_pack(1); items += 2 * 4;
  }
  while (items != lim);
}

// sse2
#define SWAP4_128_shift(i) { \
  __m128i v = *(const __m128i *)(const void *)(items + (i) * 4); \
  __m128i v2; \
  v2 = _mm_or_si128( \
        _mm_slli_si128(_mm_and_si128(v, mask), 1), \
        _mm_and_si128(_mm_srli_si128(v, 1), mask)); \
  v = _mm_or_si128( \
        _mm_slli_epi32(v, 24), \
        _mm_srli_epi32(v, 24)); \
  *(__m128i *)(void *)(items + (i) * 4) = _mm_or_si128(v2, v); }

static
#ifdef SWAP_ATTRIB_SSE2
SWAP_ATTRIB_SSE2
#endif
void
Z7_FASTCALL
SwapBytes4_128_shift(CSwapUInt32 *items, const CSwapUInt32 *lim)
{
  #define M1 0xff00
  const __m128i mask = _mm_set_epi32(M1, M1, M1, M1);
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    // SWAP4_128_shift(0)  SWAP4_128_shift(1)  items += 2 * 4;
    // SWAP4_128_shift(0)  SWAP4_128_shift(1)  items += 2 * 4;
    SWAP4_128_shift(0); items += 1 * 4;
  }
  while (items != lim);
}
*/


#if defined(USE_SWAP_SSSE3) || defined(USE_SWAP_AVX2)

#define SWAP_SHUF_REV_SEQ_2_VALS(v)                (v)+1, (v)
#define SWAP_SHUF_REV_SEQ_4_VALS(v)  (v)+3, (v)+2, (v)+1, (v)

#define SWAP2_SHUF_MASK_16_BYTES \
    SWAP_SHUF_REV_SEQ_2_VALS (0 * 2), \
    SWAP_SHUF_REV_SEQ_2_VALS (1 * 2), \
    SWAP_SHUF_REV_SEQ_2_VALS (2 * 2), \
    SWAP_SHUF_REV_SEQ_2_VALS (3 * 2), \
    SWAP_SHUF_REV_SEQ_2_VALS (4 * 2), \
    SWAP_SHUF_REV_SEQ_2_VALS (5 * 2), \
    SWAP_SHUF_REV_SEQ_2_VALS (6 * 2), \
    SWAP_SHUF_REV_SEQ_2_VALS (7 * 2)

#define SWAP4_SHUF_MASK_16_BYTES \
    SWAP_SHUF_REV_SEQ_4_VALS (0 * 4), \
    SWAP_SHUF_REV_SEQ_4_VALS (1 * 4), \
    SWAP_SHUF_REV_SEQ_4_VALS (2 * 4), \
    SWAP_SHUF_REV_SEQ_4_VALS (3 * 4)

#if defined(USE_SWAP_AVX2)
/* if we use 256_BIT_INIT_MASK, each static array mask will be larger for 16 bytes */
// #define SWAP_USE_256_BIT_INIT_MASK
#endif

#if defined(SWAP_USE_256_BIT_INIT_MASK) && defined(USE_SWAP_AVX2)
#define SWAP_MASK_INIT_SIZE 32
#else
#define SWAP_MASK_INIT_SIZE 16
#endif

MY_ALIGN(SWAP_MASK_INIT_SIZE)
static const Byte k_ShufMask_Swap2[] =
{
    SWAP2_SHUF_MASK_16_BYTES
  #if SWAP_MASK_INIT_SIZE > 16
  , SWAP2_SHUF_MASK_16_BYTES
  #endif
};

MY_ALIGN(SWAP_MASK_INIT_SIZE)
static const Byte k_ShufMask_Swap4[] =
{
    SWAP4_SHUF_MASK_16_BYTES
  #if SWAP_MASK_INIT_SIZE > 16
  , SWAP4_SHUF_MASK_16_BYTES
  #endif
};


#ifdef USE_SWAP_SSSE3

#include <tmmintrin.h> // ssse3

#define SHUF_128(i)   *(items + (i)) = \
     _mm_shuffle_epi8(*(items + (i)), mask); // SSSE3

// Z7_NO_INLINE
static
#ifdef SWAP_ATTRIB_SSSE3
SWAP_ATTRIB_SSSE3
#endif
Z7_ATTRIB_NO_VECTORIZE
void
Z7_FASTCALL
ShufBytes_128(void *items8, const void *lim8, const void *mask128_ptr)
{
  __m128i *items = (__m128i *)items8;
  const __m128i *lim = (const __m128i *)lim8;
  // const __m128i mask = _mm_set_epi8(SHUF_SWAP2_MASK_16_VALS);
  // const __m128i mask = _mm_set_epi8(SHUF_SWAP4_MASK_16_VALS);
  // const __m128i mask = _mm_load_si128((const __m128i *)(const void *)&(k_ShufMask_Swap4[0]));
  // const __m128i mask = _mm_load_si128((const __m128i *)(const void *)&(k_ShufMask_Swap4[0]));
  // const __m128i mask = *(const __m128i *)(const void *)&(k_ShufMask_Swap4[0]);
  const __m128i mask = *(const __m128i *)mask128_ptr;
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SHUF_128(0)  SHUF_128(1)  items += 2;
    SHUF_128(0)  SHUF_128(1)  items += 2;
  }
  while (items != lim);
}

#endif // USE_SWAP_SSSE3



#ifdef USE_SWAP_AVX2

#include <immintrin.h> // avx, avx2
#if defined(__clang__)
#include <avxintrin.h>
#include <avx2intrin.h>
#endif

#define SHUF_256(i)   *(items + (i)) = \
  _mm256_shuffle_epi8(*(items + (i)), mask); // AVX2

// Z7_NO_INLINE
static
#ifdef SWAP_ATTRIB_AVX2
SWAP_ATTRIB_AVX2
#endif
Z7_ATTRIB_NO_VECTORIZE
void
Z7_FASTCALL
ShufBytes_256(void *items8, const void *lim8, const void *mask128_ptr)
{
  __m256i *items = (__m256i *)items8;
  const __m256i *lim = (const __m256i *)lim8;
  /*
  UNUSED_VAR(mask128_ptr)
  __m256i mask =
  for Swap4: _mm256_setr_epi8(SWAP4_SHUF_MASK_16_BYTES, SWAP4_SHUF_MASK_16_BYTES);
  for Swap2: _mm256_setr_epi8(SWAP2_SHUF_MASK_16_BYTES, SWAP2_SHUF_MASK_16_BYTES);
  */
  const __m256i mask =
 #if SWAP_MASK_INIT_SIZE > 16
      *(const __m256i *)(const void *)mask128_ptr;
 #else
  /* msvc: broadcastsi128() version reserves the stack for no reason
     msvc 19.29-: _mm256_insertf128_si256() / _mm256_set_m128i)) versions use non-avx movdqu   xmm0,XMMWORD PTR [r8]
     msvc 19.30+ (VS2022): replaces _mm256_set_m128i(m,m) to vbroadcastf128(m) as we want
  */
  // _mm256_broadcastsi128_si256(*mask128_ptr);
#if defined(Z7_GCC_VERSION) && (Z7_GCC_VERSION < 80000)
  #define MY_mm256_set_m128i(hi, lo)  _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)
#else
  #define MY_mm256_set_m128i  _mm256_set_m128i
#endif
      MY_mm256_set_m128i(
        *(const __m128i *)mask128_ptr,
        *(const __m128i *)mask128_ptr);
 #endif
  
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SHUF_256(0)  SHUF_256(1)  items += 2;
    SHUF_256(0)  SHUF_256(1)  items += 2;
  }
  while (items != lim);
}

#endif // USE_SWAP_AVX2
#endif // USE_SWAP_SSSE3 || USE_SWAP_AVX2
#endif // USE_SWAP_128



// compile message "NEON intrinsics not available with the soft-float ABI"
#elif defined(MY_CPU_ARM_OR_ARM64) \
    && defined(MY_CPU_LE) \
    && !defined(Z7_DISABLE_ARM_NEON)

  #if defined(__clang__) && (__clang_major__ >= 8) \
    || defined(__GNUC__) && (__GNUC__ >= 6)
    #if defined(__ARM_FP)
    #if (defined(__ARM_ARCH) && (__ARM_ARCH >= 4)) \
        || defined(MY_CPU_ARM64)
    #if  defined(MY_CPU_ARM64) \
      || !defined(Z7_CLANG_VERSION) \
      || defined(__ARM_NEON)
      #define USE_SWAP_128
    #ifdef MY_CPU_ARM64
      // #define SWAP_ATTRIB_NEON __attribute__((__target__("")))
    #else
#if defined(Z7_CLANG_VERSION)
      // #define SWAP_ATTRIB_NEON __attribute__((__target__("neon")))
#else
      // #pragma message("SWAP_ATTRIB_NEON __attribute__((__target__(fpu=neon))")
      #define SWAP_ATTRIB_NEON __attribute__((__target__("fpu=neon")))
#endif
    #endif // MY_CPU_ARM64
    #endif // __ARM_NEON
    #endif // __ARM_ARCH
    #endif // __ARM_FP

  #elif defined(_MSC_VER)
    #if (_MSC_VER >= 1910)
      #define USE_SWAP_128
    #endif
  #endif

  #ifdef USE_SWAP_128
  #if defined(Z7_MSC_VER_ORIGINAL) && defined(MY_CPU_ARM64)
    #include <arm64_neon.h>
  #else

/*
#if !defined(__ARM_NEON)
#if defined(Z7_GCC_VERSION) && (__GNUC__  <   5) \
 || defined(Z7_GCC_VERSION) && (__GNUC__ ==   5) && (Z7_GCC_VERSION <  90201) \
 || defined(Z7_GCC_VERSION) && (__GNUC__ ==   5) && (Z7_GCC_VERSION < 100100)
Z7_DIAGNOSTIC_IGNORE_BEGIN_RESERVED_MACRO_IDENTIFIER
#pragma message("#define __ARM_NEON 1")
// #define __ARM_NEON 1
Z7_DIAGNOSTIC_IGNORE_END_RESERVED_MACRO_IDENTIFIER
#endif
#endif
*/
    #include <arm_neon.h>
  #endif
  #endif

#ifndef USE_SWAP_128
  #define FORCE_SWAP_MODE
#else
 
#ifdef MY_CPU_ARM64
  // for debug : comment it
  #define FORCE_SWAP_MODE
#else
  #define k_SwapBytes_Mode_NEON 1
#endif
// typedef uint8x16_t v128;
#define SWAP2_128(i)   *(uint8x16_t *)      (void *)(items + (i) * 8) = \
      vrev16q_u8(*(const uint8x16_t *)(const void *)(items + (i) * 8));
#define SWAP4_128(i)   *(uint8x16_t *)      (void *)(items + (i) * 4) = \
      vrev32q_u8(*(const uint8x16_t *)(const void *)(items + (i) * 4));

// Z7_NO_INLINE
static
#ifdef SWAP_ATTRIB_NEON
SWAP_ATTRIB_NEON
#endif
Z7_ATTRIB_NO_VECTORIZE
void
Z7_FASTCALL
SwapBytes2_128(CSwapUInt16 *items, const CSwapUInt16 *lim)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SWAP2_128(0)  SWAP2_128(1)  items += 2 * 8;
    SWAP2_128(0)  SWAP2_128(1)  items += 2 * 8;
  }
  while (items != lim);
}

// Z7_NO_INLINE
static
#ifdef SWAP_ATTRIB_NEON
SWAP_ATTRIB_NEON
#endif
Z7_ATTRIB_NO_VECTORIZE
void
Z7_FASTCALL
SwapBytes4_128(CSwapUInt32 *items, const CSwapUInt32 *lim)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SWAP4_128(0)  SWAP4_128(1)  items += 2 * 4;
    SWAP4_128(0)  SWAP4_128(1)  items += 2 * 4;
  }
  while (items != lim);
}

#endif // USE_SWAP_128

#else // MY_CPU_ARM_OR_ARM64
#define FORCE_SWAP_MODE
#endif // MY_CPU_ARM_OR_ARM64






#if defined(Z7_MSC_VER_ORIGINAL) && defined(MY_CPU_X86)
  /* _byteswap_ushort() in MSVC x86 32-bit works via slow { mov dh, al; mov dl, ah }
     So we use own versions of byteswap function */
  #if (_MSC_VER < 1400 )  // old MSVC-X86 without _rotr16() support
    #define SWAP2_16(i)  { UInt32 v = items[i];  v += (v << 16);  v >>= 8;  items[i] = (CSwapUInt16)v; }
  #else  // is new MSVC-X86 with fast _rotr16()
    #include <intrin.h>
    #define SWAP2_16(i)  { items[i] = _rotr16(items[i], 8); }
  #endif
#else  // is not MSVC-X86
  #define SWAP2_16(i)  { CSwapUInt16 v = items[i];  items[i] = Z7_BSWAP16(v); }
#endif  // MSVC-X86

#if defined(Z7_CPU_FAST_BSWAP_SUPPORTED)
  #define SWAP4_32(i)  { CSwapUInt32 v = items[i];  items[i] = Z7_BSWAP32(v); }
#else
  #define SWAP4_32(i)  \
    { UInt32 v = items[i]; \
      v = ((v & 0xff00ff) << 8) + ((v >> 8) & 0xff00ff); \
      v = rotlFixed(v, 16); \
      items[i] = v; }
#endif




#if defined(FORCE_SWAP_MODE) && defined(USE_SWAP_128)
  #define DEFAULT_Swap2  SwapBytes2_128
  #if !defined(MY_CPU_X86_OR_AMD64)
    #define DEFAULT_Swap4  SwapBytes4_128
  #endif
#endif

#if !defined(DEFAULT_Swap2) || !defined(DEFAULT_Swap4)

#define SWAP_BASE_FUNCS_PREFIXES \
Z7_FORCE_INLINE  \
static \
Z7_ATTRIB_NO_VECTOR  \
void Z7_FASTCALL


#if defined(MY_CPU_ARM_OR_ARM64)
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wlanguage-extension-token"
#endif
#endif


#ifdef MY_CPU_64BIT

#if defined(MY_CPU_ARM64) \
    && defined(__ARM_ARCH) && (__ARM_ARCH >= 8) \
    && (  (defined(__GNUC__) && (__GNUC__ >= 4)) \
       || (defined(__clang__) && (__clang_major__ >= 4)))

  #define SWAP2_64_VAR(v)  asm ("rev16 %x0,%x0" : "+r" (v));
  #define SWAP4_64_VAR(v)  asm ("rev32 %x0,%x0" : "+r" (v));

#else  // is not ARM64-GNU

#if !defined(MY_CPU_X86_OR_AMD64) || (k_SwapBytes_Mode_MIN == 0) || !defined(USE_SWAP_128)
  #define SWAP2_64_VAR(v) \
    v = ( 0x00ff00ff00ff00ff & (v >> 8))  \
      + ((0x00ff00ff00ff00ff & v) << 8);
      /* plus gives faster code in MSVC */
#endif

#ifdef Z7_CPU_FAST_BSWAP_SUPPORTED
  #define SWAP4_64_VAR(v) \
    v = Z7_BSWAP64(v); \
    v = Z7_ROTL64(v, 32);
#else
  #define SWAP4_64_VAR(v) \
    v = ( 0x000000ff000000ff & (v >> 24))  \
      + ((0x000000ff000000ff & v) << 24 )  \
      + ( 0x0000ff000000ff00 & (v >>  8))  \
      + ((0x0000ff000000ff00 & v) <<  8 )  \
      ;
#endif

#endif  // ARM64-GNU


#ifdef SWAP2_64_VAR

#define SWAP2_64(i) { \
    UInt64 v = *(const UInt64 *)(const void *)(items + (i) * 4); \
    SWAP2_64_VAR(v) \
    *(UInt64 *)(void *)(items + (i) * 4) = v; }

SWAP_BASE_FUNCS_PREFIXES
SwapBytes2_64(CSwapUInt16 *items, const CSwapUInt16 *lim)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SWAP2_64(0)  SWAP2_64(1)  items += 2 * 4;
    SWAP2_64(0)  SWAP2_64(1)  items += 2 * 4;
  }
  while (items != lim);
}

  #define DEFAULT_Swap2  SwapBytes2_64
  #if !defined(FORCE_SWAP_MODE)
    #define SWAP2_DEFAULT_MODE 0
  #endif
#else // !defined(SWAP2_64_VAR)
  #define DEFAULT_Swap2  SwapBytes2_128
  #if !defined(FORCE_SWAP_MODE)
    #define SWAP2_DEFAULT_MODE 1
  #endif
#endif // SWAP2_64_VAR


#define SWAP4_64(i) { \
    UInt64 v = *(const UInt64 *)(const void *)(items + (i) * 2); \
    SWAP4_64_VAR(v) \
    *(UInt64 *)(void *)(items + (i) * 2) = v; }

SWAP_BASE_FUNCS_PREFIXES
SwapBytes4_64(CSwapUInt32 *items, const CSwapUInt32 *lim)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SWAP4_64(0)  SWAP4_64(1)  items += 2 * 2;
    SWAP4_64(0)  SWAP4_64(1)  items += 2 * 2;
  }
  while (items != lim);
}

#define DEFAULT_Swap4  SwapBytes4_64

#else  // is not 64BIT


#if defined(MY_CPU_ARM_OR_ARM64) \
    && defined(__ARM_ARCH) && (__ARM_ARCH >= 6) \
    && (  (defined(__GNUC__) && (__GNUC__ >= 4)) \
       || (defined(__clang__) && (__clang_major__ >= 4)))

#ifdef MY_CPU_64BIT
  #define SWAP2_32_VAR(v)  asm ("rev16 %w0,%w0" : "+r" (v));
#else
  #define SWAP2_32_VAR(v)  asm ("rev16 %0,%0" : "+r" (v)); // for clang/gcc
    // asm ("rev16 %r0,%r0" : "+r" (a));  // for gcc
#endif

#elif defined(_MSC_VER) && (_MSC_VER < 1300) && defined(MY_CPU_X86) \
    || !defined(Z7_CPU_FAST_BSWAP_SUPPORTED) \
    || !defined(Z7_CPU_FAST_ROTATE_SUPPORTED)
  // old msvc doesn't support _byteswap_ulong()
  #define SWAP2_32_VAR(v) \
    v = ((v & 0xff00ff) << 8) + ((v >> 8) & 0xff00ff);

#else  // is not ARM and is not old-MSVC-X86 and fast BSWAP/ROTATE are supported
  #define SWAP2_32_VAR(v) \
    v = Z7_BSWAP32(v); \
    v = rotlFixed(v, 16);

#endif  // GNU-ARM*

#define SWAP2_32(i) { \
    UInt32 v = *(const UInt32 *)(const void *)(items + (i) * 2); \
    SWAP2_32_VAR(v); \
    *(UInt32 *)(void *)(items + (i) * 2) = v; }


SWAP_BASE_FUNCS_PREFIXES
SwapBytes2_32(CSwapUInt16 *items, const CSwapUInt16 *lim)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SWAP2_32(0)  SWAP2_32(1)  items += 2 * 2;
    SWAP2_32(0)  SWAP2_32(1)  items += 2 * 2;
  }
  while (items != lim);
}


SWAP_BASE_FUNCS_PREFIXES
SwapBytes4_32(CSwapUInt32 *items, const CSwapUInt32 *lim)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  do
  {
    SWAP4_32(0)  SWAP4_32(1)  items += 2;
    SWAP4_32(0)  SWAP4_32(1)  items += 2;
  }
  while (items != lim);
}

#define DEFAULT_Swap2  SwapBytes2_32
#define DEFAULT_Swap4  SwapBytes4_32
#if !defined(FORCE_SWAP_MODE)
  #define SWAP2_DEFAULT_MODE 0
#endif

#endif // MY_CPU_64BIT
#endif // if !defined(DEFAULT_Swap2) || !defined(DEFAULT_Swap4)



#if !defined(FORCE_SWAP_MODE)
static unsigned g_SwapBytes_Mode;
#endif

/* size of largest unrolled loop iteration: 128 bytes = 4 * 32 bytes (AVX). */
#define SWAP_ITERATION_BLOCK_SIZE_MAX  (1 << 7)

// 32 bytes for (AVX) or 2 * 16-bytes for NEON.
#define SWAP_VECTOR_ALIGN_SIZE  (1 << 5)

Z7_NO_INLINE
void z7_SwapBytes2(CSwapUInt16 *items, size_t numItems)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  for (; numItems != 0 && ((unsigned)(ptrdiff_t)items & (SWAP_VECTOR_ALIGN_SIZE - 1)) != 0; numItems--)
  {
    SWAP2_16(0)
    items++;
  }
  {
    const size_t k_Align_Mask = SWAP_ITERATION_BLOCK_SIZE_MAX / sizeof(CSwapUInt16) - 1;
    size_t numItems2 = numItems;
    CSwapUInt16 *lim;
    numItems &= k_Align_Mask;
    numItems2 &= ~(size_t)k_Align_Mask;
    lim = items + numItems2;
    if (numItems2 != 0)
    {
     #if !defined(FORCE_SWAP_MODE)
      #ifdef MY_CPU_X86_OR_AMD64
        #ifdef USE_SWAP_AVX2
          if (g_SwapBytes_Mode > k_SwapBytes_Mode_SSSE3)
            ShufBytes_256((__m256i *)(void *)items,
                (const __m256i *)(const void *)lim,
                (const __m128i *)(const void *)&(k_ShufMask_Swap2[0]));
          else
        #endif
        #ifdef USE_SWAP_SSSE3
          if (g_SwapBytes_Mode >= k_SwapBytes_Mode_SSSE3)
            ShufBytes_128((__m128i *)(void *)items,
                (const __m128i *)(const void *)lim,
                (const __m128i *)(const void *)&(k_ShufMask_Swap2[0]));
          else
        #endif
      #endif  // MY_CPU_X86_OR_AMD64
      #if SWAP2_DEFAULT_MODE == 0
          if (g_SwapBytes_Mode != 0)
            SwapBytes2_128(items, lim);
          else
      #endif
     #endif // FORCE_SWAP_MODE
            DEFAULT_Swap2(items, lim);
    }
    items = lim;
  }
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  for (; numItems != 0; numItems--)
  {
    SWAP2_16(0)
    items++;
  }
}


Z7_NO_INLINE
void z7_SwapBytes4(CSwapUInt32 *items, size_t numItems)
{
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  for (; numItems != 0 && ((unsigned)(ptrdiff_t)items & (SWAP_VECTOR_ALIGN_SIZE - 1)) != 0; numItems--)
  {
    SWAP4_32(0)
    items++;
  }
  {
    const size_t k_Align_Mask = SWAP_ITERATION_BLOCK_SIZE_MAX / sizeof(CSwapUInt32) - 1;
    size_t numItems2 = numItems;
    CSwapUInt32 *lim;
    numItems &= k_Align_Mask;
    numItems2 &= ~(size_t)k_Align_Mask;
    lim = items + numItems2;
    if (numItems2 != 0)
    {
     #if !defined(FORCE_SWAP_MODE)
      #ifdef MY_CPU_X86_OR_AMD64
        #ifdef USE_SWAP_AVX2
          if (g_SwapBytes_Mode > k_SwapBytes_Mode_SSSE3)
            ShufBytes_256((__m256i *)(void *)items,
                (const __m256i *)(const void *)lim,
                (const __m128i *)(const void *)&(k_ShufMask_Swap4[0]));
          else
        #endif
        #ifdef USE_SWAP_SSSE3
          if (g_SwapBytes_Mode >= k_SwapBytes_Mode_SSSE3)
            ShufBytes_128((__m128i *)(void *)items,
                (const __m128i *)(const void *)lim,
                (const __m128i *)(const void *)&(k_ShufMask_Swap4[0]));
          else
        #endif
      #else  // MY_CPU_X86_OR_AMD64

          if (g_SwapBytes_Mode != 0)
            SwapBytes4_128(items, lim);
          else
      #endif  // MY_CPU_X86_OR_AMD64
     #endif // FORCE_SWAP_MODE
            DEFAULT_Swap4(items, lim);
    }
    items = lim;
  }
  Z7_PRAGMA_OPT_DISABLE_LOOP_UNROLL_VECTORIZE
  for (; numItems != 0; numItems--)
  {
    SWAP4_32(0)
    items++;
  }
}


// #define SHOW_HW_STATUS

#ifdef SHOW_HW_STATUS
#include <stdio.h>
#define PRF(x) x
#else
#define PRF(x)
#endif

void z7_SwapBytesPrepare(void)
{
#ifndef FORCE_SWAP_MODE
  unsigned mode = 0; // k_SwapBytes_Mode_BASE;

#ifdef MY_CPU_ARM_OR_ARM64
  {
    if (CPU_IsSupported_NEON())
    {
      // #pragma message ("=== SwapBytes NEON")
      PRF(printf("\n=== SwapBytes NEON\n");)
      mode = k_SwapBytes_Mode_NEON;
    }
  }
#else // MY_CPU_ARM_OR_ARM64
  {
    #ifdef USE_SWAP_AVX2
      if (CPU_IsSupported_AVX2())
      {
        // #pragma message ("=== SwapBytes AVX2")
        PRF(printf("\n=== SwapBytes AVX2\n");)
        mode = k_SwapBytes_Mode_AVX2;
      }
      else
    #endif
    #ifdef USE_SWAP_SSSE3
      if (CPU_IsSupported_SSSE3())
      {
        // #pragma message ("=== SwapBytes SSSE3")
        PRF(printf("\n=== SwapBytes SSSE3\n");)
        mode = k_SwapBytes_Mode_SSSE3;
      }
      else
    #endif
    #if !defined(MY_CPU_AMD64)
      if (CPU_IsSupported_SSE2())
    #endif
      {
        // #pragma message ("=== SwapBytes SSE2")
        PRF(printf("\n=== SwapBytes SSE2\n");)
        mode = k_SwapBytes_Mode_SSE2;
      }
  }
#endif // MY_CPU_ARM_OR_ARM64
  g_SwapBytes_Mode = mode;
  // g_SwapBytes_Mode = 0; // for debug
#endif // FORCE_SWAP_MODE
  PRF(printf("\n=== SwapBytesPrepare\n");)
}

#undef PRF
