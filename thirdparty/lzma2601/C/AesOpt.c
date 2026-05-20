/* AesOpt.c -- AES optimized code for x86 AES hardware instructions
Igor Pavlov : Public domain */

#include "Precomp.h"

#include "Aes.h"
#include "CpuArch.h"

#ifdef MY_CPU_X86_OR_AMD64

  #if defined(__INTEL_COMPILER)
    #if (__INTEL_COMPILER >= 1110)
      #define USE_INTEL_AES
      #if (__INTEL_COMPILER >= 1900)
        #define USE_INTEL_VAES
      #endif
    #endif
  #elif defined(Z7_CLANG_VERSION) && (Z7_CLANG_VERSION >= 30800) \
     || defined(Z7_GCC_VERSION)   && (Z7_GCC_VERSION   >= 40400)
        #define USE_INTEL_AES
        #if !defined(__AES__)
          #define ATTRIB_AES __attribute__((__target__("aes")))
        #endif
      #if defined(__clang__) && (__clang_major__ >= 8) \
          || defined(__GNUC__) && (__GNUC__ >= 8)
        #define USE_INTEL_VAES
        #if !defined(__AES__) || !defined(__VAES__) || !defined(__AVX__) || !defined(__AVX2__)
          #define ATTRIB_VAES __attribute__((__target__("aes,vaes,avx,avx2")))
        #endif
      #endif
  #elif defined(_MSC_VER)
    #if (_MSC_VER > 1500) || (_MSC_FULL_VER >= 150030729)
      #define USE_INTEL_AES
      #if (_MSC_VER >= 1910)
        #define USE_INTEL_VAES
      #endif
    #endif
    #ifndef USE_INTEL_AES
      #define Z7_USE_AES_HW_STUB
    #endif
    #ifndef USE_INTEL_VAES
      #define Z7_USE_VAES_HW_STUB
    #endif
  #endif

    #ifndef USE_INTEL_AES
      // #define Z7_USE_AES_HW_STUB // for debug
    #endif
    #ifndef USE_INTEL_VAES
      // #define Z7_USE_VAES_HW_STUB // for debug
    #endif


#ifdef USE_INTEL_AES

#include <wmmintrin.h>

#if !defined(USE_INTEL_VAES) && defined(Z7_USE_VAES_HW_STUB)
#define AES_TYPE_keys UInt32
#define AES_TYPE_data Byte
// #define AES_TYPE_keys __m128i
// #define AES_TYPE_data __m128i
#endif

#ifndef ATTRIB_AES
  #define ATTRIB_AES
#endif

#define AES_FUNC_START(name) \
    void Z7_FASTCALL name(UInt32 *ivAes, Byte *data8, size_t numBlocks)
    // void Z7_FASTCALL name(__m128i *p, __m128i *data, size_t numBlocks)

#define AES_FUNC_START2(name) \
AES_FUNC_START (name); \
ATTRIB_AES \
AES_FUNC_START (name)

#define MM_OP(op, dest, src)  dest = op(dest, src);
#define MM_OP_m(op, src)      MM_OP(op, m, src)

#define MM_XOR( dest, src)    MM_OP(_mm_xor_si128,    dest, src)

#if 1
// use aligned SSE load/store for data.
// It is required for our Aes functions, that data is aligned for 16-bytes.
// So we can use this branch of code.
// and compiler can use fused load-op SSE instructions:
//   xorps xmm0, XMMWORD PTR [rdx]
#define LOAD_128(pp)        (*(__m128i *)(void *)(pp))
#define STORE_128(pp, _v)    *(__m128i *)(void *)(pp) = _v
// use aligned SSE load/store for data. Alternative code with direct access
// #define LOAD_128(pp)        _mm_load_si128(pp)
// #define STORE_128(pp, _v)   _mm_store_si128(pp, _v)
#else
// use unaligned load/store for data: movdqu XMMWORD PTR [rdx]
#define LOAD_128(pp)        _mm_loadu_si128(pp)
#define STORE_128(pp, _v)   _mm_storeu_si128(pp, _v)
#endif

AES_FUNC_START2 (AesCbc_Encode_HW)
{
  if (numBlocks == 0)
    return;
  {
  __m128i *p = (__m128i *)(void *)ivAes;
  __m128i *data = (__m128i *)(void *)data8;
  __m128i m = *p;
  const __m128i k0 = p[2];
  const __m128i k1 = p[3];
  const UInt32 numRounds2 = *(const UInt32 *)(p + 1) - 1;
  do
  {
    UInt32 r = numRounds2;
    const __m128i *w = p + 4;
    __m128i temp = LOAD_128(data);
    MM_XOR (temp, k0)
    MM_XOR (m, temp)
    MM_OP_m (_mm_aesenc_si128, k1)
    do
    {
      MM_OP_m (_mm_aesenc_si128, w[0])
      MM_OP_m (_mm_aesenc_si128, w[1])
      w += 2;
    }
    while (--r);
    MM_OP_m (_mm_aesenclast_si128, w[0])
    STORE_128(data, m);
    data++;
  }
  while (--numBlocks);
  *p = m;
  }
}


#define WOP_1(op)
#define WOP_2(op)   WOP_1 (op)  op (m1, 1)
#define WOP_3(op)   WOP_2 (op)  op (m2, 2)
#define WOP_4(op)   WOP_3 (op)  op (m3, 3)
#ifdef MY_CPU_AMD64
#define WOP_5(op)   WOP_4 (op)  op (m4, 4)
#define WOP_6(op)   WOP_5 (op)  op (m5, 5)
#define WOP_7(op)   WOP_6 (op)  op (m6, 6)
#define WOP_8(op)   WOP_7 (op)  op (m7, 7)
#endif
/*
#define WOP_9(op)   WOP_8 (op)  op (m8, 8);
#define WOP_10(op)  WOP_9 (op)  op (m9, 9);
#define WOP_11(op)  WOP_10(op)  op (m10, 10);
#define WOP_12(op)  WOP_11(op)  op (m11, 11);
#define WOP_13(op)  WOP_12(op)  op (m12, 12);
#define WOP_14(op)  WOP_13(op)  op (m13, 13);
*/

#ifdef MY_CPU_AMD64
  #define NUM_WAYS      8
  #define WOP_M1    WOP_8
#else
  #define NUM_WAYS      4
  #define WOP_M1    WOP_4
#endif

#define WOP(op)  op (m0, 0)  WOP_M1(op)

#define DECLARE_VAR(reg, ii)  __m128i reg;
#define LOAD_data_ii(ii)      LOAD_128(data + (ii))
#define LOAD_data(  reg, ii)  reg = LOAD_data_ii(ii);
#define STORE_data( reg, ii)  STORE_128(data + (ii), reg);
#if (NUM_WAYS > 1)
#define XOR_data_M1(reg, ii)  MM_XOR (reg, LOAD_128(data + (ii- 1)))
#endif

#define MM_OP_key(op, reg)  MM_OP(op, reg, key);

#define AES_DEC(      reg, ii)   MM_OP_key (_mm_aesdec_si128,     reg)
#define AES_DEC_LAST( reg, ii)   MM_OP_key (_mm_aesdeclast_si128, reg)
#define AES_ENC(      reg, ii)   MM_OP_key (_mm_aesenc_si128,     reg)
#define AES_ENC_LAST( reg, ii)   MM_OP_key (_mm_aesenclast_si128, reg)
#define AES_XOR(      reg, ii)   MM_OP_key (_mm_xor_si128,        reg)

#define CTR_START(reg, ii)  MM_OP (_mm_add_epi64, ctr, one)  reg = ctr;
#define CTR_END(  reg, ii)  STORE_128(data + (ii), _mm_xor_si128(reg, \
                            LOAD_128 (data + (ii))));
#define WOP_KEY(op, n) { \
    const __m128i key = w[n]; \
    WOP(op) }

#define WIDE_LOOP_START  \
    dataEnd = data + numBlocks;  \
    if (numBlocks >= NUM_WAYS)  \
    { dataEnd -= NUM_WAYS; do {  \

#define WIDE_LOOP_END  \
    data += NUM_WAYS;  \
    } while (data <= dataEnd);  \
    dataEnd += NUM_WAYS; }  \

#define SINGLE_LOOP  \
    for (; data < dataEnd; data++)



#ifdef USE_INTEL_VAES

#define AVX_XOR(dest, src)    MM_OP(_mm256_xor_si256, dest, src)
#define AVX_DECLARE_VAR(reg, ii)  __m256i reg;

#if 1
// use unaligned AVX load/store for data.
// It is required for our Aes functions, that data is aligned for 16-bytes.
// But we need 32-bytes reading.
// So we use intrinsics for unaligned AVX load/store.
// notes for _mm256_storeu_si256:
// msvc2022: uses vmovdqu and keeps the order of instruction sequence.
// new gcc11 uses vmovdqu
// old gcc9 could use pair of instructions:
//   vmovups        %xmm7, -224(%rax)
//   vextracti128   $0x1, %ymm7, -208(%rax)
#define AVX_LOAD(p)         _mm256_loadu_si256((const __m256i *)(const void *)(p))
#define AVX_STORE(p, _v)    _mm256_storeu_si256((__m256i *)(void *)(p), _v);
#else
// use aligned AVX load/store for data.
// for debug: we can use this branch, if we are sure that data is aligned for 32-bytes.
// msvc2022 uses vmovdqu still
// gcc      uses vmovdqa (that requires 32-bytes alignment)
#define AVX_LOAD(p)         (*(const __m256i *)(const void *)(p))
#define AVX_STORE(p, _v)    (*(__m256i *)(void *)(p)) = _v;
#endif

#define AVX_LOAD_data(  reg, ii)  reg = AVX_LOAD((const __m256i *)(const void *)data + (ii));
#define AVX_STORE_data( reg, ii)  AVX_STORE((__m256i *)(void *)data + (ii), reg)
/*
AVX_XOR_data_M1() needs unaligned memory load, even if (data)
is aligned for 256-bits, because we read 32-bytes chunk that
crosses (data) position: from (data - 16bytes) to (data + 16bytes).
*/
#define AVX_XOR_data_M1(reg, ii)  AVX_XOR (reg, _mm256_loadu_si256((const __m256i *)(const void *)(data - 1) + (ii)))

#define AVX_AES_DEC(      reg, ii)   MM_OP_key (_mm256_aesdec_epi128,     reg)
#define AVX_AES_DEC_LAST( reg, ii)   MM_OP_key (_mm256_aesdeclast_epi128, reg)
#define AVX_AES_ENC(      reg, ii)   MM_OP_key (_mm256_aesenc_epi128,     reg)
#define AVX_AES_ENC_LAST( reg, ii)   MM_OP_key (_mm256_aesenclast_epi128, reg)
#define AVX_AES_XOR(      reg, ii)   MM_OP_key (_mm256_xor_si256,         reg)
#define AVX_CTR_START(reg, ii)  \
    MM_OP (_mm256_add_epi64, ctr2, two) \
    reg = _mm256_xor_si256(ctr2, key);

#define AVX_CTR_END(reg, ii)  \
    AVX_STORE((__m256i *)(void *)data + (ii), _mm256_xor_si256(reg, \
    AVX_LOAD ((__m256i *)(void *)data + (ii))));

#define AVX_WOP_KEY(op, n) { \
    const __m256i key = w[n]; \
    WOP(op) }

#define NUM_AES_KEYS_MAX 15

#define WIDE_LOOP_START_AVX(OP)  \
    dataEnd = data + numBlocks;  \
    if (numBlocks >= NUM_WAYS * 2)  \
    { __m256i keys[NUM_AES_KEYS_MAX];  \
      OP  \
      { UInt32 ii; for (ii = 0; ii < numRounds; ii++)  \
        keys[ii] = _mm256_broadcastsi128_si256(p[ii]); }  \
      dataEnd -= NUM_WAYS * 2; \
      do {  \

#define WIDE_LOOP_END_AVX(OP)  \
        data += NUM_WAYS * 2;  \
      } while (data <= dataEnd);  \
      dataEnd += NUM_WAYS * 2;  \
      OP  \
      _mm256_zeroupper();  \
    }  \

/* MSVC for x86: If we don't call _mm256_zeroupper(), and -arch:IA32 is not specified,
   MSVC still can insert vzeroupper instruction. */

#endif



AES_FUNC_START2 (AesCbc_Decode_HW)
{
  __m128i *p = (__m128i *)(void *)ivAes;
  __m128i *data = (__m128i *)(void *)data8;
  __m128i iv = *p;
  const __m128i * const wStart = p + (size_t)*(const UInt32 *)(p + 1) * 2 + 2 - 1;
  const __m128i *dataEnd;
  p += 2;
  
  WIDE_LOOP_START
  {
    const __m128i *w = wStart;
    WOP (DECLARE_VAR)
    WOP (LOAD_data)
    WOP_KEY (AES_XOR, 1)
    do
    {
      WOP_KEY (AES_DEC, 0)

      w--;
    }
    while (w != p);
    WOP_KEY (AES_DEC_LAST, 0)

    MM_XOR (m0, iv)
    WOP_M1 (XOR_data_M1)
    LOAD_data(iv, NUM_WAYS - 1)
    WOP (STORE_data)
  }
  WIDE_LOOP_END

  SINGLE_LOOP
  {
    const __m128i *w = wStart - 1;
    __m128i m = _mm_xor_si128 (w[2], LOAD_data_ii(0));
    
    do
    {
      MM_OP_m (_mm_aesdec_si128, w[1])
      MM_OP_m (_mm_aesdec_si128, w[0])
      w -= 2;
    }
    while (w != p);
    MM_OP_m (_mm_aesdec_si128,     w[1])
    MM_OP_m (_mm_aesdeclast_si128, w[0])
    MM_XOR (m, iv)
    LOAD_data(iv, 0)
    STORE_data(m, 0)
  }
  
  p[-2] = iv;
}


AES_FUNC_START2 (AesCtr_Code_HW)
{
  __m128i *p = (__m128i *)(void *)ivAes;
  __m128i *data = (__m128i *)(void *)data8;
  __m128i ctr = *p;
  const UInt32 numRoundsMinus2 = *(const UInt32 *)(p + 1) * 2 - 1;
  const __m128i *dataEnd;
  const __m128i one = _mm_cvtsi32_si128(1);

  p += 2;
  
  WIDE_LOOP_START
  {
    const __m128i *w = p;
    UInt32 r = numRoundsMinus2;
    WOP (DECLARE_VAR)
    WOP (CTR_START)
    WOP_KEY (AES_XOR, 0)
    w += 1;
    do
    {
      WOP_KEY (AES_ENC, 0)
      w += 1;
    }
    while (--r);
    WOP_KEY (AES_ENC_LAST, 0)
    WOP (CTR_END)
  }
  WIDE_LOOP_END

  SINGLE_LOOP
  {
    UInt32 numRounds2 = *(const UInt32 *)(p - 2 + 1) - 1;
    const __m128i *w = p;
    __m128i m;
    MM_OP (_mm_add_epi64, ctr, one)
    m = _mm_xor_si128 (ctr, p[0]);
    w += 1;
    do
    {
      MM_OP_m (_mm_aesenc_si128, w[0])
      MM_OP_m (_mm_aesenc_si128, w[1])
      w += 2;
    }
    while (--numRounds2);
    MM_OP_m (_mm_aesenc_si128,     w[0])
    MM_OP_m (_mm_aesenclast_si128, w[1])
    CTR_END (m, 0)
  }
  
  p[-2] = ctr;
}



#ifdef USE_INTEL_VAES

/*
GCC before 2013-Jun:
  <immintrin.h>:
    #ifdef __AVX__
     #include <avxintrin.h>
    #endif
GCC after 2013-Jun:
  <immintrin.h>:
    #include <avxintrin.h>
CLANG 3.8+:
{
  <immintrin.h>:
    #if !defined(_MSC_VER) || defined(__AVX__)
      #include <avxintrin.h>
    #endif

  if (the compiler is clang for Windows and if global arch is not set for __AVX__)
    [ if (defined(_MSC_VER) && !defined(__AVX__)) ]
  {
    <immintrin.h> doesn't include <avxintrin.h>
    and we have 2 ways to fix it:
      1) we can define required __AVX__ before <immintrin.h>
      or
      2) we can include <avxintrin.h> after <immintrin.h>
  }
}

If we include <avxintrin.h> manually for GCC/CLANG, it's
required that <immintrin.h> must be included before <avxintrin.h>.
*/

/*
#if defined(__clang__) && defined(_MSC_VER)
#define __AVX__
#define __AVX2__
#define __VAES__
#endif
*/

#include <immintrin.h>
#if defined(__clang__) && defined(_MSC_VER)
  #if !defined(__AVX__)
    #include <avxintrin.h>
  #endif
  #if !defined(__AVX2__)
    #include <avx2intrin.h>
  #endif
  #if !defined(__VAES__)
    #include <vaesintrin.h>
  #endif
#endif  // __clang__ && _MSC_VER

#ifndef ATTRIB_VAES
  #define ATTRIB_VAES
#endif

#define VAES_FUNC_START2(name) \
AES_FUNC_START (name); \
ATTRIB_VAES \
AES_FUNC_START (name)

VAES_FUNC_START2 (AesCbc_Decode_HW_256)
{
  __m128i *p = (__m128i *)(void *)ivAes;
  __m128i *data = (__m128i *)(void *)data8;
  __m128i iv = *p;
  const __m128i *dataEnd;
  const UInt32 numRounds = *(const UInt32 *)(p + 1) * 2 + 1;
  p += 2;
  
  WIDE_LOOP_START_AVX(;)
  {
    const __m256i *w = keys + numRounds - 2;
    
    WOP (AVX_DECLARE_VAR)
    WOP (AVX_LOAD_data)
    AVX_WOP_KEY (AVX_AES_XOR, 1)

    do
    {
      AVX_WOP_KEY (AVX_AES_DEC, 0)
      w--;
    }
    while (w != keys);
    AVX_WOP_KEY (AVX_AES_DEC_LAST, 0)

    AVX_XOR (m0, _mm256_setr_m128i(iv, LOAD_data_ii(0)))
    WOP_M1 (AVX_XOR_data_M1)
    LOAD_data (iv, NUM_WAYS * 2 - 1)
    WOP (AVX_STORE_data)
  }
  WIDE_LOOP_END_AVX(;)

  SINGLE_LOOP
  {
    const __m128i *w = p - 2 + (size_t)*(const UInt32 *)(p + 1 - 2) * 2;
    __m128i m = _mm_xor_si128 (w[2], LOAD_data_ii(0));
    do
    {
      MM_OP_m (_mm_aesdec_si128, w[1])
      MM_OP_m (_mm_aesdec_si128, w[0])
      w -= 2;
    }
    while (w != p);
    MM_OP_m (_mm_aesdec_si128,     w[1])
    MM_OP_m (_mm_aesdeclast_si128, w[0])

    MM_XOR (m, iv)
    LOAD_data(iv, 0)
    STORE_data(m, 0)
  }
  
  p[-2] = iv;
}


/*
SSE2: _mm_cvtsi32_si128 : movd
AVX:  _mm256_setr_m128i            : vinsertf128
AVX2: _mm256_add_epi64             : vpaddq ymm, ymm, ymm
      _mm256_extracti128_si256     : vextracti128
      _mm256_broadcastsi128_si256  : vbroadcasti128
*/

#define AVX_CTR_LOOP_START  \
    ctr2 = _mm256_setr_m128i(_mm_sub_epi64(ctr, one), ctr); \
    two = _mm256_setr_m128i(one, one); \
    two = _mm256_add_epi64(two, two); \

// two = _mm256_setr_epi64x(2, 0, 2, 0);
  
#define AVX_CTR_LOOP_ENC  \
    ctr = _mm256_extracti128_si256 (ctr2, 1); \
 
VAES_FUNC_START2 (AesCtr_Code_HW_256)
{
  __m128i *p = (__m128i *)(void *)ivAes;
  __m128i *data = (__m128i *)(void *)data8;
  __m128i ctr = *p;
  const UInt32 numRounds = *(const UInt32 *)(p + 1) * 2 + 1;
  const __m128i *dataEnd;
  const __m128i one = _mm_cvtsi32_si128(1);
  __m256i ctr2, two;
  p += 2;
  
  WIDE_LOOP_START_AVX (AVX_CTR_LOOP_START)
  {
    const __m256i *w = keys;
    UInt32 r = numRounds - 2;
    WOP (AVX_DECLARE_VAR)
    AVX_WOP_KEY (AVX_CTR_START, 0)

    w += 1;
    do
    {
      AVX_WOP_KEY (AVX_AES_ENC, 0)
      w += 1;
    }
    while (--r);
    AVX_WOP_KEY (AVX_AES_ENC_LAST, 0)
   
    WOP (AVX_CTR_END)
  }
  WIDE_LOOP_END_AVX (AVX_CTR_LOOP_ENC)
  
  SINGLE_LOOP
  {
    UInt32 numRounds2 = *(const UInt32 *)(p - 2 + 1) - 1;
    const __m128i *w = p;
    __m128i m;
    MM_OP (_mm_add_epi64, ctr, one)
    m = _mm_xor_si128 (ctr, p[0]);
    w += 1;
    do
    {
      MM_OP_m (_mm_aesenc_si128, w[0])
      MM_OP_m (_mm_aesenc_si128, w[1])
      w += 2;
    }
    while (--numRounds2);
    MM_OP_m (_mm_aesenc_si128,     w[0])
    MM_OP_m (_mm_aesenclast_si128, w[1])
    CTR_END (m, 0)
  }

  p[-2] = ctr;
}

#endif // USE_INTEL_VAES

#else // USE_INTEL_AES

/* no USE_INTEL_AES */

#if defined(Z7_USE_AES_HW_STUB)
// We can compile this file with another C compiler,
// or we can compile asm version.
// So we can generate real code instead of this stub function.
// #if defined(_MSC_VER)
#pragma message("AES  HW_SW stub was used")
// #endif

#if !defined(USE_INTEL_VAES) && defined(Z7_USE_VAES_HW_STUB)
#define AES_TYPE_keys UInt32
#define AES_TYPE_data Byte
#endif

#define AES_FUNC_START(name) \
    void Z7_FASTCALL name(UInt32 *p, Byte *data, size_t numBlocks) \

#define AES_COMPAT_STUB(name) \
    AES_FUNC_START(name); \
    AES_FUNC_START(name ## _HW) \
    { name(p, data, numBlocks); }

AES_COMPAT_STUB (AesCbc_Encode)
AES_COMPAT_STUB (AesCbc_Decode)
AES_COMPAT_STUB (AesCtr_Code)
#endif // Z7_USE_AES_HW_STUB

#endif // USE_INTEL_AES


#ifndef USE_INTEL_VAES
#if defined(Z7_USE_VAES_HW_STUB)
// #if defined(_MSC_VER)
#pragma message("VAES HW_SW stub was used")
// #endif

#define VAES_COMPAT_STUB(name) \
    void Z7_FASTCALL name ## _256(UInt32 *p, Byte *data, size_t numBlocks); \
    void Z7_FASTCALL name ## _256(UInt32 *p, Byte *data, size_t numBlocks) \
    { name((AES_TYPE_keys *)(void *)p, (AES_TYPE_data *)(void *)data, numBlocks); }

VAES_COMPAT_STUB (AesCbc_Decode_HW)
VAES_COMPAT_STUB (AesCtr_Code_HW)
#endif
#endif // ! USE_INTEL_VAES




#elif defined(MY_CPU_ARM_OR_ARM64) && defined(MY_CPU_LE)

  #if   defined(__ARM_FEATURE_AES) \
     || defined(__ARM_FEATURE_CRYPTO)
    #define USE_HW_AES
  #else
    #if  defined(MY_CPU_ARM64) \
      || defined(__ARM_ARCH) && (__ARM_ARCH >= 4) \
      || defined(Z7_MSC_VER_ORIGINAL)
    #if  defined(__ARM_FP) && \
          (   defined(Z7_CLANG_VERSION) && (Z7_CLANG_VERSION >= 30800) \
           || defined(__GNUC__) && (__GNUC__ >= 6) \
          ) \
      || defined(Z7_MSC_VER_ORIGINAL) && (_MSC_VER >= 1910)
    #if  defined(MY_CPU_ARM64) \
      || !defined(Z7_CLANG_VERSION) \
      || defined(__ARM_NEON) && \
          (Z7_CLANG_VERSION < 170000 || \
           Z7_CLANG_VERSION > 170001)
      #define USE_HW_AES
    #endif
    #endif
    #endif
  #endif

#ifdef USE_HW_AES

// #pragma message("=== AES HW === ")
// __ARM_FEATURE_CRYPTO macro is deprecated in favor of the finer grained feature macro __ARM_FEATURE_AES

#if defined(__clang__) || defined(__GNUC__)
#if !defined(__ARM_FEATURE_AES) && \
    !defined(__ARM_FEATURE_CRYPTO)
  #ifdef MY_CPU_ARM64
#if defined(__clang__)
    #define ATTRIB_AES __attribute__((__target__("crypto")))
#else
    #define ATTRIB_AES __attribute__((__target__("+crypto")))
#endif
  #else
#if defined(__clang__)
    #define ATTRIB_AES __attribute__((__target__("armv8-a,aes")))
#else
    #define ATTRIB_AES __attribute__((__target__("fpu=crypto-neon-fp-armv8")))
#endif
  #endif
#endif
#else
  // _MSC_VER
  // for arm32
  #define _ARM_USE_NEW_NEON_INTRINSICS
#endif

#ifndef ATTRIB_AES
  #define ATTRIB_AES
#endif

#if defined(Z7_MSC_VER_ORIGINAL) && defined(MY_CPU_ARM64)
#include <arm64_neon.h>
#else
/*
  clang-17.0.1: error : Cannot select: intrinsic %llvm.arm.neon.aese
  clang
   3.8.1 : __ARM_NEON             :                    defined(__ARM_FEATURE_CRYPTO)
   7.0.1 : __ARM_NEON             : __ARM_ARCH >= 8 && defined(__ARM_FEATURE_CRYPTO)
  11.?.0 : __ARM_NEON && __ARM_FP : __ARM_ARCH >= 8 && defined(__ARM_FEATURE_CRYPTO)
  13.0.1 : __ARM_NEON && __ARM_FP : __ARM_ARCH >= 8 && defined(__ARM_FEATURE_AES)
  16     : __ARM_NEON && __ARM_FP : __ARM_ARCH >= 8
*/
#if defined(__clang__) && __clang_major__ < 16
#if !defined(__ARM_FEATURE_AES) && \
    !defined(__ARM_FEATURE_CRYPTO)
//     #pragma message("=== we set __ARM_FEATURE_CRYPTO 1 === ")
    Z7_DIAGNOSTIC_IGNORE_BEGIN_RESERVED_MACRO_IDENTIFIER
    #define Z7_ARM_FEATURE_CRYPTO_WAS_SET 1
// #if defined(__clang__) && __clang_major__ < 13
    #define __ARM_FEATURE_CRYPTO 1
// #else
    #define __ARM_FEATURE_AES 1
// #endif
    Z7_DIAGNOSTIC_IGNORE_END_RESERVED_MACRO_IDENTIFIER
#endif
#endif // clang

#if defined(__clang__)

#if defined(__ARM_ARCH) && __ARM_ARCH < 8
    Z7_DIAGNOSTIC_IGNORE_BEGIN_RESERVED_MACRO_IDENTIFIER
//    #pragma message("#define __ARM_ARCH 8")
    #undef  __ARM_ARCH
    #define __ARM_ARCH 8
    Z7_DIAGNOSTIC_IGNORE_END_RESERVED_MACRO_IDENTIFIER
#endif

#endif // clang

#include <arm_neon.h>

#if defined(Z7_ARM_FEATURE_CRYPTO_WAS_SET) && \
    defined(__ARM_FEATURE_CRYPTO) && \
    defined(__ARM_FEATURE_AES)
Z7_DIAGNOSTIC_IGNORE_BEGIN_RESERVED_MACRO_IDENTIFIER
    #undef __ARM_FEATURE_CRYPTO
    #undef __ARM_FEATURE_AES
    #undef Z7_ARM_FEATURE_CRYPTO_WAS_SET
Z7_DIAGNOSTIC_IGNORE_END_RESERVED_MACRO_IDENTIFIER
//    #pragma message("=== we undefine __ARM_FEATURE_CRYPTO === ")
#endif

#endif // Z7_MSC_VER_ORIGINAL

typedef uint8x16_t v128;

#define AES_FUNC_START(name) \
    void Z7_FASTCALL name(UInt32 *ivAes, Byte *data8, size_t numBlocks)
    // void Z7_FASTCALL name(v128 *p, v128 *data, size_t numBlocks)

#define AES_FUNC_START2(name) \
AES_FUNC_START (name); \
ATTRIB_AES \
AES_FUNC_START (name)

#define MM_OP(op, dest, src)  dest = op(dest, src);
#define MM_OP_m(op, src)      MM_OP(op, m, src)
#define MM_OP1_m(op)          m = op(m);

#define MM_XOR( dest, src)    MM_OP(veorq_u8, dest, src)
#define MM_XOR_m( src)        MM_XOR(m, src)

#define AES_E_m(k)     MM_OP_m (vaeseq_u8, k)
#define AES_E_MC_m(k)  AES_E_m (k)  MM_OP1_m(vaesmcq_u8)


AES_FUNC_START2 (AesCbc_Encode_HW)
{
  if (numBlocks == 0)
    return;
  {
  v128 * const p = (v128 *)(void *)ivAes;
  v128 *data = (v128 *)(void *)data8;
  v128 m = *p;
  const UInt32 numRounds2 = *(const UInt32 *)(p + 1);
  const v128 *w = p + (size_t)numRounds2 * 2;
  const v128 k0 = p[2];
  const v128 k1 = p[3];
  const v128 k2 = p[4];
  const v128 k3 = p[5];
  const v128 k4 = p[6];
  const v128 k5 = p[7];
  const v128 k6 = p[8];
  const v128 k7 = p[9];
  const v128 k8 = p[10];
  const v128 k9 = p[11];
  const v128 k_z4 = w[-2];
  const v128 k_z3 = w[-1];
  const v128 k_z2 = w[0];
  const v128 k_z1 = w[1];
  const v128 k_z0 = w[2];
  // we don't use optimization veorq_u8(*data, k_z0) that can reduce one cycle,
  // because gcc/clang compilers are not good for that optimization.
  do
  {
    MM_XOR_m (*data)
    AES_E_MC_m (k0)
    AES_E_MC_m (k1)
    AES_E_MC_m (k2)
    AES_E_MC_m (k3)
    AES_E_MC_m (k4)
    AES_E_MC_m (k5)
    if (numRounds2 >= 6)
    {
      AES_E_MC_m (k6)
      AES_E_MC_m (k7)
      if (numRounds2 != 6)
      {
        AES_E_MC_m (k8)
        AES_E_MC_m (k9)
      }
    }
    AES_E_MC_m (k_z4)
    AES_E_MC_m (k_z3)
    AES_E_MC_m (k_z2)
    AES_E_m    (k_z1)
    MM_XOR_m   (k_z0)
    *data++ = m;
  }
  while (--numBlocks);
  *p = m;
  }
}


#define WOP_1(op)
#define WOP_2(op)   WOP_1 (op)  op (m1, 1)
#define WOP_3(op)   WOP_2 (op)  op (m2, 2)
#define WOP_4(op)   WOP_3 (op)  op (m3, 3)
#define WOP_5(op)   WOP_4 (op)  op (m4, 4)
#define WOP_6(op)   WOP_5 (op)  op (m5, 5)
#define WOP_7(op)   WOP_6 (op)  op (m6, 6)
#define WOP_8(op)   WOP_7 (op)  op (m7, 7)

  #define NUM_WAYS      8
  #define WOP_M1    WOP_8

#define WOP(op)  op (m0, 0)   WOP_M1(op)

#define DECLARE_VAR(reg, ii)  v128 reg;
#define LOAD_data(  reg, ii)  reg = data[ii];
#define STORE_data( reg, ii)  data[ii] = reg;
#if (NUM_WAYS > 1)
#define XOR_data_M1(reg, ii)  MM_XOR (reg, data[ii- 1])
#endif

#define MM_OP_key(op, reg)  MM_OP (op, reg, key)

#define AES_D_m(k)      MM_OP_m (vaesdq_u8, k)
#define AES_D_IMC_m(k)  AES_D_m (k)  MM_OP1_m (vaesimcq_u8)

#define AES_XOR(   reg, ii)  MM_OP_key (veorq_u8,  reg)
#define AES_D(     reg, ii)  MM_OP_key (vaesdq_u8, reg)
#define AES_E(     reg, ii)  MM_OP_key (vaeseq_u8, reg)

#define AES_D_IMC( reg, ii)  AES_D (reg, ii)  reg = vaesimcq_u8(reg);
#define AES_E_MC(  reg, ii)  AES_E (reg, ii)  reg = vaesmcq_u8(reg);

#define CTR_START(reg, ii)  MM_OP (vaddq_u64, ctr, one)  reg = vreinterpretq_u8_u64(ctr);
#define CTR_END(  reg, ii)  MM_XOR (data[ii], reg)

#define WOP_KEY(op, n) { \
    const v128 key = w[n]; \
    WOP(op) }

#define WIDE_LOOP_START  \
    dataEnd = data + numBlocks;  \
    if (numBlocks >= NUM_WAYS)  \
    { dataEnd -= NUM_WAYS; do {  \

#define WIDE_LOOP_END  \
    data += NUM_WAYS;  \
    } while (data <= dataEnd);  \
    dataEnd += NUM_WAYS; }  \

#define SINGLE_LOOP  \
    for (; data < dataEnd; data++)


AES_FUNC_START2 (AesCbc_Decode_HW)
{
  v128 *p = (v128 *)(void *)ivAes;
  v128 *data = (v128 *)(void *)data8;
  v128 iv = *p;
  const v128 * const wStart = p + (size_t)*(const UInt32 *)(p + 1) * 2;
  const v128 *dataEnd;
  p += 2;
  
  WIDE_LOOP_START
  {
    const v128 *w = wStart;
    WOP (DECLARE_VAR)
    WOP (LOAD_data)
    WOP_KEY (AES_D_IMC, 2)
    do
    {
      WOP_KEY (AES_D_IMC, 1)
      WOP_KEY (AES_D_IMC, 0)
      w -= 2;
    }
    while (w != p);
    WOP_KEY (AES_D,   1)
    WOP_KEY (AES_XOR, 0)
    MM_XOR (m0, iv)
    WOP_M1 (XOR_data_M1)
    LOAD_data(iv, NUM_WAYS - 1)
    WOP (STORE_data)
  }
  WIDE_LOOP_END

  SINGLE_LOOP
  {
    const v128 *w = wStart;
    v128 m;  LOAD_data(m, 0)
    AES_D_IMC_m (w[2])
    do
    {
      AES_D_IMC_m (w[1])
      AES_D_IMC_m (w[0])
      w -= 2;
    }
    while (w != p);
    AES_D_m  (w[1])
    MM_XOR_m (w[0])
    MM_XOR_m (iv)
    LOAD_data(iv, 0)
    STORE_data(m, 0)
  }
  
  p[-2] = iv;
}


AES_FUNC_START2 (AesCtr_Code_HW)
{
  v128 *p = (v128 *)(void *)ivAes;
  v128 *data = (v128 *)(void *)data8;
  uint64x2_t ctr = vreinterpretq_u64_u8(*p);
  const v128 * const wEnd = p + (size_t)*(const UInt32 *)(p + 1) * 2;
  const v128 *dataEnd;
// the bug in clang:
// __builtin_neon_vsetq_lane_i64(__s0, (int8x16_t)__s1, __p2);
#if defined(__clang__) && (__clang_major__ <= 9)
#pragma GCC diagnostic ignored "-Wvector-conversion"
#endif
  const uint64x2_t one = vsetq_lane_u64(1, vdupq_n_u64(0), 0);
  p += 2;
  
  WIDE_LOOP_START
  {
    const v128 *w = p;
    WOP (DECLARE_VAR)
    WOP (CTR_START)
    do
    {
      WOP_KEY (AES_E_MC, 0)
      WOP_KEY (AES_E_MC, 1)
      w += 2;
    }
    while (w != wEnd);
    WOP_KEY (AES_E_MC, 0)
    WOP_KEY (AES_E,    1)
    WOP_KEY (AES_XOR,  2)
    WOP (CTR_END)
  }
  WIDE_LOOP_END

  SINGLE_LOOP
  {
    const v128 *w = p;
    v128 m;
    CTR_START (m, 0)
    do
    {
      AES_E_MC_m (w[0])
      AES_E_MC_m (w[1])
      w += 2;
    }
    while (w != wEnd);
    AES_E_MC_m (w[0])
    AES_E_m    (w[1])
    MM_XOR_m   (w[2])
    CTR_END (m, 0)
  }
  
  p[-2] = vreinterpretq_u8_u64(ctr);
}

#endif // USE_HW_AES

#endif // MY_CPU_ARM_OR_ARM64

#undef NUM_WAYS
#undef WOP_M1
#undef WOP
#undef DECLARE_VAR
#undef LOAD_data
#undef STORE_data
#undef USE_INTEL_AES
#undef USE_HW_AES
