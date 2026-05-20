/* 7zCrc.c -- CRC32 calculation and init
2024-03-01 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include "7zCrc.h"
#include "CpuArch.h"

// for debug:
// #define __ARM_FEATURE_CRC32 1

#ifdef __ARM_FEATURE_CRC32
// #pragma message("__ARM_FEATURE_CRC32")
#define Z7_CRC_HW_FORCE
#endif

// #define Z7_CRC_DEBUG_BE
#ifdef Z7_CRC_DEBUG_BE
#undef MY_CPU_LE
#define MY_CPU_BE
#endif

#ifdef Z7_CRC_HW_FORCE
  #define Z7_CRC_NUM_TABLES_USE  1
#else
#ifdef Z7_CRC_NUM_TABLES
  #define Z7_CRC_NUM_TABLES_USE  Z7_CRC_NUM_TABLES
#else
  #define Z7_CRC_NUM_TABLES_USE  12
#endif
#endif

#if Z7_CRC_NUM_TABLES_USE < 1
  #error Stop_Compiling_Bad_Z7_CRC_NUM_TABLES
#endif

#if defined(MY_CPU_LE) || (Z7_CRC_NUM_TABLES_USE == 1)
  #define Z7_CRC_NUM_TABLES_TOTAL  Z7_CRC_NUM_TABLES_USE
#else
  #define Z7_CRC_NUM_TABLES_TOTAL  (Z7_CRC_NUM_TABLES_USE + 1)
#endif

#ifndef Z7_CRC_HW_FORCE

#if Z7_CRC_NUM_TABLES_USE == 1 \
   || (!defined(MY_CPU_LE) && !defined(MY_CPU_BE))
#define CRC_UPDATE_BYTE_2(crc, b)   (table[((crc) ^ (b)) & 0xFF] ^ ((crc) >> 8))
#define Z7_CRC_UPDATE_T1_FUNC_NAME  CrcUpdateGT1
static UInt32 Z7_FASTCALL Z7_CRC_UPDATE_T1_FUNC_NAME(UInt32 v, const void *data, size_t size)
{
  const UInt32 *table = g_CrcTable;
  const Byte *p = (const Byte *)data;
  const Byte *lim = p + size;
  for (; p != lim; p++)
    v = CRC_UPDATE_BYTE_2(v, *p);
  return v;
}
#endif


#if Z7_CRC_NUM_TABLES_USE != 1
#ifndef MY_CPU_BE
  #define FUNC_NAME_LE_2(s)   CrcUpdateT ## s
  #define FUNC_NAME_LE_1(s)   FUNC_NAME_LE_2(s)
  #define FUNC_NAME_LE        FUNC_NAME_LE_1(Z7_CRC_NUM_TABLES_USE)
  UInt32 Z7_FASTCALL FUNC_NAME_LE (UInt32 v, const void *data, size_t size, const UInt32 *table);
#endif
#ifndef MY_CPU_LE
  #define FUNC_NAME_BE_2(s)   CrcUpdateT1_BeT ## s
  #define FUNC_NAME_BE_1(s)   FUNC_NAME_BE_2(s)
  #define FUNC_NAME_BE        FUNC_NAME_BE_1(Z7_CRC_NUM_TABLES_USE)
  UInt32 Z7_FASTCALL FUNC_NAME_BE (UInt32 v, const void *data, size_t size, const UInt32 *table);
#endif
#endif

#endif // Z7_CRC_HW_FORCE

/* ---------- hardware CRC ---------- */

#ifdef MY_CPU_LE

#if defined(MY_CPU_ARM_OR_ARM64)
// #pragma message("ARM*")

  #if (defined(__clang__) && (__clang_major__ >= 3)) \
     || defined(__GNUC__) && (__GNUC__ >= 6) && defined(MY_CPU_ARM64) \
     || defined(__GNUC__) && (__GNUC__ >= 8)
      #if !defined(__ARM_FEATURE_CRC32)
//        #pragma message("!defined(__ARM_FEATURE_CRC32)")
Z7_DIAGNOSTIC_IGNORE_BEGIN_RESERVED_MACRO_IDENTIFIER
        #define __ARM_FEATURE_CRC32 1
Z7_DIAGNOSTIC_IGNORE_END_RESERVED_MACRO_IDENTIFIER
        #define Z7_ARM_FEATURE_CRC32_WAS_SET
        #if defined(__clang__)
          #if defined(MY_CPU_ARM64)
            #define ATTRIB_CRC __attribute__((__target__("crc")))
          #else
            #define ATTRIB_CRC __attribute__((__target__("armv8-a,crc")))
          #endif
        #else
          #if defined(MY_CPU_ARM64)
#if !defined(Z7_GCC_VERSION) || (Z7_GCC_VERSION >= 60000)
            #define ATTRIB_CRC __attribute__((__target__("+crc")))
#endif
          #else
#if !defined(Z7_GCC_VERSION) || (__GNUC__  >= 8)
#if defined(__ARM_FP) && __GNUC__ >= 8
// for -mfloat-abi=hard: similar to <arm_acle.h>
            #define ATTRIB_CRC __attribute__((__target__("arch=armv8-a+crc+simd")))
#else
            #define ATTRIB_CRC __attribute__((__target__("arch=armv8-a+crc")))
#endif
#endif
          #endif
        #endif
      #endif
      #if defined(__ARM_FEATURE_CRC32)
      // #pragma message("<arm_acle.h>")
/*
arm_acle.h (GGC):
    before Nov 17, 2017:
#ifdef __ARM_FEATURE_CRC32

    Nov 17, 2017: gcc10.0  (gcc 9.2.0) checked"
#if __ARM_ARCH >= 8
#pragma GCC target ("arch=armv8-a+crc")

    Aug 22, 2019: GCC 8.4?, 9.2.1, 10.1:
#ifdef __ARM_FEATURE_CRC32
#ifdef __ARM_FP
#pragma GCC target ("arch=armv8-a+crc+simd")
#else
#pragma GCC target ("arch=armv8-a+crc")
#endif
*/
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
#if defined(Z7_GCC_VERSION) && (__GNUC__ ==   8) && (Z7_GCC_VERSION <  80400) \
 || defined(Z7_GCC_VERSION) && (__GNUC__ ==   9) && (Z7_GCC_VERSION <  90201) \
 || defined(Z7_GCC_VERSION) && (__GNUC__ ==  10) && (Z7_GCC_VERSION < 100100)
Z7_DIAGNOSTIC_IGNORE_BEGIN_RESERVED_MACRO_IDENTIFIER
// #pragma message("#define __ARM_ARCH 8")
#undef  __ARM_ARCH
#define __ARM_ARCH 8
Z7_DIAGNOSTIC_IGNORE_END_RESERVED_MACRO_IDENTIFIER
#endif
#endif
        #define Z7_CRC_HW_USE
        #include <arm_acle.h>
      #endif
  #elif defined(_MSC_VER)
    #if defined(MY_CPU_ARM64)
    #if (_MSC_VER >= 1910)
    #ifdef __clang__
       // #define Z7_CRC_HW_USE
       // #include <arm_acle.h>
    #else
       #define Z7_CRC_HW_USE
       #include <intrin.h>
    #endif
    #endif
    #endif
  #endif

#else // non-ARM*

// #define Z7_CRC_HW_USE // for debug : we can test HW-branch of code
#ifdef Z7_CRC_HW_USE
#include "7zCrcEmu.h"
#endif

#endif // non-ARM*



#if defined(Z7_CRC_HW_USE)

// #pragma message("USE ARM HW CRC")

#ifdef MY_CPU_64BIT
  #define CRC_HW_WORD_TYPE  UInt64
  #define CRC_HW_WORD_FUNC  __crc32d
#else
  #define CRC_HW_WORD_TYPE  UInt32
  #define CRC_HW_WORD_FUNC  __crc32w
#endif

#define CRC_HW_UNROLL_BYTES (sizeof(CRC_HW_WORD_TYPE) * 4)

#ifdef ATTRIB_CRC
  ATTRIB_CRC
#endif
Z7_NO_INLINE
#ifdef Z7_CRC_HW_FORCE
         UInt32 Z7_FASTCALL CrcUpdate
#else
  static UInt32 Z7_FASTCALL CrcUpdate_HW
#endif
    (UInt32 v, const void *data, size_t size)
{
  const Byte *p = (const Byte *)data;
  for (; size != 0 && ((unsigned)(ptrdiff_t)p & (CRC_HW_UNROLL_BYTES - 1)) != 0; size--)
    v = __crc32b(v, *p++);
  if (size >= CRC_HW_UNROLL_BYTES)
  {
    const Byte *lim = p + size;
    size &= CRC_HW_UNROLL_BYTES - 1;
    lim -= size;
    do
    {
      v = CRC_HW_WORD_FUNC(v, *(const CRC_HW_WORD_TYPE *)(const void *)(p));
      v = CRC_HW_WORD_FUNC(v, *(const CRC_HW_WORD_TYPE *)(const void *)(p + sizeof(CRC_HW_WORD_TYPE)));
      p += 2 * sizeof(CRC_HW_WORD_TYPE);
      v = CRC_HW_WORD_FUNC(v, *(const CRC_HW_WORD_TYPE *)(const void *)(p));
      v = CRC_HW_WORD_FUNC(v, *(const CRC_HW_WORD_TYPE *)(const void *)(p + sizeof(CRC_HW_WORD_TYPE)));
      p += 2 * sizeof(CRC_HW_WORD_TYPE);
    }
    while (p != lim);
  }
  
  for (; size != 0; size--)
    v = __crc32b(v, *p++);

  return v;
}

#ifdef Z7_ARM_FEATURE_CRC32_WAS_SET
Z7_DIAGNOSTIC_IGNORE_BEGIN_RESERVED_MACRO_IDENTIFIER
#undef __ARM_FEATURE_CRC32
Z7_DIAGNOSTIC_IGNORE_END_RESERVED_MACRO_IDENTIFIER
#undef Z7_ARM_FEATURE_CRC32_WAS_SET
#endif

#endif // defined(Z7_CRC_HW_USE)
#endif // MY_CPU_LE



#ifndef Z7_CRC_HW_FORCE

#if defined(Z7_CRC_HW_USE) || defined(Z7_CRC_UPDATE_T1_FUNC_NAME)
/*
typedef UInt32 (Z7_FASTCALL *Z7_CRC_UPDATE_WITH_TABLE_FUNC)
    (UInt32 v, const void *data, size_t size, const UInt32 *table);
Z7_CRC_UPDATE_WITH_TABLE_FUNC g_CrcUpdate;
*/
static unsigned g_Crc_Algo;
#if (!defined(MY_CPU_LE) && !defined(MY_CPU_BE))
static unsigned g_Crc_Be;
#endif
#endif // defined(Z7_CRC_HW_USE) || defined(Z7_CRC_UPDATE_T1_FUNC_NAME)



Z7_NO_INLINE
#ifdef Z7_CRC_HW_USE
  static UInt32 Z7_FASTCALL CrcUpdate_Base
#else
         UInt32 Z7_FASTCALL CrcUpdate
#endif
    (UInt32 crc, const void *data, size_t size)
{
#if Z7_CRC_NUM_TABLES_USE == 1
    return Z7_CRC_UPDATE_T1_FUNC_NAME(crc, data, size);
#else // Z7_CRC_NUM_TABLES_USE != 1
#ifdef Z7_CRC_UPDATE_T1_FUNC_NAME
  if (g_Crc_Algo == 1)
    return Z7_CRC_UPDATE_T1_FUNC_NAME(crc, data, size);
#endif

#ifdef MY_CPU_LE
    return FUNC_NAME_LE(crc, data, size, g_CrcTable);
#elif defined(MY_CPU_BE)
    return FUNC_NAME_BE(crc, data, size, g_CrcTable);
#else
  if (g_Crc_Be)
    return FUNC_NAME_BE(crc, data, size, g_CrcTable);
  else
    return FUNC_NAME_LE(crc, data, size, g_CrcTable);
#endif
#endif // Z7_CRC_NUM_TABLES_USE != 1
}


#ifdef Z7_CRC_HW_USE
Z7_NO_INLINE
UInt32 Z7_FASTCALL CrcUpdate(UInt32 crc, const void *data, size_t size)
{
  if (g_Crc_Algo == 0)
    return CrcUpdate_HW(crc, data, size);
  return CrcUpdate_Base(crc, data, size);
}
#endif

#endif // !defined(Z7_CRC_HW_FORCE)



UInt32 Z7_FASTCALL CrcCalc(const void *data, size_t size)
{
  return CrcUpdate(CRC_INIT_VAL, data, size) ^ CRC_INIT_VAL;
}


MY_ALIGN(64)
UInt32 g_CrcTable[256 * Z7_CRC_NUM_TABLES_TOTAL];


void Z7_FASTCALL CrcGenerateTable(void)
{
  UInt32 i;
  for (i = 0; i < 256; i++)
  {
#if defined(Z7_CRC_HW_FORCE)
    g_CrcTable[i] = __crc32b(i, 0);
#else
    #define kCrcPoly 0xEDB88320
    UInt32 r = i;
    unsigned j;
    for (j = 0; j < 8; j++)
      r = (r >> 1) ^ (kCrcPoly & ((UInt32)0 - (r & 1)));
    g_CrcTable[i] = r;
#endif
  }
  for (i = 256; i < 256 * Z7_CRC_NUM_TABLES_USE; i++)
  {
    const UInt32 r = g_CrcTable[(size_t)i - 256];
    g_CrcTable[i] = g_CrcTable[r & 0xFF] ^ (r >> 8);
  }

#if !defined(Z7_CRC_HW_FORCE) && \
    (defined(Z7_CRC_HW_USE) || defined(Z7_CRC_UPDATE_T1_FUNC_NAME) || defined(MY_CPU_BE))

#if Z7_CRC_NUM_TABLES_USE <= 1
    g_Crc_Algo = 1;
#else // Z7_CRC_NUM_TABLES_USE <= 1

#if defined(MY_CPU_LE)
    g_Crc_Algo = Z7_CRC_NUM_TABLES_USE;
#else // !defined(MY_CPU_LE)
  {
#ifndef MY_CPU_BE
    UInt32 k = 0x01020304;
    const Byte *p = (const Byte *)&k;
    if (p[0] == 4 && p[1] == 3)
      g_Crc_Algo = Z7_CRC_NUM_TABLES_USE;
    else if (p[0] != 1 || p[1] != 2)
      g_Crc_Algo = 1;
    else
#endif // MY_CPU_BE
    {
      for (i = 256 * Z7_CRC_NUM_TABLES_TOTAL - 1; i >= 256; i--)
      {
        const UInt32 x = g_CrcTable[(size_t)i - 256];
        g_CrcTable[i] = Z7_BSWAP32(x);
      }
#if defined(Z7_CRC_UPDATE_T1_FUNC_NAME)
      g_Crc_Algo = Z7_CRC_NUM_TABLES_USE;
#endif
#if (!defined(MY_CPU_LE) && !defined(MY_CPU_BE))
      g_Crc_Be = 1;
#endif
    }
  }
#endif  // !defined(MY_CPU_LE)

#ifdef MY_CPU_LE
#ifdef Z7_CRC_HW_USE
  if (CPU_IsSupported_CRC32())
    g_Crc_Algo = 0;
#endif // Z7_CRC_HW_USE
#endif // MY_CPU_LE

#endif // Z7_CRC_NUM_TABLES_USE <= 1
#endif // g_Crc_Algo was declared
}

Z7_CRC_UPDATE_FUNC z7_GetFunc_CrcUpdate(unsigned algo)
{
  if (algo == 0)
    return &CrcUpdate;

#if defined(Z7_CRC_HW_USE)
  if (algo == sizeof(CRC_HW_WORD_TYPE) * 8)
  {
#ifdef Z7_CRC_HW_FORCE
    return &CrcUpdate;
#else
    if (g_Crc_Algo == 0)
      return &CrcUpdate_HW;
#endif
  }
#endif

#ifndef Z7_CRC_HW_FORCE
  if (algo == Z7_CRC_NUM_TABLES_USE)
    return
  #ifdef Z7_CRC_HW_USE
      &CrcUpdate_Base;
  #else
      &CrcUpdate;
  #endif
#endif

  return NULL;
}

#undef kCrcPoly
#undef Z7_CRC_NUM_TABLES_USE
#undef Z7_CRC_NUM_TABLES_TOTAL
#undef CRC_UPDATE_BYTE_2
#undef FUNC_NAME_LE_2
#undef FUNC_NAME_LE_1
#undef FUNC_NAME_LE
#undef FUNC_NAME_BE_2
#undef FUNC_NAME_BE_1
#undef FUNC_NAME_BE

#undef CRC_HW_UNROLL_BYTES
#undef CRC_HW_WORD_FUNC
#undef CRC_HW_WORD_TYPE
