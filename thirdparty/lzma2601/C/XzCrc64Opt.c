/* XzCrc64Opt.c -- CRC64 calculation (optimized functions)
: Igor Pavlov : Public domain */

#include "Precomp.h"

#include "CpuArch.h"

#if !defined(Z7_CRC64_NUM_TABLES) || Z7_CRC64_NUM_TABLES > 1

// for debug only : define Z7_CRC64_DEBUG_BE to test big-endian code in little-endian cpu
// #define Z7_CRC64_DEBUG_BE
#ifdef Z7_CRC64_DEBUG_BE
#undef MY_CPU_LE
#define MY_CPU_BE
#endif

#if defined(MY_CPU_64BIT)
#define Z7_CRC64_USE_64BIT
#endif

// the value Z7_CRC64_NUM_TABLES_USE must be defined to same value as in XzCrc64.c
#ifdef Z7_CRC64_NUM_TABLES
#define Z7_CRC64_NUM_TABLES_USE  Z7_CRC64_NUM_TABLES
#else
#define Z7_CRC64_NUM_TABLES_USE  12
#endif

#if Z7_CRC64_NUM_TABLES_USE % 4 || \
    Z7_CRC64_NUM_TABLES_USE < 4 || \
    Z7_CRC64_NUM_TABLES_USE > 4 * 4
  #error Stop_Compiling_Bad_CRC64_NUM_TABLES
#endif


#ifndef MY_CPU_BE

#define CRC64_UPDATE_BYTE_2(crc, b)  (table[((crc) ^ (b)) & 0xFF] ^ ((crc) >> 8))

#if defined(Z7_CRC64_USE_64BIT) && (Z7_CRC64_NUM_TABLES_USE % 8 == 0)

#define Q64LE(n, d) \
    ( (table + ((n) * 8 + 7) * 0x100)[((d)         ) & 0xFF] \
    ^ (table + ((n) * 8 + 6) * 0x100)[((d) >> 1 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 5) * 0x100)[((d) >> 2 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 4) * 0x100)[((d) >> 3 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 3) * 0x100)[((d) >> 4 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 2) * 0x100)[((d) >> 5 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 1) * 0x100)[((d) >> 6 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 0) * 0x100)[((d) >> 7 * 8)] )

#define R64(a)  *((const UInt64 *)(const void *)p + (a))

#else

#define Q32LE(n, d) \
    ( (table + ((n) * 4 + 3) * 0x100)[((d)         ) & 0xFF] \
    ^ (table + ((n) * 4 + 2) * 0x100)[((d) >> 1 * 8) & 0xFF] \
    ^ (table + ((n) * 4 + 1) * 0x100)[((d) >> 2 * 8) & 0xFF] \
    ^ (table + ((n) * 4 + 0) * 0x100)[((d) >> 3 * 8)] )

#define R32(a)  *((const UInt32 *)(const void *)p + (a))

#endif


#define CRC64_FUNC_PRE_LE2(step) \
UInt64 Z7_FASTCALL XzCrc64UpdateT ## step (UInt64 v, const void *data, size_t size, const UInt64 *table)

#define CRC64_FUNC_PRE_LE(step)   \
        CRC64_FUNC_PRE_LE2(step); \
        CRC64_FUNC_PRE_LE2(step)

CRC64_FUNC_PRE_LE(Z7_CRC64_NUM_TABLES_USE)
{
  const Byte *p = (const Byte *)data;
  const Byte *lim;
  for (; size && ((unsigned)(ptrdiff_t)p & (7 - (Z7_CRC64_NUM_TABLES_USE & 4))) != 0; size--, p++)
    v = CRC64_UPDATE_BYTE_2(v, *p);
  lim = p + size;
  if (size >= Z7_CRC64_NUM_TABLES_USE)
  {
    lim -= Z7_CRC64_NUM_TABLES_USE;
    do
    {
#if Z7_CRC64_NUM_TABLES_USE == 4
      const UInt32 d = (UInt32)v ^ R32(0);
      v = (v >> 32) ^ Q32LE(0, d);
#elif Z7_CRC64_NUM_TABLES_USE == 8
#ifdef Z7_CRC64_USE_64BIT
      v ^= R64(0);
      v = Q64LE(0, v);
#else
      UInt32 v0, v1;
      v0 = (UInt32)v         ^ R32(0);
      v1 = (UInt32)(v >> 32) ^ R32(1);
      v = Q32LE(1, v0) ^ Q32LE(0, v1);
#endif
#elif Z7_CRC64_NUM_TABLES_USE == 12
      UInt32 w;
      UInt32 v0, v1;
      v0 = (UInt32)v         ^ R32(0);
      v1 = (UInt32)(v >> 32) ^ R32(1);
      w = R32(2);
      v = Q32LE(0, w);
      v ^= Q32LE(2, v0) ^ Q32LE(1, v1);
#elif Z7_CRC64_NUM_TABLES_USE == 16
#ifdef Z7_CRC64_USE_64BIT
      UInt64 w;
      UInt64 x;
      w  = R64(1);      x = Q64LE(0, w);
      v ^= R64(0);  v = x ^ Q64LE(1, v);
#else
      UInt32 v0, v1;
      UInt32 r0, r1;
      v0 = (UInt32)v         ^ R32(0);
      v1 = (UInt32)(v >> 32) ^ R32(1);
      r0 =                     R32(2);
      r1 =                     R32(3);
      v  = Q32LE(1, r0) ^ Q32LE(0, r1);
      v ^= Q32LE(3, v0) ^ Q32LE(2, v1);
#endif
#else
#error Stop_Compiling_Bad_CRC64_NUM_TABLES
#endif
      p += Z7_CRC64_NUM_TABLES_USE;
    }
    while (p <= lim);
    lim += Z7_CRC64_NUM_TABLES_USE;
  }
  for (; p < lim; p++)
    v = CRC64_UPDATE_BYTE_2(v, *p);
  return v;
}

#undef CRC64_UPDATE_BYTE_2
#undef R32
#undef R64
#undef Q32LE
#undef Q64LE
#undef CRC64_FUNC_PRE_LE
#undef CRC64_FUNC_PRE_LE2

#endif




#ifndef MY_CPU_LE

#define CRC64_UPDATE_BYTE_2_BE(crc, b)  (table[((crc) >> 56) ^ (b)] ^ ((crc) << 8))

#if defined(Z7_CRC64_USE_64BIT) && (Z7_CRC64_NUM_TABLES_USE % 8 == 0)

#define Q64BE(n, d) \
    ( (table + ((n) * 8 + 0) * 0x100)[(Byte)(d)] \
    ^ (table + ((n) * 8 + 1) * 0x100)[((d) >> 1 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 2) * 0x100)[((d) >> 2 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 3) * 0x100)[((d) >> 3 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 4) * 0x100)[((d) >> 4 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 5) * 0x100)[((d) >> 5 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 6) * 0x100)[((d) >> 6 * 8) & 0xFF] \
    ^ (table + ((n) * 8 + 7) * 0x100)[((d) >> 7 * 8)] )

#ifdef Z7_CRC64_DEBUG_BE
  #define R64BE(a)  GetBe64a((const UInt64 *)(const void *)p + (a))
#else
  #define R64BE(a)         *((const UInt64 *)(const void *)p + (a))
#endif

#else

#define Q32BE(n, d) \
    ( (table + ((n) * 4 + 0) * 0x100)[(Byte)(d)] \
    ^ (table + ((n) * 4 + 1) * 0x100)[((d) >> 1 * 8) & 0xFF] \
    ^ (table + ((n) * 4 + 2) * 0x100)[((d) >> 2 * 8) & 0xFF] \
    ^ (table + ((n) * 4 + 3) * 0x100)[((d) >> 3 * 8)] )

#ifdef Z7_CRC64_DEBUG_BE
  #define R32BE(a)  GetBe32a((const UInt32 *)(const void *)p + (a))
#else
  #define R32BE(a)         *((const UInt32 *)(const void *)p + (a))
#endif

#endif

#define CRC64_FUNC_PRE_BE2(step) \
UInt64 Z7_FASTCALL XzCrc64UpdateBeT ## step (UInt64 v, const void *data, size_t size, const UInt64 *table)

#define CRC64_FUNC_PRE_BE(step)   \
        CRC64_FUNC_PRE_BE2(step); \
        CRC64_FUNC_PRE_BE2(step)

CRC64_FUNC_PRE_BE(Z7_CRC64_NUM_TABLES_USE)
{
  const Byte *p = (const Byte *)data;
  const Byte *lim;
  v = Z7_BSWAP64(v);
  for (; size && ((unsigned)(ptrdiff_t)p & (7 - (Z7_CRC64_NUM_TABLES_USE & 4))) != 0; size--, p++)
    v = CRC64_UPDATE_BYTE_2_BE(v, *p);
  lim = p + size;
  if (size >= Z7_CRC64_NUM_TABLES_USE)
  {
    lim -= Z7_CRC64_NUM_TABLES_USE;
    do
    {
#if   Z7_CRC64_NUM_TABLES_USE == 4
      const UInt32 d = (UInt32)(v >> 32) ^ R32BE(0);
      v = (v << 32) ^ Q32BE(0, d);
#elif Z7_CRC64_NUM_TABLES_USE == 12
      const UInt32 d1 = (UInt32)(v >> 32) ^ R32BE(0);
      const UInt32 d0 = (UInt32)(v      ) ^ R32BE(1);
      const UInt32 w =                      R32BE(2);
      v  = Q32BE(0, w);
      v ^= Q32BE(2, d1) ^ Q32BE(1, d0);

#elif Z7_CRC64_NUM_TABLES_USE == 8
  #ifdef Z7_CRC64_USE_64BIT
      v ^= R64BE(0);
      v  = Q64BE(0, v);
  #else
      const UInt32 d1 = (UInt32)(v >> 32) ^ R32BE(0);
      const UInt32 d0 = (UInt32)(v      ) ^ R32BE(1);
      v = Q32BE(1, d1) ^ Q32BE(0, d0);
  #endif
#elif Z7_CRC64_NUM_TABLES_USE == 16
  #ifdef Z7_CRC64_USE_64BIT
      const UInt64 w = R64BE(1);
      v ^= R64BE(0);
      v  = Q64BE(0, w) ^ Q64BE(1, v);
  #else
      const UInt32 d1 = (UInt32)(v >> 32) ^ R32BE(0);
      const UInt32 d0 = (UInt32)(v      ) ^ R32BE(1);
      const UInt32 w1 =                     R32BE(2);
      const UInt32 w0 =                     R32BE(3);
      v  = Q32BE(1, w1) ^ Q32BE(0, w0);
      v ^= Q32BE(3, d1) ^ Q32BE(2, d0);
  #endif
#else
#error Stop_Compiling_Bad_CRC64_NUM_TABLES
#endif
      p += Z7_CRC64_NUM_TABLES_USE;
    }
    while (p <= lim);
    lim += Z7_CRC64_NUM_TABLES_USE;
  }
  for (; p < lim; p++)
    v = CRC64_UPDATE_BYTE_2_BE(v, *p);
  return Z7_BSWAP64(v);
}

#undef CRC64_UPDATE_BYTE_2_BE
#undef R32BE
#undef R64BE
#undef Q32BE
#undef Q64BE
#undef CRC64_FUNC_PRE_BE
#undef CRC64_FUNC_PRE_BE2

#endif
#undef Z7_CRC64_NUM_TABLES_USE
#endif
