/* 7zCrcOpt.c -- CRC32 calculation (optimized functions)
2023-12-07 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include "CpuArch.h"

#if !defined(Z7_CRC_NUM_TABLES) || Z7_CRC_NUM_TABLES > 1

// for debug only : define Z7_CRC_DEBUG_BE to test big-endian code in little-endian cpu
// #define Z7_CRC_DEBUG_BE
#ifdef Z7_CRC_DEBUG_BE
#undef MY_CPU_LE
#define MY_CPU_BE
#endif

// the value Z7_CRC_NUM_TABLES_USE must be defined to same value as in 7zCrc.c
#ifdef Z7_CRC_NUM_TABLES
#define Z7_CRC_NUM_TABLES_USE  Z7_CRC_NUM_TABLES
#else
#define Z7_CRC_NUM_TABLES_USE  12
#endif

#if Z7_CRC_NUM_TABLES_USE % 4     || \
    Z7_CRC_NUM_TABLES_USE < 4 * 1 || \
    Z7_CRC_NUM_TABLES_USE > 4 * 6
  #error Stop_Compiling_Bad_Z7_CRC_NUM_TABLES
#endif


#ifndef MY_CPU_BE

#define CRC_UPDATE_BYTE_2(crc, b)  (table[((crc) ^ (b)) & 0xFF] ^ ((crc) >> 8))

#define Q(n, d) \
    ( (table + ((n) * 4 + 3) * 0x100)[(Byte)(d)] \
    ^ (table + ((n) * 4 + 2) * 0x100)[((d) >> 1 * 8) & 0xFF] \
    ^ (table + ((n) * 4 + 1) * 0x100)[((d) >> 2 * 8) & 0xFF] \
    ^ (table + ((n) * 4 + 0) * 0x100)[((d) >> 3 * 8)] )

#define R(a)  *((const UInt32 *)(const void *)p + (a))

#define CRC_FUNC_PRE_LE2(step) \
UInt32 Z7_FASTCALL CrcUpdateT ## step (UInt32 v, const void *data, size_t size, const UInt32 *table)

#define CRC_FUNC_PRE_LE(step)   \
        CRC_FUNC_PRE_LE2(step); \
        CRC_FUNC_PRE_LE2(step)

CRC_FUNC_PRE_LE(Z7_CRC_NUM_TABLES_USE)
{
  const Byte *p = (const Byte *)data;
  const Byte *lim;
  for (; size && ((unsigned)(ptrdiff_t)p & (7 - (Z7_CRC_NUM_TABLES_USE & 4))) != 0; size--, p++)
    v = CRC_UPDATE_BYTE_2(v, *p);
  lim = p + size;
  if (size >= Z7_CRC_NUM_TABLES_USE)
  {
    lim -= Z7_CRC_NUM_TABLES_USE;
    do
    {
      v ^= R(0);
      {
#if Z7_CRC_NUM_TABLES_USE == 1 * 4
        v = Q(0, v);
#else
#define U2(r, op) \
        { d = R(r);  x op Q(Z7_CRC_NUM_TABLES_USE / 4 - 1 - (r), d); }
        UInt32 d, x;
        U2(1, =)
#if Z7_CRC_NUM_TABLES_USE >= 3 * 4
#define U(r)  U2(r, ^=)
        U(2)
#if Z7_CRC_NUM_TABLES_USE >= 4 * 4
        U(3)
#if Z7_CRC_NUM_TABLES_USE >= 5 * 4
        U(4)
#if Z7_CRC_NUM_TABLES_USE >= 6 * 4
        U(5)
#if Z7_CRC_NUM_TABLES_USE >= 7 * 4
#error Stop_Compiling_Bad_Z7_CRC_NUM_TABLES
#endif
#endif
#endif
#endif
#endif
#undef U
#undef U2
        v = x ^ Q(Z7_CRC_NUM_TABLES_USE / 4 - 1, v);
#endif
      }
      p += Z7_CRC_NUM_TABLES_USE;
    }
    while (p <= lim);
    lim += Z7_CRC_NUM_TABLES_USE;
  }
  for (; p < lim; p++)
    v = CRC_UPDATE_BYTE_2(v, *p);
  return v;
}

#undef CRC_UPDATE_BYTE_2
#undef R
#undef Q
#undef CRC_FUNC_PRE_LE
#undef CRC_FUNC_PRE_LE2

#endif




#ifndef MY_CPU_LE

#define CRC_UPDATE_BYTE_2_BE(crc, b)  (table[((crc) >> 24) ^ (b)] ^ ((crc) << 8))

#define Q(n, d) \
    ( (table + ((n) * 4 + 0) * 0x100)[((d)) & 0xFF] \
    ^ (table + ((n) * 4 + 1) * 0x100)[((d) >> 1 * 8) & 0xFF] \
    ^ (table + ((n) * 4 + 2) * 0x100)[((d) >> 2 * 8) & 0xFF] \
    ^ (table + ((n) * 4 + 3) * 0x100)[((d) >> 3 * 8)] )

#ifdef Z7_CRC_DEBUG_BE
  #define R(a)  GetBe32a((const UInt32 *)(const void *)p + (a))
#else
  #define R(a)         *((const UInt32 *)(const void *)p + (a))
#endif


#define CRC_FUNC_PRE_BE2(step) \
UInt32 Z7_FASTCALL CrcUpdateT1_BeT ## step (UInt32 v, const void *data, size_t size, const UInt32 *table)

#define CRC_FUNC_PRE_BE(step)   \
        CRC_FUNC_PRE_BE2(step); \
        CRC_FUNC_PRE_BE2(step)

CRC_FUNC_PRE_BE(Z7_CRC_NUM_TABLES_USE)
{
  const Byte *p = (const Byte *)data;
  const Byte *lim;
  table += 0x100;
  v = Z7_BSWAP32(v);
  for (; size && ((unsigned)(ptrdiff_t)p & (7 - (Z7_CRC_NUM_TABLES_USE & 4))) != 0; size--, p++)
    v = CRC_UPDATE_BYTE_2_BE(v, *p);
  lim = p + size;
  if (size >= Z7_CRC_NUM_TABLES_USE)
  {
    lim -= Z7_CRC_NUM_TABLES_USE;
    do
    {
      v ^= R(0);
      {
#if Z7_CRC_NUM_TABLES_USE == 1 * 4
        v = Q(0, v);
#else
#define U2(r, op) \
        { d = R(r);  x op Q(Z7_CRC_NUM_TABLES_USE / 4 - 1 - (r), d); }
        UInt32 d, x;
        U2(1, =)
#if Z7_CRC_NUM_TABLES_USE >= 3 * 4
#define U(r)  U2(r, ^=)
        U(2)
#if Z7_CRC_NUM_TABLES_USE >= 4 * 4
        U(3)
#if Z7_CRC_NUM_TABLES_USE >= 5 * 4
        U(4)
#if Z7_CRC_NUM_TABLES_USE >= 6 * 4
        U(5)
#if Z7_CRC_NUM_TABLES_USE >= 7 * 4
#error Stop_Compiling_Bad_Z7_CRC_NUM_TABLES
#endif
#endif
#endif
#endif
#endif
#undef U
#undef U2
        v = x ^ Q(Z7_CRC_NUM_TABLES_USE / 4 - 1, v);
#endif
      }
      p += Z7_CRC_NUM_TABLES_USE;
    }
    while (p <= lim);
    lim += Z7_CRC_NUM_TABLES_USE;
  }
  for (; p < lim; p++)
    v = CRC_UPDATE_BYTE_2_BE(v, *p);
  return Z7_BSWAP32(v);
}

#undef CRC_UPDATE_BYTE_2_BE
#undef R
#undef Q
#undef CRC_FUNC_PRE_BE
#undef CRC_FUNC_PRE_BE2

#endif
#undef Z7_CRC_NUM_TABLES_USE
#endif
