/* XzCrc64.c -- CRC64 calculation
2023-12-08 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include "XzCrc64.h"
#include "CpuArch.h"

#define kCrc64Poly UINT64_CONST(0xC96C5795D7870F42)

// for debug only : define Z7_CRC64_DEBUG_BE to test big-endian code in little-endian cpu
// #define Z7_CRC64_DEBUG_BE
#ifdef Z7_CRC64_DEBUG_BE
#undef MY_CPU_LE
#define MY_CPU_BE
#endif

#ifdef Z7_CRC64_NUM_TABLES
  #define Z7_CRC64_NUM_TABLES_USE  Z7_CRC64_NUM_TABLES
#else
  #define Z7_CRC64_NUM_TABLES_USE  12
#endif

#if Z7_CRC64_NUM_TABLES_USE < 1
  #error Stop_Compiling_Bad_Z7_CRC_NUM_TABLES
#endif


#if Z7_CRC64_NUM_TABLES_USE != 1

#ifndef MY_CPU_BE
  #define FUNC_NAME_LE_2(s)   XzCrc64UpdateT ## s
  #define FUNC_NAME_LE_1(s)   FUNC_NAME_LE_2(s)
  #define FUNC_NAME_LE        FUNC_NAME_LE_1(Z7_CRC64_NUM_TABLES_USE)
  UInt64 Z7_FASTCALL FUNC_NAME_LE (UInt64 v, const void *data, size_t size, const UInt64 *table);
#endif
#ifndef MY_CPU_LE
  #define FUNC_NAME_BE_2(s)   XzCrc64UpdateBeT ## s
  #define FUNC_NAME_BE_1(s)   FUNC_NAME_BE_2(s)
  #define FUNC_NAME_BE        FUNC_NAME_BE_1(Z7_CRC64_NUM_TABLES_USE)
  UInt64 Z7_FASTCALL FUNC_NAME_BE (UInt64 v, const void *data, size_t size, const UInt64 *table);
#endif

#if defined(MY_CPU_LE)
  #define FUNC_REF  FUNC_NAME_LE
#elif defined(MY_CPU_BE)
  #define FUNC_REF  FUNC_NAME_BE
#else
  #define FUNC_REF  g_Crc64Update
  static UInt64 (Z7_FASTCALL *FUNC_REF)(UInt64 v, const void *data, size_t size, const UInt64 *table);
#endif

#endif


MY_ALIGN(64)
static UInt64 g_Crc64Table[256 * Z7_CRC64_NUM_TABLES_USE];


UInt64 Z7_FASTCALL Crc64Update(UInt64 v, const void *data, size_t size)
{
#if Z7_CRC64_NUM_TABLES_USE == 1
  #define CRC64_UPDATE_BYTE_2(crc, b)  (table[((crc) ^ (b)) & 0xFF] ^ ((crc) >> 8))
  const UInt64 *table = g_Crc64Table;
  const Byte *p = (const Byte *)data;
  const Byte *lim = p + size;
  for (; p != lim; p++)
    v = CRC64_UPDATE_BYTE_2(v, *p);
  return v;
  #undef CRC64_UPDATE_BYTE_2
#else
  return FUNC_REF (v, data, size, g_Crc64Table);
#endif
}


Z7_NO_INLINE
void Z7_FASTCALL Crc64GenerateTable(void)
{
  unsigned i;
  for (i = 0; i < 256; i++)
  {
    UInt64 r = i;
    unsigned j;
    for (j = 0; j < 8; j++)
      r = (r >> 1) ^ (kCrc64Poly & ((UInt64)0 - (r & 1)));
    g_Crc64Table[i] = r;
  }

#if Z7_CRC64_NUM_TABLES_USE != 1
#if 1 || 1 && defined(MY_CPU_X86) // low register count
  for (i = 0; i < 256 * (Z7_CRC64_NUM_TABLES_USE - 1); i++)
  {
    const UInt64 r0 = g_Crc64Table[(size_t)i];
    g_Crc64Table[(size_t)i + 256] = g_Crc64Table[(Byte)r0] ^ (r0 >> 8);
  }
#else
  for (i = 0; i < 256 * (Z7_CRC64_NUM_TABLES_USE - 1); i += 2)
  {
    UInt64 r0 = g_Crc64Table[(size_t)(i)    ];
    UInt64 r1 = g_Crc64Table[(size_t)(i) + 1];
    r0 = g_Crc64Table[(Byte)r0] ^ (r0 >> 8);
    r1 = g_Crc64Table[(Byte)r1] ^ (r1 >> 8);
    g_Crc64Table[(size_t)i + 256    ] = r0;
    g_Crc64Table[(size_t)i + 256 + 1] = r1;
  }
#endif

#ifndef MY_CPU_LE
  {
#ifndef MY_CPU_BE
    UInt32 k = 1;
    if (*(const Byte *)&k == 1)
      FUNC_REF = FUNC_NAME_LE;
    else
#endif
    {
#ifndef MY_CPU_BE
      FUNC_REF = FUNC_NAME_BE;
#endif
      for (i = 0; i < 256 * Z7_CRC64_NUM_TABLES_USE; i++)
      {
        const UInt64 x = g_Crc64Table[i];
        g_Crc64Table[i] = Z7_BSWAP64(x);
      }
    }
  }
#endif // ndef MY_CPU_LE
#endif // Z7_CRC64_NUM_TABLES_USE != 1
}

#undef kCrc64Poly
#undef Z7_CRC64_NUM_TABLES_USE
#undef FUNC_REF
#undef FUNC_NAME_LE_2
#undef FUNC_NAME_LE_1
#undef FUNC_NAME_LE
#undef FUNC_NAME_BE_2
#undef FUNC_NAME_BE_1
#undef FUNC_NAME_BE
