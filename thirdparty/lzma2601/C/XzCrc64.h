/* XzCrc64.h -- CRC64 calculation
2023-12-08 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_XZ_CRC64_H
#define ZIP7_INC_XZ_CRC64_H

#include <stddef.h>

#include "7zTypes.h"

EXTERN_C_BEGIN

// extern UInt64 g_Crc64Table[];

void Z7_FASTCALL Crc64GenerateTable(void);

#define CRC64_INIT_VAL UINT64_CONST(0xFFFFFFFFFFFFFFFF)
#define CRC64_GET_DIGEST(crc) ((crc) ^ CRC64_INIT_VAL)
// #define CRC64_UPDATE_BYTE(crc, b) (g_Crc64Table[((crc) ^ (b)) & 0xFF] ^ ((crc) >> 8))

UInt64 Z7_FASTCALL Crc64Update(UInt64 crc, const void *data, size_t size);
// UInt64 Z7_FASTCALL Crc64Calc(const void *data, size_t size);

EXTERN_C_END

#endif
