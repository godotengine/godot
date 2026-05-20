/* SwapBytes.h -- Byte Swap conversion filter
2023-04-02 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_SWAP_BYTES_H
#define ZIP7_INC_SWAP_BYTES_H

#include "7zTypes.h"

EXTERN_C_BEGIN

void z7_SwapBytes2(UInt16 *data, size_t numItems);
void z7_SwapBytes4(UInt32 *data, size_t numItems);
void z7_SwapBytesPrepare(void);

EXTERN_C_END

#endif
