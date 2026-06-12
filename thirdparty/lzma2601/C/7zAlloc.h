/* 7zAlloc.h -- Allocation functions
2023-03-04 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_7Z_ALLOC_H
#define ZIP7_INC_7Z_ALLOC_H

#include "7zTypes.h"

EXTERN_C_BEGIN

void *SzAlloc(ISzAllocPtr p, size_t size);
void SzFree(ISzAllocPtr p, void *address);

void *SzAllocTemp(ISzAllocPtr p, size_t size);
void SzFreeTemp(ISzAllocPtr p, void *address);

EXTERN_C_END

#endif
