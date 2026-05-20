/* 7zBuf.h -- Byte Buffer
2023-03-04 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_7Z_BUF_H
#define ZIP7_INC_7Z_BUF_H

#include "7zTypes.h"

EXTERN_C_BEGIN

typedef struct
{
  Byte *data;
  size_t size;
} CBuf;

void Buf_Init(CBuf *p);
int Buf_Create(CBuf *p, size_t size, ISzAllocPtr alloc);
void Buf_Free(CBuf *p, ISzAllocPtr alloc);

typedef struct
{
  Byte *data;
  size_t size;
  size_t pos;
} CDynBuf;

void DynBuf_Construct(CDynBuf *p);
void DynBuf_SeekToBeg(CDynBuf *p);
int DynBuf_Write(CDynBuf *p, const Byte *buf, size_t size, ISzAllocPtr alloc);
void DynBuf_Free(CDynBuf *p, ISzAllocPtr alloc);

EXTERN_C_END

#endif
