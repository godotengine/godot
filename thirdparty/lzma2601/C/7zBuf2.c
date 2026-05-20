/* 7zBuf2.c -- Byte Buffer
2017-04-03 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include <string.h>

#include "7zBuf.h"

void DynBuf_Construct(CDynBuf *p)
{
  p->data = 0;
  p->size = 0;
  p->pos = 0;
}

void DynBuf_SeekToBeg(CDynBuf *p)
{
  p->pos = 0;
}

int DynBuf_Write(CDynBuf *p, const Byte *buf, size_t size, ISzAllocPtr alloc)
{
  if (size > p->size - p->pos)
  {
    size_t newSize = p->pos + size;
    Byte *data;
    newSize += newSize / 4;
    data = (Byte *)ISzAlloc_Alloc(alloc, newSize);
    if (!data)
      return 0;
    p->size = newSize;
    if (p->pos != 0)
      memcpy(data, p->data, p->pos);
    ISzAlloc_Free(alloc, p->data);
    p->data = data;
  }
  if (size != 0)
  {
    memcpy(p->data + p->pos, buf, size);
    p->pos += size;
  }
  return 1;
}

void DynBuf_Free(CDynBuf *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->data);
  p->data = 0;
  p->size = 0;
  p->pos = 0;
}
