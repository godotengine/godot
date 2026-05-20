/* 7zStream.c -- 7z Stream functions
2023-04-02 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include <string.h>

#include "7zTypes.h"


SRes SeqInStream_ReadMax(ISeqInStreamPtr stream, void *buf, size_t *processedSize)
{
  size_t size = *processedSize;
  *processedSize = 0;
  while (size != 0)
  {
    size_t cur = size;
    const SRes res = ISeqInStream_Read(stream, buf, &cur);
    *processedSize += cur;
    buf = (void *)((Byte *)buf + cur);
    size -= cur;
    if (res != SZ_OK)
      return res;
    if (cur == 0)
      return SZ_OK;
  }
  return SZ_OK;
}

/*
SRes SeqInStream_Read2(ISeqInStreamPtr stream, void *buf, size_t size, SRes errorType)
{
  while (size != 0)
  {
    size_t processed = size;
    RINOK(ISeqInStream_Read(stream, buf, &processed))
    if (processed == 0)
      return errorType;
    buf = (void *)((Byte *)buf + processed);
    size -= processed;
  }
  return SZ_OK;
}

SRes SeqInStream_Read(ISeqInStreamPtr stream, void *buf, size_t size)
{
  return SeqInStream_Read2(stream, buf, size, SZ_ERROR_INPUT_EOF);
}
*/


SRes SeqInStream_ReadByte(ISeqInStreamPtr stream, Byte *buf)
{
  size_t processed = 1;
  RINOK(ISeqInStream_Read(stream, buf, &processed))
  return (processed == 1) ? SZ_OK : SZ_ERROR_INPUT_EOF;
}



SRes LookInStream_SeekTo(ILookInStreamPtr stream, UInt64 offset)
{
  Int64 t = (Int64)offset;
  return ILookInStream_Seek(stream, &t, SZ_SEEK_SET);
}

SRes LookInStream_LookRead(ILookInStreamPtr stream, void *buf, size_t *size)
{
  const void *lookBuf;
  if (*size == 0)
    return SZ_OK;
  RINOK(ILookInStream_Look(stream, &lookBuf, size))
  memcpy(buf, lookBuf, *size);
  return ILookInStream_Skip(stream, *size);
}

SRes LookInStream_Read2(ILookInStreamPtr stream, void *buf, size_t size, SRes errorType)
{
  while (size != 0)
  {
    size_t processed = size;
    RINOK(ILookInStream_Read(stream, buf, &processed))
    if (processed == 0)
      return errorType;
    buf = (void *)((Byte *)buf + processed);
    size -= processed;
  }
  return SZ_OK;
}

SRes LookInStream_Read(ILookInStreamPtr stream, void *buf, size_t size)
{
  return LookInStream_Read2(stream, buf, size, SZ_ERROR_INPUT_EOF);
}



#define GET_LookToRead2  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(CLookToRead2)

static SRes LookToRead2_Look_Lookahead(ILookInStreamPtr pp, const void **buf, size_t *size)
{
  SRes res = SZ_OK;
  GET_LookToRead2
  size_t size2 = p->size - p->pos;
  if (size2 == 0 && *size != 0)
  {
    p->pos = 0;
    p->size = 0;
    size2 = p->bufSize;
    res = ISeekInStream_Read(p->realStream, p->buf, &size2);
    p->size = size2;
  }
  if (*size > size2)
    *size = size2;
  *buf = p->buf + p->pos;
  return res;
}

static SRes LookToRead2_Look_Exact(ILookInStreamPtr pp, const void **buf, size_t *size)
{
  SRes res = SZ_OK;
  GET_LookToRead2
  size_t size2 = p->size - p->pos;
  if (size2 == 0 && *size != 0)
  {
    p->pos = 0;
    p->size = 0;
    if (*size > p->bufSize)
      *size = p->bufSize;
    res = ISeekInStream_Read(p->realStream, p->buf, size);
    size2 = p->size = *size;
  }
  if (*size > size2)
    *size = size2;
  *buf = p->buf + p->pos;
  return res;
}

static SRes LookToRead2_Skip(ILookInStreamPtr pp, size_t offset)
{
  GET_LookToRead2
  p->pos += offset;
  return SZ_OK;
}

static SRes LookToRead2_Read(ILookInStreamPtr pp, void *buf, size_t *size)
{
  GET_LookToRead2
  size_t rem = p->size - p->pos;
  if (rem == 0)
    return ISeekInStream_Read(p->realStream, buf, size);
  if (rem > *size)
    rem = *size;
  memcpy(buf, p->buf + p->pos, rem);
  p->pos += rem;
  *size = rem;
  return SZ_OK;
}

static SRes LookToRead2_Seek(ILookInStreamPtr pp, Int64 *pos, ESzSeek origin)
{
  GET_LookToRead2
  p->pos = p->size = 0;
  return ISeekInStream_Seek(p->realStream, pos, origin);
}

void LookToRead2_CreateVTable(CLookToRead2 *p, int lookahead)
{
  p->vt.Look = lookahead ?
      LookToRead2_Look_Lookahead :
      LookToRead2_Look_Exact;
  p->vt.Skip = LookToRead2_Skip;
  p->vt.Read = LookToRead2_Read;
  p->vt.Seek = LookToRead2_Seek;
}



static SRes SecToLook_Read(ISeqInStreamPtr pp, void *buf, size_t *size)
{
  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(CSecToLook)
  return LookInStream_LookRead(p->realStream, buf, size);
}

void SecToLook_CreateVTable(CSecToLook *p)
{
  p->vt.Read = SecToLook_Read;
}

static SRes SecToRead_Read(ISeqInStreamPtr pp, void *buf, size_t *size)
{
  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(CSecToRead)
  return ILookInStream_Read(p->realStream, buf, size);
}

void SecToRead_CreateVTable(CSecToRead *p)
{
  p->vt.Read = SecToRead_Read;
}
