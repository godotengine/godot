/* XzIn.c - Xz input
: Igor Pavlov : Public domain */

#include "Precomp.h"

#include <string.h>

#include "7zCrc.h"
#include "Xz.h"
#include "CpuArch.h"

#define XZ_FOOTER_12B_ALIGNED16_SIG_CHECK(p) \
    (GetUi16a((const Byte *)(const void *)(p) + 10) == \
      (XZ_FOOTER_SIG_0 | (XZ_FOOTER_SIG_1 << 8)))

SRes Xz_ReadHeader(CXzStreamFlags *p, ISeqInStreamPtr inStream)
{
  UInt32 data32[XZ_STREAM_HEADER_SIZE / 4];
  size_t processedSize = XZ_STREAM_HEADER_SIZE;
  RINOK(SeqInStream_ReadMax(inStream, data32, &processedSize))
  if (processedSize != XZ_STREAM_HEADER_SIZE
      || memcmp(data32, XZ_SIG, XZ_SIG_SIZE) != 0)
    return SZ_ERROR_NO_ARCHIVE;
  return Xz_ParseHeader(p, (const Byte *)(const void *)data32);
}

#define READ_VARINT_AND_CHECK(buf, size, res) \
{ const unsigned s = Xz_ReadVarInt(buf, size, res); \
  if (s == 0) return SZ_ERROR_ARCHIVE; \
  size -= s; \
  buf += s; \
}

SRes XzBlock_ReadHeader(CXzBlock *p, ISeqInStreamPtr inStream, BoolInt *isIndex, UInt32 *headerSizeRes)
{
  MY_ALIGN(4)
  Byte header[XZ_BLOCK_HEADER_SIZE_MAX];
  unsigned headerSize;
  *headerSizeRes = 0;
  RINOK(SeqInStream_ReadByte(inStream, &header[0]))
  headerSize = header[0];
  if (headerSize == 0)
  {
    *headerSizeRes = 1;
    *isIndex = True;
    return SZ_OK;
  }

  *isIndex = False;
  headerSize = (headerSize << 2) + 4;
  *headerSizeRes = (UInt32)headerSize;
  {
    size_t processedSize = headerSize - 1;
    RINOK(SeqInStream_ReadMax(inStream, header + 1, &processedSize))
    if (processedSize != headerSize - 1)
      return SZ_ERROR_INPUT_EOF;
  }
  return XzBlock_Parse(p, header);
}


#define ADD_SIZE_CHECK(size, val) \
{ const UInt64 newSize = size + (val); \
  if (newSize < size) return XZ_SIZE_OVERFLOW; \
  size = newSize; \
}

UInt64 Xz_GetUnpackSize(const CXzStream *p)
{
  UInt64 size = 0;
  size_t i;
  for (i = 0; i < p->numBlocks; i++)
  {
    ADD_SIZE_CHECK(size, p->blocks[i].unpackSize)
  }
  return size;
}

UInt64 Xz_GetPackSize(const CXzStream *p)
{
  UInt64 size = 0;
  size_t i;
  for (i = 0; i < p->numBlocks; i++)
  {
    ADD_SIZE_CHECK(size, (p->blocks[i].totalSize + 3) & ~(UInt64)3)
  }
  return size;
}


// input;
//   CXzStream (p) is empty object.
//   size != 0
//   (size & 3) == 0
//   (buf) is aligned for at least 4 bytes.
// output:
//   p->numBlocks is number of allocated items in p->blocks
//   p->blocks[*] values must be ignored, if function returns error.
static SRes Xz_ParseIndex(CXzStream *p, const Byte *buf, size_t size, ISzAllocPtr alloc)
{
  size_t numBlocks;
  if (size < 5 || buf[0] != 0)
    return SZ_ERROR_ARCHIVE;
  size -= 4;
  {
    const UInt32 crc = CrcCalc(buf, size);
    if (crc != GetUi32a(buf + size))
      return SZ_ERROR_ARCHIVE;
  }
  buf++;
  size--;
  {
    UInt64 numBlocks64;
    READ_VARINT_AND_CHECK(buf, size, &numBlocks64)
    // (numBlocks64) is 63-bit value, so we can calculate (numBlocks64 * 2):
    if (numBlocks64 * 2 > size)
      return SZ_ERROR_ARCHIVE;
    if (numBlocks64 >= ((size_t)1 << (sizeof(size_t) * 8 - 1)) / sizeof(CXzBlockSizes))
      return SZ_ERROR_MEM; // SZ_ERROR_ARCHIVE
    numBlocks = (size_t)numBlocks64;
  }
  // Xz_Free(p, alloc); // it's optional, because (p) is empty already
  if (numBlocks)
  {
    CXzBlockSizes *blocks = (CXzBlockSizes *)ISzAlloc_Alloc(alloc, sizeof(CXzBlockSizes) * numBlocks);
    if (!blocks)
      return SZ_ERROR_MEM;
    p->blocks = blocks;
    p->numBlocks = numBlocks;
    // the caller will call Xz_Free() in case of error
    do
    {
      READ_VARINT_AND_CHECK(buf, size, &blocks->totalSize)
      READ_VARINT_AND_CHECK(buf, size, &blocks->unpackSize)
      if (blocks->totalSize == 0)
        return SZ_ERROR_ARCHIVE;
      blocks++;
    }
    while (--numBlocks);
  }
  if (size >= 4)
    return SZ_ERROR_ARCHIVE;
  while (size)
    if (buf[--size])
      return SZ_ERROR_ARCHIVE;
  return SZ_OK;
}


/*
static SRes Xz_ReadIndex(CXzStream *p, ILookInStreamPtr stream, UInt64 indexSize, ISzAllocPtr alloc)
{
  SRes res;
  size_t size;
  Byte *buf;
  if (indexSize >= ((size_t)1 << (sizeof(size_t) * 8 - 1)))
    return SZ_ERROR_MEM; // SZ_ERROR_ARCHIVE
  size = (size_t)indexSize;
  buf = (Byte *)ISzAlloc_Alloc(alloc, size);
  if (!buf)
    return SZ_ERROR_MEM;
  res = LookInStream_Read2(stream, buf, size, SZ_ERROR_UNSUPPORTED);
  if (res == SZ_OK)
    res = Xz_ParseIndex(p, buf, size, alloc);
  ISzAlloc_Free(alloc, buf);
  return res;
}
*/

static SRes LookInStream_SeekRead_ForArc(ILookInStreamPtr stream, UInt64 offset, void *buf, size_t size)
{
  RINOK(LookInStream_SeekTo(stream, offset))
  return LookInStream_Read(stream, buf, size);
  /* return LookInStream_Read2(stream, buf, size, SZ_ERROR_NO_ARCHIVE); */
}


/*
in:
  (*startOffset) is position in (stream) where xz_stream must be finished.
out:
  if returns SZ_OK, then (*startOffset) is position in stream that shows start of xz_stream.
*/
static SRes Xz_ReadBackward(CXzStream *p, ILookInStreamPtr stream, Int64 *startOffset, ISzAllocPtr alloc)
{
  #define TEMP_BUF_SIZE  (1 << 10)
  UInt32 buf32[TEMP_BUF_SIZE / 4];
  UInt64 pos = (UInt64)*startOffset;

  if ((pos & 3) || pos < XZ_STREAM_FOOTER_SIZE)
    return SZ_ERROR_NO_ARCHIVE;
  pos -= XZ_STREAM_FOOTER_SIZE;
  RINOK(LookInStream_SeekRead_ForArc(stream, pos, buf32, XZ_STREAM_FOOTER_SIZE))
  
  if (!XZ_FOOTER_12B_ALIGNED16_SIG_CHECK(buf32))
  {
    pos += XZ_STREAM_FOOTER_SIZE;
    for (;;)
    {
      // pos != 0
      // (pos & 3) == 0
      size_t i = pos >= TEMP_BUF_SIZE ? TEMP_BUF_SIZE : (size_t)pos;
      pos -= i;
      RINOK(LookInStream_SeekRead_ForArc(stream, pos, buf32, i))
      i /= 4;
      do
        if (buf32[i - 1] != 0)
          break;
      while (--i);

      pos += i * 4;
      #define XZ_STREAM_BACKWARD_READING_PAD_MAX (1 << 16)
      // here we don't support rare case with big padding for xz stream.
      // so we have padding limit for backward reading.
      if ((UInt64)*startOffset - pos > XZ_STREAM_BACKWARD_READING_PAD_MAX)
        return SZ_ERROR_NO_ARCHIVE;
      if (i)
        break;
    }
    // we try to open xz stream after skipping zero padding.
    // ((UInt64)*startOffset == pos) is possible here!
    if (pos < XZ_STREAM_FOOTER_SIZE)
      return SZ_ERROR_NO_ARCHIVE;
    pos -= XZ_STREAM_FOOTER_SIZE;
    RINOK(LookInStream_SeekRead_ForArc(stream, pos, buf32, XZ_STREAM_FOOTER_SIZE))
    if (!XZ_FOOTER_12B_ALIGNED16_SIG_CHECK(buf32))
      return SZ_ERROR_NO_ARCHIVE;
  }
  
  p->flags = (CXzStreamFlags)GetBe16a(buf32 + 2);
  if (!XzFlags_IsSupported(p->flags))
    return SZ_ERROR_UNSUPPORTED;
  {
    /* to eliminate GCC 6.3 warning:
       dereferencing type-punned pointer will break strict-aliasing rules */
    const UInt32 *buf_ptr = buf32;
    if (GetUi32a(buf_ptr) != CrcCalc(buf32 + 1, 6))
      return SZ_ERROR_ARCHIVE;
  }
  {
    const UInt64 indexSize = ((UInt64)GetUi32a(buf32 + 1) + 1) << 2;
    if (pos < indexSize)
      return SZ_ERROR_ARCHIVE;
    pos -= indexSize;
    // v25.00: relaxed indexSize check. We allow big index table.
    // if (indexSize > ((UInt32)1 << 31))
    if (indexSize >= ((size_t)1 << (sizeof(size_t) * 8 - 1)))
      return SZ_ERROR_MEM; // SZ_ERROR_ARCHIVE
    RINOK(LookInStream_SeekTo(stream, pos))
    // RINOK(Xz_ReadIndex(p, stream, indexSize, alloc))
    {
      SRes res;
      const size_t size = (size_t)indexSize;
      // if (size != indexSize) return SZ_ERROR_UNSUPPORTED;
      Byte *buf = (Byte *)ISzAlloc_Alloc(alloc, size);
      if (!buf)
        return SZ_ERROR_MEM;
      res = LookInStream_Read2(stream, buf, size, SZ_ERROR_UNSUPPORTED);
      if (res == SZ_OK)
        res = Xz_ParseIndex(p, buf, size, alloc);
      ISzAlloc_Free(alloc, buf);
      RINOK(res)
    }
  }
  {
    UInt64 total = Xz_GetPackSize(p);
    if (total == XZ_SIZE_OVERFLOW || total >= ((UInt64)1 << 63))
      return SZ_ERROR_ARCHIVE;
    total += XZ_STREAM_HEADER_SIZE;
    if (pos < total)
      return SZ_ERROR_ARCHIVE;
    pos -= total;
    RINOK(LookInStream_SeekTo(stream, pos))
    *startOffset = (Int64)pos;
  }
  {
    CXzStreamFlags headerFlags;
    CSecToRead secToRead;
    SecToRead_CreateVTable(&secToRead);
    secToRead.realStream = stream;
    RINOK(Xz_ReadHeader(&headerFlags, &secToRead.vt))
    return (p->flags == headerFlags) ? SZ_OK : SZ_ERROR_ARCHIVE;
  }
}


/* ---------- Xz Streams ---------- */

void Xzs_Construct(CXzs *p)
{
  Xzs_CONSTRUCT(p)
}

void Xzs_Free(CXzs *p, ISzAllocPtr alloc)
{
  size_t i;
  for (i = 0; i < p->num; i++)
    Xz_Free(&p->streams[i], alloc);
  ISzAlloc_Free(alloc, p->streams);
  p->num = p->numAllocated = 0;
  p->streams = NULL;
}

UInt64 Xzs_GetNumBlocks(const CXzs *p)
{
  UInt64 num = 0;
  size_t i;
  for (i = 0; i < p->num; i++)
    num += p->streams[i].numBlocks;
  return num;
}

UInt64 Xzs_GetUnpackSize(const CXzs *p)
{
  UInt64 size = 0;
  size_t i;
  for (i = 0; i < p->num; i++)
  {
    ADD_SIZE_CHECK(size, Xz_GetUnpackSize(&p->streams[i]))
  }
  return size;
}

/*
UInt64 Xzs_GetPackSize(const CXzs *p)
{
  UInt64 size = 0;
  size_t i;
  for (i = 0; i < p->num; i++)
  {
    ADD_SIZE_CHECK(size, Xz_GetTotalSize(&p->streams[i]))
  }
  return size;
}
*/

SRes Xzs_ReadBackward(CXzs *p, ILookInStreamPtr stream, Int64 *startOffset, ICompressProgressPtr progress, ISzAllocPtr alloc)
{
  Int64 endOffset = 0;
  // it's supposed that CXzs object is empty here.
  // if CXzs object is not empty, it will add new streams to that non-empty object.
  // Xzs_Free(p, alloc); // it's optional call to empty CXzs object.
  RINOK(ILookInStream_Seek(stream, &endOffset, SZ_SEEK_END))
  *startOffset = endOffset;
  for (;;)
  {
    CXzStream st;
    SRes res;
    Xz_CONSTRUCT(&st)
    res = Xz_ReadBackward(&st, stream, startOffset, alloc);
    // if (res == SZ_OK), then (*startOffset) is start offset of new stream if
    // if (res != SZ_OK), then (*startOffset) is unchend or it's expected start offset of stream with error
    st.startOffset = (UInt64)*startOffset;
    // we must store (st) object to array, or we must free (st) local object.
    if (res != SZ_OK)
    {
      Xz_Free(&st, alloc);
      return res;
    }
    if (p->num == p->numAllocated)
    {
      const size_t newNum = p->num + p->num / 4 + 1;
      void *data = ISzAlloc_Alloc(alloc, newNum * sizeof(CXzStream));
      if (!data)
      {
        Xz_Free(&st, alloc);
        return SZ_ERROR_MEM;
      }
      p->numAllocated = newNum;
      if (p->num != 0)
        memcpy(data, p->streams, p->num * sizeof(CXzStream));
      ISzAlloc_Free(alloc, p->streams);
      p->streams = (CXzStream *)data;
    }
    // we use direct copying of raw data from local variable (st) to object in array.
    // so we don't need to call Xz_Free(&st, alloc) after copying and after p->num++
    p->streams[p->num++] = st;
    if (*startOffset == 0)
      return SZ_OK;
    // seek operation is optional:
    // RINOK(LookInStream_SeekTo(stream, (UInt64)*startOffset))
    if (progress && ICompressProgress_Progress(progress, (UInt64)(endOffset - *startOffset), (UInt64)(Int64)-1) != SZ_OK)
      return SZ_ERROR_PROGRESS;
  }
}
