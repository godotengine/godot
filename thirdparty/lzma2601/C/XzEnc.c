/* XzEnc.c -- Xz Encode
: Igor Pavlov : Public domain */

#include "Precomp.h"

#include <stdlib.h>
#include <string.h>

#include "7zCrc.h"
#include "Bra.h"
#include "CpuArch.h"

#ifdef USE_SUBBLOCK
#include "Bcj3Enc.c"
#include "SbFind.c"
#include "SbEnc.c"
#endif

#include "XzEnc.h"

// #define Z7_ST

#ifndef Z7_ST
#include "MtCoder.h"
#else
#define MTCODER_THREADS_MAX 1
#define MTCODER_BLOCKS_MAX 1
#endif

#define XZ_GET_PAD_SIZE(dataSize) ((4 - ((unsigned)(dataSize) & 3)) & 3)

#define XZ_CHECK_SIZE_MAX 64
/* max pack size for LZMA2 block + pad4 + check_size: */
#define XZ_GET_MAX_BLOCK_PACK_SIZE(unpackSize) ((unpackSize) + ((unpackSize) >> 10) + 16 + XZ_CHECK_SIZE_MAX)

#define XZ_GET_ESTIMATED_BLOCK_TOTAL_PACK_SIZE(unpackSize) (XZ_BLOCK_HEADER_SIZE_MAX + XZ_GET_MAX_BLOCK_PACK_SIZE(unpackSize))


// #define XzBlock_ClearFlags(p)       (p)->flags = 0;
#define XzBlock_ClearFlags_SetNumFilters(p, n) (p)->flags = (Byte)((n) - 1);
#define XzBlock_SetHasPackSize(p)   (p)->flags |= XZ_BF_PACK_SIZE;
#define XzBlock_SetHasUnpackSize(p) (p)->flags |= XZ_BF_UNPACK_SIZE;


static SRes WriteBytes(ISeqOutStreamPtr s, const void *buf, size_t size)
{
  return (ISeqOutStream_Write(s, buf, size) == size) ? SZ_OK : SZ_ERROR_WRITE;
}

static SRes WriteBytes_UpdateCrc(ISeqOutStreamPtr s, const void *buf, size_t size, UInt32 *crc)
{
  *crc = CrcUpdate(*crc, buf, size);
  return WriteBytes(s, buf, size);
}


static SRes Xz_WriteHeader(CXzStreamFlags f, ISeqOutStreamPtr s)
{
  UInt32 crc;
  Byte header[XZ_STREAM_HEADER_SIZE];
  memcpy(header, XZ_SIG, XZ_SIG_SIZE);
  header[XZ_SIG_SIZE] = (Byte)(f >> 8);
  header[XZ_SIG_SIZE + 1] = (Byte)(f & 0xFF);
  crc = CrcCalc(header + XZ_SIG_SIZE, XZ_STREAM_FLAGS_SIZE);
  SetUi32(header + XZ_SIG_SIZE + XZ_STREAM_FLAGS_SIZE, crc)
  return WriteBytes(s, header, XZ_STREAM_HEADER_SIZE);
}


static SRes XzBlock_WriteHeader(const CXzBlock *p, ISeqOutStreamPtr s)
{
  Byte header[XZ_BLOCK_HEADER_SIZE_MAX];

  unsigned pos = 1;
  unsigned numFilters, i;
  header[pos++] = p->flags;

  if (XzBlock_HasPackSize(p)) pos += Xz_WriteVarInt(header + pos, p->packSize);
  if (XzBlock_HasUnpackSize(p)) pos += Xz_WriteVarInt(header + pos, p->unpackSize);
  numFilters = XzBlock_GetNumFilters(p);
  
  for (i = 0; i < numFilters; i++)
  {
    const CXzFilter *f = &p->filters[i];
    pos += Xz_WriteVarInt(header + pos, f->id);
    pos += Xz_WriteVarInt(header + pos, f->propsSize);
    memcpy(header + pos, f->props, f->propsSize);
    pos += f->propsSize;
  }

  while ((pos & 3) != 0)
    header[pos++] = 0;

  header[0] = (Byte)(pos >> 2);
  SetUi32(header + pos, CrcCalc(header, pos))
  return WriteBytes(s, header, pos + 4);
}




typedef struct
{
  size_t numBlocks;
  size_t size;
  size_t allocated;
  Byte *blocks;
} CXzEncIndex;


static void XzEncIndex_Construct(CXzEncIndex *p)
{
  p->numBlocks = 0;
  p->size = 0;
  p->allocated = 0;
  p->blocks = NULL;
}

static void XzEncIndex_Init(CXzEncIndex *p)
{
  p->numBlocks = 0;
  p->size = 0;
}

static void XzEncIndex_Free(CXzEncIndex *p, ISzAllocPtr alloc)
{
  if (p->blocks)
  {
    ISzAlloc_Free(alloc, p->blocks);
    p->blocks = NULL;
  }
  p->numBlocks = 0;
  p->size = 0;
  p->allocated = 0;
}


static SRes XzEncIndex_ReAlloc(CXzEncIndex *p, size_t newSize, ISzAllocPtr alloc)
{
  Byte *blocks = (Byte *)ISzAlloc_Alloc(alloc, newSize);
  if (!blocks)
    return SZ_ERROR_MEM;
  if (p->size != 0)
    memcpy(blocks, p->blocks, p->size);
  if (p->blocks)
    ISzAlloc_Free(alloc, p->blocks);
  p->blocks = blocks;
  p->allocated = newSize;
  return SZ_OK;
}


static SRes XzEncIndex_PreAlloc(CXzEncIndex *p, UInt64 numBlocks, UInt64 unpackSize, UInt64 totalSize, ISzAllocPtr alloc)
{
  UInt64 pos;
  {
    Byte buf[32];
    unsigned pos2 = Xz_WriteVarInt(buf, totalSize);
    pos2 += Xz_WriteVarInt(buf + pos2, unpackSize);
    pos = numBlocks * pos2;
  }
  
  if (pos <= p->allocated - p->size)
    return SZ_OK;
  {
    UInt64 newSize64 = p->size + pos;
    size_t newSize = (size_t)newSize64;
    if (newSize != newSize64)
      return SZ_ERROR_MEM;
    return XzEncIndex_ReAlloc(p, newSize, alloc);
  }
}


static SRes XzEncIndex_AddIndexRecord(CXzEncIndex *p, UInt64 unpackSize, UInt64 totalSize, ISzAllocPtr alloc)
{
  Byte buf[32];
  unsigned pos = Xz_WriteVarInt(buf, totalSize);
  pos += Xz_WriteVarInt(buf + pos, unpackSize);

  if (pos > p->allocated - p->size)
  {
    size_t newSize = p->allocated * 2 + 16 * 2;
    if (newSize < p->size + pos)
      return SZ_ERROR_MEM;
    RINOK(XzEncIndex_ReAlloc(p, newSize, alloc))
  }
  memcpy(p->blocks + p->size, buf, pos);
  p->size += pos;
  p->numBlocks++;
  return SZ_OK;
}


static SRes XzEncIndex_WriteFooter(const CXzEncIndex *p, CXzStreamFlags flags, ISeqOutStreamPtr s)
{
  Byte buf[32];
  UInt64 globalPos;
  UInt32 crc = CRC_INIT_VAL;
  unsigned pos = 1 + Xz_WriteVarInt(buf + 1, p->numBlocks);
  
  globalPos = pos;
  buf[0] = 0;
  RINOK(WriteBytes_UpdateCrc(s, buf, pos, &crc))
  RINOK(WriteBytes_UpdateCrc(s, p->blocks, p->size, &crc))
  globalPos += p->size;
  
  pos = XZ_GET_PAD_SIZE(globalPos);
  buf[1] = 0;
  buf[2] = 0;
  buf[3] = 0;
  globalPos += pos;
  
  crc = CrcUpdate(crc, buf + 4 - pos, pos);
  SetUi32(buf + 4, CRC_GET_DIGEST(crc))
  
  SetUi32(buf + 8 + 4, (UInt32)(globalPos >> 2))
  buf[8 + 8] = (Byte)(flags >> 8);
  buf[8 + 9] = (Byte)(flags & 0xFF);
  SetUi32(buf + 8, CrcCalc(buf + 8 + 4, 6))
  buf[8 + 10] = XZ_FOOTER_SIG_0;
  buf[8 + 11] = XZ_FOOTER_SIG_1;
  
  return WriteBytes(s, buf + 4 - pos, pos + 4 + 12);
}



/* ---------- CSeqCheckInStream ---------- */

typedef struct
{
  ISeqInStream vt;
  ISeqInStreamPtr realStream;
  const Byte *data;
  UInt64 limit;
  UInt64 processed;
  int realStreamFinished;
  CXzCheck check;
} CSeqCheckInStream;

static void SeqCheckInStream_Init(CSeqCheckInStream *p, unsigned checkMode)
{
  p->limit = (UInt64)(Int64)-1;
  p->processed = 0;
  p->realStreamFinished = 0;
  XzCheck_Init(&p->check, checkMode);
}

static void SeqCheckInStream_GetDigest(CSeqCheckInStream *p, Byte *digest)
{
  XzCheck_Final(&p->check, digest);
}

static SRes SeqCheckInStream_Read(ISeqInStreamPtr pp, void *data, size_t *size)
{
  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(CSeqCheckInStream)
  size_t size2 = *size;
  SRes res = SZ_OK;
  
  if (p->limit != (UInt64)(Int64)-1)
  {
    UInt64 rem = p->limit - p->processed;
    if (size2 > rem)
      size2 = (size_t)rem;
  }
  if (size2 != 0)
  {
    if (p->realStream)
    {
      res = ISeqInStream_Read(p->realStream, data, &size2);
      p->realStreamFinished = (size2 == 0) ? 1 : 0;
    }
    else
      memcpy(data, p->data + (size_t)p->processed, size2);
    XzCheck_Update(&p->check, data, size2);
    p->processed += size2;
  }
  *size = size2;
  return res;
}


/* ---------- CSeqSizeOutStream ---------- */

typedef struct
{
  ISeqOutStream vt;
  ISeqOutStreamPtr realStream;
  Byte *outBuf;
  size_t outBufLimit;
  UInt64 processed;
} CSeqSizeOutStream;

static size_t SeqSizeOutStream_Write(ISeqOutStreamPtr pp, const void *data, size_t size)
{
  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(CSeqSizeOutStream)
  if (p->realStream)
    size = ISeqOutStream_Write(p->realStream, data, size);
  else
  {
    if (size > p->outBufLimit - (size_t)p->processed)
      return 0;
    memcpy(p->outBuf + (size_t)p->processed, data, size);
  }
  p->processed += size;
  return size;
}


/* ---------- CSeqInFilter ---------- */

#define FILTER_BUF_SIZE (1 << 20)

typedef struct
{
  ISeqInStream vt;
  ISeqInStreamPtr realStream;
  IStateCoder StateCoder;
  Byte *buf;
  size_t curPos;
  size_t endPos;
  int srcWasFinished;
} CSeqInFilter;


static const z7_Func_BranchConv g_Funcs_BranchConv_RISC_Enc[] =
{
  Z7_BRANCH_CONV_ENC_2 (BranchConv_PPC),
  Z7_BRANCH_CONV_ENC_2 (BranchConv_IA64),
  Z7_BRANCH_CONV_ENC_2 (BranchConv_ARM),
  Z7_BRANCH_CONV_ENC_2 (BranchConv_ARMT),
  Z7_BRANCH_CONV_ENC_2 (BranchConv_SPARC),
  Z7_BRANCH_CONV_ENC_2 (BranchConv_ARM64),
  Z7_BRANCH_CONV_ENC_2 (BranchConv_RISCV)
};

static SizeT XzBcFilterStateBase_Filter_Enc(CXzBcFilterStateBase *p, Byte *data, SizeT size)
{
  switch (p->methodId)
  {
    case XZ_ID_Delta:
      Delta_Encode(p->delta_State, p->delta, data, size);
      break;
    case XZ_ID_X86:
      size = (SizeT)(z7_BranchConvSt_X86_Enc(data, size, p->ip, &p->X86_State) - data);
      break;
    default:
      if (p->methodId >= XZ_ID_PPC)
      {
        const UInt32 i = p->methodId - XZ_ID_PPC;
        if (i < Z7_ARRAY_SIZE(g_Funcs_BranchConv_RISC_Enc))
          size = (SizeT)(g_Funcs_BranchConv_RISC_Enc[i](data, size, p->ip) - data);
      }
      break;
  }
  p->ip += (UInt32)size;
  return size;
}


static SRes SeqInFilter_Init(CSeqInFilter *p, const CXzFilter *props, ISzAllocPtr alloc)
{
  if (!p->buf)
  {
    p->buf = (Byte *)ISzAlloc_Alloc(alloc, FILTER_BUF_SIZE);
    if (!p->buf)
      return SZ_ERROR_MEM;
  }
  p->curPos = p->endPos = 0;
  p->srcWasFinished = 0;
  RINOK(Xz_StateCoder_Bc_SetFromMethod_Func(&p->StateCoder, props->id, XzBcFilterStateBase_Filter_Enc, alloc))
  RINOK(p->StateCoder.SetProps(p->StateCoder.p, props->props, props->propsSize, alloc))
  p->StateCoder.Init(p->StateCoder.p);
  return SZ_OK;
}


static SRes SeqInFilter_Read(ISeqInStreamPtr pp, void *data, size_t *size)
{
  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(CSeqInFilter)
  const size_t sizeOriginal = *size;
  if (sizeOriginal == 0)
    return SZ_OK;
  *size = 0;
  
  for (;;)
  {
    if (!p->srcWasFinished && p->curPos == p->endPos)
    {
      p->curPos = 0;
      p->endPos = FILTER_BUF_SIZE;
      RINOK(ISeqInStream_Read(p->realStream, p->buf, &p->endPos))
      if (p->endPos == 0)
        p->srcWasFinished = 1;
    }
    {
      SizeT srcLen = p->endPos - p->curPos;
      ECoderStatus status;
      SRes res;
      *size = sizeOriginal;
      res = p->StateCoder.Code2(p->StateCoder.p,
          (Byte *)data, size,
          p->buf + p->curPos, &srcLen,
          p->srcWasFinished, CODER_FINISH_ANY,
          &status);
      p->curPos += srcLen;
      if (*size != 0 || srcLen == 0 || res != SZ_OK)
        return res;
    }
  }
}

Z7_FORCE_INLINE
static void SeqInFilter_Construct(CSeqInFilter *p)
{
  p->buf = NULL;
  p->StateCoder.p = NULL;
  p->vt.Read = SeqInFilter_Read;
}

Z7_FORCE_INLINE
static void SeqInFilter_Free(CSeqInFilter *p, ISzAllocPtr alloc)
{
  if (p->StateCoder.p)
  {
    p->StateCoder.Free(p->StateCoder.p, alloc);
    p->StateCoder.p = NULL;
  }
  if (p->buf)
  {
    ISzAlloc_Free(alloc, p->buf);
    p->buf = NULL;
  }
}


/* ---------- CSbEncInStream ---------- */

#ifdef USE_SUBBLOCK

typedef struct
{
  ISeqInStream vt;
  ISeqInStreamPtr inStream;
  CSbEnc enc;
} CSbEncInStream;

static SRes SbEncInStream_Read(ISeqInStreamPtr pp, void *data, size_t *size)
{
  CSbEncInStream *p = Z7_CONTAINER_FROM_VTBL(pp, CSbEncInStream, vt);
  size_t sizeOriginal = *size;
  if (sizeOriginal == 0)
    return SZ_OK;
  
  for (;;)
  {
    if (p->enc.needRead && !p->enc.readWasFinished)
    {
      size_t processed = p->enc.needReadSizeMax;
      RINOK(p->inStream->Read(p->inStream, p->enc.buf + p->enc.readPos, &processed))
      p->enc.readPos += processed;
      if (processed == 0)
      {
        p->enc.readWasFinished = True;
        p->enc.isFinalFinished = True;
      }
      p->enc.needRead = False;
    }
  
    *size = sizeOriginal;
    RINOK(SbEnc_Read(&p->enc, data, size))
    if (*size != 0 || !p->enc.needRead)
      return SZ_OK;
  }
}

void SbEncInStream_Construct(CSbEncInStream *p, ISzAllocPtr alloc)
{
  SbEnc_Construct(&p->enc, alloc);
  p->vt.Read = SbEncInStream_Read;
}

SRes SbEncInStream_Init(CSbEncInStream *p)
{
  return SbEnc_Init(&p->enc);
}

void SbEncInStream_Free(CSbEncInStream *p)
{
  SbEnc_Free(&p->enc);
}

#endif



/* ---------- CXzProps ---------- */


void XzFilterProps_Init(CXzFilterProps *p)
{
  p->id = 0;
  p->delta = 0;
  p->ip = 0;
  p->ipDefined = False;
}

void XzProps_Init(CXzProps *p)
{
  p->checkId = XZ_CHECK_CRC32;
  p->numThreadGroups = 0;
  p->blockSize = XZ_PROPS_BLOCK_SIZE_AUTO;
  p->numBlockThreads_Reduced = -1;
  p->numBlockThreads_Max = -1;
  p->numTotalThreads = -1;
  p->reduceSize = (UInt64)(Int64)-1;
  p->forceWriteSizesInHeader = 0;
  // p->forceWriteSizesInHeader = 1;

  XzFilterProps_Init(&p->filterProps);
  Lzma2EncProps_Init(&p->lzma2Props);
}


static void XzEncProps_Normalize_Fixed(CXzProps *p)
{
  UInt64 fileSize;
  int t1, t1n, t2, t2r, t3;
  {
    CLzma2EncProps tp = p->lzma2Props;
    if (tp.numTotalThreads <= 0)
      tp.numTotalThreads = p->numTotalThreads;
    Lzma2EncProps_Normalize(&tp);
    t1n = tp.numTotalThreads;
  }

  t1 = p->lzma2Props.numTotalThreads;
  t2 = p->numBlockThreads_Max;
  t3 = p->numTotalThreads;

  if (t2 > MTCODER_THREADS_MAX)
    t2 = MTCODER_THREADS_MAX;

  if (t3 <= 0)
  {
    if (t2 <= 0)
      t2 = 1;
    t3 = t1n * t2;
  }
  else if (t2 <= 0)
  {
    t2 = t3 / t1n;
    if (t2 == 0)
    {
      t1 = 1;
      t2 = t3;
    }
    if (t2 > MTCODER_THREADS_MAX)
      t2 = MTCODER_THREADS_MAX;
  }
  else if (t1 <= 0)
  {
    t1 = t3 / t2;
    if (t1 == 0)
      t1 = 1;
  }
  else
    t3 = t1n * t2;

  p->lzma2Props.numTotalThreads = t1;

  t2r = t2;

  fileSize = p->reduceSize;

  if ((p->blockSize < fileSize || fileSize == (UInt64)(Int64)-1))
    p->lzma2Props.lzmaProps.reduceSize = p->blockSize;

  Lzma2EncProps_Normalize(&p->lzma2Props);

  t1 = p->lzma2Props.numTotalThreads;

  {
    if (t2 > 1 && fileSize != (UInt64)(Int64)-1)
    {
      UInt64 numBlocks = fileSize / p->blockSize;
      if (numBlocks * p->blockSize != fileSize)
        numBlocks++;
      if (numBlocks < (unsigned)t2)
      {
        t2r = (int)numBlocks;
        if (t2r == 0)
          t2r = 1;
        t3 = t1 * t2r;
      }
    }
  }
  
  p->numBlockThreads_Max = t2;
  p->numBlockThreads_Reduced = t2r;
  p->numTotalThreads = t3;
}


static void XzProps_Normalize(CXzProps *p)
{
  /* we normalize xzProps properties, but we normalize only some of CXzProps::lzma2Props properties.
     Lzma2Enc_SetProps() will normalize lzma2Props later. */
  
  if (p->blockSize == XZ_PROPS_BLOCK_SIZE_SOLID)
  {
    p->lzma2Props.lzmaProps.reduceSize = p->reduceSize;
    p->numBlockThreads_Reduced = 1;
    p->numBlockThreads_Max = 1;
    if (p->lzma2Props.numTotalThreads <= 0)
      p->lzma2Props.numTotalThreads = p->numTotalThreads;
    return;
  }
  else
  {
    CLzma2EncProps *lzma2 = &p->lzma2Props;
    if (p->blockSize == LZMA2_ENC_PROPS_BLOCK_SIZE_AUTO)
    {
      // xz-auto
      p->lzma2Props.lzmaProps.reduceSize = p->reduceSize;

      if (lzma2->blockSize == LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID)
      {
        // if (xz-auto && lzma2-solid) - we use solid for both
        p->blockSize = XZ_PROPS_BLOCK_SIZE_SOLID;
        p->numBlockThreads_Reduced = 1;
        p->numBlockThreads_Max = 1;
        if (p->lzma2Props.numTotalThreads <= 0)
          p->lzma2Props.numTotalThreads = p->numTotalThreads;
      }
      else
      {
        // if (xz-auto && (lzma2-auto || lzma2-fixed_)
        //   we calculate block size for lzma2 and use that block size for xz, lzma2 uses single-chunk per block
        CLzma2EncProps tp = p->lzma2Props;
        if (tp.numTotalThreads <= 0)
          tp.numTotalThreads = p->numTotalThreads;
        
        Lzma2EncProps_Normalize(&tp);
        
        p->blockSize = tp.blockSize; // fixed or solid
        p->numBlockThreads_Reduced = tp.numBlockThreads_Reduced;
        p->numBlockThreads_Max = tp.numBlockThreads_Max;
        if (lzma2->blockSize == LZMA2_ENC_PROPS_BLOCK_SIZE_AUTO)
          lzma2->blockSize = tp.blockSize; // fixed or solid, LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID
        if (lzma2->lzmaProps.reduceSize > tp.blockSize && tp.blockSize != LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID)
          lzma2->lzmaProps.reduceSize = tp.blockSize;
        lzma2->numBlockThreads_Reduced = 1;
        lzma2->numBlockThreads_Max = 1;
        return;
      }
    }
    else
    {
      // xz-fixed
      // we can use xz::reduceSize or xz::blockSize as base for lzmaProps::reduceSize
      
      p->lzma2Props.lzmaProps.reduceSize = p->reduceSize;
      {
        UInt64 r = p->reduceSize;
        if (r > p->blockSize || r == (UInt64)(Int64)-1)
          r = p->blockSize;
        lzma2->lzmaProps.reduceSize = r;
      }
      if (lzma2->blockSize == LZMA2_ENC_PROPS_BLOCK_SIZE_AUTO)
        lzma2->blockSize = LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID;
      else if (lzma2->blockSize > p->blockSize && lzma2->blockSize != LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID)
        lzma2->blockSize = p->blockSize;
      
      XzEncProps_Normalize_Fixed(p);
    }
  }
}


/* ---------- CLzma2WithFilters ---------- */

typedef struct
{
  CLzma2EncHandle lzma2;
  CSeqInFilter filter;

  #ifdef USE_SUBBLOCK
  CSbEncInStream sb;
  #endif
} CLzma2WithFilters;


Z7_FORCE_INLINE
static void Lzma2WithFilters_Construct(CLzma2WithFilters *p)
{
  p->lzma2 = NULL;
  SeqInFilter_Construct(&p->filter);

  #ifdef USE_SUBBLOCK
  SbEncInStream_Construct(&p->sb, alloc);
  #endif
}


static SRes Lzma2WithFilters_Create(CLzma2WithFilters *p, ISzAllocPtr alloc, ISzAllocPtr bigAlloc)
{
  if (!p->lzma2)
  {
    p->lzma2 = Lzma2Enc_Create(alloc, bigAlloc);
    if (!p->lzma2)
      return SZ_ERROR_MEM;
  }
  return SZ_OK;
}


Z7_FORCE_INLINE
static void Lzma2WithFilters_Free(CLzma2WithFilters *p, ISzAllocPtr alloc)
{
  #ifdef USE_SUBBLOCK
  SbEncInStream_Free(&p->sb);
  #endif

  SeqInFilter_Free(&p->filter, alloc);
  if (p->lzma2)
  {
    Lzma2Enc_Destroy(p->lzma2);
    p->lzma2 = NULL;
  }
}


typedef struct
{
  UInt64 unpackSize;
  UInt64 totalSize;
  size_t headerSize;
} CXzEncBlockInfo;


static SRes Xz_CompressBlock(
    CLzma2WithFilters *lzmaf,
    
    ISeqOutStreamPtr outStream,
    Byte *outBufHeader,
    Byte *outBufData, size_t outBufDataLimit,

    ISeqInStreamPtr inStream,
    // UInt64 expectedSize,
    const Byte *inBuf, // used if (!inStream)
    size_t inBufSize,  // used if (!inStream), it's block size, props->blockSize is ignored

    const CXzProps *props,
    ICompressProgressPtr progress,
    int *inStreamFinished,  /* only for inStream version */
    CXzEncBlockInfo *blockSizes,
    ISzAllocPtr alloc,
    ISzAllocPtr allocBig)
{
  CSeqCheckInStream checkInStream;
  CSeqSizeOutStream seqSizeOutStream;
  CXzBlock block;
  unsigned filterIndex = 0;
  CXzFilter *filter = NULL;
  const CXzFilterProps *fp = &props->filterProps;
  if (fp->id == 0)
    fp = NULL;
  
  *inStreamFinished = False;
  
  RINOK(Lzma2WithFilters_Create(lzmaf, alloc, allocBig))
  
  RINOK(Lzma2Enc_SetProps(lzmaf->lzma2, &props->lzma2Props))
  
  // XzBlock_ClearFlags(&block)
  XzBlock_ClearFlags_SetNumFilters(&block, 1 + (fp ? 1 : 0))
  
  if (fp)
  {
    filter = &block.filters[filterIndex++];
    filter->id = fp->id;
    filter->propsSize = 0;
    
    if (fp->id == XZ_ID_Delta)
    {
      filter->props[0] = (Byte)(fp->delta - 1);
      filter->propsSize = 1;
    }
    else if (fp->ipDefined)
    {
      Byte *ptr = filter->props;
      SetUi32(ptr, fp->ip)
      filter->propsSize = 4;
    }
  }
  
  {
    CXzFilter *f = &block.filters[filterIndex++];
    f->id = XZ_ID_LZMA2;
    f->propsSize = 1;
    f->props[0] = Lzma2Enc_WriteProperties(lzmaf->lzma2);
  }
  
  seqSizeOutStream.vt.Write = SeqSizeOutStream_Write;
  seqSizeOutStream.realStream = outStream;
  seqSizeOutStream.outBuf = outBufData;
  seqSizeOutStream.outBufLimit = outBufDataLimit;
  seqSizeOutStream.processed = 0;
    
  /*
  if (expectedSize != (UInt64)(Int64)-1)
  {
    block.unpackSize = expectedSize;
    if (props->blockSize != (UInt64)(Int64)-1)
      if (expectedSize > props->blockSize)
        block.unpackSize = props->blockSize;
    XzBlock_SetHasUnpackSize(&block)
  }
  */

  if (outStream)
  {
    RINOK(XzBlock_WriteHeader(&block, &seqSizeOutStream.vt))
  }
  
  checkInStream.vt.Read = SeqCheckInStream_Read;
  SeqCheckInStream_Init(&checkInStream, props->checkId);
  
  checkInStream.realStream = inStream;
  checkInStream.data = inBuf;
  checkInStream.limit = props->blockSize;
  if (!inStream)
    checkInStream.limit = inBufSize;

  if (fp)
  {
    #ifdef USE_SUBBLOCK
    if (fp->id == XZ_ID_Subblock)
    {
      lzmaf->sb.inStream = &checkInStream.vt;
      RINOK(SbEncInStream_Init(&lzmaf->sb))
    }
    else
    #endif
    {
      lzmaf->filter.realStream = &checkInStream.vt;
      RINOK(SeqInFilter_Init(&lzmaf->filter, filter, alloc))
    }
  }

  {
    SRes res;
    Byte *outBuf = NULL;
    size_t outSize = 0;
    BoolInt useStream = (fp || inStream);
    // useStream = True;
    
    if (!useStream)
    {
      XzCheck_Update(&checkInStream.check, inBuf, inBufSize);
      checkInStream.processed = inBufSize;
    }
    
    if (!outStream)
    {
      outBuf = seqSizeOutStream.outBuf; //  + (size_t)seqSizeOutStream.processed;
      outSize = seqSizeOutStream.outBufLimit; // - (size_t)seqSizeOutStream.processed;
    }
    
    res = Lzma2Enc_Encode2(lzmaf->lzma2,
        outBuf ? NULL : &seqSizeOutStream.vt,
        outBuf,
        outBuf ? &outSize : NULL,
      
        useStream ?
          (fp ?
            (
            #ifdef USE_SUBBLOCK
            (fp->id == XZ_ID_Subblock) ? &lzmaf->sb.vt:
            #endif
            &lzmaf->filter.vt) :
            &checkInStream.vt) : NULL,
      
        useStream ? NULL : inBuf,
        useStream ? 0 : inBufSize,
        
        progress);
    
    if (outBuf)
      seqSizeOutStream.processed += outSize;
    
    RINOK(res)
    blockSizes->unpackSize = checkInStream.processed;
  }
  {
    Byte buf[4 + XZ_CHECK_SIZE_MAX];
    const unsigned padSize = XZ_GET_PAD_SIZE(seqSizeOutStream.processed);
    const UInt64 packSize = seqSizeOutStream.processed;
    
    buf[0] = 0;
    buf[1] = 0;
    buf[2] = 0;
    buf[3] = 0;
    
    SeqCheckInStream_GetDigest(&checkInStream, buf + 4);
    RINOK(WriteBytes(&seqSizeOutStream.vt, buf + (4 - padSize),
        padSize + XzFlags_GetCheckSize((CXzStreamFlags)props->checkId)))
    
    blockSizes->totalSize = seqSizeOutStream.processed - padSize;
    
    if (!outStream)
    {
      seqSizeOutStream.outBuf = outBufHeader;
      seqSizeOutStream.outBufLimit = XZ_BLOCK_HEADER_SIZE_MAX;
      seqSizeOutStream.processed = 0;
      
      block.unpackSize = blockSizes->unpackSize;
      XzBlock_SetHasUnpackSize(&block)
      
      block.packSize = packSize;
      XzBlock_SetHasPackSize(&block)
      
      RINOK(XzBlock_WriteHeader(&block, &seqSizeOutStream.vt))
      
      blockSizes->headerSize = (size_t)seqSizeOutStream.processed;
      blockSizes->totalSize += seqSizeOutStream.processed;
    }
  }
  
  if (inStream)
    *inStreamFinished = checkInStream.realStreamFinished;
  else
  {
    *inStreamFinished = False;
    if (checkInStream.processed != inBufSize)
      return SZ_ERROR_FAIL;
  }

  return SZ_OK;
}



typedef struct
{
  ICompressProgress vt;
  ICompressProgressPtr progress;
  UInt64 inOffset;
  UInt64 outOffset;
} CCompressProgress_XzEncOffset;


static SRes CompressProgress_XzEncOffset_Progress(ICompressProgressPtr pp, UInt64 inSize, UInt64 outSize)
{
  const CCompressProgress_XzEncOffset *p = Z7_CONTAINER_FROM_VTBL_CONST(pp, CCompressProgress_XzEncOffset, vt);
  inSize += p->inOffset;
  outSize += p->outOffset;
  return ICompressProgress_Progress(p->progress, inSize, outSize);
}




struct CXzEnc
{
  ISzAllocPtr alloc;
  ISzAllocPtr allocBig;

  CXzProps xzProps;
  UInt64 expectedDataSize;

  CXzEncIndex xzIndex;

  CLzma2WithFilters lzmaf_Items[MTCODER_THREADS_MAX];
  
  size_t outBufSize;       /* size of allocated outBufs[i] */
  Byte *outBufs[MTCODER_BLOCKS_MAX];

  #ifndef Z7_ST
  unsigned checkType;
  ISeqOutStreamPtr outStream;
  BoolInt mtCoder_WasConstructed;
  CMtCoder mtCoder;
  CXzEncBlockInfo EncBlocks[MTCODER_BLOCKS_MAX];
  #endif
};


static void XzEnc_Construct(CXzEnc *p)
{
  unsigned i;

  XzEncIndex_Construct(&p->xzIndex);

  for (i = 0; i < MTCODER_THREADS_MAX; i++)
    Lzma2WithFilters_Construct(&p->lzmaf_Items[i]);

  #ifndef Z7_ST
  p->mtCoder_WasConstructed = False;
  {
    for (i = 0; i < MTCODER_BLOCKS_MAX; i++)
      p->outBufs[i] = NULL;
    p->outBufSize = 0;
  }
  #endif
}


static void XzEnc_FreeOutBufs(CXzEnc *p)
{
  unsigned i;
  for (i = 0; i < MTCODER_BLOCKS_MAX; i++)
    if (p->outBufs[i])
    {
      ISzAlloc_Free(p->alloc, p->outBufs[i]);
      p->outBufs[i] = NULL;
    }
  p->outBufSize = 0;
}


static void XzEnc_Free(CXzEnc *p, ISzAllocPtr alloc)
{
  unsigned i;

  XzEncIndex_Free(&p->xzIndex, alloc);

  for (i = 0; i < MTCODER_THREADS_MAX; i++)
    Lzma2WithFilters_Free(&p->lzmaf_Items[i], alloc);
  
  #ifndef Z7_ST
  if (p->mtCoder_WasConstructed)
  {
    MtCoder_Destruct(&p->mtCoder);
    p->mtCoder_WasConstructed = False;
  }
  XzEnc_FreeOutBufs(p);
  #endif
}


CXzEncHandle XzEnc_Create(ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  CXzEnc *p = (CXzEnc *)ISzAlloc_Alloc(alloc, sizeof(CXzEnc));
  if (!p)
    return NULL;
  XzEnc_Construct(p);
  XzProps_Init(&p->xzProps);
  XzProps_Normalize(&p->xzProps);
  p->expectedDataSize = (UInt64)(Int64)-1;
  p->alloc = alloc;
  p->allocBig = allocBig;
  return (CXzEncHandle)p;
}

// #define GET_CXzEnc_p  CXzEnc *p = (CXzEnc *)(void *)pp;

void XzEnc_Destroy(CXzEncHandle p)
{
  // GET_CXzEnc_p
  XzEnc_Free(p, p->alloc);
  ISzAlloc_Free(p->alloc, p);
}


SRes XzEnc_SetProps(CXzEncHandle p, const CXzProps *props)
{
  // GET_CXzEnc_p
  p->xzProps = *props;
  XzProps_Normalize(&p->xzProps);
  return SZ_OK;
}


void XzEnc_SetDataSize(CXzEncHandle p, UInt64 expectedDataSiize)
{
  // GET_CXzEnc_p
  p->expectedDataSize = expectedDataSiize;
}




#ifndef Z7_ST

static SRes XzEnc_MtCallback_Code(void *pp, unsigned coderIndex, unsigned outBufIndex,
    const Byte *src, size_t srcSize, int finished)
{
  CXzEnc *me = (CXzEnc *)pp;
  SRes res;
  CMtProgressThunk progressThunk;
  Byte *dest;
  UNUSED_VAR(finished)
  {
    CXzEncBlockInfo *bInfo = &me->EncBlocks[outBufIndex];
    bInfo->totalSize = 0;
    bInfo->unpackSize = 0;
    bInfo->headerSize = 0;
    // v23.02: we don't compress empty blocks
    // also we must ignore that empty block in XzEnc_MtCallback_Write()
    if (srcSize == 0)
      return SZ_OK;
  }
  dest = me->outBufs[outBufIndex];
  if (!dest)
  {
    dest = (Byte *)ISzAlloc_Alloc(me->alloc, me->outBufSize);
    if (!dest)
      return SZ_ERROR_MEM;
    me->outBufs[outBufIndex] = dest;
  }
  
  MtProgressThunk_CreateVTable(&progressThunk);
  progressThunk.mtProgress = &me->mtCoder.mtProgress;
  MtProgressThunk_INIT(&progressThunk)

  {
    CXzEncBlockInfo blockSizes;
    int inStreamFinished;

    res = Xz_CompressBlock(
        &me->lzmaf_Items[coderIndex],
        
        NULL,
        dest,
        dest + XZ_BLOCK_HEADER_SIZE_MAX, me->outBufSize - XZ_BLOCK_HEADER_SIZE_MAX,

        NULL,
        // srcSize, // expectedSize
        src, srcSize,

        &me->xzProps,
        &progressThunk.vt,
        &inStreamFinished,
        &blockSizes,
        me->alloc,
        me->allocBig);
    
    if (res == SZ_OK)
      me->EncBlocks[outBufIndex] = blockSizes;

    return res;
  }
}


static SRes XzEnc_MtCallback_Write(void *pp, unsigned outBufIndex)
{
  CXzEnc *me = (CXzEnc *)pp;
  const CXzEncBlockInfo *bInfo = &me->EncBlocks[outBufIndex];
  // v23.02: we don't write empty blocks
  // note: if (bInfo->unpackSize == 0) then there is no compressed data of block
  if (bInfo->unpackSize == 0)
    return SZ_OK;
  {
    const Byte *data = me->outBufs[outBufIndex];
    RINOK(WriteBytes(me->outStream, data, bInfo->headerSize))
    {
      const UInt64 totalPackFull = bInfo->totalSize + XZ_GET_PAD_SIZE(bInfo->totalSize);
      RINOK(WriteBytes(me->outStream, data + XZ_BLOCK_HEADER_SIZE_MAX, (size_t)totalPackFull - bInfo->headerSize))
    }
    return XzEncIndex_AddIndexRecord(&me->xzIndex, bInfo->unpackSize, bInfo->totalSize, me->alloc);
  }
}

#endif



SRes XzEnc_Encode(CXzEncHandle p, ISeqOutStreamPtr outStream, ISeqInStreamPtr inStream, ICompressProgressPtr progress)
{
  // GET_CXzEnc_p

  const CXzProps *props = &p->xzProps;

  XzEncIndex_Init(&p->xzIndex);
  {
    UInt64 numBlocks = 1;
    UInt64 blockSize = props->blockSize;
    
    if (blockSize != XZ_PROPS_BLOCK_SIZE_SOLID
        && props->reduceSize != (UInt64)(Int64)-1)
    {
      numBlocks = props->reduceSize / blockSize;
      if (numBlocks * blockSize != props->reduceSize)
        numBlocks++;
    }
    else
      blockSize = (UInt64)1 << 62;
    
    RINOK(XzEncIndex_PreAlloc(&p->xzIndex, numBlocks, blockSize, XZ_GET_ESTIMATED_BLOCK_TOTAL_PACK_SIZE(blockSize), p->alloc))
  }

  RINOK(Xz_WriteHeader((CXzStreamFlags)props->checkId, outStream))


  #ifndef Z7_ST
  if (props->numBlockThreads_Reduced > 1)
  {
    IMtCoderCallback2 vt;

    if (!p->mtCoder_WasConstructed)
    {
      p->mtCoder_WasConstructed = True;
      MtCoder_Construct(&p->mtCoder);
    }

    vt.Code = XzEnc_MtCallback_Code;
    vt.Write = XzEnc_MtCallback_Write;

    p->checkType = props->checkId;
    p->xzProps = *props;
    
    p->outStream = outStream;

    p->mtCoder.allocBig = p->allocBig;
    p->mtCoder.progress = progress;
    p->mtCoder.inStream = inStream;
    p->mtCoder.inData = NULL;
    p->mtCoder.inDataSize = 0;
    p->mtCoder.mtCallback = &vt;
    p->mtCoder.mtCallbackObject = p;

    if (   props->blockSize == XZ_PROPS_BLOCK_SIZE_SOLID
        || props->blockSize == XZ_PROPS_BLOCK_SIZE_AUTO)
      return SZ_ERROR_FAIL;

    p->mtCoder.blockSize = (size_t)props->blockSize;
    if (p->mtCoder.blockSize != props->blockSize)
      return SZ_ERROR_PARAM; /* SZ_ERROR_MEM */

    {
      size_t destBlockSize = XZ_BLOCK_HEADER_SIZE_MAX + XZ_GET_MAX_BLOCK_PACK_SIZE(p->mtCoder.blockSize);
      if (destBlockSize < p->mtCoder.blockSize)
        return SZ_ERROR_PARAM;
      if (p->outBufSize != destBlockSize)
        XzEnc_FreeOutBufs(p);
      p->outBufSize = destBlockSize;
    }

    p->mtCoder.numThreadsMax = (unsigned)props->numBlockThreads_Max;
    p->mtCoder.numThreadGroups = props->numThreadGroups;
    p->mtCoder.expectedDataSize = p->expectedDataSize;
    
    RINOK(MtCoder_Code(&p->mtCoder))
  }
  else
  #endif
  {
    int writeStartSizes;
    CCompressProgress_XzEncOffset progress2;
    Byte *bufData = NULL;
    size_t bufSize = 0;

    progress2.vt.Progress = CompressProgress_XzEncOffset_Progress;
    progress2.inOffset = 0;
    progress2.outOffset = 0;
    progress2.progress = progress;
    
    writeStartSizes = 0;
    
    if (props->blockSize != XZ_PROPS_BLOCK_SIZE_SOLID)
    {
      writeStartSizes = (props->forceWriteSizesInHeader > 0);
      
      if (writeStartSizes)
      {
        size_t t2;
        size_t t = (size_t)props->blockSize;
        if (t != props->blockSize)
          return SZ_ERROR_PARAM;
        t = XZ_GET_MAX_BLOCK_PACK_SIZE(t);
        if (t < props->blockSize)
          return SZ_ERROR_PARAM;
        t2 = XZ_BLOCK_HEADER_SIZE_MAX + t;
        if (!p->outBufs[0] || t2 != p->outBufSize)
        {
          XzEnc_FreeOutBufs(p);
          p->outBufs[0] = (Byte *)ISzAlloc_Alloc(p->alloc, t2);
          if (!p->outBufs[0])
            return SZ_ERROR_MEM;
          p->outBufSize = t2;
        }
        bufData = p->outBufs[0] + XZ_BLOCK_HEADER_SIZE_MAX;
        bufSize = t;
      }
    }
    
    for (;;)
    {
      CXzEncBlockInfo blockSizes;
      int inStreamFinished;
      
      /*
      UInt64 rem = (UInt64)(Int64)-1;
      if (props->reduceSize != (UInt64)(Int64)-1
          && props->reduceSize >= progress2.inOffset)
        rem = props->reduceSize - progress2.inOffset;
      */

      blockSizes.headerSize = 0; // for GCC
      
      RINOK(Xz_CompressBlock(
          &p->lzmaf_Items[0],
          
          writeStartSizes ? NULL : outStream,
          writeStartSizes ? p->outBufs[0] : NULL,
          bufData, bufSize,
          
          inStream,
          // rem,
          NULL, 0,
          
          props,
          progress ? &progress2.vt : NULL,
          &inStreamFinished,
          &blockSizes,
          p->alloc,
          p->allocBig))

      {
        UInt64 totalPackFull = blockSizes.totalSize + XZ_GET_PAD_SIZE(blockSizes.totalSize);
      
        if (writeStartSizes)
        {
          RINOK(WriteBytes(outStream, p->outBufs[0], blockSizes.headerSize))
          RINOK(WriteBytes(outStream, bufData, (size_t)totalPackFull - blockSizes.headerSize))
        }
        
        RINOK(XzEncIndex_AddIndexRecord(&p->xzIndex, blockSizes.unpackSize, blockSizes.totalSize, p->alloc))
        
        progress2.inOffset += blockSizes.unpackSize;
        progress2.outOffset += totalPackFull;
      }
        
      if (inStreamFinished)
        break;
    }
  }

  return XzEncIndex_WriteFooter(&p->xzIndex, (CXzStreamFlags)props->checkId, outStream);
}


#include "Alloc.h"

SRes Xz_Encode(ISeqOutStreamPtr outStream, ISeqInStreamPtr inStream,
    const CXzProps *props, ICompressProgressPtr progress)
{
  SRes res;
  CXzEncHandle xz = XzEnc_Create(&g_Alloc, &g_BigAlloc);
  if (!xz)
    return SZ_ERROR_MEM;
  res = XzEnc_SetProps(xz, props);
  if (res == SZ_OK)
    res = XzEnc_Encode(xz, outStream, inStream, progress);
  XzEnc_Destroy(xz);
  return res;
}


SRes Xz_EncodeEmpty(ISeqOutStreamPtr outStream)
{
  SRes res;
  CXzEncIndex xzIndex;
  XzEncIndex_Construct(&xzIndex);
  res = Xz_WriteHeader((CXzStreamFlags)0, outStream);
  if (res == SZ_OK)
    res = XzEncIndex_WriteFooter(&xzIndex, (CXzStreamFlags)0, outStream);
  XzEncIndex_Free(&xzIndex, NULL); // g_Alloc
  return res;
}
