/* Lzma2Enc.c -- LZMA2 Encoder
: Igor Pavlov : Public domain */

#include "Precomp.h"

#include <string.h>

/* #define Z7_ST */

#include "Lzma2Enc.h"

#ifndef Z7_ST
#include "MtCoder.h"
#else
#define MTCODER_THREADS_MAX 1
#endif

#define LZMA2_CONTROL_LZMA (1 << 7)
#define LZMA2_CONTROL_COPY_NO_RESET 2
#define LZMA2_CONTROL_COPY_RESET_DIC 1
#define LZMA2_CONTROL_EOF 0

#define LZMA2_LCLP_MAX 4

#define LZMA2_DIC_SIZE_FROM_PROP(p) (((UInt32)2 | ((p) & 1)) << ((p) / 2 + 11))

#define LZMA2_PACK_SIZE_MAX (1 << 16)
#define LZMA2_COPY_CHUNK_SIZE LZMA2_PACK_SIZE_MAX
#define LZMA2_UNPACK_SIZE_MAX (1 << 21)
#define LZMA2_KEEP_WINDOW_SIZE LZMA2_UNPACK_SIZE_MAX

#define LZMA2_CHUNK_SIZE_COMPRESSED_MAX ((1 << 16) + 16)


#define PRF(x) /* x */


/* ---------- CLimitedSeqInStream ---------- */

typedef struct
{
  ISeqInStream vt;
  ISeqInStreamPtr realStream;
  UInt64 limit;
  UInt64 processed;
  int finished;
} CLimitedSeqInStream;

static void LimitedSeqInStream_Init(CLimitedSeqInStream *p)
{
  p->limit = (UInt64)(Int64)-1;
  p->processed = 0;
  p->finished = 0;
}

static SRes LimitedSeqInStream_Read(ISeqInStreamPtr pp, void *data, size_t *size)
{
  Z7_CONTAINER_FROM_VTBL_TO_DECL_VAR_pp_vt_p(CLimitedSeqInStream)
  size_t size2 = *size;
  SRes res = SZ_OK;
  
  if (p->limit != (UInt64)(Int64)-1)
  {
    const UInt64 rem = p->limit - p->processed;
    if (size2 > rem)
      size2 = (size_t)rem;
  }
  if (size2 != 0)
  {
    res = ISeqInStream_Read(p->realStream, data, &size2);
    p->finished = (size2 == 0 ? 1 : 0);
    p->processed += size2;
  }
  *size = size2;
  return res;
}


/* ---------- CLzma2EncInt ---------- */

typedef struct
{
  CLzmaEncHandle enc;
  Byte propsAreSet;
  Byte propsByte;
  Byte needInitState;
  Byte needInitProp;
  UInt64 srcPos;
} CLzma2EncInt;


static SRes Lzma2EncInt_InitStream(CLzma2EncInt *p, const CLzma2EncProps *props)
{
  if (!p->propsAreSet)
  {
    SizeT propsSize = LZMA_PROPS_SIZE;
    Byte propsEncoded[LZMA_PROPS_SIZE];
    RINOK(LzmaEnc_SetProps(p->enc, &props->lzmaProps))
    RINOK(LzmaEnc_WriteProperties(p->enc, propsEncoded, &propsSize))
    p->propsByte = propsEncoded[0];
    p->propsAreSet = True;
  }
  return SZ_OK;
}

static void Lzma2EncInt_InitBlock(CLzma2EncInt *p)
{
  p->srcPos = 0;
  p->needInitState = True;
  p->needInitProp = True;
}


SRes LzmaEnc_PrepareForLzma2(CLzmaEncHandle p, ISeqInStreamPtr inStream, UInt32 keepWindowSize,
    ISzAllocPtr alloc, ISzAllocPtr allocBig);
SRes LzmaEnc_MemPrepare(CLzmaEncHandle p, const Byte *src, SizeT srcLen,
    UInt32 keepWindowSize, ISzAllocPtr alloc, ISzAllocPtr allocBig);
SRes LzmaEnc_CodeOneMemBlock(CLzmaEncHandle p, BoolInt reInit,
    Byte *dest, size_t *destLen, UInt32 desiredPackSize, UInt32 *unpackSize);
const Byte *LzmaEnc_GetCurBuf(CLzmaEncHandle p);
void LzmaEnc_Finish(CLzmaEncHandle p);
void LzmaEnc_SaveState(CLzmaEncHandle p);
void LzmaEnc_RestoreState(CLzmaEncHandle p);

/*
UInt32 LzmaEnc_GetNumAvailableBytes(CLzmaEncHandle p);
*/

static SRes Lzma2EncInt_EncodeSubblock(CLzma2EncInt *p, Byte *outBuf,
    size_t *packSizeRes, ISeqOutStreamPtr outStream)
{
  size_t packSizeLimit = *packSizeRes;
  size_t packSize = packSizeLimit;
  UInt32 unpackSize = LZMA2_UNPACK_SIZE_MAX;
  unsigned lzHeaderSize = 5 + (p->needInitProp ? 1 : 0);
  BoolInt useCopyBlock;
  SRes res;

  *packSizeRes = 0;
  if (packSize < lzHeaderSize)
    return SZ_ERROR_OUTPUT_EOF;
  packSize -= lzHeaderSize;
  
  LzmaEnc_SaveState(p->enc);
  res = LzmaEnc_CodeOneMemBlock(p->enc, p->needInitState,
      outBuf + lzHeaderSize, &packSize, LZMA2_PACK_SIZE_MAX, &unpackSize);
  
  PRF(printf("\npackSize = %7d unpackSize = %7d  ", packSize, unpackSize));

  if (unpackSize == 0)
    return res;

  if (res == SZ_OK)
    useCopyBlock = (packSize + 2 >= unpackSize || packSize > (1 << 16));
  else
  {
    if (res != SZ_ERROR_OUTPUT_EOF)
      return res;
    res = SZ_OK;
    useCopyBlock = True;
  }

  if (useCopyBlock)
  {
    size_t destPos = 0;
    PRF(printf("################# COPY           "));

    while (unpackSize > 0)
    {
      const UInt32 u = (unpackSize < LZMA2_COPY_CHUNK_SIZE) ? unpackSize : LZMA2_COPY_CHUNK_SIZE;
      if (packSizeLimit - destPos < u + 3)
        return SZ_ERROR_OUTPUT_EOF;
      outBuf[destPos++] = (Byte)(p->srcPos == 0 ? LZMA2_CONTROL_COPY_RESET_DIC : LZMA2_CONTROL_COPY_NO_RESET);
      outBuf[destPos++] = (Byte)((u - 1) >> 8);
      outBuf[destPos++] = (Byte)(u - 1);
      memcpy(outBuf + destPos, LzmaEnc_GetCurBuf(p->enc) - unpackSize, u);
      unpackSize -= u;
      destPos += u;
      p->srcPos += u;
      
      if (outStream)
      {
        *packSizeRes += destPos;
        if (ISeqOutStream_Write(outStream, outBuf, destPos) != destPos)
          return SZ_ERROR_WRITE;
        destPos = 0;
      }
      else
        *packSizeRes = destPos;
      /* needInitState = True; */
    }
    
    LzmaEnc_RestoreState(p->enc);
    return SZ_OK;
  }

  {
    size_t destPos = 0;
    const UInt32 u = unpackSize - 1;
    const UInt32 pm = (UInt32)(packSize - 1);
    const unsigned mode = (p->srcPos == 0) ? 3 : (p->needInitState ? (p->needInitProp ? 2 : 1) : 0);

    PRF(printf("               "));

    outBuf[destPos++] = (Byte)(LZMA2_CONTROL_LZMA | (mode << 5) | ((u >> 16) & 0x1F));
    outBuf[destPos++] = (Byte)(u >> 8);
    outBuf[destPos++] = (Byte)u;
    outBuf[destPos++] = (Byte)(pm >> 8);
    outBuf[destPos++] = (Byte)pm;
    
    if (p->needInitProp)
      outBuf[destPos++] = p->propsByte;
    
    p->needInitProp = False;
    p->needInitState = False;
    destPos += packSize;
    p->srcPos += unpackSize;

    if (outStream)
      if (ISeqOutStream_Write(outStream, outBuf, destPos) != destPos)
        return SZ_ERROR_WRITE;
    
    *packSizeRes = destPos;
    return SZ_OK;
  }
}


/* ---------- Lzma2 Props ---------- */

void Lzma2EncProps_Init(CLzma2EncProps *p)
{
  LzmaEncProps_Init(&p->lzmaProps);
  p->blockSize = LZMA2_ENC_PROPS_BLOCK_SIZE_AUTO;
  p->numBlockThreads_Reduced = -1;
  p->numBlockThreads_Max = -1;
  p->numTotalThreads = -1;
  p->numThreadGroups = 0;
}

void Lzma2EncProps_Normalize(CLzma2EncProps *p)
{
  UInt64 fileSize;
  int t1, t1n, t2, t2r, t3;
  {
    CLzmaEncProps lzmaProps = p->lzmaProps;
    LzmaEncProps_Normalize(&lzmaProps);
    t1n = lzmaProps.numThreads;
  }

  t1 = p->lzmaProps.numThreads;
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

  p->lzmaProps.numThreads = t1;

  t2r = t2;

  fileSize = p->lzmaProps.reduceSize;

  if (   p->blockSize != LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID
      && p->blockSize != LZMA2_ENC_PROPS_BLOCK_SIZE_AUTO
      && (p->blockSize < fileSize || fileSize == (UInt64)(Int64)-1))
    p->lzmaProps.reduceSize = p->blockSize;

  LzmaEncProps_Normalize(&p->lzmaProps);

  p->lzmaProps.reduceSize = fileSize;

  t1 = p->lzmaProps.numThreads;

  if (p->blockSize == LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID)
  {
    t2r = t2 = 1;
    t3 = t1;
  }
  else if (p->blockSize == LZMA2_ENC_PROPS_BLOCK_SIZE_AUTO && t2 <= 1)
  {
    /* if there is no block multi-threading, we use SOLID block */
    p->blockSize = LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID;
  }
  else
  {
    if (p->blockSize == LZMA2_ENC_PROPS_BLOCK_SIZE_AUTO)
    {
      const UInt32 kMinSize = (UInt32)1 << 20;
      const UInt32 kMaxSize = (UInt32)1 << 28;
      const UInt32 dictSize = p->lzmaProps.dictSize;
      UInt64 blockSize = (UInt64)dictSize << 2;
      if (blockSize < kMinSize) blockSize = kMinSize;
      if (blockSize > kMaxSize) blockSize = kMaxSize;
      if (blockSize < dictSize) blockSize = dictSize;
      blockSize += (kMinSize - 1);
      blockSize &= ~(UInt64)(kMinSize - 1);
      p->blockSize = blockSize;
    }
    
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


static SRes Progress(ICompressProgressPtr p, UInt64 inSize, UInt64 outSize)
{
  return (p && ICompressProgress_Progress(p, inSize, outSize) != SZ_OK) ? SZ_ERROR_PROGRESS : SZ_OK;
}


/* ---------- Lzma2 ---------- */

struct CLzma2Enc
{
  Byte propEncoded;
  CLzma2EncProps props;
  UInt64 expectedDataSize;
  
  Byte *tempBufLzma;

  ISzAllocPtr alloc;
  ISzAllocPtr allocBig;

  CLzma2EncInt coders[MTCODER_THREADS_MAX];

  #ifndef Z7_ST
  
  ISeqOutStreamPtr outStream;
  Byte *outBuf;
  size_t outBuf_Rem;   /* remainder in outBuf */

  size_t outBufSize;   /* size of allocated outBufs[i] */
  size_t outBufsDataSizes[MTCODER_BLOCKS_MAX];
  BoolInt mtCoder_WasConstructed;
  CMtCoder mtCoder;
  Byte *outBufs[MTCODER_BLOCKS_MAX];

  #endif
};



CLzma2EncHandle Lzma2Enc_Create(ISzAllocPtr alloc, ISzAllocPtr allocBig)
{
  CLzma2Enc *p = (CLzma2Enc *)ISzAlloc_Alloc(alloc, sizeof(CLzma2Enc));
  if (!p)
    return NULL;
  Lzma2EncProps_Init(&p->props);
  Lzma2EncProps_Normalize(&p->props);
  p->expectedDataSize = (UInt64)(Int64)-1;
  p->tempBufLzma = NULL;
  p->alloc = alloc;
  p->allocBig = allocBig;
  {
    unsigned i;
    for (i = 0; i < MTCODER_THREADS_MAX; i++)
      p->coders[i].enc = NULL;
  }
  
  #ifndef Z7_ST
  p->mtCoder_WasConstructed = False;
  {
    unsigned i;
    for (i = 0; i < MTCODER_BLOCKS_MAX; i++)
      p->outBufs[i] = NULL;
    p->outBufSize = 0;
  }
  #endif

  return (CLzma2EncHandle)p;
}


#ifndef Z7_ST

static void Lzma2Enc_FreeOutBufs(CLzma2Enc *p)
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

#endif

// #define GET_CLzma2Enc_p  CLzma2Enc *p = (CLzma2Enc *)(void *)p;

void Lzma2Enc_Destroy(CLzma2EncHandle p)
{
  // GET_CLzma2Enc_p
  unsigned i;
  for (i = 0; i < MTCODER_THREADS_MAX; i++)
  {
    CLzma2EncInt *t = &p->coders[i];
    if (t->enc)
    {
      LzmaEnc_Destroy(t->enc, p->alloc, p->allocBig);
      t->enc = NULL;
    }
  }


  #ifndef Z7_ST
  if (p->mtCoder_WasConstructed)
  {
    MtCoder_Destruct(&p->mtCoder);
    p->mtCoder_WasConstructed = False;
  }
  Lzma2Enc_FreeOutBufs(p);
  #endif

  ISzAlloc_Free(p->alloc, p->tempBufLzma);
  p->tempBufLzma = NULL;

  ISzAlloc_Free(p->alloc, p);
}


SRes Lzma2Enc_SetProps(CLzma2EncHandle p, const CLzma2EncProps *props)
{
  // GET_CLzma2Enc_p
  CLzmaEncProps lzmaProps = props->lzmaProps;
  LzmaEncProps_Normalize(&lzmaProps);
  if (lzmaProps.lc + lzmaProps.lp > LZMA2_LCLP_MAX)
    return SZ_ERROR_PARAM;
  p->props = *props;
  Lzma2EncProps_Normalize(&p->props);
  return SZ_OK;
}


void Lzma2Enc_SetDataSize(CLzma2EncHandle p, UInt64 expectedDataSiize)
{
  // GET_CLzma2Enc_p
  p->expectedDataSize = expectedDataSiize;
}


Byte Lzma2Enc_WriteProperties(CLzma2EncHandle p)
{
  // GET_CLzma2Enc_p
  unsigned i;
  UInt32 dicSize = LzmaEncProps_GetDictSize(&p->props.lzmaProps);
  for (i = 0; i < 40; i++)
    if (dicSize <= LZMA2_DIC_SIZE_FROM_PROP(i))
      break;
  return (Byte)i;
}


static SRes Lzma2Enc_EncodeMt1(
    CLzma2Enc *me,
    CLzma2EncInt *p,
    ISeqOutStreamPtr outStream,
    Byte *outBuf, size_t *outBufSize,
    ISeqInStreamPtr inStream,
    const Byte *inData, size_t inDataSize,
    int finished,
    ICompressProgressPtr progress)
{
  UInt64 unpackTotal = 0;
  UInt64 packTotal = 0;
  size_t outLim = 0;
  CLimitedSeqInStream limitedInStream;

  if (outBuf)
  {
    outLim = *outBufSize;
    *outBufSize = 0;
  }

  if (!p->enc)
  {
    p->propsAreSet = False;
    p->enc = LzmaEnc_Create(me->alloc);
    if (!p->enc)
      return SZ_ERROR_MEM;
  }

  limitedInStream.realStream = inStream;
  if (inStream)
  {
    limitedInStream.vt.Read = LimitedSeqInStream_Read;
  }
  
  if (!outBuf)
  {
    // outStream version works only in one thread. So we use CLzma2Enc::tempBufLzma
    if (!me->tempBufLzma)
    {
      me->tempBufLzma = (Byte *)ISzAlloc_Alloc(me->alloc, LZMA2_CHUNK_SIZE_COMPRESSED_MAX);
      if (!me->tempBufLzma)
        return SZ_ERROR_MEM;
    }
  }

  RINOK(Lzma2EncInt_InitStream(p, &me->props))

  for (;;)
  {
    SRes res = SZ_OK;
    SizeT inSizeCur = 0;

    Lzma2EncInt_InitBlock(p);
    
    LimitedSeqInStream_Init(&limitedInStream);
    limitedInStream.limit = me->props.blockSize;

    if (inStream)
    {
      UInt64 expected = (UInt64)(Int64)-1;
      // inStream version works only in one thread. So we use CLzma2Enc::expectedDataSize
      if (me->expectedDataSize != (UInt64)(Int64)-1
          && me->expectedDataSize >= unpackTotal)
        expected = me->expectedDataSize - unpackTotal;
      if (me->props.blockSize != LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID
          && expected > me->props.blockSize)
        expected = (size_t)me->props.blockSize;

      LzmaEnc_SetDataSize(p->enc, expected);

      RINOK(LzmaEnc_PrepareForLzma2(p->enc,
          &limitedInStream.vt,
          LZMA2_KEEP_WINDOW_SIZE,
          me->alloc,
          me->allocBig))
    }
    else
    {
      inSizeCur = (SizeT)(inDataSize - (size_t)unpackTotal);
      if (me->props.blockSize != LZMA2_ENC_PROPS_BLOCK_SIZE_SOLID
          && inSizeCur > me->props.blockSize)
        inSizeCur = (SizeT)(size_t)me->props.blockSize;
    
      // LzmaEnc_SetDataSize(p->enc, inSizeCur);
      
      RINOK(LzmaEnc_MemPrepare(p->enc,
          inData + (size_t)unpackTotal, inSizeCur,
          LZMA2_KEEP_WINDOW_SIZE,
          me->alloc,
          me->allocBig))
    }

    for (;;)
    {
      size_t packSize = LZMA2_CHUNK_SIZE_COMPRESSED_MAX;
      if (outBuf)
        packSize = outLim - (size_t)packTotal;
      
      res = Lzma2EncInt_EncodeSubblock(p,
          outBuf ? outBuf + (size_t)packTotal : me->tempBufLzma, &packSize,
          outBuf ? NULL : outStream);
      
      if (res != SZ_OK)
        break;

      packTotal += packSize;
      if (outBuf)
        *outBufSize = (size_t)packTotal;
      
      res = Progress(progress, unpackTotal + p->srcPos, packTotal);
      if (res != SZ_OK)
        break;

      /*
      if (LzmaEnc_GetNumAvailableBytes(p->enc) == 0)
        break;
      */

      if (packSize == 0)
        break;
    }
    
    LzmaEnc_Finish(p->enc);
    
    unpackTotal += p->srcPos;
    
    RINOK(res)

    if (p->srcPos != (inStream ? limitedInStream.processed : inSizeCur))
      return SZ_ERROR_FAIL;
    
    if (inStream ? limitedInStream.finished : (unpackTotal == inDataSize))
    {
      if (finished)
      {
        if (outBuf)
        {
          const size_t destPos = *outBufSize;
          if (destPos >= outLim)
            return SZ_ERROR_OUTPUT_EOF;
          outBuf[destPos] = LZMA2_CONTROL_EOF; // 0
          *outBufSize = destPos + 1;
        }
        else
        {
          const Byte b = LZMA2_CONTROL_EOF; // 0;
          if (ISeqOutStream_Write(outStream, &b, 1) != 1)
            return SZ_ERROR_WRITE;
        }
      }
      return SZ_OK;
    }
  }
}



#ifndef Z7_ST

static SRes Lzma2Enc_MtCallback_Code(void *p, unsigned coderIndex, unsigned outBufIndex,
    const Byte *src, size_t srcSize, int finished)
{
  CLzma2Enc *me = (CLzma2Enc *)p;
  size_t destSize = me->outBufSize;
  SRes res;
  CMtProgressThunk progressThunk;

  Byte *dest = me->outBufs[outBufIndex];

  me->outBufsDataSizes[outBufIndex] = 0;

  if (!dest)
  {
    dest = (Byte *)ISzAlloc_Alloc(me->alloc, me->outBufSize);
    if (!dest)
      return SZ_ERROR_MEM;
    me->outBufs[outBufIndex] = dest;
  }

  MtProgressThunk_CreateVTable(&progressThunk);
  progressThunk.mtProgress = &me->mtCoder.mtProgress;
  progressThunk.inSize = 0;
  progressThunk.outSize = 0;

  res = Lzma2Enc_EncodeMt1(me,
      &me->coders[coderIndex],
      NULL, dest, &destSize,
      NULL, src, srcSize,
      finished,
      &progressThunk.vt);

  me->outBufsDataSizes[outBufIndex] = destSize;

  return res;
}


static SRes Lzma2Enc_MtCallback_Write(void *p, unsigned outBufIndex)
{
  CLzma2Enc *me = (CLzma2Enc *)p;
  size_t size = me->outBufsDataSizes[outBufIndex];
  const Byte *data = me->outBufs[outBufIndex];
  
  if (me->outStream)
    return ISeqOutStream_Write(me->outStream, data, size) == size ? SZ_OK : SZ_ERROR_WRITE;
  
  if (size > me->outBuf_Rem)
    return SZ_ERROR_OUTPUT_EOF;
  memcpy(me->outBuf, data, size);
  me->outBuf_Rem -= size;
  me->outBuf += size;
  return SZ_OK;
}

#endif



SRes Lzma2Enc_Encode2(CLzma2EncHandle p,
    ISeqOutStreamPtr outStream,
    Byte *outBuf, size_t *outBufSize,
    ISeqInStreamPtr inStream,
    const Byte *inData, size_t inDataSize,
    ICompressProgressPtr progress)
{
  // GET_CLzma2Enc_p

  if (inStream && inData)
    return SZ_ERROR_PARAM;

  if (outStream && outBuf)
    return SZ_ERROR_PARAM;

  {
    unsigned i;
    for (i = 0; i < MTCODER_THREADS_MAX; i++)
      p->coders[i].propsAreSet = False;
  }

  #ifndef Z7_ST
  
  if (p->props.numBlockThreads_Reduced > 1)
  {
    IMtCoderCallback2 vt;

    if (!p->mtCoder_WasConstructed)
    {
      p->mtCoder_WasConstructed = True;
      MtCoder_Construct(&p->mtCoder);
    }

    vt.Code = Lzma2Enc_MtCallback_Code;
    vt.Write = Lzma2Enc_MtCallback_Write;

    p->outStream = outStream;
    p->outBuf = NULL;
    p->outBuf_Rem = 0;
    if (!outStream)
    {
      p->outBuf = outBuf;
      p->outBuf_Rem = *outBufSize;
      *outBufSize = 0;
    }

    p->mtCoder.allocBig = p->allocBig;
    p->mtCoder.progress = progress;
    p->mtCoder.inStream = inStream;
    p->mtCoder.inData = inData;
    p->mtCoder.inDataSize = inDataSize;
    p->mtCoder.mtCallback = &vt;
    p->mtCoder.mtCallbackObject = p;

    p->mtCoder.blockSize = (size_t)p->props.blockSize;
    if (p->mtCoder.blockSize != p->props.blockSize)
      return SZ_ERROR_PARAM; /* SZ_ERROR_MEM */

    {
      const size_t destBlockSize = p->mtCoder.blockSize + (p->mtCoder.blockSize >> 10) + 16;
      if (destBlockSize < p->mtCoder.blockSize)
        return SZ_ERROR_PARAM;
      if (p->outBufSize != destBlockSize)
        Lzma2Enc_FreeOutBufs(p);
      p->outBufSize = destBlockSize;
    }

    p->mtCoder.numThreadsMax = (unsigned)p->props.numBlockThreads_Max;
    p->mtCoder.numThreadGroups = p->props.numThreadGroups;
    p->mtCoder.expectedDataSize = p->expectedDataSize;
    
    {
      const SRes res = MtCoder_Code(&p->mtCoder);
      if (!outStream)
        *outBufSize = (size_t)(p->outBuf - outBuf);
      return res;
    }
  }

  #endif


  return Lzma2Enc_EncodeMt1(p,
      &p->coders[0],
      outStream, outBuf, outBufSize,
      inStream, inData, inDataSize,
      True, /* finished */
      progress);
}

#undef PRF
