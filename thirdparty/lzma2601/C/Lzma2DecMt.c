/* Lzma2DecMt.c -- LZMA2 Decoder Multi-thread
2023-04-13 : Igor Pavlov : Public domain */

#include "Precomp.h"

// #define SHOW_DEBUG_INFO
// #define Z7_ST

#ifdef SHOW_DEBUG_INFO
#include <stdio.h>
#endif

#include "Alloc.h"

#include "Lzma2Dec.h"
#include "Lzma2DecMt.h"

#ifndef Z7_ST
#include "MtDec.h"

#define LZMA2DECMT_OUT_BLOCK_MAX_DEFAULT (1 << 28)
#endif


#ifndef Z7_ST
#ifdef SHOW_DEBUG_INFO
#define PRF(x) x
#else
#define PRF(x)
#endif
#define PRF_STR(s) PRF(printf("\n" s "\n");)
#define PRF_STR_INT_2(s, d1, d2) PRF(printf("\n" s " %d %d\n", (unsigned)d1, (unsigned)d2);)
#endif


void Lzma2DecMtProps_Init(CLzma2DecMtProps *p)
{
  p->inBufSize_ST = 1 << 20;
  p->outStep_ST = 1 << 20;

  #ifndef Z7_ST
  p->numThreads = 1;
  p->inBufSize_MT = 1 << 18;
  p->outBlockMax = LZMA2DECMT_OUT_BLOCK_MAX_DEFAULT;
  p->inBlockMax = p->outBlockMax + p->outBlockMax / 16;
  #endif
}



#ifndef Z7_ST

/* ---------- CLzma2DecMtThread ---------- */

typedef struct
{
  CLzma2Dec dec;
  Byte dec_created;
  Byte needInit;
  
  Byte *outBuf;
  size_t outBufSize;

  EMtDecParseState state;
  ELzma2ParseStatus parseStatus;

  size_t inPreSize;
  size_t outPreSize;

  size_t inCodeSize;
  size_t outCodeSize;
  SRes codeRes;

  CAlignOffsetAlloc alloc;

  Byte mtPad[1 << 7];
} CLzma2DecMtThread;

#endif


/* ---------- CLzma2DecMt ---------- */

struct CLzma2DecMt
{
  // ISzAllocPtr alloc;
  ISzAllocPtr allocMid;

  CAlignOffsetAlloc alignOffsetAlloc;
  CLzma2DecMtProps props;
  Byte prop;
  
  ISeqInStreamPtr inStream;
  ISeqOutStreamPtr outStream;
  ICompressProgressPtr progress;

  BoolInt finishMode;
  BoolInt outSize_Defined;
  UInt64 outSize;

  UInt64 outProcessed;
  UInt64 inProcessed;
  BoolInt readWasFinished;
  SRes readRes;

  Byte *inBuf;
  size_t inBufSize;
  Byte dec_created;
  CLzma2Dec dec;

  size_t inPos;
  size_t inLim;

  #ifndef Z7_ST
  UInt64 outProcessed_Parse;
  BoolInt mtc_WasConstructed;
  CMtDec mtc;
  CLzma2DecMtThread coders[MTDEC_THREADS_MAX];
  #endif
};



CLzma2DecMtHandle Lzma2DecMt_Create(ISzAllocPtr alloc, ISzAllocPtr allocMid)
{
  CLzma2DecMt *p = (CLzma2DecMt *)ISzAlloc_Alloc(alloc, sizeof(CLzma2DecMt));
  if (!p)
    return NULL;
  
  // p->alloc = alloc;
  p->allocMid = allocMid;

  AlignOffsetAlloc_CreateVTable(&p->alignOffsetAlloc);
  p->alignOffsetAlloc.numAlignBits = 7;
  p->alignOffsetAlloc.offset = 0;
  p->alignOffsetAlloc.baseAlloc = alloc;

  p->inBuf = NULL;
  p->inBufSize = 0;
  p->dec_created = False;

  // Lzma2DecMtProps_Init(&p->props);

  #ifndef Z7_ST
  p->mtc_WasConstructed = False;
  {
    unsigned i;
    for (i = 0; i < MTDEC_THREADS_MAX; i++)
    {
      CLzma2DecMtThread *t = &p->coders[i];
      t->dec_created = False;
      t->outBuf = NULL;
      t->outBufSize = 0;
    }
  }
  #endif

  return (CLzma2DecMtHandle)(void *)p;
}


#ifndef Z7_ST

static void Lzma2DecMt_FreeOutBufs(CLzma2DecMt *p)
{
  unsigned i;
  for (i = 0; i < MTDEC_THREADS_MAX; i++)
  {
    CLzma2DecMtThread *t = &p->coders[i];
    if (t->outBuf)
    {
      ISzAlloc_Free(p->allocMid, t->outBuf);
      t->outBuf = NULL;
      t->outBufSize = 0;
    }
  }
}

#endif


static void Lzma2DecMt_FreeSt(CLzma2DecMt *p)
{
  if (p->dec_created)
  {
    Lzma2Dec_Free(&p->dec, &p->alignOffsetAlloc.vt);
    p->dec_created = False;
  }
  if (p->inBuf)
  {
    ISzAlloc_Free(p->allocMid, p->inBuf);
    p->inBuf = NULL;
  }
  p->inBufSize = 0;
}


// #define GET_CLzma2DecMt_p CLzma2DecMt *p = (CLzma2DecMt *)(void *)pp;

void Lzma2DecMt_Destroy(CLzma2DecMtHandle p)
{
  // GET_CLzma2DecMt_p

  Lzma2DecMt_FreeSt(p);

  #ifndef Z7_ST

  if (p->mtc_WasConstructed)
  {
    MtDec_Destruct(&p->mtc);
    p->mtc_WasConstructed = False;
  }
  {
    unsigned i;
    for (i = 0; i < MTDEC_THREADS_MAX; i++)
    {
      CLzma2DecMtThread *t = &p->coders[i];
      if (t->dec_created)
      {
        // we don't need to free dict here
        Lzma2Dec_FreeProbs(&t->dec, &t->alloc.vt); // p->alloc !!!
        t->dec_created = False;
      }
    }
  }
  Lzma2DecMt_FreeOutBufs(p);

  #endif

  ISzAlloc_Free(p->alignOffsetAlloc.baseAlloc, p);
}



#ifndef Z7_ST

static void Lzma2DecMt_MtCallback_Parse(void *obj, unsigned coderIndex, CMtDecCallbackInfo *cc)
{
  CLzma2DecMt *me = (CLzma2DecMt *)obj;
  CLzma2DecMtThread *t = &me->coders[coderIndex];

  PRF_STR_INT_2("Parse", coderIndex, cc->srcSize)

  cc->state = MTDEC_PARSE_CONTINUE;

  if (cc->startCall)
  {
    if (!t->dec_created)
    {
      Lzma2Dec_CONSTRUCT(&t->dec)
      t->dec_created = True;
      AlignOffsetAlloc_CreateVTable(&t->alloc);
      {
        /* (1 << 12) is expected size of one way in data cache.
           We optimize alignment for cache line size of 128 bytes and smaller */
        const unsigned kNumAlignBits = 12;
        const unsigned kNumCacheLineBits = 7; /* <= kNumAlignBits */
        t->alloc.numAlignBits = kNumAlignBits;
        t->alloc.offset = ((UInt32)coderIndex * (((unsigned)1 << 11) + (1 << 8) + (1 << 6))) & (((unsigned)1 << kNumAlignBits) - ((unsigned)1 << kNumCacheLineBits));
        t->alloc.baseAlloc = me->alignOffsetAlloc.baseAlloc;
      }
    }
    Lzma2Dec_Init(&t->dec);
    
    t->inPreSize = 0;
    t->outPreSize = 0;
    // t->blockWasFinished = False;
    // t->finishedWithMark = False;
    t->parseStatus = (ELzma2ParseStatus)LZMA_STATUS_NOT_SPECIFIED;
    t->state = MTDEC_PARSE_CONTINUE;

    t->inCodeSize = 0;
    t->outCodeSize = 0;
    t->codeRes = SZ_OK;

    // (cc->srcSize == 0) is allowed
  }

  {
    ELzma2ParseStatus status;
    BoolInt overflow;
    UInt32 unpackRem = 0;
    
    int checkFinishBlock = True;
    size_t limit = me->props.outBlockMax;
    if (me->outSize_Defined)
    {
      UInt64 rem = me->outSize - me->outProcessed_Parse;
      if (limit >= rem)
      {
        limit = (size_t)rem;
        if (!me->finishMode)
          checkFinishBlock = False;
      }
    }

    // checkFinishBlock = False, if we want to decode partial data
    // that must be finished at position <= outBlockMax.

    {
      const size_t srcOrig = cc->srcSize;
      SizeT srcSize_Point = 0;
      SizeT dicPos_Point = 0;
      
      cc->srcSize = 0;
      overflow = False;

      for (;;)
      {
        SizeT srcCur = (SizeT)(srcOrig - cc->srcSize);
        
        status = Lzma2Dec_Parse(&t->dec,
            (SizeT)limit - t->dec.decoder.dicPos,
            cc->src + cc->srcSize, &srcCur,
            checkFinishBlock);

        cc->srcSize += srcCur;

        if (status == LZMA2_PARSE_STATUS_NEW_CHUNK)
        {
          if (t->dec.unpackSize > me->props.outBlockMax - t->dec.decoder.dicPos)
          {
            overflow = True;
            break;
          }
          continue;
        }
        
        if (status == LZMA2_PARSE_STATUS_NEW_BLOCK)
        {
          if (t->dec.decoder.dicPos == 0)
            continue;
          // we decode small blocks in one thread
          if (t->dec.decoder.dicPos >= (1 << 14))
            break;
          dicPos_Point = t->dec.decoder.dicPos;
          srcSize_Point = (SizeT)cc->srcSize;
          continue;
        }

        if ((int)status == LZMA_STATUS_NOT_FINISHED && checkFinishBlock
            // && limit == t->dec.decoder.dicPos
            // && limit == me->props.outBlockMax
            )
        {
          overflow = True;
          break;
        }
        
        unpackRem = Lzma2Dec_GetUnpackExtra(&t->dec);
        break;
      }

      if (dicPos_Point != 0
          && (int)status != LZMA2_PARSE_STATUS_NEW_BLOCK
          && (int)status != LZMA_STATUS_FINISHED_WITH_MARK
          && (int)status != LZMA_STATUS_NOT_SPECIFIED)
      {
        // we revert to latest newBlock state
        status = LZMA2_PARSE_STATUS_NEW_BLOCK;
        unpackRem = 0;
        t->dec.decoder.dicPos = dicPos_Point;
        cc->srcSize = srcSize_Point;
        overflow = False;
      }
    }

    t->inPreSize += cc->srcSize;
    t->parseStatus = status;

    if (overflow)
      cc->state = MTDEC_PARSE_OVERFLOW;
    else
    {
      size_t dicPos = t->dec.decoder.dicPos;

      if ((int)status != LZMA_STATUS_NEEDS_MORE_INPUT)
      {
        if (status == LZMA2_PARSE_STATUS_NEW_BLOCK)
        {
          cc->state = MTDEC_PARSE_NEW;
          cc->srcSize--; // we don't need control byte of next block
          t->inPreSize--;
        }
        else
        {
          cc->state = MTDEC_PARSE_END;
          if ((int)status != LZMA_STATUS_FINISHED_WITH_MARK)
          {
            // (status == LZMA_STATUS_NOT_SPECIFIED)
            // (status == LZMA_STATUS_NOT_FINISHED)
            if (unpackRem != 0)
            {
              /* we also reserve space for max possible number of output bytes of current LZMA chunk */
              size_t rem = limit - dicPos;
              if (rem > unpackRem)
                rem = unpackRem;
              dicPos += rem;
            }
          }
        }
    
        me->outProcessed_Parse += dicPos;
      }
      
      cc->outPos = dicPos;
      t->outPreSize = (size_t)dicPos;
    }

    t->state = cc->state;
    return;
  }
}


static SRes Lzma2DecMt_MtCallback_PreCode(void *pp, unsigned coderIndex)
{
  CLzma2DecMt *me = (CLzma2DecMt *)pp;
  CLzma2DecMtThread *t = &me->coders[coderIndex];
  Byte *dest = t->outBuf;

  if (t->inPreSize == 0)
  {
    t->codeRes = SZ_ERROR_DATA;
    return t->codeRes;
  }

  if (!dest || t->outBufSize < t->outPreSize)
  {
    if (dest)
    {
      ISzAlloc_Free(me->allocMid, dest);
      t->outBuf = NULL;
      t->outBufSize = 0;
    }

    dest = (Byte *)ISzAlloc_Alloc(me->allocMid, t->outPreSize
        // + (1 << 28)
        );
    // Sleep(200);
    if (!dest)
      return SZ_ERROR_MEM;
    t->outBuf = dest;
    t->outBufSize = t->outPreSize;
  }

  t->dec.decoder.dic = dest;
  t->dec.decoder.dicBufSize = (SizeT)t->outPreSize;

  t->needInit = True;

  return Lzma2Dec_AllocateProbs(&t->dec, me->prop, &t->alloc.vt); // alloc.vt
}


static SRes Lzma2DecMt_MtCallback_Code(void *pp, unsigned coderIndex,
    const Byte *src, size_t srcSize, int srcFinished,
    // int finished, int blockFinished,
    UInt64 *inCodePos, UInt64 *outCodePos, int *stop)
{
  CLzma2DecMt *me = (CLzma2DecMt *)pp;
  CLzma2DecMtThread *t = &me->coders[coderIndex];

  UNUSED_VAR(srcFinished)

  PRF_STR_INT_2("Code", coderIndex, srcSize)

  *inCodePos = t->inCodeSize;
  *outCodePos = 0;
  *stop = True;

  if (t->needInit)
  {
    Lzma2Dec_Init(&t->dec);
    t->needInit = False;
  }

  {
    ELzmaStatus status;
    SizeT srcProcessed = (SizeT)srcSize;
    BoolInt blockWasFinished =
        ((int)t->parseStatus == LZMA_STATUS_FINISHED_WITH_MARK
        || t->parseStatus == LZMA2_PARSE_STATUS_NEW_BLOCK);
    
    SRes res = Lzma2Dec_DecodeToDic(&t->dec,
        (SizeT)t->outPreSize,
        src, &srcProcessed,
        blockWasFinished ? LZMA_FINISH_END : LZMA_FINISH_ANY,
        &status);

    t->codeRes = res;

    t->inCodeSize += srcProcessed;
    *inCodePos = t->inCodeSize;
    t->outCodeSize = t->dec.decoder.dicPos;
    *outCodePos = t->dec.decoder.dicPos;

    if (res != SZ_OK)
      return res;

    if (srcProcessed == srcSize)
      *stop = False;

    if (blockWasFinished)
    {
      if (srcSize != srcProcessed)
        return SZ_ERROR_FAIL;
      
      if (t->inPreSize == t->inCodeSize)
      {
        if (t->outPreSize != t->outCodeSize)
          return SZ_ERROR_FAIL;
        *stop = True;
      }
    }
    else
    {
      if (t->outPreSize == t->outCodeSize)
        *stop = True;
    }

    return SZ_OK;
  }
}


#define LZMA2DECMT_STREAM_WRITE_STEP (1 << 24)

static SRes Lzma2DecMt_MtCallback_Write(void *pp, unsigned coderIndex,
    BoolInt needWriteToStream,
    const Byte *src, size_t srcSize, BoolInt isCross,
    BoolInt *needContinue, BoolInt *canRecode)
{
  CLzma2DecMt *me = (CLzma2DecMt *)pp;
  const CLzma2DecMtThread *t = &me->coders[coderIndex];
  size_t size = t->outCodeSize;
  const Byte *data = t->outBuf;
  BoolInt needContinue2 = True;

  UNUSED_VAR(src)
  UNUSED_VAR(srcSize)
  UNUSED_VAR(isCross)

  PRF_STR_INT_2("Write", coderIndex, srcSize)

  *needContinue = False;
  *canRecode = True;

  if (
      // t->parseStatus == LZMA_STATUS_FINISHED_WITH_MARK
         t->state == MTDEC_PARSE_OVERFLOW
      || t->state == MTDEC_PARSE_END)
    needContinue2 = False;


  if (!needWriteToStream)
    return SZ_OK;

  me->mtc.inProcessed += t->inCodeSize;

  if (t->codeRes == SZ_OK)
  if ((int)t->parseStatus == LZMA_STATUS_FINISHED_WITH_MARK
      || t->parseStatus == LZMA2_PARSE_STATUS_NEW_BLOCK)
  if (t->outPreSize != t->outCodeSize
      || t->inPreSize != t->inCodeSize)
    return SZ_ERROR_FAIL;

  *canRecode = False;
    
  if (me->outStream)
  {
    for (;;)
    {
      size_t cur = size;
      size_t written;
      if (cur > LZMA2DECMT_STREAM_WRITE_STEP)
        cur = LZMA2DECMT_STREAM_WRITE_STEP;

      written = ISeqOutStream_Write(me->outStream, data, cur);
      
      me->outProcessed += written;
      // me->mtc.writtenTotal += written;
      if (written != cur)
        return SZ_ERROR_WRITE;
      data += cur;
      size -= cur;
      if (size == 0)
      {
        *needContinue = needContinue2;
        return SZ_OK;
      }
      RINOK(MtProgress_ProgressAdd(&me->mtc.mtProgress, 0, 0))
    }
  }
  
  return SZ_ERROR_FAIL;
  /*
  if (size > me->outBufSize)
    return SZ_ERROR_OUTPUT_EOF;
  memcpy(me->outBuf, data, size);
  me->outBufSize -= size;
  me->outBuf += size;
  *needContinue = needContinue2;
  return SZ_OK;
  */
}

#endif


static SRes Lzma2Dec_Prepare_ST(CLzma2DecMt *p)
{
  if (!p->dec_created)
  {
    Lzma2Dec_CONSTRUCT(&p->dec)
    p->dec_created = True;
  }

  RINOK(Lzma2Dec_Allocate(&p->dec, p->prop, &p->alignOffsetAlloc.vt))

  if (!p->inBuf || p->inBufSize != p->props.inBufSize_ST)
  {
    ISzAlloc_Free(p->allocMid, p->inBuf);
    p->inBufSize = 0;
    p->inBuf = (Byte *)ISzAlloc_Alloc(p->allocMid, p->props.inBufSize_ST);
    if (!p->inBuf)
      return SZ_ERROR_MEM;
    p->inBufSize = p->props.inBufSize_ST;
  }

  Lzma2Dec_Init(&p->dec);
  
  return SZ_OK;
}


static SRes Lzma2Dec_Decode_ST(CLzma2DecMt *p
    #ifndef Z7_ST
    , BoolInt tMode
    #endif
    )
{
  SizeT wrPos;
  size_t inPos, inLim;
  const Byte *inData;
  UInt64 inPrev, outPrev;

  CLzma2Dec *dec;

  #ifndef Z7_ST
  if (tMode)
  {
    Lzma2DecMt_FreeOutBufs(p);
    tMode = MtDec_PrepareRead(&p->mtc);
  }
  #endif

  RINOK(Lzma2Dec_Prepare_ST(p))

  dec = &p->dec;

  inPrev = p->inProcessed;
  outPrev = p->outProcessed;

  inPos = 0;
  inLim = 0;
  inData = NULL;
  wrPos = dec->decoder.dicPos;

  for (;;)
  {
    SizeT dicPos;
    SizeT size;
    ELzmaFinishMode finishMode;
    SizeT inProcessed;
    ELzmaStatus status;
    SRes res;

    SizeT outProcessed;
    BoolInt outFinished;
    BoolInt needStop;

    if (inPos == inLim)
    {
      #ifndef Z7_ST
      if (tMode)
      {
        inData = MtDec_Read(&p->mtc, &inLim);
        inPos = 0;
        if (inData)
          continue;
        tMode = False;
        inLim = 0;
      }
      #endif
      
      if (!p->readWasFinished)
      {
        inPos = 0;
        inLim = p->inBufSize;
        inData = p->inBuf;
        p->readRes = ISeqInStream_Read(p->inStream, (void *)(p->inBuf), &inLim);
        // p->readProcessed += inLim;
        // inLim -= 5; p->readWasFinished = True; // for test
        if (inLim == 0 || p->readRes != SZ_OK)
          p->readWasFinished = True;
      }
    }

    dicPos = dec->decoder.dicPos;
    {
      SizeT next = dec->decoder.dicBufSize;
      if (next - wrPos > p->props.outStep_ST)
        next = wrPos + (SizeT)p->props.outStep_ST;
      size = next - dicPos;
    }

    finishMode = LZMA_FINISH_ANY;
    if (p->outSize_Defined)
    {
      const UInt64 rem = p->outSize - p->outProcessed;
      if (size >= rem)
      {
        size = (SizeT)rem;
        if (p->finishMode)
          finishMode = LZMA_FINISH_END;
      }
    }

    inProcessed = (SizeT)(inLim - inPos);
    
    res = Lzma2Dec_DecodeToDic(dec, dicPos + size, inData + inPos, &inProcessed, finishMode, &status);

    inPos += inProcessed;
    p->inProcessed += inProcessed;
    outProcessed = dec->decoder.dicPos - dicPos;
    p->outProcessed += outProcessed;

    outFinished = (p->outSize_Defined && p->outSize <= p->outProcessed);

    needStop = (res != SZ_OK
        || (inProcessed == 0 && outProcessed == 0)
        || status == LZMA_STATUS_FINISHED_WITH_MARK
        || (!p->finishMode && outFinished));

    if (needStop || outProcessed >= size)
    {
      SRes res2;
      {
        size_t writeSize = dec->decoder.dicPos - wrPos;
        size_t written = ISeqOutStream_Write(p->outStream, dec->decoder.dic + wrPos, writeSize);
        res2 = (written == writeSize) ? SZ_OK : SZ_ERROR_WRITE;
      }

      if (dec->decoder.dicPos == dec->decoder.dicBufSize)
        dec->decoder.dicPos = 0;
      wrPos = dec->decoder.dicPos;

      RINOK(res2)

      if (needStop)
      {
        if (res != SZ_OK)
          return res;

        if (status == LZMA_STATUS_FINISHED_WITH_MARK)
        {
          if (p->finishMode)
          {
            if (p->outSize_Defined && p->outSize != p->outProcessed)
              return SZ_ERROR_DATA;
          }
          return SZ_OK;
        }

        if (!p->finishMode && outFinished)
          return SZ_OK;

        if (status == LZMA_STATUS_NEEDS_MORE_INPUT)
          return SZ_ERROR_INPUT_EOF;
        
        return SZ_ERROR_DATA;
      }
    }
    
    if (p->progress)
    {
      UInt64 inDelta = p->inProcessed - inPrev;
      UInt64 outDelta = p->outProcessed - outPrev;
      if (inDelta >= (1 << 22) || outDelta >= (1 << 22))
      {
        RINOK(ICompressProgress_Progress(p->progress, p->inProcessed, p->outProcessed))
        inPrev = p->inProcessed;
        outPrev = p->outProcessed;
      }
    }
  }
}



SRes Lzma2DecMt_Decode(CLzma2DecMtHandle p,
    Byte prop,
    const CLzma2DecMtProps *props,
    ISeqOutStreamPtr outStream, const UInt64 *outDataSize, int finishMode,
    // Byte *outBuf, size_t *outBufSize,
    ISeqInStreamPtr inStream,
    // const Byte *inData, size_t inDataSize,
    UInt64 *inProcessed,
    // UInt64 *outProcessed,
    int *isMT,
    ICompressProgressPtr progress)
{
  // GET_CLzma2DecMt_p
  #ifndef Z7_ST
  BoolInt tMode;
  #endif

  *inProcessed = 0;

  if (prop > 40)
    return SZ_ERROR_UNSUPPORTED;

  p->prop = prop;
  p->props = *props;

  p->inStream = inStream;
  p->outStream = outStream;
  p->progress = progress;

  p->outSize = 0;
  p->outSize_Defined = False;
  if (outDataSize)
  {
    p->outSize_Defined = True;
    p->outSize = *outDataSize;
  }
  p->finishMode = finishMode;

  p->outProcessed = 0;
  p->inProcessed = 0;

  p->readWasFinished = False;
  p->readRes = SZ_OK;

  *isMT = False;

  
  #ifndef Z7_ST

  tMode = False;

  // p->mtc.parseRes = SZ_OK;

  // p->mtc.numFilledThreads = 0;
  // p->mtc.crossStart = 0;
  // p->mtc.crossEnd = 0;
  // p->mtc.allocError_for_Read_BlockIndex = 0;
  // p->mtc.isAllocError = False;

  if (p->props.numThreads > 1)
  {
    IMtDecCallback2 vt;

    Lzma2DecMt_FreeSt(p);

    p->outProcessed_Parse = 0;

    if (!p->mtc_WasConstructed)
    {
      p->mtc_WasConstructed = True;
      MtDec_Construct(&p->mtc);
    }
    
    p->mtc.progress = progress;
    p->mtc.inStream = inStream;

    // p->outBuf = NULL;
    // p->outBufSize = 0;
    /*
    if (!outStream)
    {
      // p->outBuf = outBuf;
      // p->outBufSize = *outBufSize;
      // *outBufSize = 0;
      return SZ_ERROR_PARAM;
    }
    */

    // p->mtc.inBlockMax = p->props.inBlockMax;
    p->mtc.alloc = &p->alignOffsetAlloc.vt;
      // p->alignOffsetAlloc.baseAlloc;
    // p->mtc.inData = inData;
    // p->mtc.inDataSize = inDataSize;
    p->mtc.mtCallback = &vt;
    p->mtc.mtCallbackObject = p;

    p->mtc.inBufSize = p->props.inBufSize_MT;

    p->mtc.numThreadsMax = p->props.numThreads;

    *isMT = True;

    vt.Parse = Lzma2DecMt_MtCallback_Parse;
    vt.PreCode = Lzma2DecMt_MtCallback_PreCode;
    vt.Code = Lzma2DecMt_MtCallback_Code;
    vt.Write = Lzma2DecMt_MtCallback_Write;

    {
      BoolInt needContinue = False;

      SRes res = MtDec_Code(&p->mtc);

      /*
      if (!outStream)
        *outBufSize = p->outBuf - outBuf;
      */

      *inProcessed = p->mtc.inProcessed;

      needContinue = False;

      if (res == SZ_OK)
      {
        if (p->mtc.mtProgress.res != SZ_OK)
          res = p->mtc.mtProgress.res;
        else
          needContinue = p->mtc.needContinue;
      }

      if (!needContinue)
      {
        if (res == SZ_OK)
          return p->mtc.readRes;
        return res;
      }

      tMode = True;
      p->readRes = p->mtc.readRes;
      p->readWasFinished = p->mtc.readWasFinished;
      p->inProcessed = p->mtc.inProcessed;
      
      PRF_STR("----- decoding ST -----")
    }
  }

  #endif


  *isMT = False;

  {
    SRes res = Lzma2Dec_Decode_ST(p
        #ifndef Z7_ST
        , tMode
        #endif
        );

    *inProcessed = p->inProcessed;

    // res = SZ_OK; // for test
    if (res == SZ_ERROR_INPUT_EOF)
    {
      if (p->readRes != SZ_OK)
        res = p->readRes;
    }
    else if (res == SZ_OK && p->readRes != SZ_OK)
      res = p->readRes;
    
    /*
    #ifndef Z7_ST
    if (res == SZ_OK && tMode && p->mtc.parseRes != SZ_OK)
      res = p->mtc.parseRes;
    #endif
    */
    
    return res;
  }
}


/* ---------- Read from CLzma2DecMtHandle Interface ---------- */

SRes Lzma2DecMt_Init(CLzma2DecMtHandle p,
    Byte prop,
    const CLzma2DecMtProps *props,
    const UInt64 *outDataSize, int finishMode,
    ISeqInStreamPtr inStream)
{
  // GET_CLzma2DecMt_p

  if (prop > 40)
    return SZ_ERROR_UNSUPPORTED;

  p->prop = prop;
  p->props = *props;

  p->inStream = inStream;

  p->outSize = 0;
  p->outSize_Defined = False;
  if (outDataSize)
  {
    p->outSize_Defined = True;
    p->outSize = *outDataSize;
  }
  p->finishMode = finishMode;

  p->outProcessed = 0;
  p->inProcessed = 0;

  p->inPos = 0;
  p->inLim = 0;

  return Lzma2Dec_Prepare_ST(p);
}


SRes Lzma2DecMt_Read(CLzma2DecMtHandle p,
    Byte *data, size_t *outSize,
    UInt64 *inStreamProcessed)
{
  // GET_CLzma2DecMt_p
  ELzmaFinishMode finishMode;
  SRes readRes;
  size_t size = *outSize;

  *outSize = 0;
  *inStreamProcessed = 0;

  finishMode = LZMA_FINISH_ANY;
  if (p->outSize_Defined)
  {
    const UInt64 rem = p->outSize - p->outProcessed;
    if (size >= rem)
    {
      size = (size_t)rem;
      if (p->finishMode)
        finishMode = LZMA_FINISH_END;
    }
  }

  readRes = SZ_OK;

  for (;;)
  {
    SizeT inCur;
    SizeT outCur;
    ELzmaStatus status;
    SRes res;

    if (p->inPos == p->inLim && readRes == SZ_OK)
    {
      p->inPos = 0;
      p->inLim = p->props.inBufSize_ST;
      readRes = ISeqInStream_Read(p->inStream, p->inBuf, &p->inLim);
    }

    inCur = (SizeT)(p->inLim - p->inPos);
    outCur = (SizeT)size;

    res = Lzma2Dec_DecodeToBuf(&p->dec, data, &outCur,
        p->inBuf + p->inPos, &inCur, finishMode, &status);
    
    p->inPos += inCur;
    p->inProcessed += inCur;
    *inStreamProcessed += inCur;
    p->outProcessed += outCur;
    *outSize += outCur;
    size -= outCur;
    data += outCur;
    
    if (res != 0)
      return res;
    
    /*
    if (status == LZMA_STATUS_FINISHED_WITH_MARK)
      return readRes;

    if (size == 0 && status != LZMA_STATUS_NEEDS_MORE_INPUT)
    {
      if (p->finishMode && p->outSize_Defined && p->outProcessed >= p->outSize)
        return SZ_ERROR_DATA;
      return readRes;
    }
    */

    if (inCur == 0 && outCur == 0)
      return readRes;
  }
}

#undef PRF
#undef PRF_STR
#undef PRF_STR_INT_2
