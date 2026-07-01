/* Lzma2Dec.c -- LZMA2 Decoder
2024-03-01 : Igor Pavlov : Public domain */

/* #define SHOW_DEBUG_INFO */

#include "Precomp.h"

#ifdef SHOW_DEBUG_INFO
#include <stdio.h>
#endif

#include <string.h>

#include "Lzma2Dec.h"

/*
00000000  -  End of data
00000001 U U  -  Uncompressed, reset dic, need reset state and set new prop
00000010 U U  -  Uncompressed, no reset
100uuuuu U U P P  -  LZMA, no reset
101uuuuu U U P P  -  LZMA, reset state
110uuuuu U U P P S  -  LZMA, reset state + set new prop
111uuuuu U U P P S  -  LZMA, reset state + set new prop, reset dic

  u, U - Unpack Size
  P - Pack Size
  S - Props
*/

#define LZMA2_CONTROL_COPY_RESET_DIC 1

#define LZMA2_IS_UNCOMPRESSED_STATE(p) (((p)->control & (1 << 7)) == 0)

#define LZMA2_LCLP_MAX 4
#define LZMA2_DIC_SIZE_FROM_PROP(p) (((UInt32)2 | ((p) & 1)) << ((p) / 2 + 11))

#ifdef SHOW_DEBUG_INFO
#define PRF(x) x
#else
#define PRF(x)
#endif

typedef enum
{
  LZMA2_STATE_CONTROL,
  LZMA2_STATE_UNPACK0,
  LZMA2_STATE_UNPACK1,
  LZMA2_STATE_PACK0,
  LZMA2_STATE_PACK1,
  LZMA2_STATE_PROP,
  LZMA2_STATE_DATA,
  LZMA2_STATE_DATA_CONT,
  LZMA2_STATE_FINISHED,
  LZMA2_STATE_ERROR
} ELzma2State;

static SRes Lzma2Dec_GetOldProps(Byte prop, Byte *props)
{
  UInt32 dicSize;
  if (prop > 40)
    return SZ_ERROR_UNSUPPORTED;
  dicSize = (prop == 40) ? 0xFFFFFFFF : LZMA2_DIC_SIZE_FROM_PROP(prop);
  props[0] = (Byte)LZMA2_LCLP_MAX;
  props[1] = (Byte)(dicSize);
  props[2] = (Byte)(dicSize >> 8);
  props[3] = (Byte)(dicSize >> 16);
  props[4] = (Byte)(dicSize >> 24);
  return SZ_OK;
}

SRes Lzma2Dec_AllocateProbs(CLzma2Dec *p, Byte prop, ISzAllocPtr alloc)
{
  Byte props[LZMA_PROPS_SIZE];
  RINOK(Lzma2Dec_GetOldProps(prop, props))
  return LzmaDec_AllocateProbs(&p->decoder, props, LZMA_PROPS_SIZE, alloc);
}

SRes Lzma2Dec_Allocate(CLzma2Dec *p, Byte prop, ISzAllocPtr alloc)
{
  Byte props[LZMA_PROPS_SIZE];
  RINOK(Lzma2Dec_GetOldProps(prop, props))
  return LzmaDec_Allocate(&p->decoder, props, LZMA_PROPS_SIZE, alloc);
}

void Lzma2Dec_Init(CLzma2Dec *p)
{
  p->state = LZMA2_STATE_CONTROL;
  p->needInitLevel = 0xE0;
  p->isExtraMode = False;
  p->unpackSize = 0;
  
  // p->decoder.dicPos = 0; // we can use it instead of full init
  LzmaDec_Init(&p->decoder);
}

// ELzma2State
static unsigned Lzma2Dec_UpdateState(CLzma2Dec *p, Byte b)
{
  switch (p->state)
  {
    case LZMA2_STATE_CONTROL:
      p->isExtraMode = False;
      p->control = b;
      PRF(printf("\n %8X", (unsigned)p->decoder.dicPos));
      PRF(printf(" %02X", (unsigned)b));
      if (b == 0)
        return LZMA2_STATE_FINISHED;
      if (LZMA2_IS_UNCOMPRESSED_STATE(p))
      {
        if (b == LZMA2_CONTROL_COPY_RESET_DIC)
          p->needInitLevel = 0xC0;
        else if (b > 2 || p->needInitLevel == 0xE0)
          return LZMA2_STATE_ERROR;
      }
      else
      {
        if (b < p->needInitLevel)
          return LZMA2_STATE_ERROR;
        p->needInitLevel = 0;
        p->unpackSize = (UInt32)(b & 0x1F) << 16;
      }
      return LZMA2_STATE_UNPACK0;
    
    case LZMA2_STATE_UNPACK0:
      p->unpackSize |= (UInt32)b << 8;
      return LZMA2_STATE_UNPACK1;
    
    case LZMA2_STATE_UNPACK1:
      p->unpackSize |= (UInt32)b;
      p->unpackSize++;
      PRF(printf(" %7u", (unsigned)p->unpackSize));
      return LZMA2_IS_UNCOMPRESSED_STATE(p) ? LZMA2_STATE_DATA : LZMA2_STATE_PACK0;
    
    case LZMA2_STATE_PACK0:
      p->packSize = (UInt32)b << 8;
      return LZMA2_STATE_PACK1;

    case LZMA2_STATE_PACK1:
      p->packSize |= (UInt32)b;
      p->packSize++;
      // if (p->packSize < 5) return LZMA2_STATE_ERROR;
      PRF(printf(" %5u", (unsigned)p->packSize));
      return (p->control & 0x40) ? LZMA2_STATE_PROP : LZMA2_STATE_DATA;

    case LZMA2_STATE_PROP:
    {
      unsigned lc, lp;
      if (b >= (9 * 5 * 5))
        return LZMA2_STATE_ERROR;
      lc = b % 9;
      b /= 9;
      p->decoder.prop.pb = (Byte)(b / 5);
      lp = b % 5;
      if (lc + lp > LZMA2_LCLP_MAX)
        return LZMA2_STATE_ERROR;
      p->decoder.prop.lc = (Byte)lc;
      p->decoder.prop.lp = (Byte)lp;
      return LZMA2_STATE_DATA;
    }
    
    default:
      return LZMA2_STATE_ERROR;
  }
}

static void LzmaDec_UpdateWithUncompressed(CLzmaDec *p, const Byte *src, SizeT size)
{
  memcpy(p->dic + p->dicPos, src, size);
  p->dicPos += size;
  if (p->checkDicSize == 0 && p->prop.dicSize - p->processedPos <= size)
    p->checkDicSize = p->prop.dicSize;
  p->processedPos += (UInt32)size;
}

void LzmaDec_InitDicAndState(CLzmaDec *p, BoolInt initDic, BoolInt initState);


SRes Lzma2Dec_DecodeToDic(CLzma2Dec *p, SizeT dicLimit,
    const Byte *src, SizeT *srcLen, ELzmaFinishMode finishMode, ELzmaStatus *status)
{
  SizeT inSize = *srcLen;
  *srcLen = 0;
  *status = LZMA_STATUS_NOT_SPECIFIED;

  while (p->state != LZMA2_STATE_ERROR)
  {
    SizeT dicPos;

    if (p->state == LZMA2_STATE_FINISHED)
    {
      *status = LZMA_STATUS_FINISHED_WITH_MARK;
      return SZ_OK;
    }
    
    dicPos = p->decoder.dicPos;
    
    if (dicPos == dicLimit && finishMode == LZMA_FINISH_ANY)
    {
      *status = LZMA_STATUS_NOT_FINISHED;
      return SZ_OK;
    }

    if (p->state != LZMA2_STATE_DATA && p->state != LZMA2_STATE_DATA_CONT)
    {
      if (*srcLen == inSize)
      {
        *status = LZMA_STATUS_NEEDS_MORE_INPUT;
        return SZ_OK;
      }
      (*srcLen)++;
      p->state = Lzma2Dec_UpdateState(p, *src++);
      if (dicPos == dicLimit && p->state != LZMA2_STATE_FINISHED)
        break;
      continue;
    }
    
    {
      SizeT inCur = inSize - *srcLen;
      SizeT outCur = dicLimit - dicPos;
      ELzmaFinishMode curFinishMode = LZMA_FINISH_ANY;
      
      if (outCur >= p->unpackSize)
      {
        outCur = (SizeT)p->unpackSize;
        curFinishMode = LZMA_FINISH_END;
      }

      if (LZMA2_IS_UNCOMPRESSED_STATE(p))
      {
        if (inCur == 0)
        {
          *status = LZMA_STATUS_NEEDS_MORE_INPUT;
          return SZ_OK;
        }

        if (p->state == LZMA2_STATE_DATA)
        {
          BoolInt initDic = (p->control == LZMA2_CONTROL_COPY_RESET_DIC);
          LzmaDec_InitDicAndState(&p->decoder, initDic, False);
        }

        if (inCur > outCur)
          inCur = outCur;
        if (inCur == 0)
          break;

        LzmaDec_UpdateWithUncompressed(&p->decoder, src, inCur);

        src += inCur;
        *srcLen += inCur;
        p->unpackSize -= (UInt32)inCur;
        p->state = (p->unpackSize == 0) ? LZMA2_STATE_CONTROL : LZMA2_STATE_DATA_CONT;
      }
      else
      {
        SRes res;

        if (p->state == LZMA2_STATE_DATA)
        {
          BoolInt initDic = (p->control >= 0xE0);
          BoolInt initState = (p->control >= 0xA0);
          LzmaDec_InitDicAndState(&p->decoder, initDic, initState);
          p->state = LZMA2_STATE_DATA_CONT;
        }
  
        if (inCur > p->packSize)
          inCur = (SizeT)p->packSize;
        
        res = LzmaDec_DecodeToDic(&p->decoder, dicPos + outCur, src, &inCur, curFinishMode, status);

        src += inCur;
        *srcLen += inCur;
        p->packSize -= (UInt32)inCur;
        outCur = p->decoder.dicPos - dicPos;
        p->unpackSize -= (UInt32)outCur;

        if (res != 0)
          break;
        
        if (*status == LZMA_STATUS_NEEDS_MORE_INPUT)
        {
          if (p->packSize == 0)
            break;
          return SZ_OK;
        }

        if (inCur == 0 && outCur == 0)
        {
          if (*status != LZMA_STATUS_MAYBE_FINISHED_WITHOUT_MARK
              || p->unpackSize != 0
              || p->packSize != 0)
            break;
          p->state = LZMA2_STATE_CONTROL;
        }
        
        *status = LZMA_STATUS_NOT_SPECIFIED;
      }
    }
  }
  
  *status = LZMA_STATUS_NOT_SPECIFIED;
  p->state = LZMA2_STATE_ERROR;
  return SZ_ERROR_DATA;
}




ELzma2ParseStatus Lzma2Dec_Parse(CLzma2Dec *p,
    SizeT outSize,
    const Byte *src, SizeT *srcLen,
    int checkFinishBlock)
{
  SizeT inSize = *srcLen;
  *srcLen = 0;

  while (p->state != LZMA2_STATE_ERROR)
  {
    if (p->state == LZMA2_STATE_FINISHED)
      return (ELzma2ParseStatus)LZMA_STATUS_FINISHED_WITH_MARK;

    if (outSize == 0 && !checkFinishBlock)
      return (ELzma2ParseStatus)LZMA_STATUS_NOT_FINISHED;
    
    if (p->state != LZMA2_STATE_DATA && p->state != LZMA2_STATE_DATA_CONT)
    {
      if (*srcLen == inSize)
        return (ELzma2ParseStatus)LZMA_STATUS_NEEDS_MORE_INPUT;
      (*srcLen)++;

      p->state = Lzma2Dec_UpdateState(p, *src++);

      if (p->state == LZMA2_STATE_UNPACK0)
      {
        // if (p->decoder.dicPos != 0)
        if (p->control == LZMA2_CONTROL_COPY_RESET_DIC || p->control >= 0xE0)
          return LZMA2_PARSE_STATUS_NEW_BLOCK;
        // if (outSize == 0) return LZMA_STATUS_NOT_FINISHED;
      }

      // The following code can be commented.
      // It's not big problem, if we read additional input bytes.
      // It will be stopped later in LZMA2_STATE_DATA / LZMA2_STATE_DATA_CONT state.

      if (outSize == 0 && p->state != LZMA2_STATE_FINISHED)
      {
        // checkFinishBlock is true. So we expect that block must be finished,
        // We can return LZMA_STATUS_NOT_SPECIFIED or LZMA_STATUS_NOT_FINISHED here
        // break;
        return (ELzma2ParseStatus)LZMA_STATUS_NOT_FINISHED;
      }

      if (p->state == LZMA2_STATE_DATA)
        return LZMA2_PARSE_STATUS_NEW_CHUNK;

      continue;
    }

    if (outSize == 0)
      return (ELzma2ParseStatus)LZMA_STATUS_NOT_FINISHED;

    {
      SizeT inCur = inSize - *srcLen;

      if (LZMA2_IS_UNCOMPRESSED_STATE(p))
      {
        if (inCur == 0)
          return (ELzma2ParseStatus)LZMA_STATUS_NEEDS_MORE_INPUT;
        if (inCur > p->unpackSize)
          inCur = p->unpackSize;
        if (inCur > outSize)
          inCur = outSize;
        p->decoder.dicPos += inCur;
        src += inCur;
        *srcLen += inCur;
        outSize -= inCur;
        p->unpackSize -= (UInt32)inCur;
        p->state = (p->unpackSize == 0) ? LZMA2_STATE_CONTROL : LZMA2_STATE_DATA_CONT;
      }
      else
      {
        p->isExtraMode = True;

        if (inCur == 0)
        {
          if (p->packSize != 0)
            return (ELzma2ParseStatus)LZMA_STATUS_NEEDS_MORE_INPUT;
        }
        else if (p->state == LZMA2_STATE_DATA)
        {
          p->state = LZMA2_STATE_DATA_CONT;
          if (*src != 0)
          {
            // first byte of lzma chunk must be Zero
            *srcLen += 1;
            p->packSize--;
            break;
          }
        }
  
        if (inCur > p->packSize)
          inCur = (SizeT)p->packSize;

        src += inCur;
        *srcLen += inCur;
        p->packSize -= (UInt32)inCur;

        if (p->packSize == 0)
        {
          SizeT rem = outSize;
          if (rem > p->unpackSize)
            rem = p->unpackSize;
          p->decoder.dicPos += rem;
          p->unpackSize -= (UInt32)rem;
          outSize -= rem;
          if (p->unpackSize == 0)
            p->state = LZMA2_STATE_CONTROL;
        }
      }
    }
  }
  
  p->state = LZMA2_STATE_ERROR;
  return (ELzma2ParseStatus)LZMA_STATUS_NOT_SPECIFIED;
}




SRes Lzma2Dec_DecodeToBuf(CLzma2Dec *p, Byte *dest, SizeT *destLen, const Byte *src, SizeT *srcLen, ELzmaFinishMode finishMode, ELzmaStatus *status)
{
  SizeT outSize = *destLen, inSize = *srcLen;
  *srcLen = *destLen = 0;
  
  for (;;)
  {
    SizeT inCur = inSize, outCur, dicPos;
    ELzmaFinishMode curFinishMode;
    SRes res;
    
    if (p->decoder.dicPos == p->decoder.dicBufSize)
      p->decoder.dicPos = 0;
    dicPos = p->decoder.dicPos;
    curFinishMode = LZMA_FINISH_ANY;
    outCur = p->decoder.dicBufSize - dicPos;
    
    if (outCur >= outSize)
    {
      outCur = outSize;
      curFinishMode = finishMode;
    }

    res = Lzma2Dec_DecodeToDic(p, dicPos + outCur, src, &inCur, curFinishMode, status);
    
    src += inCur;
    inSize -= inCur;
    *srcLen += inCur;
    outCur = p->decoder.dicPos - dicPos;
    memcpy(dest, p->decoder.dic + dicPos, outCur);
    dest += outCur;
    outSize -= outCur;
    *destLen += outCur;
    if (res != 0)
      return res;
    if (outCur == 0 || outSize == 0)
      return SZ_OK;
  }
}


SRes Lzma2Decode(Byte *dest, SizeT *destLen, const Byte *src, SizeT *srcLen,
    Byte prop, ELzmaFinishMode finishMode, ELzmaStatus *status, ISzAllocPtr alloc)
{
  CLzma2Dec p;
  SRes res;
  SizeT outSize = *destLen, inSize = *srcLen;
  *destLen = *srcLen = 0;
  *status = LZMA_STATUS_NOT_SPECIFIED;
  Lzma2Dec_CONSTRUCT(&p)
  RINOK(Lzma2Dec_AllocateProbs(&p, prop, alloc))
  p.decoder.dic = dest;
  p.decoder.dicBufSize = outSize;
  Lzma2Dec_Init(&p);
  *srcLen = inSize;
  res = Lzma2Dec_DecodeToDic(&p, outSize, src, srcLen, finishMode, status);
  *destLen = p.decoder.dicPos;
  if (res == SZ_OK && *status == LZMA_STATUS_NEEDS_MORE_INPUT)
    res = SZ_ERROR_INPUT_EOF;
  Lzma2Dec_FreeProbs(&p, alloc);
  return res;
}

#undef PRF
