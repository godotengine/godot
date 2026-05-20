/* Ppmd7Enc.c -- Ppmd7z (PPMdH with 7z Range Coder) Encoder
2023-09-07 : Igor Pavlov : Public domain
This code is based on:
  PPMd var.H (2001): Dmitry Shkarin : Public domain */


#include "Precomp.h"

#include "Ppmd7.h"

#define kTopValue ((UInt32)1 << 24)

#define R (&p->rc.enc)

void Ppmd7z_Init_RangeEnc(CPpmd7 *p)
{
  R->Low = 0;
  R->Range = 0xFFFFFFFF;
  R->Cache = 0;
  R->CacheSize = 1;
}

Z7_NO_INLINE
static void Ppmd7z_RangeEnc_ShiftLow(CPpmd7 *p)
{
  if ((UInt32)R->Low < (UInt32)0xFF000000 || (unsigned)(R->Low >> 32) != 0)
  {
    Byte temp = R->Cache;
    do
    {
      IByteOut_Write(R->Stream, (Byte)(temp + (Byte)(R->Low >> 32)));
      temp = 0xFF;
    }
    while (--R->CacheSize != 0);
    R->Cache = (Byte)((UInt32)R->Low >> 24);
  }
  R->CacheSize++;
  R->Low = (UInt32)((UInt32)R->Low << 8);
}

#define RC_NORM_BASE(p) if (R->Range < kTopValue) { R->Range <<= 8;  Ppmd7z_RangeEnc_ShiftLow(p);
#define RC_NORM_1(p)    RC_NORM_BASE(p) }
#define RC_NORM(p)      RC_NORM_BASE(p)  RC_NORM_BASE(p) }}

// we must use only one type of Normalization from two: LOCAL or REMOTE
#define RC_NORM_LOCAL(p)    // RC_NORM(p)
#define RC_NORM_REMOTE(p)   RC_NORM(p)

/*
#define Ppmd7z_RangeEnc_Encode(p, start, _size_) \
  { UInt32 size = _size_; \
    R->Low += start * R->Range; \
    R->Range *= size; \
    RC_NORM_LOCAL(p); }
*/

Z7_FORCE_INLINE
// Z7_NO_INLINE
static void Ppmd7z_RangeEnc_Encode(CPpmd7 *p, UInt32 start, UInt32 size)
{
  R->Low += start * R->Range;
  R->Range *= size;
  RC_NORM_LOCAL(p)
}

void Ppmd7z_Flush_RangeEnc(CPpmd7 *p)
{
  unsigned i;
  for (i = 0; i < 5; i++)
    Ppmd7z_RangeEnc_ShiftLow(p);
}



#define RC_Encode(start, size)  Ppmd7z_RangeEnc_Encode(p, start, size);
#define RC_EncodeFinal(start, size)  RC_Encode(start, size) RC_NORM_REMOTE(p)

#define CTX(ref) ((CPpmd7_Context *)Ppmd7_GetContext(p, ref))
#define SUFFIX(ctx) CTX((ctx)->Suffix)
// typedef CPpmd7_Context * CTX_PTR;
#define SUCCESSOR(p) Ppmd_GET_SUCCESSOR(p)

void Ppmd7_UpdateModel(CPpmd7 *p);

#define MASK(sym)  ((Byte *)charMask)[sym]

Z7_FORCE_INLINE
static
void Ppmd7z_EncodeSymbol(CPpmd7 *p, int symbol)
{
  size_t charMask[256 / sizeof(size_t)];
  
  if (p->MinContext->NumStats != 1)
  {
    CPpmd_State *s = Ppmd7_GetStats(p, p->MinContext);
    UInt32 sum;
    unsigned i;
   

    
    
    R->Range /= p->MinContext->Union2.SummFreq;
    
    if (s->Symbol == symbol)
    {
      // R->Range /= p->MinContext->Union2.SummFreq;
      RC_EncodeFinal(0, s->Freq)
      p->FoundState = s;
      Ppmd7_Update1_0(p);
      return;
    }
    p->PrevSuccess = 0;
    sum = s->Freq;
    i = (unsigned)p->MinContext->NumStats - 1;
    do
    {
      if ((++s)->Symbol == symbol)
      {
        // R->Range /= p->MinContext->Union2.SummFreq;
        RC_EncodeFinal(sum, s->Freq)
        p->FoundState = s;
        Ppmd7_Update1(p);
        return;
      }
      sum += s->Freq;
    }
    while (--i);

    // R->Range /= p->MinContext->Union2.SummFreq;
    RC_Encode(sum, p->MinContext->Union2.SummFreq - sum)
    
    p->HiBitsFlag = PPMD7_HiBitsFlag_3(p->FoundState->Symbol);
    PPMD_SetAllBitsIn256Bytes(charMask)
    // MASK(s->Symbol) = 0;
    // i = p->MinContext->NumStats - 1;
    // do { MASK((--s)->Symbol) = 0; } while (--i);
    {
      CPpmd_State *s2 = Ppmd7_GetStats(p, p->MinContext);
      MASK(s->Symbol) = 0;
      do
      {
        const unsigned sym0 = s2[0].Symbol;
        const unsigned sym1 = s2[1].Symbol;
        s2 += 2;
        MASK(sym0) = 0;
        MASK(sym1) = 0;
      }
      while (s2 < s);
    }
  }
  else
  {
    UInt16 *prob = Ppmd7_GetBinSumm(p);
    CPpmd_State *s = Ppmd7Context_OneState(p->MinContext);
    UInt32 pr = *prob;
    const UInt32 bound = (R->Range >> 14) * pr;
    pr = PPMD_UPDATE_PROB_1(pr);
    if (s->Symbol == symbol)
    {
      *prob = (UInt16)(pr + (1 << PPMD_INT_BITS));
      // RangeEnc_EncodeBit_0(p, bound);
      R->Range = bound;
      RC_NORM_1(p)
      
      // p->FoundState = s;
      // Ppmd7_UpdateBin(p);
      {
        const unsigned freq = s->Freq;
        CPpmd7_Context *c = CTX(SUCCESSOR(s));
        p->FoundState = s;
        p->PrevSuccess = 1;
        p->RunLength++;
        s->Freq = (Byte)(freq + (freq < 128));
        // NextContext(p);
        if (p->OrderFall == 0 && (const Byte *)c > p->Text)
          p->MaxContext = p->MinContext = c;
        else
          Ppmd7_UpdateModel(p);
      }
      return;
    }

    *prob = (UInt16)pr;
    p->InitEsc = p->ExpEscape[pr >> 10];
    // RangeEnc_EncodeBit_1(p, bound);
    R->Low += bound;
    R->Range -= bound;
    RC_NORM_LOCAL(p)
    
    PPMD_SetAllBitsIn256Bytes(charMask)
    MASK(s->Symbol) = 0;
    p->PrevSuccess = 0;
  }

  for (;;)
  {
    CPpmd_See *see;
    CPpmd_State *s;
    UInt32 sum, escFreq;
    CPpmd7_Context *mc;
    unsigned i, numMasked;
    
    RC_NORM_REMOTE(p)

    mc = p->MinContext;
    numMasked = mc->NumStats;

    do
    {
      p->OrderFall++;
      if (!mc->Suffix)
        return; /* EndMarker (symbol = -1) */
      mc = Ppmd7_GetContext(p, mc->Suffix);
      i = mc->NumStats;
    }
    while (i == numMasked);

    p->MinContext = mc;
    
    // see = Ppmd7_MakeEscFreq(p, numMasked, &escFreq);
    {
      if (i != 256)
      {
        unsigned nonMasked = i - numMasked;
        see = p->See[(unsigned)p->NS2Indx[(size_t)nonMasked - 1]]
            + p->HiBitsFlag
            + (nonMasked < (unsigned)SUFFIX(mc)->NumStats - i)
            + 2 * (unsigned)(mc->Union2.SummFreq < 11 * i)
            + 4 * (unsigned)(numMasked > nonMasked);
        {
          // if (see->Summ) field is larger than 16-bit, we need only low 16 bits of Summ
          unsigned summ = (UInt16)see->Summ; // & 0xFFFF
          unsigned r = (summ >> see->Shift);
          see->Summ = (UInt16)(summ - r);
          escFreq = r + (r == 0);
        }
      }
      else
      {
        see = &p->DummySee;
        escFreq = 1;
      }
    }

    s = Ppmd7_GetStats(p, mc);
    sum = 0;
    // i = mc->NumStats;

    do
    {
      const unsigned cur = s->Symbol;
      if ((int)cur == symbol)
      {
        const UInt32 low = sum;
        const UInt32 freq = s->Freq;
        unsigned num2;

        Ppmd_See_UPDATE(see)
        p->FoundState = s;
        sum += escFreq;

        num2 = i / 2;
        i &= 1;
        sum += freq & (0 - (UInt32)i);
        if (num2 != 0)
        {
          s += i;
          do
          {
            const unsigned sym0 = s[0].Symbol;
            const unsigned sym1 = s[1].Symbol;
            s += 2;
            sum += (s[-2].Freq & (unsigned)(MASK(sym0)));
            sum += (s[-1].Freq & (unsigned)(MASK(sym1)));
          }
          while (--num2);
        }

        
        R->Range /= sum;
        RC_EncodeFinal(low, freq)
        Ppmd7_Update2(p);
        return;
      }
      sum += (s->Freq & (unsigned)(MASK(cur)));
      s++;
    }
    while (--i);
    
    {
      const UInt32 total = sum + escFreq;
      see->Summ = (UInt16)(see->Summ + total);

      R->Range /= total;
      RC_Encode(sum, escFreq)
    }

    {
      const CPpmd_State *s2 = Ppmd7_GetStats(p, p->MinContext);
      s--;
      MASK(s->Symbol) = 0;
      do
      {
        const unsigned sym0 = s2[0].Symbol;
        const unsigned sym1 = s2[1].Symbol;
        s2 += 2;
        MASK(sym0) = 0;
        MASK(sym1) = 0;
      }
      while (s2 < s);
    }
  }
}


void Ppmd7z_EncodeSymbols(CPpmd7 *p, const Byte *buf, const Byte *lim)
{
  for (; buf < lim; buf++)
  {
    Ppmd7z_EncodeSymbol(p, *buf);
  }
}

#undef kTopValue
#undef WRITE_BYTE
#undef RC_NORM_BASE
#undef RC_NORM_1
#undef RC_NORM
#undef RC_NORM_LOCAL
#undef RC_NORM_REMOTE
#undef R
#undef RC_Encode
#undef RC_EncodeFinal
#undef SUFFIX
#undef CTX
#undef SUCCESSOR
#undef MASK
