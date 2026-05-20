/* Bcj2Enc.c -- BCJ2 Encoder converter for x86 code (Branch CALL/JUMP variant2)
2023-04-02 : Igor Pavlov : Public domain */

#include "Precomp.h"

/* #define SHOW_STAT */
#ifdef SHOW_STAT
#include <stdio.h>
#define PRF2(s) printf("%s ip=%8x  tempPos=%d  src= %8x\n", s, (unsigned)p->ip64, p->tempPos, (unsigned)(p->srcLim - p->src));
#else
#define PRF2(s)
#endif

#include "Bcj2.h"
#include "CpuArch.h"

#define kTopValue ((UInt32)1 << 24)
#define kNumBitModelTotalBits 11
#define kBitModelTotal (1 << kNumBitModelTotalBits)
#define kNumMoveBits 5

void Bcj2Enc_Init(CBcj2Enc *p)
{
  unsigned i;
  p->state = BCJ2_ENC_STATE_ORIG;
  p->finishMode = BCJ2_ENC_FINISH_MODE_CONTINUE;
  p->context = 0;
  p->flushRem = 5;
  p->isFlushState = 0;
  p->cache = 0;
  p->range = 0xffffffff;
  p->low = 0;
  p->cacheSize = 1;
  p->ip64 = 0;
  p->fileIp64 = 0;
  p->fileSize64_minus1 = BCJ2_ENC_FileSizeField_UNLIMITED;
  p->relatLimit = BCJ2_ENC_RELAT_LIMIT_DEFAULT;
  // p->relatExcludeBits = 0;
  p->tempPos = 0;
  for (i = 0; i < sizeof(p->probs) / sizeof(p->probs[0]); i++)
    p->probs[i] = kBitModelTotal >> 1;
}

// Z7_NO_INLINE
Z7_FORCE_INLINE
static BoolInt Bcj2_RangeEnc_ShiftLow(CBcj2Enc *p)
{
  const UInt32 low = (UInt32)p->low;
  const unsigned high = (unsigned)
    #if defined(Z7_MSC_VER_ORIGINAL) \
        && defined(MY_CPU_X86) \
        && defined(MY_CPU_LE) \
        && !defined(MY_CPU_64BIT)
      // we try to rid of __aullshr() call in MSVS-x86
      (((const UInt32 *)&p->low)[1]); // [1] : for little-endian only
    #else
      (p->low >> 32);
    #endif
  if (low < (UInt32)0xff000000 || high != 0)
  {
    Byte *buf = p->bufs[BCJ2_STREAM_RC];
    do
    {
      if (buf == p->lims[BCJ2_STREAM_RC])
      {
        p->state = BCJ2_STREAM_RC;
        p->bufs[BCJ2_STREAM_RC] = buf;
        return True;
      }
      *buf++ = (Byte)(p->cache + high);
      p->cache = 0xff;
    }
    while (--p->cacheSize);
    p->bufs[BCJ2_STREAM_RC] = buf;
    p->cache = (Byte)(low >> 24);
  }
  p->cacheSize++;
  p->low = low << 8;
  return False;
}


/*
We can use 2 alternative versions of code:
1) non-marker version:
  Byte CBcj2Enc::context
  Byte temp[8];
  Last byte of marker (e8/e9/[0f]8x) can be written to temp[] buffer.
  Encoder writes last byte of marker (e8/e9/[0f]8x) to dest, only in conjunction
  with writing branch symbol to range coder in same Bcj2Enc_Encode_2() call.

2) marker version:
  UInt32 CBcj2Enc::context
  Byte CBcj2Enc::temp[4];
  MARKER_FLAG in CBcj2Enc::context shows that CBcj2Enc::context contains finded marker.
  it's allowed that
    one call of Bcj2Enc_Encode_2() writes last byte of marker (e8/e9/[0f]8x) to dest,
    and another call of Bcj2Enc_Encode_2() does offset conversion.
    So different values of (fileIp) and (fileSize) are possible
    in these different Bcj2Enc_Encode_2() calls.

Also marker version requires additional if((v & MARKER_FLAG) == 0) check in main loop.
So we use non-marker version.
*/

/*
  Corner cases with overlap in multi-block.
  before v23: there was one corner case, where converted instruction
    could start in one sub-stream and finish in next sub-stream.
  If multi-block (solid) encoding is used,
    and BCJ2_ENC_FINISH_MODE_END_BLOCK is used for each sub-stream.
    and (0f) is last byte of previous sub-stream
    and (8x) is first byte of current sub-stream
  then (0f 8x) pair is treated as marker by BCJ2 encoder and decoder.
  BCJ2 encoder can converts 32-bit offset for that (0f 8x) cortage,
  if that offset meets limit requirements.
  If encoder allows 32-bit offset conversion for such overlap case,
  then the data in 3 uncompressed BCJ2 streams for some sub-stream
  can depend from data of previous sub-stream.
  That corner case is not big problem, and it's rare case.
  Since v23.00 we do additional check to prevent conversions in such overlap cases.
*/

/*
  Bcj2Enc_Encode_2() output variables at exit:
  {
    if (Bcj2Enc_Encode_2() exits with (p->state == BCJ2_ENC_STATE_ORIG))
    {
      it means that encoder needs more input data.
      if (p->srcLim == p->src) at exit, then
      {
        (p->finishMode != BCJ2_ENC_FINISH_MODE_END_STREAM)
        all input data were read and processed, and we are ready for
        new input data.
      }
      else
      {
        (p->srcLim != p->src)
        (p->finishMode == BCJ2_ENC_FINISH_MODE_CONTINUE)
          The encoder have found e8/e9/0f_8x marker,
          and p->src points to last byte of that marker,
          Bcj2Enc_Encode_2() needs more input data to get totally
          5 bytes (last byte of marker and 32-bit branch offset)
          as continuous array starting from p->src.
        (p->srcLim - p->src < 5) requirement is met after exit.
          So non-processed resedue from p->src to p->srcLim is always less than 5 bytes.
      }
    }
  }
*/

Z7_NO_INLINE
static void Bcj2Enc_Encode_2(CBcj2Enc *p)
{
  if (!p->isFlushState)
  {
    const Byte *src;
    UInt32 v;
    {
      const unsigned state = p->state;
      if (BCJ2_IS_32BIT_STREAM(state))
      {
        Byte *cur = p->bufs[state];
        if (cur == p->lims[state])
          return;
        SetBe32a(cur, p->tempTarget)
        p->bufs[state] = cur + 4;
      }
    }
    p->state = BCJ2_ENC_STATE_ORIG; // for main reason of exit
    src = p->src;
    v = p->context;
    
    // #define WRITE_CONTEXT  p->context = v; // for marker version
    #define WRITE_CONTEXT           p->context = (Byte)v;
    #define WRITE_CONTEXT_AND_SRC   p->src = src;  WRITE_CONTEXT

    for (;;)
    {
      // const Byte *src;
      // UInt32 v;
      CBcj2Enc_ip_unsigned ip;
      if (p->range < kTopValue)
      {
        // to reduce register pressure and code size: we save and restore local variables.
        WRITE_CONTEXT_AND_SRC
        if (Bcj2_RangeEnc_ShiftLow(p))
          return;
        p->range <<= 8;
        src = p->src;
        v = p->context;
      }
      // src = p->src;
      // #define MARKER_FLAG  ((UInt32)1 << 17)
      // if ((v & MARKER_FLAG) == 0) // for marker version
      {
        const Byte *srcLim;
        Byte *dest = p->bufs[BCJ2_STREAM_MAIN];
        {
          const SizeT remSrc = (SizeT)(p->srcLim - src);
          SizeT rem = (SizeT)(p->lims[BCJ2_STREAM_MAIN] - dest);
          if (rem >= remSrc)
            rem = remSrc;
          srcLim = src + rem;
        }
        /* p->context contains context of previous byte:
           bits [0 : 7]  : src[-1], if (src) was changed in this call
           bits [8 : 31] : are undefined for non-marker version
        */
        // v = p->context;
        #define NUM_SHIFT_BITS  24
        #define CONV_FLAG  ((UInt32)1 << 16)
        #define ONE_ITER { \
          b = src[0]; \
          *dest++ = (Byte)b; \
          v = (v << NUM_SHIFT_BITS) | b; \
          if (((b + (0x100 - 0xe8)) & 0xfe) == 0) break; \
          if (((v - (((UInt32)0x0f << (NUM_SHIFT_BITS)) + 0x80)) & \
              ((((UInt32)1 << (4 + NUM_SHIFT_BITS)) - 0x1) << 4)) == 0) break; \
          src++; if (src == srcLim) { break; } }

        if (src != srcLim)
        for (;;)
        {
          /* clang can generate ineffective code with setne instead of two jcc instructions.
             we can use 2 iterations and external (unsigned b) to avoid that ineffective code genaration. */
          unsigned b;
          ONE_ITER
          ONE_ITER
        }
        
        ip = p->ip64 + (CBcj2Enc_ip_unsigned)(SizeT)(dest - p->bufs[BCJ2_STREAM_MAIN]);
        p->bufs[BCJ2_STREAM_MAIN] = dest;
        p->ip64 = ip;

        if (src == srcLim)
        {
          WRITE_CONTEXT_AND_SRC
          if (src != p->srcLim)
          {
            p->state = BCJ2_STREAM_MAIN;
            return;
          }
          /* (p->src == p->srcLim)
          (p->state == BCJ2_ENC_STATE_ORIG) */
          if (p->finishMode != BCJ2_ENC_FINISH_MODE_END_STREAM)
            return;
          /* (p->finishMode == BCJ2_ENC_FINISH_MODE_END_STREAM */
          // (p->flushRem == 5);
          p->isFlushState = 1;
          break;
        }
        src++;
        // p->src = src;
      }
      // ip = p->ip; // for marker version
      /* marker was found */
      /* (v) contains marker that was found:
           bits [NUM_SHIFT_BITS : NUM_SHIFT_BITS + 7]
                         : value of src[-2] : xx/xx/0f
           bits [0 : 7]  : value of src[-1] : e8/e9/8x
      */
      {
        {
        #if NUM_SHIFT_BITS != 24
          v &= ~(UInt32)CONV_FLAG;
        #endif
          // UInt32 relat = 0;
          if ((SizeT)(p->srcLim - src) >= 4)
          {
            /*
            if (relat != 0 || (Byte)v != 0xe8)
            BoolInt isBigOffset = True;
            */
            const UInt32 relat = GetUi32(src);
            /*
            #define EXCLUDE_FLAG  ((UInt32)1 << 4)
            #define NEED_CONVERT(rel) ((((rel) + EXCLUDE_FLAG) & (0 - EXCLUDE_FLAG * 2)) != 0)
            if (p->relatExcludeBits != 0)
            {
              const UInt32 flag = (UInt32)1 << (p->relatExcludeBits - 1);
              isBigOffset = (((relat + flag) & (0 - flag * 2)) != 0);
            }
            // isBigOffset = False; // for debug
            */
            ip -= p->fileIp64;
            // Use the following if check, if (ip) is 64-bit:
            if (ip > (((v + 0x20) >> 5) & 1))  // 23.00 : we eliminate milti-block overlap for (Of 80) and (e8/e9)
            if ((CBcj2Enc_ip_unsigned)((CBcj2Enc_ip_signed)ip + 4 + (Int32)relat) <= p->fileSize64_minus1)
            if (((UInt32)(relat + p->relatLimit) >> 1) < p->relatLimit)
              v |= CONV_FLAG;
          }
          else if (p->finishMode == BCJ2_ENC_FINISH_MODE_CONTINUE)
          {
            // (p->srcLim - src < 4)
            // /*
            // for non-marker version
            p->ip64--; // p->ip = ip - 1;
            p->bufs[BCJ2_STREAM_MAIN]--;
            src--;
            v >>= NUM_SHIFT_BITS;
            // (0 < p->srcLim - p->src <= 4)
            // */
            // v |= MARKER_FLAG; // for marker version
            /* (p->state == BCJ2_ENC_STATE_ORIG) */
            WRITE_CONTEXT_AND_SRC
            return;
          }
          {
            const unsigned c = ((v + 0x17) >> 6) & 1;
            CBcj2Prob *prob = p->probs + (unsigned)
                (((0 - c) & (Byte)(v >> NUM_SHIFT_BITS)) + c + ((v >> 5) & 1));
            /*
                ((Byte)v == 0xe8 ? 2 + ((Byte)(v >> 8)) :
                ((Byte)v < 0xe8 ? 0 : 1));  // ((v >> 5) & 1));
            */
            const unsigned ttt = *prob;
            const UInt32 bound = (p->range >> kNumBitModelTotalBits) * ttt;
            if ((v & CONV_FLAG) == 0)
            {
              // static int yyy = 0; yyy++; printf("\n!needConvert = %d\n", yyy);
              // v = (Byte)v; // for marker version
              p->range = bound;
              *prob = (CBcj2Prob)(ttt + ((kBitModelTotal - ttt) >> kNumMoveBits));
              // WRITE_CONTEXT_AND_SRC
              continue;
            }
            p->low += bound;
            p->range -= bound;
            *prob = (CBcj2Prob)(ttt - (ttt >> kNumMoveBits));
          }
          // p->context = src[3];
          {
            // const unsigned cj = ((Byte)v == 0xe8 ? BCJ2_STREAM_CALL : BCJ2_STREAM_JUMP);
            const unsigned cj = (((v + 0x57) >> 6) & 1) + BCJ2_STREAM_CALL;
            ip = p->ip64;
            v = GetUi32(src); // relat
            ip += 4;
            p->ip64 = ip;
            src += 4;
            // p->src = src;
            {
              const UInt32 absol = (UInt32)ip + v;
              Byte *cur = p->bufs[cj];
              v >>= 24;
              // WRITE_CONTEXT
              if (cur == p->lims[cj])
              {
                p->state = cj;
                p->tempTarget = absol;
                WRITE_CONTEXT_AND_SRC
                return;
              }
              SetBe32a(cur, absol)
              p->bufs[cj] = cur + 4;
            }
          }
        }
      }
    } // end of loop
  }

  for (; p->flushRem != 0; p->flushRem--)
    if (Bcj2_RangeEnc_ShiftLow(p))
      return;
  p->state = BCJ2_ENC_STATE_FINISHED;
}


/*
BCJ2 encoder needs look ahead for up to 4 bytes in (src) buffer.
So base function Bcj2Enc_Encode_2()
  in BCJ2_ENC_FINISH_MODE_CONTINUE mode can return with
  (p->state == BCJ2_ENC_STATE_ORIG && p->src < p->srcLim)
Bcj2Enc_Encode() solves that look ahead problem by using p->temp[] buffer.
  so if (p->state == BCJ2_ENC_STATE_ORIG) after Bcj2Enc_Encode(),
    then (p->src == p->srcLim).
  And the caller's code is simpler with Bcj2Enc_Encode().
*/

Z7_NO_INLINE
void Bcj2Enc_Encode(CBcj2Enc *p)
{
  PRF2("\n----")
  if (p->tempPos != 0)
  {
    /* extra: number of bytes that were copied from (src) to (temp) buffer in this call */
    unsigned extra = 0;
    /* We will touch only minimal required number of bytes in input (src) stream.
       So we will add input bytes from (src) stream to temp[] with step of 1 byte.
       We don't add new bytes to temp[] before Bcj2Enc_Encode_2() call
         in first loop iteration because
         - previous call of Bcj2Enc_Encode() could use another (finishMode),
         - previous call could finish with (p->state != BCJ2_ENC_STATE_ORIG).
       the case with full temp[] buffer (p->tempPos == 4) is possible here.
    */
    for (;;)
    {
      // (0 < p->tempPos <= 5) // in non-marker version
      /* p->src : the current src data position including extra bytes
                  that were copied to temp[] buffer in this call */
      const Byte *src = p->src;
      const Byte *srcLim = p->srcLim;
      const EBcj2Enc_FinishMode finishMode = p->finishMode;
      if (src != srcLim)
      {
        /* if there are some src data after the data copied to temp[],
           then we use MODE_CONTINUE for temp data */
        p->finishMode = BCJ2_ENC_FINISH_MODE_CONTINUE;
      }
      p->src = p->temp;
      p->srcLim = p->temp + p->tempPos;
      PRF2("    ")
      Bcj2Enc_Encode_2(p);
      {
        const unsigned num = (unsigned)(p->src - p->temp);
        const unsigned tempPos = p->tempPos - num;
        unsigned i;
        p->tempPos = tempPos;
        for (i = 0; i < tempPos; i++)
          p->temp[i] = p->temp[(SizeT)i + num];
        // tempPos : number of bytes in temp buffer
        p->src = src;
        p->srcLim = srcLim;
        p->finishMode = finishMode;
        if (p->state != BCJ2_ENC_STATE_ORIG)
        {
          // (p->tempPos <= 4) // in non-marker version
          /* if (the reason of exit from Bcj2Enc_Encode_2()
                 is not BCJ2_ENC_STATE_ORIG),
             then we exit from Bcj2Enc_Encode() with same reason */
          // optional code begin : we rollback (src) and tempPos, if it's possible:
          if (extra >= tempPos)
            extra = tempPos;
          p->src = src - extra;
          p->tempPos = tempPos - extra;
          // optional code end : rollback of (src) and tempPos
          return;
        }
        /* (p->tempPos <= 4)
           (p->state == BCJ2_ENC_STATE_ORIG)
             so encoder needs more data than in temp[] */
        if (src == srcLim)
          return; // src buffer has no more input data.
        /* (src != srcLim)
           so we can provide more input data from src for Bcj2Enc_Encode_2() */
        if (extra >= tempPos)
        {
          /* (extra >= tempPos) means that temp buffer contains
             only data from src buffer of this call.
             So now we can encode without temp buffer */
          p->src = src - tempPos; // rollback (src)
          p->tempPos = 0;
          break;
        }
        // we append one additional extra byte from (src) to temp[] buffer:
        p->temp[tempPos] = *src;
        p->tempPos = tempPos + 1;
        // (0 < p->tempPos <= 5) // in non-marker version
        p->src = src + 1;
        extra++;
      }
    }
  }

  PRF2("++++")
  // (p->tempPos == 0)
  Bcj2Enc_Encode_2(p);
  PRF2("====")
  
  if (p->state == BCJ2_ENC_STATE_ORIG)
  {
    const Byte *src = p->src;
    const Byte *srcLim = p->srcLim;
    const unsigned rem = (unsigned)(srcLim - src);
    /* (rem <= 4) here.
       if (p->src != p->srcLim), then
         - we copy non-processed bytes from (p->src) to temp[] buffer,
         - we set p->src equal to p->srcLim.
    */
    if (rem)
    {
      unsigned i = 0;
      p->src = srcLim;
      p->tempPos = rem;
      // (0 < p->tempPos <= 4)
      do
        p->temp[i] = src[i];
      while (++i != rem);
    }
    // (p->tempPos <= 4)
    // (p->src == p->srcLim)
  }
}

#undef PRF2
#undef CONV_FLAG
#undef MARKER_FLAG
#undef WRITE_CONTEXT
#undef WRITE_CONTEXT_AND_SRC
#undef ONE_ITER
#undef NUM_SHIFT_BITS
#undef kTopValue
#undef kNumBitModelTotalBits
#undef kBitModelTotal
#undef kNumMoveBits
