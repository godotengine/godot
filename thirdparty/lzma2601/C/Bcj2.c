/* Bcj2.c -- BCJ2 Decoder (Converter for x86 code)
2023-03-01 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include "Bcj2.h"
#include "CpuArch.h"

#define kTopValue ((UInt32)1 << 24)
#define kNumBitModelTotalBits 11
#define kBitModelTotal (1 << kNumBitModelTotalBits)
#define kNumMoveBits 5

// UInt32 bcj2_stats[256 + 2][2];

void Bcj2Dec_Init(CBcj2Dec *p)
{
  unsigned i;
  p->state = BCJ2_STREAM_RC; // BCJ2_DEC_STATE_OK;
  p->ip = 0;
  p->temp = 0;
  p->range = 0;
  p->code = 0;
  for (i = 0; i < sizeof(p->probs) / sizeof(p->probs[0]); i++)
    p->probs[i] = kBitModelTotal >> 1;
}

SRes Bcj2Dec_Decode(CBcj2Dec *p)
{
  UInt32 v = p->temp;
  // const Byte *src;
  if (p->range <= 5)
  {
    UInt32 code = p->code;
    p->state = BCJ2_DEC_STATE_ERROR; /* for case if we return SZ_ERROR_DATA; */
    for (; p->range != 5; p->range++)
    {
      if (p->range == 1 && code != 0)
        return SZ_ERROR_DATA;
      if (p->bufs[BCJ2_STREAM_RC] == p->lims[BCJ2_STREAM_RC])
      {
        p->state = BCJ2_STREAM_RC;
        return SZ_OK;
      }
      code = (code << 8) | *(p->bufs[BCJ2_STREAM_RC])++;
      p->code = code;
    }
    if (code == 0xffffffff)
      return SZ_ERROR_DATA;
    p->range = 0xffffffff;
  }
  // else
  {
    unsigned state = p->state;
    // we check BCJ2_IS_32BIT_STREAM() here instead of check in the main loop
    if (BCJ2_IS_32BIT_STREAM(state))
    {
      const Byte *cur = p->bufs[state];
      if (cur == p->lims[state])
        return SZ_OK;
      p->bufs[state] = cur + 4;
      {
        const UInt32 ip = p->ip + 4;
        v = GetBe32a(cur) - ip;
        p->ip = ip;
      }
      state = BCJ2_DEC_STATE_ORIG_0;
    }
    if ((unsigned)(state - BCJ2_DEC_STATE_ORIG_0) < 4)
    {
      Byte *dest = p->dest;
      for (;;)
      {
        if (dest == p->destLim)
        {
          p->state = state;
          p->temp = v;
          return SZ_OK;
        }
        *dest++ = (Byte)v;
        p->dest = dest;
        if (++state == BCJ2_DEC_STATE_ORIG_3 + 1)
          break;
        v >>= 8;
      }
    }
  }

  // src = p->bufs[BCJ2_STREAM_MAIN];
  for (;;)
  {
    /*
    if (BCJ2_IS_32BIT_STREAM(p->state))
      p->state = BCJ2_DEC_STATE_OK;
    else
    */
    {
      if (p->range < kTopValue)
      {
        if (p->bufs[BCJ2_STREAM_RC] == p->lims[BCJ2_STREAM_RC])
        {
          p->state = BCJ2_STREAM_RC;
          p->temp = v;
          return SZ_OK;
        }
        p->range <<= 8;
        p->code = (p->code << 8) | *(p->bufs[BCJ2_STREAM_RC])++;
      }
      {
        const Byte *src = p->bufs[BCJ2_STREAM_MAIN];
        const Byte *srcLim;
        Byte *dest = p->dest;
        {
          const SizeT rem = (SizeT)(p->lims[BCJ2_STREAM_MAIN] - src);
          SizeT num = (SizeT)(p->destLim - dest);
          if (num >= rem)
            num = rem;
        #define NUM_ITERS 4
        #if (NUM_ITERS & (NUM_ITERS - 1)) == 0
          num &= ~((SizeT)NUM_ITERS - 1);   // if (NUM_ITERS == (1 << x))
        #else
          num -= num % NUM_ITERS; // if (NUM_ITERS != (1 << x))
        #endif
          srcLim = src + num;
        }

        #define NUM_SHIFT_BITS  24
        #define ONE_ITER(indx) { \
          const unsigned b = src[indx]; \
          *dest++ = (Byte)b; \
          v = (v << NUM_SHIFT_BITS) | b; \
          if (((b + (0x100 - 0xe8)) & 0xfe) == 0) break; \
          if (((v - (((UInt32)0x0f << (NUM_SHIFT_BITS)) + 0x80)) & \
              ((((UInt32)1 << (4 + NUM_SHIFT_BITS)) - 0x1) << 4)) == 0) break; \
            /* ++dest */; /* v = b; */ }
          
        if (src != srcLim)
        for (;;)
        {
            /* The dependency chain of 2-cycle for (v) calculation is not big problem here.
               But we can remove dependency chain with v = b in the end of loop. */
          ONE_ITER(0)
          #if (NUM_ITERS > 1)
            ONE_ITER(1)
          #if (NUM_ITERS > 2)
            ONE_ITER(2)
          #if (NUM_ITERS > 3)
            ONE_ITER(3)
          #if (NUM_ITERS > 4)
            ONE_ITER(4)
          #if (NUM_ITERS > 5)
            ONE_ITER(5)
          #if (NUM_ITERS > 6)
            ONE_ITER(6)
          #if (NUM_ITERS > 7)
            ONE_ITER(7)
          #endif
          #endif
          #endif
          #endif
          #endif
          #endif
          #endif
          
          src += NUM_ITERS;
          if (src == srcLim)
            break;
        }

        if (src == srcLim)
      #if (NUM_ITERS > 1)
        for (;;)
      #endif
        {
        #if (NUM_ITERS > 1)
          if (src == p->lims[BCJ2_STREAM_MAIN] || dest == p->destLim)
        #endif
          {
            const SizeT num = (SizeT)(src - p->bufs[BCJ2_STREAM_MAIN]);
            p->bufs[BCJ2_STREAM_MAIN] = src;
            p->dest = dest;
            p->ip += (UInt32)num;
            /* state BCJ2_STREAM_MAIN has more priority than BCJ2_STATE_ORIG */
            p->state =
              src == p->lims[BCJ2_STREAM_MAIN] ?
                (unsigned)BCJ2_STREAM_MAIN :
                (unsigned)BCJ2_DEC_STATE_ORIG;
            p->temp = v;
            return SZ_OK;
          }
        #if (NUM_ITERS > 1)
          ONE_ITER(0)
          src++;
        #endif
        }

        {
          const SizeT num = (SizeT)(dest - p->dest);
          p->dest = dest; // p->dest += num;
          p->bufs[BCJ2_STREAM_MAIN] += num; // = src;
          p->ip += (UInt32)num;
        }
        {
          UInt32 bound, ttt;
          CBcj2Prob *prob; // unsigned index;
          /*
          prob = p->probs + (unsigned)((Byte)v == 0xe8 ?
              2 + (Byte)(v >> 8) :
              ((v >> 5) & 1));  // ((Byte)v < 0xe8 ? 0 : 1));
          */
          {
            const unsigned c = ((v + 0x17) >> 6) & 1;
            prob = p->probs + (unsigned)
                (((0 - c) & (Byte)(v >> NUM_SHIFT_BITS)) + c + ((v >> 5) & 1));
                // (Byte)
                // 8x->0     : e9->1     : xxe8->xx+2
                // 8x->0x100 : e9->0x101 : xxe8->xx
                // (((0x100 - (e & ~v)) & (0x100 | (v >> 8))) + (e & v));
                // (((0x101 + (~e | v)) & (0x100 | (v >> 8))) + (e & v));
          }
          ttt = *prob;
          bound = (p->range >> kNumBitModelTotalBits) * ttt;
          if (p->code < bound)
          {
            // bcj2_stats[prob - p->probs][0]++;
            p->range = bound;
            *prob = (CBcj2Prob)(ttt + ((kBitModelTotal - ttt) >> kNumMoveBits));
            continue;
          }
          {
            // bcj2_stats[prob - p->probs][1]++;
            p->range -= bound;
            p->code -= bound;
            *prob = (CBcj2Prob)(ttt - (ttt >> kNumMoveBits));
          }
        }
      }
    }
    {
      /* (v == 0xe8 ? 0 : 1) uses setcc instruction with additional zero register usage in x64 MSVC. */
      // const unsigned cj = ((Byte)v == 0xe8) ? BCJ2_STREAM_CALL : BCJ2_STREAM_JUMP;
      const unsigned cj = (((v + 0x57) >> 6) & 1) + BCJ2_STREAM_CALL;
      const Byte *cur = p->bufs[cj];
      Byte *dest;
      SizeT rem;
      if (cur == p->lims[cj])
      {
        p->state = cj;
        break;
      }
      v = GetBe32a(cur);
      p->bufs[cj] = cur + 4;
      {
        const UInt32 ip = p->ip + 4;
        v -= ip;
        p->ip = ip;
      }
      dest = p->dest;
      rem = (SizeT)(p->destLim - dest);
      if (rem < 4)
      {
        if ((unsigned)rem > 0) { dest[0] = (Byte)v;  v >>= 8;
        if ((unsigned)rem > 1) { dest[1] = (Byte)v;  v >>= 8;
        if ((unsigned)rem > 2) { dest[2] = (Byte)v;  v >>= 8; }}}
        p->temp = v;
        p->dest = dest + rem;
        p->state = BCJ2_DEC_STATE_ORIG_0 + (unsigned)rem;
        break;
      }
      SetUi32(dest, v)
      v >>= 24;
      p->dest = dest + 4;
    }
  }

  if (p->range < kTopValue && p->bufs[BCJ2_STREAM_RC] != p->lims[BCJ2_STREAM_RC])
  {
    p->range <<= 8;
    p->code = (p->code << 8) | *(p->bufs[BCJ2_STREAM_RC])++;
  }
  return SZ_OK;
}

#undef NUM_ITERS
#undef ONE_ITER
#undef NUM_SHIFT_BITS
#undef kTopValue
#undef kNumBitModelTotalBits
#undef kBitModelTotal
#undef kNumMoveBits
