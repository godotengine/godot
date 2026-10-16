/* Copyright (c) 2009-2010 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "celt_lpc.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "pitch.h"

void _celt_lpc(
      opus_val16       *_lpc, /* out: [0...p-1] LPC coefficients      */
const opus_val32 *ac,  /* in:  [0...p] autocorrelation values  */
int          p
)
{
   int i, j;
   opus_val32 r;
   opus_val32 error = ac[0];
#ifdef FIXED_POINT
   opus_val32 lpc[CELT_LPC_ORDER];
#else
   float *lpc = _lpc;
#endif

   OPUS_CLEAR(lpc, p);
#ifdef FIXED_POINT
   if (ac[0] != 0)
#else
   if (ac[0] > 1e-10f)
#endif
   {
      for (i = 0; i < p; i++) {
         /* Sum up this iteration's reflection coefficient */
         opus_val32 rr = 0;
#if defined (FIXED_POINT) && OPUS_FAST_INT64
         opus_int64 acc = 0;
         for (j = 0; j < i; j++)
            acc += (opus_int64)(lpc[j]) * (opus_int64)(ac[i - j]);
         rr = (opus_val32)SHR64(acc, 31);
#else
         for (j = 0; j < i; j++)
            rr += MULT32_32_Q31(lpc[j],ac[i - j]);
#endif
         rr += SHR32(ac[i + 1],6);
         r = -frac_div32(SHL32(rr,6), error);
         /*  Update LPC coefficients and total error */
         lpc[i] = SHR32(r,6);
         for (j = 0; j < (i+1)>>1; j++)
         {
            opus_val32 tmp1, tmp2;
            tmp1 = lpc[j];
            tmp2 = lpc[i-1-j];
            lpc[j]     = tmp1 + MULT32_32_Q31(r,tmp2);
            lpc[i-1-j] = tmp2 + MULT32_32_Q31(r,tmp1);
         }

         error = error - MULT32_32_Q31(MULT32_32_Q31(r,r),error);
         /* Bail out once we get 30 dB gain */
#ifdef FIXED_POINT
         if (error<=SHR32(ac[0],10))
            break;
#else
         if (error<=.001f*ac[0])
            break;
#endif
      }
   }
#ifdef FIXED_POINT
   {
      /* Convert the int32 lpcs to int16 and ensure there are no wrap-arounds.
         This reuses the logic in silk_LPC_fit() and silk_bwexpander_32(). Any bug
         fixes should also be applied there. */
      int iter, idx = 0;
      opus_val32 maxabs, absval, chirp_Q16, chirp_minus_one_Q16;

      for (iter = 0; iter < 10; iter++) {
         maxabs = 0;
         for (i = 0; i < p; i++) {
            absval = ABS32(lpc[i]);
            if (absval > maxabs) {
               maxabs = absval;
               idx = i;
            }
         }
         maxabs = PSHR32(maxabs, 13);  /* Q25->Q12 */

         if (maxabs > 32767) {
            maxabs = MIN32(maxabs, 163838);
            chirp_Q16 = QCONST32(0.999, 16) - DIV32(SHL32(maxabs - 32767, 14),
                                                    SHR32(MULT32_32_32(maxabs, idx + 1), 2));
            chirp_minus_one_Q16 = chirp_Q16 - 65536;

            /* Apply bandwidth expansion. */
            for (i = 0; i < p - 1; i++) {
               lpc[i] = MULT32_32_Q16(chirp_Q16, lpc[i]);
               chirp_Q16 += PSHR32(MULT32_32_32(chirp_Q16, chirp_minus_one_Q16), 16);
            }
            lpc[p - 1] = MULT32_32_Q16(chirp_Q16, lpc[p - 1]);
         } else {
            break;
         }
      }

      if (iter == 10) {
         /* If the coeffs still do not fit into the 16 bit range after 10 iterations,
            fall back to the A(z)=1 filter. */
         OPUS_CLEAR(lpc, p);
         _lpc[0] = 4096;  /* Q12 */
      } else {
         for (i = 0; i < p; i++) {
            _lpc[i] = EXTRACT16(PSHR32(lpc[i], 13));  /* Q25->Q12 */
         }
      }
   }
#endif
}


void celt_fir_c(
         const opus_val16 *x,
         const opus_val16 *num,
         opus_val16 *y,
         int N,
         int ord,
         int arch)
{
   int i,j;
   VARDECL(opus_val16, rnum);
   SAVE_STACK;
   celt_assert(x != y);
   ALLOC(rnum, ord, opus_val16);
   for(i=0;i<ord;i++)
      rnum[i] = num[ord-i-1];
   for (i=0;i<N-3;i+=4)
   {
      opus_val32 sum[4];
      sum[0] = SHL32(EXTEND32(x[i  ]), SIG_SHIFT);
      sum[1] = SHL32(EXTEND32(x[i+1]), SIG_SHIFT);
      sum[2] = SHL32(EXTEND32(x[i+2]), SIG_SHIFT);
      sum[3] = SHL32(EXTEND32(x[i+3]), SIG_SHIFT);
#if defined(OPUS_CHECK_ASM) && defined(FIXED_POINT)
      {
         opus_val32 sum_c[4];
         memcpy(sum_c, sum, sizeof(sum_c));
         xcorr_kernel_c(rnum, x+i-ord, sum_c, ord);
#endif
         xcorr_kernel(rnum, x+i-ord, sum, ord, arch);
#if defined(OPUS_CHECK_ASM) && defined(FIXED_POINT)
         celt_assert(memcmp(sum, sum_c, sizeof(sum)) == 0);
      }
#endif
      y[i  ] = SROUND16(sum[0], SIG_SHIFT);
      y[i+1] = SROUND16(sum[1], SIG_SHIFT);
      y[i+2] = SROUND16(sum[2], SIG_SHIFT);
      y[i+3] = SROUND16(sum[3], SIG_SHIFT);
   }
   for (;i<N;i++)
   {
      opus_val32 sum = SHL32(EXTEND32(x[i]), SIG_SHIFT);
      for (j=0;j<ord;j++)
         sum = MAC16_16(sum,rnum[j],x[i+j-ord]);
      y[i] = SROUND16(sum, SIG_SHIFT);
   }
   RESTORE_STACK;
}

void celt_iir(const opus_val32 *_x,
         const opus_val16 *den,
         opus_val32 *_y,
         int N,
         int ord,
         opus_val16 *mem,
         int arch)
{
#ifdef SMALL_FOOTPRINT
   int i,j;
   (void)arch;
   for (i=0;i<N;i++)
   {
      opus_val32 sum = _x[i];
      for (j=0;j<ord;j++)
      {
         sum -= MULT16_16(den[j],mem[j]);
      }
      for (j=ord-1;j>=1;j--)
      {
         mem[j]=mem[j-1];
      }
      mem[0] = SROUND16(sum, SIG_SHIFT);
      _y[i] = sum;
   }
#else
   int i,j;
   VARDECL(opus_val16, rden);
   VARDECL(opus_val16, y);
   SAVE_STACK;

   celt_assert((ord&3)==0);
   ALLOC(rden, ord, opus_val16);
   ALLOC(y, N+ord, opus_val16);
   for(i=0;i<ord;i++)
      rden[i] = den[ord-i-1];
   for(i=0;i<ord;i++)
      y[i] = -mem[ord-i-1];
   for(;i<N+ord;i++)
      y[i]=0;
   for (i=0;i<N-3;i+=4)
   {
      /* Unroll by 4 as if it were an FIR filter */
      opus_val32 sum[4];
      sum[0]=_x[i];
      sum[1]=_x[i+1];
      sum[2]=_x[i+2];
      sum[3]=_x[i+3];
#if defined(OPUS_CHECK_ASM) && defined(FIXED_POINT)
      {
         opus_val32 sum_c[4];
         memcpy(sum_c, sum, sizeof(sum_c));
         xcorr_kernel_c(rden, y+i, sum_c, ord);
#endif
         xcorr_kernel(rden, y+i, sum, ord, arch);
#if defined(OPUS_CHECK_ASM) && defined(FIXED_POINT)
         celt_assert(memcmp(sum, sum_c, sizeof(sum)) == 0);
      }
#endif
      /* Patch up the result to compensate for the fact that this is an IIR */
      y[i+ord  ] = -SROUND16(sum[0],SIG_SHIFT);
      _y[i  ] = sum[0];
      sum[1] = MAC16_16(sum[1], y[i+ord  ], den[0]);
      y[i+ord+1] = -SROUND16(sum[1],SIG_SHIFT);
      _y[i+1] = sum[1];
      sum[2] = MAC16_16(sum[2], y[i+ord+1], den[0]);
      sum[2] = MAC16_16(sum[2], y[i+ord  ], den[1]);
      y[i+ord+2] = -SROUND16(sum[2],SIG_SHIFT);
      _y[i+2] = sum[2];

      sum[3] = MAC16_16(sum[3], y[i+ord+2], den[0]);
      sum[3] = MAC16_16(sum[3], y[i+ord+1], den[1]);
      sum[3] = MAC16_16(sum[3], y[i+ord  ], den[2]);
      y[i+ord+3] = -SROUND16(sum[3],SIG_SHIFT);
      _y[i+3] = sum[3];
   }
   for (;i<N;i++)
   {
      opus_val32 sum = _x[i];
      for (j=0;j<ord;j++)
         sum -= MULT16_16(rden[j],y[i+j]);
      y[i+ord] = SROUND16(sum,SIG_SHIFT);
      _y[i] = sum;
   }
   for(i=0;i<ord;i++)
      mem[i] = _y[N-i-1];
   RESTORE_STACK;
#endif
}

int _celt_autocorr(
                   const opus_val16 *x,   /*  in: [0...n-1] samples x   */
                   opus_val32       *ac,  /* out: [0...lag-1] ac values */
                   const celt_coef  *window,
                   int          overlap,
                   int          lag,
                   int          n,
                   int          arch
                  )
{
   opus_val32 d;
   int i, k;
   int fastN=n-lag;
   int shift;
   const opus_val16 *xptr;
   VARDECL(opus_val16, xx);
   SAVE_STACK;
   ALLOC(xx, n, opus_val16);
   celt_assert(n>0);
   celt_assert(overlap>=0);
   if (overlap == 0)
   {
      xptr = x;
   } else {
      for (i=0;i<n;i++)
         xx[i] = x[i];
      for (i=0;i<overlap;i++)
      {
         opus_val16 w = COEF2VAL16(window[i]);
         xx[i] = MULT16_16_Q15(x[i],w);
         xx[n-i-1] = MULT16_16_Q15(x[n-i-1],w);
      }
      xptr = xx;
   }
   shift=0;
#ifdef FIXED_POINT
   {
      opus_val32 ac0;
      int ac0_shift = celt_ilog2(n + (n>>4));
      ac0 = 1+(n<<7);
      if (n&1) ac0 += SHR32(MULT16_16(xptr[0],xptr[0]),ac0_shift);
      for(i=(n&1);i<n;i+=2)
      {
         ac0 += SHR32(MULT16_16(xptr[i],xptr[i]),ac0_shift);
         ac0 += SHR32(MULT16_16(xptr[i+1],xptr[i+1]),ac0_shift);
      }
      /* Consider the effect of rounding-to-nearest when scaling the signal. */
      ac0 += SHR32(ac0,7);

      shift = celt_ilog2(ac0)-30+ac0_shift+1;
      shift = (shift)/2;
      if (shift>0)
      {
         for(i=0;i<n;i++)
            xx[i] = PSHR32(xptr[i], shift);
         xptr = xx;
      } else
         shift = 0;
   }
#endif
   celt_pitch_xcorr(xptr, xptr, ac, fastN, lag+1, arch);
   for (k=0;k<=lag;k++)
   {
      for (i = k+fastN, d = 0; i < n; i++)
         d = MAC16_16(d, xptr[i], xptr[i-k]);
      ac[k] += d;
   }
#ifdef FIXED_POINT
   shift = 2*shift;
   if (shift<=0)
      ac[0] += SHL32((opus_int32)1, -shift);
   if (ac[0] < 268435456)
   {
      int shift2 = 29 - EC_ILOG(ac[0]);
      for (i=0;i<=lag;i++)
         ac[i] = SHL32(ac[i], shift2);
      shift -= shift2;
   } else if (ac[0] >= 536870912)
   {
      int shift2=1;
      if (ac[0] >= 1073741824)
         shift2++;
      for (i=0;i<=lag;i++)
         ac[i] = SHR32(ac[i], shift2);
      shift += shift2;
   }
#endif

   RESTORE_STACK;
   return shift;
}
