/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2008-2009 Gregory Maxwell
   Written by Jean-Marc Valin and Gregory Maxwell */
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

#include <math.h>
#include "bands.h"
#include "modes.h"
#include "vq.h"
#include "cwrs.h"
#include "stack_alloc.h"
#include "os_support.h"
#include "mathops.h"
#include "rate.h"
#include "quant_bands.h"
#include "pitch.h"

int hysteresis_decision(opus_val16 val, const opus_val16 *thresholds, const opus_val16 *hysteresis, int N, int prev)
{
   int i;
   for (i=0;i<N;i++)
   {
      if (val < thresholds[i])
         break;
   }
   if (i>prev && val < thresholds[prev]+hysteresis[prev])
      i=prev;
   if (i<prev && val > thresholds[prev-1]-hysteresis[prev-1])
      i=prev;
   return i;
}

opus_uint32 celt_lcg_rand(opus_uint32 seed)
{
   return 1664525 * seed + 1013904223;
}

/* This is a cos() approximation designed to be bit-exact on any platform. Bit exactness
   with this approximation is important because it has an impact on the bit allocation */
opus_int16 bitexact_cos(opus_int16 x)
{
   opus_int32 tmp;
   opus_int16 x2;
   tmp = (4096+((opus_int32)(x)*(x)))>>13;
   celt_sig_assert(tmp<=32767);
   x2 = tmp;
   x2 = (32767-x2) + FRAC_MUL16(x2, (-7651 + FRAC_MUL16(x2, (8277 + FRAC_MUL16(-626, x2)))));
   celt_sig_assert(x2<=32766);
   return 1+x2;
}

int bitexact_log2tan(int isin,int icos)
{
   int lc;
   int ls;
   lc=EC_ILOG(icos);
   ls=EC_ILOG(isin);
   icos<<=15-lc;
   isin<<=15-ls;
   return (ls-lc)*(1<<11)
         +FRAC_MUL16(isin, FRAC_MUL16(isin, -2597) + 7932)
         -FRAC_MUL16(icos, FRAC_MUL16(icos, -2597) + 7932);
}

#ifdef FIXED_POINT
/* Compute the amplitude (sqrt energy) in each of the bands */
void compute_band_energies(const CELTMode *m, const celt_sig *X, celt_ener *bandE, int end, int C, int LM, int arch)
{
   int i, c, N;
   const opus_int16 *eBands = m->eBands;
   (void)arch;
   N = m->shortMdctSize<<LM;
   c=0; do {
      for (i=0;i<end;i++)
      {
         int j;
         opus_val32 maxval=0;
         opus_val32 sum = 0;

         maxval = celt_maxabs32(&X[c*N+(eBands[i]<<LM)], (eBands[i+1]-eBands[i])<<LM);
         if (maxval > 0)
         {
            int shift = IMAX(0, 30 - celt_ilog2(maxval+(maxval>>14)+1) - ((((m->logN[i]+7)>>BITRES)+LM+1)>>1));
            j=eBands[i]<<LM; do {
               opus_val32 x = SHL32(X[j+c*N],shift);
               sum = ADD32(sum, MULT32_32_Q31(x, x));
            } while (++j<eBands[i+1]<<LM);
            bandE[i+c*m->nbEBands] = MAX32(maxval, PSHR32(celt_sqrt32(SHR32(sum,1)), shift));
         } else {
            bandE[i+c*m->nbEBands] = EPSILON;
         }
      }
   } while (++c<C);
}

/* Normalise each band such that the energy is one. */
void normalise_bands(const CELTMode *m, const celt_sig * OPUS_RESTRICT freq, celt_norm * OPUS_RESTRICT X, const celt_ener *bandE, int end, int C, int M)
{
   int i, c, N;
   const opus_int16 *eBands = m->eBands;
   N = M*m->shortMdctSize;
   c=0; do {
      i=0; do {
         int j,shift;
         opus_val32 E;
         opus_val32 g;
         E = bandE[i+c*m->nbEBands];
         /* For very low energies, we need this to make sure not to prevent energy rounding from
            blowing up the normalized signal. */
         if (E < 10) E += EPSILON;
         shift = 30-celt_zlog2(E);
         E = SHL32(E, shift);
         g = celt_rcp_norm32(E);
         j=M*eBands[i]; do {
            X[j+c*N] = PSHR32(MULT32_32_Q31(g, SHL32(freq[j+c*N], shift)), 30-NORM_SHIFT);
         } while (++j<M*eBands[i+1]);
      } while (++i<end);
   } while (++c<C);
}

#else /* FIXED_POINT */
/* Compute the amplitude (sqrt energy) in each of the bands */
void compute_band_energies(const CELTMode *m, const celt_sig *X, celt_ener *bandE, int end, int C, int LM, int arch)
{
   int i, c, N;
   const opus_int16 *eBands = m->eBands;
   N = m->shortMdctSize<<LM;
   c=0; do {
      for (i=0;i<end;i++)
      {
         opus_val32 sum;
         sum = 1e-27f + celt_inner_prod(&X[c*N+(eBands[i]<<LM)], &X[c*N+(eBands[i]<<LM)], (eBands[i+1]-eBands[i])<<LM, arch);
         bandE[i+c*m->nbEBands] = celt_sqrt(sum);
         /*printf ("%f ", bandE[i+c*m->nbEBands]);*/
      }
   } while (++c<C);
   /*printf ("\n");*/
}

/* Normalise each band such that the energy is one. */
void normalise_bands(const CELTMode *m, const celt_sig * OPUS_RESTRICT freq, celt_norm * OPUS_RESTRICT X, const celt_ener *bandE, int end, int C, int M)
{
   int i, c, N;
   const opus_int16 *eBands = m->eBands;
   N = M*m->shortMdctSize;
   c=0; do {
      for (i=0;i<end;i++)
      {
         int j;
         opus_val16 g = 1.f/(1e-27f+bandE[i+c*m->nbEBands]);
         for (j=M*eBands[i];j<M*eBands[i+1];j++)
            X[j+c*N] = freq[j+c*N]*g;
      }
   } while (++c<C);
}

#endif /* FIXED_POINT */

/* De-normalise the energy to produce the synthesis from the unit-energy bands */
void denormalise_bands(const CELTMode *m, const celt_norm * OPUS_RESTRICT X,
      celt_sig * OPUS_RESTRICT freq, const celt_glog *bandLogE, int start,
      int end, int M, int downsample, int silence)
{
   int i, N;
   int bound;
   celt_sig * OPUS_RESTRICT f;
   const celt_norm * OPUS_RESTRICT x;
   const opus_int16 *eBands = m->eBands;
   N = M*m->shortMdctSize;
   bound = M*eBands[end];
   if (downsample!=1)
      bound = IMIN(bound, N/downsample);
   if (silence)
   {
      bound = 0;
      start = end = 0;
   }
   f = freq;
   x = X+M*eBands[start];
   if (start != 0)
   {
      for (i=0;i<M*eBands[start];i++)
         *f++ = 0;
   } else {
      f += M*eBands[start];
   }
   for (i=start;i<end;i++)
   {
      int j, band_end;
      opus_val32 g;
      celt_glog lg;
#ifdef FIXED_POINT
      int shift;
#endif
      j=M*eBands[i];
      band_end = M*eBands[i+1];
      lg = ADD32(bandLogE[i], SHL32((opus_val32)eMeans[i],DB_SHIFT-4));
#ifndef FIXED_POINT
      g = celt_exp2_db(MIN32(32.f, lg));
#else
      /* Handle the integer part of the log energy */
      shift = 17-(lg>>DB_SHIFT);
      if (shift>=31)
      {
         shift=0;
         g=0;
      } else {
         /* Handle the fractional part. */
         g = SHL32(celt_exp2_db_frac((lg&((1<<DB_SHIFT)-1))), 2);
      }
      /* Handle extreme gains with negative shift. */
      if (shift<0)
      {
         /* To avoid overflow, we're
            capping the gain here, which is equivalent to a cap of 18 on lg.
            This shouldn't trigger unless the bitstream is already corrupted. */
         g = 2147483647;
         shift = 0;
      }
#endif
      do {
         *f++ = PSHR32(MULT32_32_Q31(SHL32(*x, 30-NORM_SHIFT), g), shift);
         x++;
      } while (++j<band_end);
   }
   celt_assert(start <= end);
   OPUS_CLEAR(&freq[bound], N-bound);
}

/* This prevents energy collapse for transients with multiple short MDCTs */
void anti_collapse(const CELTMode *m, celt_norm *X_, unsigned char *collapse_masks, int LM, int C, int size,
      int start, int end, const celt_glog *logE, const celt_glog *prev1logE,
      const celt_glog *prev2logE, const int *pulses, opus_uint32 seed, int encode, int arch)
{
   int c, i, j, k;
   for (i=start;i<end;i++)
   {
      int N0;
      opus_val16 thresh, sqrt_1;
      int depth;
#ifdef FIXED_POINT
      int shift;
      opus_val32 thresh32;
#endif

      N0 = m->eBands[i+1]-m->eBands[i];
      /* depth in 1/8 bits */
      celt_sig_assert(pulses[i]>=0);
      depth = celt_udiv(1+pulses[i], (m->eBands[i+1]-m->eBands[i]))>>LM;

#ifdef FIXED_POINT
      thresh32 = SHR32(celt_exp2(-SHL16(depth, 10-BITRES)),1);
      thresh = MULT16_32_Q15(QCONST16(0.5f, 15), MIN32(32767,thresh32));
      {
         opus_val32 t;
         t = N0<<LM;
         shift = celt_ilog2(t)>>1;
         t = SHL32(t, (7-shift)<<1);
         sqrt_1 = celt_rsqrt_norm(t);
      }
#else
      thresh = .5f*celt_exp2(-.125f*depth);
      sqrt_1 = celt_rsqrt(N0<<LM);
#endif

      c=0; do
      {
         celt_norm *X;
         celt_glog prev1;
         celt_glog prev2;
         opus_val32 Ediff;
         celt_norm r;
         int renormalize=0;
         prev1 = prev1logE[c*m->nbEBands+i];
         prev2 = prev2logE[c*m->nbEBands+i];
         if (!encode && C==1)
         {
            prev1 = MAXG(prev1,prev1logE[m->nbEBands+i]);
            prev2 = MAXG(prev2,prev2logE[m->nbEBands+i]);
         }
         Ediff = logE[c*m->nbEBands+i]-MING(prev1,prev2);
         Ediff = MAX32(0, Ediff);

#ifdef FIXED_POINT
         if (Ediff < GCONST(16.f))
         {
            opus_val32 r32 = SHR32(celt_exp2_db(-Ediff),1);
            r = 2*MIN16(16383,r32);
         } else {
            r = 0;
         }
         if (LM==3)
            r = MULT16_16_Q14(23170, MIN32(23169, r));
         r = SHR16(MIN16(thresh, r),1);
         r = VSHR32(MULT16_16_Q15(sqrt_1, r),shift+14-NORM_SHIFT);
#else
         /* r needs to be multiplied by 2 or 2*sqrt(2) depending on LM because
            short blocks don't have the same energy as long */
         r = 2.f*celt_exp2_db(-Ediff);
         if (LM==3)
            r *= 1.41421356f;
         r = MIN16(thresh, r);
         r = r*sqrt_1;
#endif
         X = X_+c*size+(m->eBands[i]<<LM);
         for (k=0;k<1<<LM;k++)
         {
            /* Detect collapse */
            if (!(collapse_masks[i*C+c]&1<<k))
            {
               /* Fill with noise */
               for (j=0;j<N0;j++)
               {
                  seed = celt_lcg_rand(seed);
                  X[(j<<LM)+k] = (seed&0x8000 ? r : -r);
               }
               renormalize = 1;
            }
         }
         /* We just added some energy, so we need to renormalise */
         if (renormalize)
            renormalise_vector(X, N0<<LM, Q31ONE, arch);
      } while (++c<C);
   }
}

/* Compute the weights to use for optimizing normalized distortion across
   channels. We use the amplitude to weight square distortion, which means
   that we use the square root of the value we would have been using if we
   wanted to minimize the MSE in the non-normalized domain. This roughly
   corresponds to some quick-and-dirty perceptual experiments I ran to
   measure inter-aural masking (there doesn't seem to be any published data
   on the topic). */
static void compute_channel_weights(celt_ener Ex, celt_ener Ey, opus_val16 w[2])
{
   celt_ener minE;
#ifdef FIXED_POINT
   int shift;
#endif
   minE = MIN32(Ex, Ey);
   /* Adjustment to make the weights a bit more conservative. */
   Ex = ADD32(Ex, minE/3);
   Ey = ADD32(Ey, minE/3);
#ifdef FIXED_POINT
   shift = celt_ilog2(EPSILON+MAX32(Ex, Ey))-14;
#endif
   w[0] = VSHR32(Ex, shift);
   w[1] = VSHR32(Ey, shift);
}

static void intensity_stereo(const CELTMode *m, celt_norm * OPUS_RESTRICT X, const celt_norm * OPUS_RESTRICT Y, const celt_ener *bandE, int bandID, int N)
{
   int i = bandID;
   int j;
   opus_val16 a1, a2;
   opus_val16 left, right;
   opus_val16 norm;
#ifdef FIXED_POINT
   int shift = celt_zlog2(MAX32(bandE[i], bandE[i+m->nbEBands]))-13;
#endif
   left = VSHR32(bandE[i],shift);
   right = VSHR32(bandE[i+m->nbEBands],shift);
   norm = EPSILON + celt_sqrt(EPSILON+MULT16_16(left,left)+MULT16_16(right,right));
#ifdef FIXED_POINT
   left = MIN32(left, norm-1);
   right = MIN32(right, norm-1);
#endif
   a1 = DIV32_16(SHL32(EXTEND32(left),15),norm);
   a2 = DIV32_16(SHL32(EXTEND32(right),15),norm);
   for (j=0;j<N;j++)
   {
      X[j] = ADD32(MULT16_32_Q15(a1, X[j]), MULT16_32_Q15(a2, Y[j]));
      /* Side is not encoded, no need to calculate */
   }
}

static void stereo_split(celt_norm * OPUS_RESTRICT X, celt_norm * OPUS_RESTRICT Y, int N)
{
   int j;
   for (j=0;j<N;j++)
   {
      opus_val32 r, l;
      l = MULT32_32_Q31(QCONST32(.70710678f,31), X[j]);
      r = MULT32_32_Q31(QCONST32(.70710678f,31), Y[j]);
      X[j] = ADD32(l, r);
      Y[j] = SUB32(r, l);
   }
}

static void stereo_merge(celt_norm * OPUS_RESTRICT X, celt_norm * OPUS_RESTRICT Y, opus_val32 mid, int N, int arch)
{
   int j;
   opus_val32 xp=0, side=0;
   opus_val32 El, Er;
#ifdef FIXED_POINT
   int kl, kr;
#endif
   opus_val32 t, lgain, rgain;

   /* Compute the norm of X+Y and X-Y as |X|^2 + |Y|^2 +/- sum(xy) */
   xp = celt_inner_prod_norm_shift(Y, X, N, arch);
   side = celt_inner_prod_norm_shift(Y, Y, N, arch);
   /* Compensating for the mid normalization */
   xp = MULT32_32_Q31(mid, xp);
   /* mid and side are in Q15, not Q14 like X and Y */
   El = SHR32(MULT32_32_Q31(mid, mid),3) + side - 2*xp;
   Er = SHR32(MULT32_32_Q31(mid, mid),3) + side + 2*xp;
   if (Er < QCONST32(6e-4f, 28) || El < QCONST32(6e-4f, 28))
   {
      OPUS_COPY(Y, X, N);
      return;
   }

#ifdef FIXED_POINT
   kl = celt_ilog2(El)>>1;
   kr = celt_ilog2(Er)>>1;
#endif
   t = VSHR32(El, (kl<<1)-29);
   lgain = celt_rsqrt_norm32(t);
   t = VSHR32(Er, (kr<<1)-29);
   rgain = celt_rsqrt_norm32(t);

#ifdef FIXED_POINT
   if (kl < 7)
      kl = 7;
   if (kr < 7)
      kr = 7;
#endif

   for (j=0;j<N;j++)
   {
      celt_norm r, l;
      /* Apply mid scaling (side is already scaled) */
      l = MULT32_32_Q31(mid, X[j]);
      r = Y[j];
      X[j] = VSHR32(MULT32_32_Q31(lgain, SUB32(l,r)), kl-15);
      Y[j] = VSHR32(MULT32_32_Q31(rgain, ADD32(l,r)), kr-15);
   }
}

/* Decide whether we should spread the pulses in the current frame */
int spreading_decision(const CELTMode *m, const celt_norm *X, int *average,
      int last_decision, int *hf_average, int *tapset_decision, int update_hf,
      int end, int C, int M, const int *spread_weight)
{
   int i, c, N0;
   int sum = 0, nbBands=0;
   const opus_int16 * OPUS_RESTRICT eBands = m->eBands;
   int decision;
   int hf_sum=0;

   celt_assert(end>0);

   N0 = M*m->shortMdctSize;

   if (M*(eBands[end]-eBands[end-1]) <= 8)
      return SPREAD_NONE;
   c=0; do {
      for (i=0;i<end;i++)
      {
         int j, N, tmp=0;
         int tcount[3] = {0,0,0};
         const celt_norm * OPUS_RESTRICT x = X+M*eBands[i]+c*N0;
         N = M*(eBands[i+1]-eBands[i]);
         if (N<=8)
            continue;
         /* Compute rough CDF of |x[j]| */
         for (j=0;j<N;j++)
         {
            opus_val32 x2N; /* Q13 */

            x2N = MULT16_16(MULT16_16_Q15(SHR32(x[j], NORM_SHIFT-14), SHR32(x[j], NORM_SHIFT-14)), N);
            if (x2N < QCONST16(0.25f,13))
               tcount[0]++;
            if (x2N < QCONST16(0.0625f,13))
               tcount[1]++;
            if (x2N < QCONST16(0.015625f,13))
               tcount[2]++;
         }

         /* Only include four last bands (8 kHz and up) */
         if (i>m->nbEBands-4)
            hf_sum += celt_udiv(32*(tcount[1]+tcount[0]), N);
         tmp = (2*tcount[2] >= N) + (2*tcount[1] >= N) + (2*tcount[0] >= N);
         sum += tmp*spread_weight[i];
         nbBands+=spread_weight[i];
      }
   } while (++c<C);

   if (update_hf)
   {
      if (hf_sum)
         hf_sum = celt_udiv(hf_sum, C*(4-m->nbEBands+end));
      *hf_average = (*hf_average+hf_sum)>>1;
      hf_sum = *hf_average;
      if (*tapset_decision==2)
         hf_sum += 4;
      else if (*tapset_decision==0)
         hf_sum -= 4;
      if (hf_sum > 22)
         *tapset_decision=2;
      else if (hf_sum > 18)
         *tapset_decision=1;
      else
         *tapset_decision=0;
   }
   /*printf("%d %d %d\n", hf_sum, *hf_average, *tapset_decision);*/
   celt_assert(nbBands>0); /* end has to be non-zero */
   celt_assert(sum>=0);
   sum = celt_udiv((opus_int32)sum<<8, nbBands);
   /* Recursive averaging */
   sum = (sum+*average)>>1;
   *average = sum;
   /* Hysteresis */
   sum = (3*sum + (((3-last_decision)<<7) + 64) + 2)>>2;
   if (sum < 80)
   {
      decision = SPREAD_AGGRESSIVE;
   } else if (sum < 256)
   {
      decision = SPREAD_NORMAL;
   } else if (sum < 384)
   {
      decision = SPREAD_LIGHT;
   } else {
      decision = SPREAD_NONE;
   }
#ifdef FUZZING
   decision = rand()&0x3;
   *tapset_decision=rand()%3;
#endif
   return decision;
}

/* Indexing table for converting from natural Hadamard to ordery Hadamard
   This is essentially a bit-reversed Gray, on top of which we've added
   an inversion of the order because we want the DC at the end rather than
   the beginning. The lines are for N=2, 4, 8, 16 */
static const int ordery_table[] = {
       1,  0,
       3,  0,  2,  1,
       7,  0,  4,  3,  6,  1,  5,  2,
      15,  0,  8,  7, 12,  3, 11,  4, 14,  1,  9,  6, 13,  2, 10,  5,
};

static void deinterleave_hadamard(celt_norm *X, int N0, int stride, int hadamard)
{
   int i,j;
   VARDECL(celt_norm, tmp);
   int N;
   SAVE_STACK;
   N = N0*stride;
   ALLOC(tmp, N, celt_norm);
   celt_assert(stride>0);
   if (hadamard)
   {
      const int *ordery = ordery_table+stride-2;
      for (i=0;i<stride;i++)
      {
         for (j=0;j<N0;j++)
            tmp[ordery[i]*N0+j] = X[j*stride+i];
      }
   } else {
      for (i=0;i<stride;i++)
         for (j=0;j<N0;j++)
            tmp[i*N0+j] = X[j*stride+i];
   }
   OPUS_COPY(X, tmp, N);
   RESTORE_STACK;
}

static void interleave_hadamard(celt_norm *X, int N0, int stride, int hadamard)
{
   int i,j;
   VARDECL(celt_norm, tmp);
   int N;
   SAVE_STACK;
   N = N0*stride;
   ALLOC(tmp, N, celt_norm);
   if (hadamard)
   {
      const int *ordery = ordery_table+stride-2;
      for (i=0;i<stride;i++)
         for (j=0;j<N0;j++)
            tmp[j*stride+i] = X[ordery[i]*N0+j];
   } else {
      for (i=0;i<stride;i++)
         for (j=0;j<N0;j++)
            tmp[j*stride+i] = X[i*N0+j];
   }
   OPUS_COPY(X, tmp, N);
   RESTORE_STACK;
}

void haar1(celt_norm *X, int N0, int stride)
{
   int i, j;
   N0 >>= 1;
   for (i=0;i<stride;i++)
      for (j=0;j<N0;j++)
      {
         opus_val32 tmp1, tmp2;
         tmp1 = MULT32_32_Q31(QCONST32(.70710678f,31), X[stride*2*j+i]);
         tmp2 = MULT32_32_Q31(QCONST32(.70710678f,31), X[stride*(2*j+1)+i]);
         X[stride*2*j+i] = ADD32(tmp1, tmp2);
         X[stride*(2*j+1)+i] = SUB32(tmp1, tmp2);
      }
}

static int compute_qn(int N, int b, int offset, int pulse_cap, int stereo)
{
   static const opus_int16 exp2_table8[8] =
      {16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048};
   int qn, qb;
   int N2 = 2*N-1;
   if (stereo && N==2)
      N2--;
   /* The upper limit ensures that in a stereo split with itheta==16384, we'll
       always have enough bits left over to code at least one pulse in the
       side; otherwise it would collapse, since it doesn't get folded. */
   qb = celt_sudiv(b+N2*offset, N2);
   qb = IMIN(b-pulse_cap-(4<<BITRES), qb);

   qb = IMIN(8<<BITRES, qb);

   if (qb<(1<<BITRES>>1)) {
      qn = 1;
   } else {
      qn = exp2_table8[qb&0x7]>>(14-(qb>>BITRES));
      qn = (qn+1)>>1<<1;
   }
   celt_assert(qn <= 256);
   return qn;
}

struct band_ctx {
   int encode;
   int resynth;
   const CELTMode *m;
   int i;
   int intensity;
   int spread;
   int tf_change;
   ec_ctx *ec;
   opus_int32 remaining_bits;
   const celt_ener *bandE;
   opus_uint32 seed;
   int arch;
   int theta_round;
   int disable_inv;
   int avoid_split_noise;
#ifdef ENABLE_QEXT
   ec_ctx *ext_ec;
   int extra_bits;
   opus_int32 ext_total_bits;
   int extra_bands;
#endif
};

struct split_ctx {
   int inv;
   int imid;
   int iside;
   int delta;
   int itheta;
#ifdef ENABLE_QEXT
   int itheta_q30;
#endif
   int qalloc;
};

static void compute_theta(struct band_ctx *ctx, struct split_ctx *sctx,
      celt_norm *X, celt_norm *Y, int N, int *b, int B, int B0,
      int LM,
      int stereo, int *fill ARG_QEXT(int *ext_b))
{
   int qn;
   int itheta=0;
   int itheta_q30=0;
   int delta;
   int imid, iside;
   int qalloc;
   int pulse_cap;
   int offset;
   opus_int32 tell;
   int inv=0;
   int encode;
   const CELTMode *m;
   int i;
   int intensity;
   ec_ctx *ec;
   const celt_ener *bandE;

   encode = ctx->encode;
   m = ctx->m;
   i = ctx->i;
   intensity = ctx->intensity;
   ec = ctx->ec;
   bandE = ctx->bandE;

   /* Decide on the resolution to give to the split parameter theta */
   pulse_cap = m->logN[i]+LM*(1<<BITRES);
   offset = (pulse_cap>>1) - (stereo&&N==2 ? QTHETA_OFFSET_TWOPHASE : QTHETA_OFFSET);
   qn = compute_qn(N, *b, offset, pulse_cap, stereo);
   if (stereo && i>=intensity)
      qn = 1;
   if (encode)
   {
      /* theta is the atan() of the ratio between the (normalized)
         side and mid. With just that parameter, we can re-scale both
         mid and side because we know that 1) they have unit norm and
         2) they are orthogonal. */
      itheta_q30 = stereo_itheta(X, Y, stereo, N, ctx->arch);
      itheta = itheta_q30>>16;
   }
   tell = ec_tell_frac(ec);
   if (qn!=1)
   {
      if (encode)
      {
         if (!stereo || ctx->theta_round == 0)
         {
            itheta = (itheta*(opus_int32)qn+8192)>>14;
            if (!stereo && ctx->avoid_split_noise && itheta > 0 && itheta < qn)
            {
               /* Check if the selected value of theta will cause the bit allocation
                  to inject noise on one side. If so, make sure the energy of that side
                  is zero. */
               int unquantized = celt_udiv((opus_int32)itheta*16384, qn);
               imid = bitexact_cos((opus_int16)unquantized);
               iside = bitexact_cos((opus_int16)(16384-unquantized));
               delta = FRAC_MUL16((N-1)<<7,bitexact_log2tan(iside,imid));
               if (delta > *b)
                  itheta = qn;
               else if (delta < -*b)
                  itheta = 0;
            }
         } else {
            int down;
            /* Bias quantization towards itheta=0 and itheta=16384. */
            int bias = itheta > 8192 ? 32767/qn : -32767/qn;
            down = IMIN(qn-1, IMAX(0, (itheta*(opus_int32)qn + bias)>>14));
            if (ctx->theta_round < 0)
               itheta = down;
            else
               itheta = down+1;
         }
      }
      /* Entropy coding of the angle. We use a uniform pdf for the
         time split, a step for stereo, and a triangular one for the rest. */
      if (stereo && N>2)
      {
         int p0 = 3;
         int x = itheta;
         int x0 = qn/2;
         int ft = p0*(x0+1) + x0;
         /* Use a probability of p0 up to itheta=8192 and then use 1 after */
         if (encode)
         {
            ec_encode(ec,x<=x0?p0*x:(x-1-x0)+(x0+1)*p0,x<=x0?p0*(x+1):(x-x0)+(x0+1)*p0,ft);
         } else {
            int fs;
            fs=ec_decode(ec,ft);
            if (fs<(x0+1)*p0)
               x=fs/p0;
            else
               x=x0+1+(fs-(x0+1)*p0);
            ec_dec_update(ec,x<=x0?p0*x:(x-1-x0)+(x0+1)*p0,x<=x0?p0*(x+1):(x-x0)+(x0+1)*p0,ft);
            itheta = x;
         }
      } else if (B0>1 || stereo) {
         /* Uniform pdf */
         if (encode)
            ec_enc_uint(ec, itheta, qn+1);
         else
            itheta = ec_dec_uint(ec, qn+1);
      } else {
         int fs=1, ft;
         ft = ((qn>>1)+1)*((qn>>1)+1);
         if (encode)
         {
            int fl;

            fs = itheta <= (qn>>1) ? itheta + 1 : qn + 1 - itheta;
            fl = itheta <= (qn>>1) ? itheta*(itheta + 1)>>1 :
             ft - ((qn + 1 - itheta)*(qn + 2 - itheta)>>1);

            ec_encode(ec, fl, fl+fs, ft);
         } else {
            /* Triangular pdf */
            int fl=0;
            int fm;
            fm = ec_decode(ec, ft);

            if (fm < ((qn>>1)*((qn>>1) + 1)>>1))
            {
               itheta = (isqrt32(8*(opus_uint32)fm + 1) - 1)>>1;
               fs = itheta + 1;
               fl = itheta*(itheta + 1)>>1;
            }
            else
            {
               itheta = (2*(qn + 1)
                - isqrt32(8*(opus_uint32)(ft - fm - 1) + 1))>>1;
               fs = qn + 1 - itheta;
               fl = ft - ((qn + 1 - itheta)*(qn + 2 - itheta)>>1);
            }

            ec_dec_update(ec, fl, fl+fs, ft);
         }
      }
      celt_assert(itheta>=0);
      itheta = celt_udiv((opus_int32)itheta*16384, qn);
#ifdef ENABLE_QEXT
      *ext_b = IMIN(*ext_b, ctx->ext_total_bits - (opus_int32)ec_tell_frac(ctx->ext_ec));
      if (*ext_b >= 2*N<<BITRES && ctx->ext_total_bits-ec_tell_frac(ctx->ext_ec)-1 > 2<<BITRES) {
         int extra_bits;
         int ext_tell = ec_tell_frac(ctx->ext_ec);
         extra_bits = IMIN(12, IMAX(2, celt_sudiv(*ext_b, (2*N-1)<<BITRES)));
         if (encode) {
            itheta_q30 = itheta_q30 - (itheta<<16);
            itheta_q30 = (itheta_q30*(opus_int64)qn*((1<<extra_bits)-1)+(1<<29))>>30;
            itheta_q30 += (1<<(extra_bits-1))-1;
            itheta_q30 = IMAX(0, IMIN((1<<extra_bits)-2, itheta_q30));
            ec_enc_uint(ctx->ext_ec, itheta_q30, (1<<extra_bits)-1);
         } else {
            itheta_q30 = ec_dec_uint(ctx->ext_ec, (1<<extra_bits)-1);
         }
         itheta_q30 -= (1<<(extra_bits-1))-1;
         itheta_q30 = (itheta<<16) + itheta_q30*(opus_int64)(1<<30)/(qn*((1<<extra_bits)-1));
         /* Hard bounds on itheta (can only trigger on corrupted bitstreams). */
         itheta_q30 = IMAX(0, IMIN(itheta_q30, 1073741824));
         *ext_b -= ec_tell_frac(ctx->ext_ec) - ext_tell;
      } else {
         itheta_q30 = (opus_int32)itheta<<16;
      }
#endif
      if (encode && stereo)
      {
         if (itheta==0)
            intensity_stereo(m, X, Y, bandE, i, N);
         else
            stereo_split(X, Y, N);
      }
      /* NOTE: Renormalising X and Y *may* help fixed-point a bit at very high rate.
               Let's do that at higher complexity */
   } else if (stereo) {
      if (encode)
      {
         inv = itheta > 8192 && !ctx->disable_inv;
         if (inv)
         {
            int j;
            for (j=0;j<N;j++)
               Y[j] = -Y[j];
         }
         intensity_stereo(m, X, Y, bandE, i, N);
      }
      if (*b>2<<BITRES && ctx->remaining_bits > 2<<BITRES)
      {
         if (encode)
            ec_enc_bit_logp(ec, inv, 2);
         else
            inv = ec_dec_bit_logp(ec, 2);
      } else
         inv = 0;
      /* inv flag override to avoid problems with downmixing. */
      if (ctx->disable_inv)
         inv = 0;
      itheta = 0;
      itheta_q30 = 0;
   }
   qalloc = ec_tell_frac(ec) - tell;
   *b -= qalloc;

   if (itheta == 0)
   {
      imid = 32767;
      iside = 0;
      *fill &= (1<<B)-1;
      delta = -16384;
   } else if (itheta == 16384)
   {
      imid = 0;
      iside = 32767;
      *fill &= ((1<<B)-1)<<B;
      delta = 16384;
   } else {
      imid = bitexact_cos((opus_int16)itheta);
      iside = bitexact_cos((opus_int16)(16384-itheta));
      /* This is the mid vs side allocation that minimizes squared error
         in that band. */
      delta = FRAC_MUL16((N-1)<<7,bitexact_log2tan(iside,imid));
   }

   sctx->inv = inv;
   sctx->imid = imid;
   sctx->iside = iside;
   sctx->delta = delta;
   sctx->itheta = itheta;
#ifdef ENABLE_QEXT
   sctx->itheta_q30 = itheta_q30;
#endif
   sctx->qalloc = qalloc;
}
static unsigned quant_band_n1(struct band_ctx *ctx, celt_norm *X, celt_norm *Y,
      celt_norm *lowband_out)
{
   int c;
   int stereo;
   celt_norm *x = X;
   int encode;
   ec_ctx *ec;

   encode = ctx->encode;
   ec = ctx->ec;

   stereo = Y != NULL;
   c=0; do {
      int sign=0;
      if (ctx->remaining_bits>=1<<BITRES)
      {
         if (encode)
         {
            sign = x[0]<0;
            ec_enc_bits(ec, sign, 1);
         } else {
            sign = ec_dec_bits(ec, 1);
         }
         ctx->remaining_bits -= 1<<BITRES;
      }
      if (ctx->resynth)
         x[0] = sign ? -NORM_SCALING : NORM_SCALING;
      x = Y;
   } while (++c<1+stereo);
   if (lowband_out)
      lowband_out[0] = SHR32(X[0],4);
   return 1;
}

/* This function is responsible for encoding and decoding a mono partition.
   It can split the band in two and transmit the energy difference with
   the two half-bands. It can be called recursively so bands can end up being
   split in 8 parts. */
static unsigned quant_partition(struct band_ctx *ctx, celt_norm *X,
      int N, int b, int B, celt_norm *lowband,
      int LM,
      opus_val32 gain, int fill
      ARG_QEXT(int ext_b))
{
   const unsigned char *cache;
   int q;
   int curr_bits;
   int imid=0, iside=0;
   int B0=B;
   opus_val32 mid=0, side=0;
   unsigned cm=0;
   celt_norm *Y=NULL;
   int encode;
   const CELTMode *m;
   int i;
   int spread;
   ec_ctx *ec;

   encode = ctx->encode;
   m = ctx->m;
   i = ctx->i;
   spread = ctx->spread;
   ec = ctx->ec;

   /* If we need 1.5 more bit than we can produce, split the band in two. */
   cache = m->cache.bits + m->cache.index[(LM+1)*m->nbEBands+i];
   if (LM != -1 && b > cache[cache[0]]+12 && N>2)
   {
      int mbits, sbits, delta;
      int itheta;
      int qalloc;
      struct split_ctx sctx;
      celt_norm *next_lowband2=NULL;
      opus_int32 rebalance;

      N >>= 1;
      Y = X+N;
      LM -= 1;
      if (B==1)
         fill = (fill&1)|(fill<<1);
      B = (B+1)>>1;

      compute_theta(ctx, &sctx, X, Y, N, &b, B, B0, LM, 0, &fill ARG_QEXT(&ext_b));
      imid = sctx.imid;
      iside = sctx.iside;
      delta = sctx.delta;
      itheta = sctx.itheta;
      qalloc = sctx.qalloc;
#ifdef FIXED_POINT
# ifdef ENABLE_QEXT
      (void)imid;
      (void)iside;
      mid = celt_cos_norm32(sctx.itheta_q30);
      side = celt_cos_norm32((1<<30)-sctx.itheta_q30);
# else
      mid = SHL32(EXTEND32(imid), 16);
      side = SHL32(EXTEND32(iside), 16);
# endif
#else
# ifdef ENABLE_QEXT
      (void)imid;
      (void)iside;
      mid = celt_cos_norm2(sctx.itheta_q30*(1.f/(1<<30)));
      side = celt_cos_norm2(1.f-sctx.itheta_q30*(1.f/(1<<30)));
# else
      mid = (1.f/32768)*imid;
      side = (1.f/32768)*iside;
# endif
#endif

      /* Give more bits to low-energy MDCTs than they would otherwise deserve */
      if (B0>1 && (itheta&0x3fff))
      {
         if (itheta > 8192)
            /* Rough approximation for pre-echo masking */
            delta -= delta>>(4-LM);
         else
            /* Corresponds to a forward-masking slope of 1.5 dB per 10 ms */
            delta = IMIN(0, delta + (N<<BITRES>>(5-LM)));
      }
      mbits = IMAX(0, IMIN(b, (b-delta)/2));
      sbits = b-mbits;
      ctx->remaining_bits -= qalloc;

      if (lowband)
         next_lowband2 = lowband+N; /* >32-bit split case */

      rebalance = ctx->remaining_bits;
      if (mbits >= sbits)
      {
         cm = quant_partition(ctx, X, N, mbits, B, lowband, LM,
               MULT32_32_Q31(gain,mid), fill ARG_QEXT(ext_b/2));
         rebalance = mbits - (rebalance-ctx->remaining_bits);
         if (rebalance > 3<<BITRES && itheta!=0)
            sbits += rebalance - (3<<BITRES);
         cm |= quant_partition(ctx, Y, N, sbits, B, next_lowband2, LM,
               MULT32_32_Q31(gain,side), fill>>B ARG_QEXT(ext_b/2))<<(B0>>1);
      } else {
         cm = quant_partition(ctx, Y, N, sbits, B, next_lowband2, LM,
               MULT32_32_Q31(gain,side), fill>>B ARG_QEXT(ext_b/2))<<(B0>>1);
         rebalance = sbits - (rebalance-ctx->remaining_bits);
         if (rebalance > 3<<BITRES && itheta!=16384)
            mbits += rebalance - (3<<BITRES);
         cm |= quant_partition(ctx, X, N, mbits, B, lowband, LM,
               MULT32_32_Q31(gain,mid), fill ARG_QEXT(ext_b/2));
      }
   } else {
#ifdef ENABLE_QEXT
      int extra_bits;
      int ext_remaining_bits;
      extra_bits = ext_b/(N-1)>>BITRES;
      ext_remaining_bits = ctx->ext_total_bits-(opus_int32)ec_tell_frac(ctx->ext_ec);
      if (ext_remaining_bits < ((extra_bits+1)*(N-1)+N)<<BITRES) {
         extra_bits = (ext_remaining_bits-(N<<BITRES))/(N-1)>>BITRES;
         extra_bits = IMAX(extra_bits-1, 0);
      }
      extra_bits = IMIN(12, extra_bits);
#endif
      /* This is the basic no-split case */
      q = bits2pulses(m, i, LM, b);
      curr_bits = pulses2bits(m, i, LM, q);
      ctx->remaining_bits -= curr_bits;

      /* Ensures we can never bust the budget */
      while (ctx->remaining_bits < 0 && q > 0)
      {
         ctx->remaining_bits += curr_bits;
         q--;
         curr_bits = pulses2bits(m, i, LM, q);
         ctx->remaining_bits -= curr_bits;
      }

      if (q!=0)
      {
         int K = get_pulses(q);

         /* Finally do the actual quantization */
         if (encode)
         {
            cm = alg_quant(X, N, K, spread, B, ec, gain, ctx->resynth
                           ARG_QEXT(ctx->ext_ec) ARG_QEXT(extra_bits),
                           ctx->arch);
         } else {
            cm = alg_unquant(X, N, K, spread, B, ec, gain
                             ARG_QEXT(ctx->ext_ec) ARG_QEXT(extra_bits));
         }
#ifdef ENABLE_QEXT
      } else if (ext_b > 2*N<<BITRES)
      {
         extra_bits = ext_b/(N-1)>>BITRES;
         ext_remaining_bits = ctx->ext_total_bits-ec_tell_frac(ctx->ext_ec);
         if (ext_remaining_bits < ((extra_bits+1)*(N-1)+N)<<BITRES) {
            extra_bits = (ext_remaining_bits-(N<<BITRES))/(N-1)>>BITRES;
            extra_bits = IMAX(extra_bits-1, 0);
         }
         extra_bits = IMIN(14, extra_bits);
         if (encode) cm = cubic_quant(X, N, extra_bits, B, ctx->ext_ec, gain, ctx->resynth);
         else cm = cubic_unquant(X, N, extra_bits, B, ctx->ext_ec, gain);
#endif
      } else {
         /* If there's no pulse, fill the band anyway */
         int j;
         if (ctx->resynth)
         {
            unsigned cm_mask;
            /* B can be as large as 16, so this shift might overflow an int on a
               16-bit platform; use a long to get defined behavior.*/
            cm_mask = (unsigned)(1UL<<B)-1;
            fill &= cm_mask;
            if (!fill)
            {
               OPUS_CLEAR(X, N);
            } else {
               if (lowband == NULL)
               {
                  /* Noise */
                  for (j=0;j<N;j++)
                  {
                     ctx->seed = celt_lcg_rand(ctx->seed);
                     X[j] = SHL32((celt_norm)((opus_int32)ctx->seed>>20), NORM_SHIFT-14);
                  }
                  cm = cm_mask;
               } else {
                  /* Folded spectrum */
                  for (j=0;j<N;j++)
                  {
                     opus_val16 tmp;
                     ctx->seed = celt_lcg_rand(ctx->seed);
                     /* About 48 dB below the "normal" folding level */
                     tmp = QCONST16(1.0f/256, NORM_SHIFT-4);
                     tmp = (ctx->seed)&0x8000 ? tmp : -tmp;
                     X[j] = lowband[j]+tmp;
                  }
                  cm = fill;
               }
               renormalise_vector(X, N, gain, ctx->arch);
            }
         }
      }
   }

   return cm;
}

#ifdef ENABLE_QEXT
static unsigned cubic_quant_partition(struct band_ctx *ctx, celt_norm *X, int N, int b, int B, ec_ctx *ec, int LM, opus_val32 gain, int resynth, int encode)
{
   celt_assert(LM>=0);
   ctx->remaining_bits = ctx->ec->storage*8*8 - ec_tell_frac(ctx->ec);
   b = IMIN(b, ctx->remaining_bits);
   /* As long as we have at least two bits of depth, split all the way to LM=0 (not -1 like PVQ). */
   if (LM==0 || b<=2*N<<BITRES) {
      int res, ret;
      b = IMIN(b + ((N-1)<<BITRES)/2, ctx->remaining_bits);
      /* Resolution left after taking into account coding the cube face. */
      res = (b-(1<<BITRES)-ctx->m->logN[ctx->i]-(LM<<BITRES)-1)/(N-1)>>BITRES;
      res = IMIN(14, IMAX(0, res));
      if (encode) ret = cubic_quant(X, N, res, B, ec, gain, resynth);
      else ret = cubic_unquant(X, N, res, B, ec, gain);
      ctx->remaining_bits = ctx->ec->storage*8*8 - ec_tell_frac(ctx->ec);
      return ret;
   } else {
      celt_norm *Y;
      opus_int32 itheta_q30;
      opus_val32 g1, g2;
      opus_int32 theta_res;
      opus_int32 qtheta;
      int delta;
      int b1, b2;
      int cm;
      int N0;
      N0 = N;
      N >>= 1;
      Y = X+N;
      LM -= 1;
      B = (B+1)>>1;
      theta_res = IMIN(16, (b>>BITRES)/(N0-1) + 1);
      if (encode) {
         itheta_q30 = stereo_itheta(X, Y, 0, N, ctx->arch);
         qtheta = (itheta_q30+(1<<(29-theta_res)))>>(30-theta_res);
         ec_enc_uint(ec, qtheta, (1<<theta_res)+1);
      } else {
         qtheta = ec_dec_uint(ec, (1<<theta_res)+1);
      }
      itheta_q30 = qtheta<<(30-theta_res);
      b -= theta_res<<BITRES;
      delta = (N0-1) * 23 * ((itheta_q30>>16)-8192) >> (17-BITRES);

#ifdef FIXED_POINT
      g1 = celt_cos_norm32(itheta_q30);
      g2 = celt_cos_norm32((1<<30)-itheta_q30);
#else
      g1 = celt_cos_norm2(itheta_q30*(1.f/(1<<30)));
      g2 = celt_cos_norm2(1.f-itheta_q30*(1.f/(1<<30)));
#endif
      if (itheta_q30 == 0) {
         b1=b;
         b2=0;
      } else if (itheta_q30==1073741824) {
         b1=0;
         b2=b;
      } else {
         b1 = IMIN(b, IMAX(0, (b-delta)/2));
         b2 = b-b1;
      }
      cm  = cubic_quant_partition(ctx, X, N, b1, B, ec, LM, MULT32_32_Q31(gain, g1), resynth, encode);
      cm |= cubic_quant_partition(ctx, Y, N, b2, B, ec, LM, MULT32_32_Q31(gain, g2), resynth, encode);
      return cm;
   }
}
#endif

/* This function is responsible for encoding and decoding a band for the mono case. */
static unsigned quant_band(struct band_ctx *ctx, celt_norm *X,
      int N, int b, int B, celt_norm *lowband,
      int LM, celt_norm *lowband_out,
      opus_val32 gain, celt_norm *lowband_scratch, int fill
      ARG_QEXT(int ext_b))
{
   int N0=N;
   int N_B=N;
   int N_B0;
   int B0=B;
   int time_divide=0;
   int recombine=0;
   int longBlocks;
   unsigned cm=0;
   int k;
   int encode;
   int tf_change;

   encode = ctx->encode;
   tf_change = ctx->tf_change;

   longBlocks = B0==1;

   N_B = celt_udiv(N_B, B);

   /* Special case for one sample */
   if (N==1)
   {
      return quant_band_n1(ctx, X, NULL, lowband_out);
   }

   if (tf_change>0)
      recombine = tf_change;
   /* Band recombining to increase frequency resolution */

   if (lowband_scratch && lowband && (recombine || ((N_B&1) == 0 && tf_change<0) || B0>1))
   {
      OPUS_COPY(lowband_scratch, lowband, N);
      lowband = lowband_scratch;
   }

   for (k=0;k<recombine;k++)
   {
      static const unsigned char bit_interleave_table[16]={
            0,1,1,1,2,3,3,3,2,3,3,3,2,3,3,3
      };
      if (encode)
         haar1(X, N>>k, 1<<k);
      if (lowband)
         haar1(lowband, N>>k, 1<<k);
      fill = bit_interleave_table[fill&0xF]|bit_interleave_table[fill>>4]<<2;
   }
   B>>=recombine;
   N_B<<=recombine;

   /* Increasing the time resolution */
   while ((N_B&1) == 0 && tf_change<0)
   {
      if (encode)
         haar1(X, N_B, B);
      if (lowband)
         haar1(lowband, N_B, B);
      fill |= fill<<B;
      B <<= 1;
      N_B >>= 1;
      time_divide++;
      tf_change++;
   }
   B0=B;
   N_B0 = N_B;

   /* Reorganize the samples in time order instead of frequency order */
   if (B0>1)
   {
      if (encode)
         deinterleave_hadamard(X, N_B>>recombine, B0<<recombine, longBlocks);
      if (lowband)
         deinterleave_hadamard(lowband, N_B>>recombine, B0<<recombine, longBlocks);
   }

#ifdef ENABLE_QEXT
   if (ctx->extra_bands && b > (3*N<<BITRES)+(ctx->m->logN[ctx->i]+8+8*LM)) {
      cm = cubic_quant_partition(ctx, X, N, b, B, ctx->ec, LM, gain, ctx->resynth, encode);
   } else
#endif
   {
      cm = quant_partition(ctx, X, N, b, B, lowband, LM, gain, fill ARG_QEXT(ext_b));
   }

   /* This code is used by the decoder and by the resynthesis-enabled encoder */
   if (ctx->resynth)
   {
      /* Undo the sample reorganization going from time order to frequency order */
      if (B0>1)
         interleave_hadamard(X, N_B>>recombine, B0<<recombine, longBlocks);

      /* Undo time-freq changes that we did earlier */
      N_B = N_B0;
      B = B0;
      for (k=0;k<time_divide;k++)
      {
         B >>= 1;
         N_B <<= 1;
         cm |= cm>>B;
         haar1(X, N_B, B);
      }

      for (k=0;k<recombine;k++)
      {
         static const unsigned char bit_deinterleave_table[16]={
               0x00,0x03,0x0C,0x0F,0x30,0x33,0x3C,0x3F,
               0xC0,0xC3,0xCC,0xCF,0xF0,0xF3,0xFC,0xFF
         };
         cm = bit_deinterleave_table[cm];
         haar1(X, N0>>k, 1<<k);
      }
      B<<=recombine;

      /* Scale output for later folding */
      if (lowband_out)
      {
         int j;
         opus_val16 n;
         n = celt_sqrt(SHL32(EXTEND32(N0),22));
         for (j=0;j<N0;j++)
            lowband_out[j] = MULT16_32_Q15(n,X[j]);
      }
      cm &= (1<<B)-1;
   }
   return cm;
}

#ifdef FIXED_POINT
#define MIN_STEREO_ENERGY 2
#else
#define MIN_STEREO_ENERGY 1e-10f
#endif

/* This function is responsible for encoding and decoding a band for the stereo case. */
static unsigned quant_band_stereo(struct band_ctx *ctx, celt_norm *X, celt_norm *Y,
      int N, int b, int B, celt_norm *lowband,
      int LM, celt_norm *lowband_out,
      celt_norm *lowband_scratch, int fill
      ARG_QEXT(int ext_b) ARG_QEXT(const int *cap))
{
   int imid=0, iside=0;
   int inv = 0;
   opus_val32 mid=0, side=0;
   unsigned cm=0;
   int mbits, sbits, delta;
   int itheta;
   int qalloc;
   struct split_ctx sctx;
   int orig_fill;
   int encode;
   ec_ctx *ec;

   encode = ctx->encode;
   ec = ctx->ec;

   /* Special case for one sample */
   if (N==1)
   {
      return quant_band_n1(ctx, X, Y, lowband_out);
   }

   orig_fill = fill;

   if (encode) {
      if (ctx->bandE[ctx->i] < MIN_STEREO_ENERGY || ctx->bandE[ctx->m->nbEBands+ctx->i] < MIN_STEREO_ENERGY) {
         if (ctx->bandE[ctx->i] > ctx->bandE[ctx->m->nbEBands+ctx->i]) OPUS_COPY(Y, X, N);
         else OPUS_COPY(X, Y, N);
      }
   }
   compute_theta(ctx, &sctx, X, Y, N, &b, B, B, LM, 1, &fill ARG_QEXT(&ext_b));
   inv = sctx.inv;
   imid = sctx.imid;
   iside = sctx.iside;
   delta = sctx.delta;
   itheta = sctx.itheta;
   qalloc = sctx.qalloc;
#ifdef FIXED_POINT
# ifdef ENABLE_QEXT
   (void)imid;
   (void)iside;
   mid = celt_cos_norm32(sctx.itheta_q30);
   side = celt_cos_norm32((1<<30)-sctx.itheta_q30);
# else
   mid = SHL32(EXTEND32(imid), 16);
   side = SHL32(EXTEND32(iside), 16);
# endif
#else
# ifdef ENABLE_QEXT
   (void)imid;
   (void)iside;
   mid = celt_cos_norm2(sctx.itheta_q30*(1.f/(1<<30)));
   side = celt_cos_norm2(1.f-sctx.itheta_q30*(1.f/(1<<30)));
# else
   mid = (1.f/32768)*imid;
   side = (1.f/32768)*iside;
# endif
#endif

   /* This is a special case for N=2 that only works for stereo and takes
      advantage of the fact that mid and side are orthogonal to encode
      the side with just one bit. */
   if (N==2)
   {
      int c;
      int sign=0;
      celt_norm *x2, *y2;
      mbits = b;
      sbits = 0;
      /* Only need one bit for the side. */
      if (itheta != 0 && itheta != 16384)
         sbits = 1<<BITRES;
      mbits -= sbits;
      c = itheta > 8192;
      ctx->remaining_bits -= qalloc+sbits;

      x2 = c ? Y : X;
      y2 = c ? X : Y;
      if (sbits)
      {
         if (encode)
         {
            /* Here we only need to encode a sign for the side. */
            /* FIXME: Need to increase fixed-point precision? */
            sign = MULT32_32_Q31(x2[0],y2[1]) - MULT32_32_Q31(x2[1],y2[0]) < 0;
            ec_enc_bits(ec, sign, 1);
         } else {
            sign = ec_dec_bits(ec, 1);
         }
      }
      sign = 1-2*sign;
      /* We use orig_fill here because we want to fold the side, but if
         itheta==16384, we'll have cleared the low bits of fill. */
      cm = quant_band(ctx, x2, N, mbits, B, lowband, LM, lowband_out, Q31ONE,
            lowband_scratch, orig_fill ARG_QEXT(ext_b));
      /* We don't split N=2 bands, so cm is either 1 or 0 (for a fold-collapse),
         and there's no need to worry about mixing with the other channel. */
      y2[0] = -sign*x2[1];
      y2[1] = sign*x2[0];
      if (ctx->resynth)
      {
         celt_norm tmp;
         X[0] = MULT32_32_Q31(mid, X[0]);
         X[1] = MULT32_32_Q31(mid, X[1]);
         Y[0] = MULT32_32_Q31(side, Y[0]);
         Y[1] = MULT32_32_Q31(side, Y[1]);
         tmp = X[0];
         X[0] = SUB32(tmp,Y[0]);
         Y[0] = ADD32(tmp,Y[0]);
         tmp = X[1];
         X[1] = SUB32(tmp,Y[1]);
         Y[1] = ADD32(tmp,Y[1]);
      }
   } else {
      /* "Normal" split code */
      opus_int32 rebalance;

      mbits = IMAX(0, IMIN(b, (b-delta)/2));
      sbits = b-mbits;
      ctx->remaining_bits -= qalloc;

      rebalance = ctx->remaining_bits;
      if (mbits >= sbits)
      {
#ifdef ENABLE_QEXT
         int qext_extra = 0;
         /* Reallocate any mid bits that cannot be used to extra mid bits. */
         if (cap != NULL && ext_b != 0) qext_extra = IMAX(0, IMIN(ext_b/2, mbits - cap[ctx->i]/2));
#endif
         /* In stereo mode, we do not apply a scaling to the mid because we need the normalized
            mid for folding later. */
         cm = quant_band(ctx, X, N, mbits, B, lowband, LM, lowband_out, Q31ONE,
               lowband_scratch, fill ARG_QEXT(ext_b/2+qext_extra));
         rebalance = mbits - (rebalance-ctx->remaining_bits);
         if (rebalance > 3<<BITRES && itheta!=0)
            sbits += rebalance - (3<<BITRES);
#ifdef ENABLE_QEXT
         /* Guard against overflowing the EC with the angle if the cubic quant used too many bits for the mid. */
         if (ctx->extra_bands) sbits = IMIN(sbits, ctx->remaining_bits);
#endif
         /* For a stereo split, the high bits of fill are always zero, so no
            folding will be done to the side. */
         cm |= quant_band(ctx, Y, N, sbits, B, NULL, LM, NULL, side, NULL, fill>>B ARG_QEXT(ext_b/2-qext_extra));
      } else {
#ifdef ENABLE_QEXT
         int qext_extra = 0;
         /* Reallocate any side bits that cannot be used to extra side bits. */
         if (cap != NULL && ext_b != 0) qext_extra = IMAX(0, IMIN(ext_b/2, sbits - cap[ctx->i]/2));
#endif
         /* For a stereo split, the high bits of fill are always zero, so no
            folding will be done to the side. */
         cm = quant_band(ctx, Y, N, sbits, B, NULL, LM, NULL, side, NULL, fill>>B ARG_QEXT(ext_b/2+qext_extra));
         rebalance = sbits - (rebalance-ctx->remaining_bits);
         if (rebalance > 3<<BITRES && itheta!=16384)
            mbits += rebalance - (3<<BITRES);
#ifdef ENABLE_QEXT
         /* Guard against overflowing the EC with the angle if the cubic quant used too many bits for the side. */
         if (ctx->extra_bands) mbits = IMIN(mbits, ctx->remaining_bits);
#endif
         /* In stereo mode, we do not apply a scaling to the mid because we need the normalized
            mid for folding later. */
         cm |= quant_band(ctx, X, N, mbits, B, lowband, LM, lowband_out, Q31ONE,
               lowband_scratch, fill ARG_QEXT(ext_b/2-qext_extra));
      }
   }


   /* This code is used by the decoder and by the resynthesis-enabled encoder */
   if (ctx->resynth)
   {
      if (N!=2)
         stereo_merge(X, Y, mid, N, ctx->arch);
      if (inv)
      {
         int j;
         for (j=0;j<N;j++)
            Y[j] = -Y[j];
      }
   }
   return cm;
}

#ifndef DISABLE_UPDATE_DRAFT
static void special_hybrid_folding(const CELTMode *m, celt_norm *norm, celt_norm *norm2, int start, int M, int dual_stereo)
{
   int n1, n2;
   const opus_int16 * OPUS_RESTRICT eBands = m->eBands;
   n1 = M*(eBands[start+1]-eBands[start]);
   n2 = M*(eBands[start+2]-eBands[start+1]);
   /* Duplicate enough of the first band folding data to be able to fold the second band.
      Copies no data for CELT-only mode. */
   OPUS_COPY(&norm[n1], &norm[2*n1 - n2], n2-n1);
   if (dual_stereo)
      OPUS_COPY(&norm2[n1], &norm2[2*n1 - n2], n2-n1);
}
#endif

void quant_all_bands(int encode, const CELTMode *m, int start, int end,
      celt_norm *X_, celt_norm *Y_, unsigned char *collapse_masks,
      const celt_ener *bandE, int *pulses, int shortBlocks, int spread,
      int dual_stereo, int intensity, int *tf_res, opus_int32 total_bits,
      opus_int32 balance, ec_ctx *ec, int LM, int codedBands,
      opus_uint32 *seed, int complexity, int arch, int disable_inv
      ARG_QEXT(ec_ctx *ext_ec) ARG_QEXT(int *extra_pulses)
      ARG_QEXT(opus_int32 ext_total_bits) ARG_QEXT(const int *cap))
{
   int i;
   opus_int32 remaining_bits;
   const opus_int16 * OPUS_RESTRICT eBands = m->eBands;
   celt_norm * OPUS_RESTRICT norm, * OPUS_RESTRICT norm2;
   VARDECL(celt_norm, _norm);
   VARDECL(celt_norm, _lowband_scratch);
   VARDECL(celt_norm, X_save);
   VARDECL(celt_norm, Y_save);
   VARDECL(celt_norm, X_save2);
   VARDECL(celt_norm, Y_save2);
   VARDECL(celt_norm, norm_save2);
   VARDECL(unsigned char, bytes_save);
   int resynth_alloc;
   celt_norm *lowband_scratch;
   int B;
   int M;
   int lowband_offset;
   int update_lowband = 1;
   int C = Y_ != NULL ? 2 : 1;
   int norm_offset;
   int theta_rdo = encode && Y_!=NULL && !dual_stereo && complexity>=8;
#ifdef RESYNTH
   int resynth = 1;
#else
   int resynth = !encode || theta_rdo;
#endif
   struct band_ctx ctx;
#ifdef ENABLE_QEXT
   int ext_b;
   opus_int32 ext_balance=0;
   opus_int32 ext_tell=0;
   VARDECL(unsigned char, ext_bytes_save);
#endif
   SAVE_STACK;

   M = 1<<LM;
   B = shortBlocks ? M : 1;
   norm_offset = M*eBands[start];
   /* No need to allocate norm for the last band because we don't need an
      output in that band. */
   ALLOC(_norm, C*(M*eBands[m->nbEBands-1]-norm_offset), celt_norm);
   norm = _norm;
   norm2 = norm + M*eBands[m->nbEBands-1]-norm_offset;

   /* For decoding, we can use the last band as scratch space because we don't need that
      scratch space for the last band and we don't care about the data there until we're
      decoding the last band. */
   if (encode && resynth)
      resynth_alloc = M*(eBands[m->nbEBands]-eBands[m->nbEBands-1]);
   else
      resynth_alloc = ALLOC_NONE;
   ALLOC(_lowband_scratch, resynth_alloc, celt_norm);
   if (encode && resynth)
      lowband_scratch = _lowband_scratch;
   else
      lowband_scratch = X_+M*eBands[m->effEBands-1];
   ALLOC(X_save, resynth_alloc, celt_norm);
   ALLOC(Y_save, resynth_alloc, celt_norm);
   ALLOC(X_save2, resynth_alloc, celt_norm);
   ALLOC(Y_save2, resynth_alloc, celt_norm);
   ALLOC(norm_save2, resynth_alloc, celt_norm);

   lowband_offset = 0;
   ctx.bandE = bandE;
   ctx.ec = ec;
   ctx.encode = encode;
   ctx.intensity = intensity;
   ctx.m = m;
   ctx.seed = *seed;
   ctx.spread = spread;
   ctx.arch = arch;
   ctx.disable_inv = disable_inv;
   ctx.resynth = resynth;
   ctx.theta_round = 0;
#ifdef ENABLE_QEXT
   ctx.ext_ec = ext_ec;
   ctx.ext_total_bits = ext_total_bits;
   ctx.extra_bands = end == NB_QEXT_BANDS || end == 2;
   if (ctx.extra_bands) theta_rdo = 0;
   ALLOC(ext_bytes_save, theta_rdo ? QEXT_PACKET_SIZE_CAP : ALLOC_NONE, unsigned char);
#endif
   ALLOC(bytes_save, theta_rdo ? 1275 : ALLOC_NONE, unsigned char);

   /* Avoid injecting noise in the first band on transients. */
   ctx.avoid_split_noise = B > 1;
   for (i=start;i<end;i++)
   {
      opus_int32 tell;
      int b;
      int N;
      opus_int32 curr_balance;
      int effective_lowband=-1;
      celt_norm * OPUS_RESTRICT X, * OPUS_RESTRICT Y;
      int tf_change=0;
      unsigned x_cm;
      unsigned y_cm;
      int last;

      ctx.i = i;
      last = (i==end-1);

      X = X_+M*eBands[i];
      if (Y_!=NULL)
         Y = Y_+M*eBands[i];
      else
         Y = NULL;
      N = M*eBands[i+1]-M*eBands[i];
      celt_assert(N > 0);
      tell = ec_tell_frac(ec);

      /* Compute how many bits we want to allocate to this band */
      if (i != start)
         balance -= tell;
      remaining_bits = total_bits-tell-1;
      ctx.remaining_bits = remaining_bits;
#ifdef ENABLE_QEXT
      if (i != start) {
         ext_balance += extra_pulses[i-1] + ext_tell;
      }
      ext_tell = ec_tell_frac(ext_ec);
      ctx.extra_bits = extra_pulses[i];
      if (i != start)
         ext_balance -= ext_tell;
      if (i <= codedBands-1)
      {
         opus_int32 ext_curr_balance = celt_sudiv(ext_balance, IMIN(3, codedBands-i));
         ext_b = IMAX(0, IMIN(16383, IMIN(ext_total_bits-ext_tell,extra_pulses[i]+ext_curr_balance)));
      } else {
         ext_b = 0;
      }
#endif
      if (i <= codedBands-1)
      {
         curr_balance = celt_sudiv(balance, IMIN(3, codedBands-i));
         b = IMAX(0, IMIN(16383, IMIN(remaining_bits+1,pulses[i]+curr_balance)));
      } else {
         b = 0;
      }

#ifndef DISABLE_UPDATE_DRAFT
      if (resynth && (M*eBands[i]-N >= M*eBands[start] || i==start+1) && (update_lowband || lowband_offset==0))
            lowband_offset = i;
      if (i == start+1)
         special_hybrid_folding(m, norm, norm2, start, M, dual_stereo);
#else
      if (resynth && M*eBands[i]-N >= M*eBands[start] && (update_lowband || lowband_offset==0))
            lowband_offset = i;
#endif

      tf_change = tf_res[i];
      ctx.tf_change = tf_change;
      if (i>=m->effEBands)
      {
         X=norm;
         if (Y_!=NULL)
            Y = norm;
         lowband_scratch = NULL;
      }
      if (last && !theta_rdo)
         lowband_scratch = NULL;

      /* Get a conservative estimate of the collapse_mask's for the bands we're
         going to be folding from. */
      if (lowband_offset != 0 && (spread!=SPREAD_AGGRESSIVE || B>1 || tf_change<0))
      {
         int fold_start;
         int fold_end;
         int fold_i;
         /* This ensures we never repeat spectral content within one band */
         effective_lowband = IMAX(0, M*eBands[lowband_offset]-norm_offset-N);
         fold_start = lowband_offset;
         while(M*eBands[--fold_start] > effective_lowband+norm_offset);
         fold_end = lowband_offset-1;
#ifndef DISABLE_UPDATE_DRAFT
         while(++fold_end < i && M*eBands[fold_end] < effective_lowband+norm_offset+N);
#else
         while(M*eBands[++fold_end] < effective_lowband+norm_offset+N);
#endif
         x_cm = y_cm = 0;
         fold_i = fold_start; do {
           x_cm |= collapse_masks[fold_i*C+0];
           y_cm |= collapse_masks[fold_i*C+C-1];
         } while (++fold_i<fold_end);
      }
      /* Otherwise, we'll be using the LCG to fold, so all blocks will (almost
         always) be non-zero. */
      else
         x_cm = y_cm = (1<<B)-1;

      if (dual_stereo && i==intensity)
      {
         int j;

         /* Switch off dual stereo to do intensity. */
         dual_stereo = 0;
         if (resynth)
            for (j=0;j<M*eBands[i]-norm_offset;j++)
               norm[j] = HALF32(norm[j]+norm2[j]);
      }
      if (dual_stereo)
      {
         x_cm = quant_band(&ctx, X, N, b/2, B,
               effective_lowband != -1 ? norm+effective_lowband : NULL, LM,
               last?NULL:norm+M*eBands[i]-norm_offset, Q31ONE, lowband_scratch, x_cm ARG_QEXT(ext_b/2));
         y_cm = quant_band(&ctx, Y, N, b/2, B,
               effective_lowband != -1 ? norm2+effective_lowband : NULL, LM,
               last?NULL:norm2+M*eBands[i]-norm_offset, Q31ONE, lowband_scratch, y_cm ARG_QEXT(ext_b/2));
      } else {
         if (Y!=NULL)
         {
            if (theta_rdo && i < intensity)
            {
               ec_ctx ec_save, ec_save2;
               struct band_ctx ctx_save, ctx_save2;
               opus_val32 dist0, dist1;
               unsigned cm, cm2;
               int nstart_bytes, nend_bytes, save_bytes;
               unsigned char *bytes_buf;
#ifdef ENABLE_QEXT
               ec_ctx ext_ec_save, ext_ec_save2;
               unsigned char *ext_bytes_buf;
               int ext_nstart_bytes, ext_nend_bytes, ext_save_bytes;
#endif
               opus_val16 w[2];
               compute_channel_weights(bandE[i], bandE[i+m->nbEBands], w);
               /* Make a copy. */
               cm = x_cm|y_cm;
               ec_save = *ec;
#ifdef ENABLE_QEXT
               ext_ec_save = *ext_ec;
#endif
               ctx_save = ctx;
               OPUS_COPY(X_save, X, N);
               OPUS_COPY(Y_save, Y, N);
               /* Encode and round down. */
               ctx.theta_round = -1;
               x_cm = quant_band_stereo(&ctx, X, Y, N, b, B,
                     effective_lowband != -1 ? norm+effective_lowband : NULL, LM,
                     last?NULL:norm+M*eBands[i]-norm_offset, lowband_scratch, cm ARG_QEXT(ext_b) ARG_QEXT(cap));
               dist0 = MULT16_32_Q15(w[0], celt_inner_prod_norm_shift(X_save, X, N, arch)) + MULT16_32_Q15(w[1], celt_inner_prod_norm_shift(Y_save, Y, N, arch));

               /* Save first result. */
               cm2 = x_cm;
               ec_save2 = *ec;
#ifdef ENABLE_QEXT
               ext_ec_save2 = *ext_ec;
#endif
               ctx_save2 = ctx;
               OPUS_COPY(X_save2, X, N);
               OPUS_COPY(Y_save2, Y, N);
               if (!last)
                  OPUS_COPY(norm_save2, norm+M*eBands[i]-norm_offset, N);
               nstart_bytes = ec_save.offs;
               nend_bytes = ec_save.storage;
               bytes_buf = ec_save.buf+nstart_bytes;
               save_bytes = nend_bytes-nstart_bytes;
               OPUS_COPY(bytes_save, bytes_buf, save_bytes);
#ifdef ENABLE_QEXT
               ext_nstart_bytes = ext_ec_save.offs;
               ext_nend_bytes = ext_ec_save.storage;
               ext_bytes_buf = ext_ec_save.buf!=NULL ? ext_ec_save.buf+ext_nstart_bytes : NULL;
               ext_save_bytes = ext_nend_bytes-ext_nstart_bytes;
               if (ext_save_bytes) OPUS_COPY(ext_bytes_save, ext_bytes_buf, ext_save_bytes);
#endif
               /* Restore */
               *ec = ec_save;
#ifdef ENABLE_QEXT
               *ext_ec = ext_ec_save;
#endif
               ctx = ctx_save;
               OPUS_COPY(X, X_save, N);
               OPUS_COPY(Y, Y_save, N);
#ifndef DISABLE_UPDATE_DRAFT
               if (i == start+1)
                  special_hybrid_folding(m, norm, norm2, start, M, dual_stereo);
#endif
               /* Encode and round up. */
               ctx.theta_round = 1;
               x_cm = quant_band_stereo(&ctx, X, Y, N, b, B,
                     effective_lowband != -1 ? norm+effective_lowband : NULL, LM,
                     last?NULL:norm+M*eBands[i]-norm_offset, lowband_scratch, cm ARG_QEXT(ext_b) ARG_QEXT(cap));
               dist1 = MULT16_32_Q15(w[0], celt_inner_prod_norm_shift(X_save, X, N, arch)) + MULT16_32_Q15(w[1], celt_inner_prod_norm_shift(Y_save, Y, N, arch));
               if (dist0 >= dist1) {
                  x_cm = cm2;
                  *ec = ec_save2;
#ifdef ENABLE_QEXT
                  *ext_ec = ext_ec_save2;
#endif
                  ctx = ctx_save2;
                  OPUS_COPY(X, X_save2, N);
                  OPUS_COPY(Y, Y_save2, N);
                  if (!last)
                     OPUS_COPY(norm+M*eBands[i]-norm_offset, norm_save2, N);
                  OPUS_COPY(bytes_buf, bytes_save, save_bytes);
#ifdef ENABLE_QEXT
                  if (ext_save_bytes) OPUS_COPY(ext_bytes_buf, ext_bytes_save, ext_save_bytes);
#endif
               }
            } else {
               ctx.theta_round = 0;
               x_cm = quant_band_stereo(&ctx, X, Y, N, b, B,
                     effective_lowband != -1 ? norm+effective_lowband : NULL, LM,
                     last?NULL:norm+M*eBands[i]-norm_offset, lowband_scratch, x_cm|y_cm ARG_QEXT(ext_b) ARG_QEXT(cap));
            }
         } else {
            x_cm = quant_band(&ctx, X, N, b, B,
                  effective_lowband != -1 ? norm+effective_lowband : NULL, LM,
                  last?NULL:norm+M*eBands[i]-norm_offset, Q31ONE, lowband_scratch, x_cm|y_cm ARG_QEXT(ext_b));
         }
         y_cm = x_cm;
      }
      collapse_masks[i*C+0] = (unsigned char)x_cm;
      collapse_masks[i*C+C-1] = (unsigned char)y_cm;
      balance += pulses[i] + tell;

      /* Update the folding position only as long as we have 1 bit/sample depth. */
      update_lowband = b>(N<<BITRES);
      /* We only need to avoid noise on a split for the first band. After that, we
         have folding. */
      ctx.avoid_split_noise = 0;
   }
   *seed = ctx.seed;

   RESTORE_STACK;
}
