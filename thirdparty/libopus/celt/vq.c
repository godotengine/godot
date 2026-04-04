/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
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

#include "mathops.h"
#include "cwrs.h"
#include "vq.h"
#include "arch.h"
#include "os_support.h"
#include "bands.h"
#include "rate.h"
#include "pitch.h"
#include "SigProc_FIX.h"

#if defined(FIXED_POINT)
void norm_scaleup(celt_norm *X, int N, int shift) {
   int i;
   celt_assert(shift >= 0);
   if (shift <= 0) return;
   for (i=0;i<N;i++) X[i] = SHL32(X[i], shift);
}

void norm_scaledown(celt_norm *X, int N, int shift) {
   int i;
   celt_assert(shift >= 0);
   if (shift <= 0) return;
   for (i=0;i<N;i++) X[i] = PSHR32(X[i], shift);
}

opus_val32 celt_inner_prod_norm(const celt_norm *x, const celt_norm *y, int len, int arch) {
   int i;
   opus_val32 sum = 0;
   (void)arch;
   for (i=0;i<len;i++) sum += x[i]*y[i];
   return sum;
}
opus_val32 celt_inner_prod_norm_shift(const celt_norm *x, const celt_norm *y, int len, int arch) {
   int i;
   opus_val64 sum = 0;
   (void)arch;
   for (i=0;i<len;i++) sum += x[i]*(opus_val64)y[i];
   return sum>>2*(NORM_SHIFT-14);
}
#endif

#ifndef OVERRIDE_vq_exp_rotation1
static void exp_rotation1(celt_norm *X, int len, int stride, opus_val16 c, opus_val16 s)
{
   int i;
   opus_val16 ms;
   celt_norm *Xptr;
   Xptr = X;
   ms = NEG16(s);
   norm_scaledown(X, len, NORM_SHIFT-14);
   for (i=0;i<len-stride;i++)
   {
      celt_norm x1, x2;
      x1 = Xptr[0];
      x2 = Xptr[stride];
      Xptr[stride] = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, x2),  s, x1), 15));
      *Xptr++      = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, x1), ms, x2), 15));
   }
   Xptr = &X[len-2*stride-1];
   for (i=len-2*stride-1;i>=0;i--)
   {
      celt_norm x1, x2;
      x1 = Xptr[0];
      x2 = Xptr[stride];
      Xptr[stride] = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, x2),  s, x1), 15));
      *Xptr--      = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, x1), ms, x2), 15));
   }
   norm_scaleup(X, len, NORM_SHIFT-14);
}
#endif /* OVERRIDE_vq_exp_rotation1 */

void exp_rotation(celt_norm *X, int len, int dir, int stride, int K, int spread)
{
   static const int SPREAD_FACTOR[3]={15,10,5};
   int i;
   opus_val16 c, s;
   opus_val16 gain, theta;
   int stride2=0;
   int factor;

   if (2*K>=len || spread==SPREAD_NONE)
      return;
   factor = SPREAD_FACTOR[spread-1];

   gain = celt_div((opus_val32)MULT16_16(Q15_ONE,len),(opus_val32)(len+factor*K));
   theta = HALF16(MULT16_16_Q15(gain,gain));

   c = celt_cos_norm(EXTEND32(theta));
   s = celt_cos_norm(EXTEND32(SUB16(Q15ONE,theta))); /*  sin(theta) */

   if (len>=8*stride)
   {
      stride2 = 1;
      /* This is just a simple (equivalent) way of computing sqrt(len/stride) with rounding.
         It's basically incrementing long as (stride2+0.5)^2 < len/stride. */
      while ((stride2*stride2+stride2)*stride + (stride>>2) < len)
         stride2++;
   }
   /*NOTE: As a minor optimization, we could be passing around log2(B), not B, for both this and for
      extract_collapse_mask().*/
   len = celt_udiv(len, stride);
   for (i=0;i<stride;i++)
   {
      if (dir < 0)
      {
         if (stride2)
            exp_rotation1(X+i*len, len, stride2, s, c);
         exp_rotation1(X+i*len, len, 1, c, s);
      } else {
         exp_rotation1(X+i*len, len, 1, c, -s);
         if (stride2)
            exp_rotation1(X+i*len, len, stride2, s, -c);
      }
   }
}

/** Normalizes the decoded integer pvq codeword to unit norm. */
static void normalise_residual(int * OPUS_RESTRICT iy, celt_norm * OPUS_RESTRICT X,
      int N, opus_val32 Ryy, opus_val32 gain, int shift)
{
   int i;
#ifdef FIXED_POINT
   int k;
#endif
   opus_val32 t;
   opus_val32 g;

#ifdef FIXED_POINT
   k = celt_ilog2(Ryy)>>1;
#endif
   t = VSHR32(Ryy, 2*(k-7)-15);
   g = MULT32_32_Q31(celt_rsqrt_norm32(t),gain);
   i=0;
   (void)shift;
#if defined(FIXED_POINT) && defined(ENABLE_QEXT)
   if (shift>0) {
      int tot_shift = NORM_SHIFT+1-k-shift;
      if (tot_shift >= 0) {
         do X[i] = MULT32_32_Q31(g, SHL32(iy[i], tot_shift));
         while (++i < N);
      } else {
         do X[i] = MULT32_32_Q31(g, PSHR32(iy[i], -tot_shift));
         while (++i < N);
      }
   } else
#endif
   do X[i] = VSHR32(MULT16_32_Q15(iy[i], g), k+15-NORM_SHIFT);
   while (++i < N);
}

static unsigned extract_collapse_mask(int *iy, int N, int B)
{
   unsigned collapse_mask;
   int N0;
   int i;
   if (B<=1)
      return 1;
   /*NOTE: As a minor optimization, we could be passing around log2(B), not B, for both this and for
      exp_rotation().*/
   N0 = celt_udiv(N, B);
   collapse_mask = 0;
   i=0; do {
      int j;
      unsigned tmp=0;
      j=0; do {
         tmp |= iy[i*N0+j];
      } while (++j<N0);
      collapse_mask |= (tmp!=0)<<i;
   } while (++i<B);
   return collapse_mask;
}

opus_val16 op_pvq_search_c(celt_norm *X, int *iy, int K, int N, int arch)
{
   VARDECL(celt_norm, y);
   VARDECL(int, signx);
   int i, j;
   int pulsesLeft;
   opus_val32 sum;
   opus_val32 xy;
   opus_val16 yy;
   SAVE_STACK;

   (void)arch;
   ALLOC(y, N, celt_norm);
   ALLOC(signx, N, int);
#ifdef FIXED_POINT
   {
      int shift = (celt_ilog2(1+celt_inner_prod_norm_shift(X, X, N, arch))+1)/2;
      shift = IMAX(0, shift+(NORM_SHIFT-14)-14);
      norm_scaledown(X, N, shift);
   }
#endif
   /* Get rid of the sign */
   sum = 0;
   j=0; do {
      signx[j] = X[j]<0;
      /* OPT: Make sure the compiler doesn't use a branch on ABS16(). */
      X[j] = ABS16(X[j]);
      iy[j] = 0;
      y[j] = 0;
   } while (++j<N);

   xy = yy = 0;

   pulsesLeft = K;

   /* Do a pre-search by projecting on the pyramid */
   if (K > (N>>1))
   {
      opus_val16 rcp;
      j=0; do {
         sum += X[j];
      }  while (++j<N);

      /* If X is too small, just replace it with a pulse at 0 */
#ifdef FIXED_POINT
      if (sum <= K)
#else
      /* Prevents infinities and NaNs from causing too many pulses
         to be allocated. 64 is an approximation of infinity here. */
      if (!(sum > EPSILON && sum < 64))
#endif
      {
         X[0] = QCONST16(1.f,14);
         j=1; do
            X[j]=0;
         while (++j<N);
         sum = QCONST16(1.f,14);
      }
#ifdef FIXED_POINT
      rcp = EXTRACT16(MULT16_32_Q16(K, celt_rcp(sum)));
#else
      /* Using K+e with e < 1 guarantees we cannot get more than K pulses. */
      rcp = EXTRACT16(MULT16_32_Q16(K+0.8f, celt_rcp(sum)));
#endif
      j=0; do {
#ifdef FIXED_POINT
         /* It's really important to round *towards zero* here */
         iy[j] = MULT16_16_Q15(X[j],rcp);
#else
         iy[j] = (int)floor(rcp*X[j]);
#endif
         y[j] = (celt_norm)iy[j];
         yy = MAC16_16(yy, y[j],y[j]);
         xy = MAC16_16(xy, X[j],y[j]);
         y[j] *= 2;
         pulsesLeft -= iy[j];
      }  while (++j<N);
   }
   celt_sig_assert(pulsesLeft>=0);

   /* This should never happen, but just in case it does (e.g. on silence)
      we fill the first bin with pulses. */
#ifdef FIXED_POINT_DEBUG
   celt_sig_assert(pulsesLeft<=N+3);
#endif
   if (pulsesLeft > N+3)
   {
      opus_val16 tmp = (opus_val16)pulsesLeft;
      yy = MAC16_16(yy, tmp, tmp);
      yy = MAC16_16(yy, tmp, y[0]);
      iy[0] += pulsesLeft;
      pulsesLeft=0;
   }

   for (i=0;i<pulsesLeft;i++)
   {
      opus_val16 Rxy, Ryy;
      int best_id;
      opus_val32 best_num;
      opus_val16 best_den;
#ifdef FIXED_POINT
      int rshift;
#endif
#ifdef FIXED_POINT
      rshift = 1+celt_ilog2(K-pulsesLeft+i+1);
#endif
      best_id = 0;
      /* The squared magnitude term gets added anyway, so we might as well
         add it outside the loop */
      yy = ADD16(yy, 1);

      /* Calculations for position 0 are out of the loop, in part to reduce
         mispredicted branches (since the if condition is usually false)
         in the loop. */
      /* Temporary sums of the new pulse(s) */
      Rxy = EXTRACT16(SHR32(ADD32(xy, EXTEND32(X[0])),rshift));
      /* We're multiplying y[j] by two so we don't have to do it here */
      Ryy = ADD16(yy, y[0]);

      /* Approximate score: we maximise Rxy/sqrt(Ryy) (we're guaranteed that
         Rxy is positive because the sign is pre-computed) */
      Rxy = MULT16_16_Q15(Rxy,Rxy);
      best_den = Ryy;
      best_num = Rxy;
      j=1;
      do {
         /* Temporary sums of the new pulse(s) */
         Rxy = EXTRACT16(SHR32(ADD32(xy, EXTEND32(X[j])),rshift));
         /* We're multiplying y[j] by two so we don't have to do it here */
         Ryy = ADD16(yy, y[j]);

         /* Approximate score: we maximise Rxy/sqrt(Ryy) (we're guaranteed that
            Rxy is positive because the sign is pre-computed) */
         Rxy = MULT16_16_Q15(Rxy,Rxy);
         /* The idea is to check for num/den >= best_num/best_den, but that way
            we can do it without any division */
         /* OPT: It's not clear whether a cmov is faster than a branch here
            since the condition is more often false than true and using
            a cmov introduces data dependencies across iterations. The optimal
            choice may be architecture-dependent. */
         if (opus_unlikely(MULT16_16(best_den, Rxy) > MULT16_16(Ryy, best_num)))
         {
            best_den = Ryy;
            best_num = Rxy;
            best_id = j;
         }
      } while (++j<N);

      /* Updating the sums of the new pulse(s) */
      xy = ADD32(xy, EXTEND32(X[best_id]));
      /* We're multiplying y[j] by two so we don't have to do it here */
      yy = ADD16(yy, y[best_id]);

      /* Only now that we've made the final choice, update y/iy */
      /* Multiplying y[j] by 2 so we don't have to do it everywhere else */
      y[best_id] += 2;
      iy[best_id]++;
   }

   /* Put the original sign back */
   j=0;
   do {
      /*iy[j] = signx[j] ? -iy[j] : iy[j];*/
      /* OPT: The is more likely to be compiled without a branch than the code above
         but has the same performance otherwise. */
      iy[j] = (iy[j]^-signx[j]) + signx[j];
   } while (++j<N);
   RESTORE_STACK;
   return yy;
}

#ifdef ENABLE_QEXT
#include "macros.h"

static opus_val32 op_pvq_search_N2(const celt_norm *X, int *iy, int *up_iy, int K, int up, int *refine, int shift) {
   opus_val32 sum;
   opus_val32 rcp_sum;
   int offset;
   sum = ABS32(X[0]) + ABS32(X[1]);
   if (sum < EPSILON) {
      iy[0] = K;
      up_iy[0] = up*K;
      iy[1]=up_iy[1]=0;
      *refine=0;
#ifdef FIXED_POINT
      return (opus_val64)K*K*up*up>>2*shift;
#else
      (void)shift;
      return K*(float)K*up*up;
#endif
   }
#ifdef FIXED_POINT
   int sum_shift;
   opus_val32 X0;
   sum_shift = 30-celt_ilog2(sum);
   rcp_sum = celt_rcp_norm32(SHL32(sum, sum_shift));
   X0 = MULT32_32_Q31(SHL32(X[0], sum_shift), rcp_sum);
   iy[0] = PSHR32(MULT32_32_Q31(SHL32(K, 8), X0), 7);
   up_iy[0] = PSHR32(MULT32_32_Q31(SHL32(up*K, 8), X0), 7);
#else
   rcp_sum = 1.f/sum;
   iy[0] = (int)floor(.5f+K*X[0]*rcp_sum);
   up_iy[0] = (int)floor(.5f+up*K*X[0]*rcp_sum);
#endif
   up_iy[0] = IMAX(up*iy[0] - (up-1)/2, IMIN(up*iy[0] + (up-1)/2, up_iy[0]));
   offset = up_iy[0] - up*iy[0];
   iy[1] = K-abs(iy[0]);
   up_iy[1] = up*K-abs(up_iy[0]);
   if (X[1] < 0) {
      iy[1] = -iy[1];
      up_iy[1] = -up_iy[1];
      offset = -offset;
   }
   *refine = offset;
#ifdef FIXED_POINT
   return (up_iy[0]*(opus_val64)up_iy[0] + up_iy[1]*(opus_val64)up_iy[1] + (1<<2*shift>>1))>>2*shift;
#else
   return up_iy[0]*(opus_val64)up_iy[0] + up_iy[1]*(opus_val64)up_iy[1];
#endif
}

static int op_pvq_refine(const opus_val32 *Xn, int *iy, int *iy0, int K, int up, int margin, int N) {
   int i;
   int dir;
   VARDECL(opus_val32, rounding);
   int iysum = 0;
   SAVE_STACK;
   ALLOC(rounding, N, opus_val32);
   for (i=0;i<N;i++) {
      opus_val32 tmp;
      tmp = MULT32_32_Q31(SHL32(K, 8), Xn[i]);
#ifdef FIXED_POINT
      iy[i] = (tmp+64) >> 7;
#else
      iy[i] = (int)floor(.5+tmp);
#endif
      rounding[i] = tmp - SHL32(iy[i], 7);
   }
   if (iy != iy0) {
      for (i=0;i<N;i++) iy[i] = IMIN(up*iy0[i]+up-1, IMAX(up*iy0[i]-up+1, iy[i]));
   }
   for (i=0;i<N;i++) iysum += iy[i];
   if (abs(iysum - K) > 32) {
      RESTORE_STACK;
      return 1;
   }
   dir = iysum < K ? 1 : -1;
   while (iysum != K) {
      opus_val32 roundval=-1000000*dir;
      int roundpos=0;
      for (i=0;i<N;i++) {
         if ((rounding[i]-roundval)*dir > 0 && abs(iy[i]-up*iy0[i]) < (margin-1) && !(dir==-1 && iy[i] == 0)) {
            roundval = rounding[i];
            roundpos = i;
         }
      }
      iy[roundpos] += dir;
      rounding[roundpos] -= SHL32(dir, 15);
      iysum+=dir;
   }
   RESTORE_STACK;
   return 0;
}

static opus_val32 op_pvq_search_extra(const celt_norm *X, int *iy, int *up_iy, int K, int up, int *refine, int N, int shift) {
   opus_val32 rcp_sum;
   opus_val32 sum=0;
   int i;
   int failed=0;
   opus_val64 yy=0;
   VARDECL(opus_val32, Xn);
   SAVE_STACK;
   for (i=0;i<N;i++) sum += ABS32(X[i]);
   ALLOC(Xn, N, opus_val32);
   if (sum < EPSILON)
      failed = 1;
   else {
#ifdef FIXED_POINT
      int sum_shift = 30-celt_ilog2(sum);
      rcp_sum = celt_rcp_norm32(SHL32(sum, sum_shift));
      for (i=0;i<N;i++) {
         Xn[i] = MULT32_32_Q31(SHL32(ABS32(X[i]), sum_shift), rcp_sum);
      }
#else
      rcp_sum = celt_rcp(sum);
      for (i=0;i<N;i++) {
         Xn[i] = ABS32(X[i])*rcp_sum;
      }
#endif
   }
   failed = failed || op_pvq_refine(Xn, iy, iy, K, 1, K+1, N);
   failed = failed || op_pvq_refine(Xn, up_iy, iy, up*K, up, up, N);
   if (failed) {
      iy[0] = K;
      for (i=1;i<N;i++) iy[i] = 0;
      up_iy[0] = up*K;
      for (i=1;i<N;i++) up_iy[i] = 0;
   }
   for (i=0;i<N;i++) {
      yy += up_iy[i]*(opus_val64)up_iy[i];
      if (X[i] < 0) {
         iy[i] = -iy[i];
         up_iy[i] = -up_iy[i];
      }
      refine[i] = up_iy[i]-up*iy[i];
   }
   RESTORE_STACK;
#ifdef FIXED_POINT
   return (yy + (1<<2*shift>>1))>>2*shift;
#else
   (void)shift;
   return yy;
#endif
}
#endif

#ifdef ENABLE_QEXT
/* Take advantage of the fact that "large" refine values are much less likely
   than smaller ones. */
static void ec_enc_refine(ec_enc *enc, opus_int32 refine, opus_int32 up, int extra_bits, int use_entropy) {
   int large;
   large = abs(refine)>up/2;
   ec_enc_bit_logp(enc, large, use_entropy ? 3 : 1);
   if (large) {
      ec_enc_bits(enc, refine < 0, 1);
      ec_enc_bits(enc, abs(refine)-up/2-1, extra_bits-1);
   } else {
      ec_enc_bits(enc, refine+up/2, extra_bits);
   }
}

static int ec_dec_refine(ec_enc *dec, opus_int32 up, int extra_bits, int use_entropy) {
   int large, refine;
   large = ec_dec_bit_logp(dec, use_entropy ? 3 : 1);
   if (large) {
      int sign = ec_dec_bits(dec, 1);
      refine = ec_dec_bits(dec, extra_bits-1) + up/2+1;
      if (sign) refine = -refine;
   } else {
      refine = (opus_int32)ec_dec_bits(dec, extra_bits)-up/2;
   }
   return refine;
}
#endif

unsigned alg_quant(celt_norm *X, int N, int K, int spread, int B, ec_enc *enc,
      opus_val32 gain, int resynth
      ARG_QEXT(ec_enc *ext_enc) ARG_QEXT(int extra_bits), int arch)
{
   VARDECL(int, iy);
   opus_val32 yy;
   unsigned collapse_mask;
#ifdef ENABLE_QEXT
   int yy_shift = 0;
#endif
   SAVE_STACK;

   celt_assert2(K>0, "alg_quant() needs at least one pulse");
   celt_assert2(N>1, "alg_quant() needs at least two dimensions");

   /* Covers vectorization by up to 4. */
   ALLOC(iy, N+3, int);

   exp_rotation(X, N, 1, B, K, spread);

#ifdef ENABLE_QEXT
   if (N==2 && extra_bits >= 2) {
      int refine;
      int up_iy[2];
      int up;
      yy_shift = IMAX(0, extra_bits-7);
      up = (1<<extra_bits)-1;
      yy = op_pvq_search_N2(X, iy, up_iy, K, up, &refine, yy_shift);
      collapse_mask = extract_collapse_mask(up_iy, N, B);
      encode_pulses(iy, N, K, enc);
      ec_enc_uint(ext_enc, refine+(up-1)/2, up);
      if (resynth)
         normalise_residual(up_iy, X, N, yy, gain, yy_shift);
   } else if (extra_bits >= 2) {
      int i;
      VARDECL(int, up_iy);
      VARDECL(int, refine);
      int up, use_entropy;
      ALLOC(up_iy, N, int);
      ALLOC(refine, N, int);
      yy_shift = IMAX(0, extra_bits-7);
      up = (1<<extra_bits)-1;
      yy = op_pvq_search_extra(X, iy, up_iy, K, up, refine, N, yy_shift);
      collapse_mask = extract_collapse_mask(up_iy, N, B);
      encode_pulses(iy, N, K, enc);
      use_entropy = (ext_enc->storage*8 - ec_tell(ext_enc)) > (unsigned)(N-1)*(extra_bits+3)+1;
      for (i=0;i<N-1;i++) ec_enc_refine(ext_enc, refine[i], up, extra_bits, use_entropy);
      if (iy[N-1]==0) ec_enc_bits(ext_enc, up_iy[N-1]<0, 1);
      if (resynth)
         normalise_residual(up_iy, X, N, yy, gain, yy_shift);
   } else
#endif
   {
      yy = op_pvq_search(X, iy, K, N, arch);
      collapse_mask = extract_collapse_mask(iy, N, B);
      encode_pulses(iy, N, K, enc);
      if (resynth)
         normalise_residual(iy, X, N, yy, gain, 0);
   }

   if (resynth)
      exp_rotation(X, N, -1, B, K, spread);

   RESTORE_STACK;
   return collapse_mask;
}

/** Decode pulse vector and combine the result with the pitch vector to produce
    the final normalised signal in the current band. */
unsigned alg_unquant(celt_norm *X, int N, int K, int spread, int B,
      ec_dec *dec, opus_val32 gain
      ARG_QEXT(ec_enc *ext_dec) ARG_QEXT(int extra_bits))
{
   opus_val32 Ryy;
   unsigned collapse_mask;
   VARDECL(int, iy);
   int yy_shift=0;
   SAVE_STACK;

   celt_assert2(K>0, "alg_unquant() needs at least one pulse");
   celt_assert2(N>1, "alg_unquant() needs at least two dimensions");
   ALLOC(iy, N, int);
   Ryy = decode_pulses(iy, N, K, dec);
#ifdef ENABLE_QEXT
   if (N==2 && extra_bits >= 2) {
      int up;
      int refine;
      yy_shift = IMAX(0, extra_bits-7);
      up = (1<<extra_bits)-1;
      refine = (opus_int32)ec_dec_uint(ext_dec, up) - (up-1)/2;
      iy[0] *= up;
      iy[1] *= up;
      if (iy[1] == 0) {
         iy[1] = (iy[0] > 0) ? -refine : refine;
         iy[0] += (refine*(opus_int64)iy[0] > 0) ? -refine : refine;
      } else if (iy[1] > 0) {
         iy[0] += refine;
         iy[1] -= refine*(iy[0]>0?1:-1);
      } else {
         iy[0] -= refine;
         iy[1] -= refine*(iy[0]>0?1:-1);
      }
#ifdef FIXED_POINT
      Ryy = (iy[0]*(opus_val64)iy[0] + iy[1]*(opus_val64)iy[1] + (1<<2*yy_shift>>1)) >> 2*yy_shift;
#else
      Ryy = iy[0]*(opus_val64)iy[0] + iy[1]*(opus_val64)iy[1];
#endif
   } else if (extra_bits >= 2) {
      int i;
      opus_val64 yy64;
      VARDECL(int, refine);
      int up, use_entropy;
      int sign=0;
      ALLOC(refine, N, int);
      yy_shift = IMAX(0, extra_bits-7);
      up = (1<<extra_bits)-1;
      use_entropy = (ext_dec->storage*8 - ec_tell(ext_dec)) > (unsigned)(N-1)*(extra_bits+3)+1;
      for (i=0;i<N-1;i++) refine[i] = ec_dec_refine(ext_dec, up, extra_bits, use_entropy);
      if (iy[N-1]==0) sign = ec_dec_bits(ext_dec, 1);
      else sign = iy[N-1] < 0;
      for (i=0;i<N-1;i++) {
         iy[i] = iy[i]*up + refine[i];
      }
      iy[N-1] = up*K;
      for (i=0;i<N-1;i++) iy[N-1] -= abs(iy[i]);
      if (sign) iy[N-1] = -iy[N-1];
      yy64 = 0;
      for (i=0;i<N;i++) yy64 += iy[i]*(opus_val64)iy[i];
#ifdef FIXED_POINT
      Ryy = (yy64 + (1<<2*yy_shift>>1)) >> 2*yy_shift;
#else
      Ryy = yy64;
#endif
   }
#endif
   normalise_residual(iy, X, N, Ryy, gain, yy_shift);
   exp_rotation(X, N, -1, B, K, spread);
   collapse_mask = extract_collapse_mask(iy, N, B);
   RESTORE_STACK;
   return collapse_mask;
}

#ifndef OVERRIDE_renormalise_vector
void renormalise_vector(celt_norm *X, int N, opus_val32 gain, int arch)
{
   int i;
#ifdef FIXED_POINT
   int k;
#endif
   opus_val32 E;
   opus_val16 g;
   opus_val32 t;
   celt_norm *xptr;
   norm_scaledown(X, N, NORM_SHIFT-14);
   E = EPSILON + celt_inner_prod_norm(X, X, N, arch);
#ifdef FIXED_POINT
   k = celt_ilog2(E)>>1;
#endif
   t = VSHR32(E, 2*(k-7));
   g = MULT32_32_Q31(celt_rsqrt_norm(t),gain);

   xptr = X;
   for (i=0;i<N;i++)
   {
      *xptr = EXTRACT16(PSHR32(MULT16_16(g, *xptr), k+15-14));
      xptr++;
   }
   norm_scaleup(X, N, NORM_SHIFT-14);
   /*return celt_sqrt(E);*/
}
#endif /* OVERRIDE_renormalise_vector */

opus_int32 stereo_itheta(const celt_norm *X, const celt_norm *Y, int stereo, int N, int arch)
{
   int i;
   int itheta;
   opus_val32 mid, side;
   opus_val32 Emid, Eside;

   Emid = Eside = 0;
   if (stereo)
   {
      for (i=0;i<N;i++)
      {
         celt_norm m, s;
         m = PSHR32(ADD32(X[i], Y[i]), NORM_SHIFT-13);
         s = PSHR32(SUB32(X[i], Y[i]), NORM_SHIFT-13);
         Emid = MAC16_16(Emid, m, m);
         Eside = MAC16_16(Eside, s, s);
      }
   } else {
      Emid += celt_inner_prod_norm_shift(X, X, N, arch);
      Eside += celt_inner_prod_norm_shift(Y, Y, N, arch);
   }
   mid = celt_sqrt32(Emid);
   side = celt_sqrt32(Eside);
#if defined(FIXED_POINT)
   itheta = celt_atan2p_norm(side, mid);
#else
   itheta = (int)floor(.5f+65536.f*16384*celt_atan2p_norm(side,mid));
#endif

   return itheta;
}

#ifdef ENABLE_QEXT

static void cubic_synthesis(celt_norm *X, int *iy, int N, int K, int face, int sign, opus_val32 gain) {
   int i;
   opus_val32 sum=0;
   opus_val32 mag;
#ifdef FIXED_POINT
   int sum_shift;
   int shift = IMAX(celt_ilog2(K) + celt_ilog2(N)/2 - 13, 0);
#endif
   for (i=0;i<N;i++) {
      X[i] = (1+2*iy[i])-K;
   }
   X[face] = sign ? -K : K;
   for (i=0;i<N;i++) {
      sum += PSHR32(MULT16_16(X[i],X[i]), 2*shift);
   }
#ifdef FIXED_POINT
   sum_shift = (29-celt_ilog2(sum))>>1;
   mag = celt_rsqrt_norm32(SHL32(sum, 2*sum_shift+1));
   for (i=0;i<N;i++) {
      X[i] = VSHR32(MULT16_32_Q15(X[i],MULT32_32_Q31(mag,gain)), shift-sum_shift+29-NORM_SHIFT);
   }
#else
   mag = 1.f/sqrt(sum);
   for (i=0;i<N;i++) {
      X[i] *= mag*gain;
   }
#endif
}

unsigned cubic_quant(celt_norm *X, int N, int res, int B, ec_enc *enc, opus_val32 gain, int resynth) {
   int i;
   int face=0;
   int K;
   VARDECL(int, iy);
   celt_norm faceval=-1;
   opus_val32 norm;
   int sign;
   SAVE_STACK;
   ALLOC(iy, N, int);
   K = 1<<res;
   /* Using odd K on transients to avoid adding pre-echo. */
   if (B!=1) K=IMAX(1, K-1);
   if (K==1) {
      if (resynth) OPUS_CLEAR(X, N);
      RESTORE_STACK;
      return 0;
   }
   for (i=0;i<N;i++) {
      if (ABS32(X[i]) > faceval) {
         faceval = ABS32(X[i]);
         face = i;
      }
   }
   sign = X[face]<0;
   ec_enc_uint(enc, face, N);
   ec_enc_bits(enc, sign, 1);
#ifdef FIXED_POINT
   if (faceval != 0) {
      int face_shift = 30-celt_ilog2(faceval);
      norm = celt_rcp_norm32(SHL32(faceval, face_shift));
      norm = MULT16_32_Q15(K, norm);
      for (i=0;i<N;i++) {
         /* By computing X[i]+faceval inside the shift, the result is guaranteed non-negative. */
         iy[i] = IMIN(K-1, (MULT32_32_Q31(SHL32(X[i]+faceval, face_shift-1), norm)) >> 15);
      }
   } else {
      OPUS_CLEAR(iy, N);
   }
#else
   norm = .5f*K/(faceval+EPSILON);
   for (i=0;i<N;i++) {
      iy[i] = IMIN(K-1, (int)floor((X[i]+faceval)*norm));
   }
#endif
   for (i=0;i<N;i++) {
      if (i != face) ec_enc_bits(enc, iy[i], res);
   }
   if (resynth) {
      cubic_synthesis(X, iy, N, K, face, sign, gain);
   }
   RESTORE_STACK;
   return (1<<B)-1;
}

unsigned cubic_unquant(celt_norm *X, int N, int res, int B, ec_dec *dec, opus_val32 gain) {
   int i;
   int face;
   int sign;
   int K;
   VARDECL(int, iy);
   SAVE_STACK;
   ALLOC(iy, N, int);
   K = 1<<res;
   /* Using odd K on transients to avoid adding pre-echo. */
   if (B!=1) K=IMAX(1, K-1);
   if (K==1) {
      OPUS_CLEAR(X, N);
      RESTORE_STACK;
      return 0;
   }
   face = ec_dec_uint(dec, N);
   sign = ec_dec_bits(dec, 1);
   for (i=0;i<N;i++) {
      if (i != face) iy[i] = ec_dec_bits(dec, res);
   }
   iy[face]=0;
   cubic_synthesis(X, iy, N, K, face, sign, gain);
   RESTORE_STACK;
   return (1<<B)-1;
}
#endif
