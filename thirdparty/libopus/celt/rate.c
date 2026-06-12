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

#include <math.h>
#include "modes.h"
#include "cwrs.h"
#include "arch.h"
#include "os_support.h"

#include "entcode.h"
#include "rate.h"
#include "quant_bands.h"

static const unsigned char LOG2_FRAC_TABLE[24]={
   0,
   8,13,
  16,19,21,23,
  24,26,27,28,29,30,31,32,
  32,33,34,34,35,36,36,37,37
};

#if defined(CUSTOM_MODES)

/*Determines if V(N,K) fits in a 32-bit unsigned integer.
  N and K are themselves limited to 15 bits.*/
static int fits_in32(int _n, int _k)
{
   static const opus_int16 maxN[15] = {
      32767, 32767, 32767, 1476, 283, 109,  60,  40,
       29,  24,  20,  18,  16,  14,  13};
   static const opus_int16 maxK[15] = {
      32767, 32767, 32767, 32767, 1172, 238,  95,  53,
       36,  27,  22,  18,  16,  15,  13};
   if (_n>=14)
   {
      if (_k>=14)
         return 0;
      else
         return _n <= maxN[_k];
   } else {
      return _k <= maxK[_n];
   }
}

void compute_pulse_cache(CELTMode *m, int LM)
{
   int C;
   int i;
   int j;
   int curr=0;
   int nbEntries=0;
   int entryN[100], entryK[100], entryI[100];
   const opus_int16 *eBands = m->eBands;
   PulseCache *cache = &m->cache;
   opus_int16 *cindex;
   unsigned char *bits;
   unsigned char *cap;

   cindex = (opus_int16 *)opus_alloc(sizeof(cache->index[0])*m->nbEBands*(LM+2));
   cache->index = cindex;

   /* Scan for all unique band sizes */
   for (i=0;i<=LM+1;i++)
   {
      for (j=0;j<m->nbEBands;j++)
      {
         int k;
         int N = (eBands[j+1]-eBands[j])<<i>>1;
         cindex[i*m->nbEBands+j] = -1;
         /* Find other bands that have the same size */
         for (k=0;k<=i;k++)
         {
            int n;
            for (n=0;n<m->nbEBands && (k!=i || n<j);n++)
            {
               if (N == (eBands[n+1]-eBands[n])<<k>>1)
               {
                  cindex[i*m->nbEBands+j] = cindex[k*m->nbEBands+n];
                  break;
               }
            }
         }
         if (cache->index[i*m->nbEBands+j] == -1 && N!=0)
         {
            int K;
            entryN[nbEntries] = N;
            K = 0;
            while (fits_in32(N,get_pulses(K+1)) && K<MAX_PSEUDO)
               K++;
            entryK[nbEntries] = K;
            cindex[i*m->nbEBands+j] = curr;
            entryI[nbEntries] = curr;

            curr += K+1;
            nbEntries++;
         }
      }
   }
   bits = (unsigned char *)opus_alloc(sizeof(unsigned char)*curr);
   cache->bits = bits;
   cache->size = curr;
   /* Compute the cache for all unique sizes */
   for (i=0;i<nbEntries;i++)
   {
      unsigned char *ptr = bits+entryI[i];
      opus_int16 tmp[CELT_MAX_PULSES+1];
      get_required_bits(tmp, entryN[i], get_pulses(entryK[i]), BITRES);
      for (j=1;j<=entryK[i];j++)
         ptr[j] = tmp[get_pulses(j)]-1;
      ptr[0] = entryK[i];
   }

   /* Compute the maximum rate for each band at which we'll reliably use as
       many bits as we ask for. */
   cache->caps = cap = (unsigned char *)opus_alloc(sizeof(cache->caps[0])*(LM+1)*2*m->nbEBands);
   for (i=0;i<=LM;i++)
   {
      for (C=1;C<=2;C++)
      {
         for (j=0;j<m->nbEBands;j++)
         {
            int N0;
            int max_bits;
            N0 = m->eBands[j+1]-m->eBands[j];
            /* N=1 bands only have a sign bit and fine bits. */
            if (N0<<i == 1)
               max_bits = C*(1+MAX_FINE_BITS)<<BITRES;
            else
            {
               const unsigned char *pcache;
               opus_int32           num;
               opus_int32           den;
               int                  LM0;
               int                  N;
               int                  offset;
               int                  ndof;
               int                  qb;
               int                  k;
               LM0 = 0;
               /* Even-sized bands bigger than N=2 can be split one more time.
                  As of commit 44203907 all bands >1 are even, including custom modes.*/
               if (N0 > 2)
               {
                  N0>>=1;
                  LM0--;
               }
               /* N0=1 bands can't be split down to N<2. */
               else if (N0 <= 1)
               {
                  LM0=IMIN(i,1);
                  N0<<=LM0;
               }
               /* Compute the cost for the lowest-level PVQ of a fully split
                   band. */
               pcache = bits + cindex[(LM0+1)*m->nbEBands+j];
               max_bits = pcache[pcache[0]]+1;
               /* Add in the cost of coding regular splits. */
               N = N0;
               for(k=0;k<i-LM0;k++){
                  max_bits <<= 1;
                  /* Offset the number of qtheta bits by log2(N)/2
                      + QTHETA_OFFSET compared to their "fair share" of
                      total/N */
                  offset = ((m->logN[j]+(opus_int32)((opus_uint32)(LM0+k)<<BITRES))>>1)-QTHETA_OFFSET;
                  /* The number of qtheta bits we'll allocate if the remainder
                      is to be max_bits.
                     The average measured cost for theta is 0.89701 times qb,
                      approximated here as 459/512. */
                  num=459*(opus_int32)((2*N-1)*offset+max_bits);
                  den=((opus_int32)(2*N-1)<<9)-459;
                  qb = IMIN((num+(den>>1))/den, 57);
                  celt_assert(qb >= 0);
                  max_bits += qb;
                  N <<= 1;
               }
               /* Add in the cost of a stereo split, if necessary. */
               if (C==2)
               {
                  max_bits <<= 1;
                  offset = ((m->logN[j]+(i<<BITRES))>>1)-(N==2?QTHETA_OFFSET_TWOPHASE:QTHETA_OFFSET);
                  ndof = 2*N-1-(N==2);
                  /* The average measured cost for theta with the step PDF is
                      0.95164 times qb, approximated here as 487/512. */
                  num = (N==2?512:487)*(opus_int32)(max_bits+ndof*offset);
                  den = ((opus_int32)ndof<<9)-(N==2?512:487);
                  qb = IMIN((num+(den>>1))/den, (N==2?64:61));
                  celt_assert(qb >= 0);
                  max_bits += qb;
               }
               /* Add the fine bits we'll use. */
               /* Compensate for the extra DoF in stereo */
               ndof = C*N + ((C==2 && N>2) ? 1 : 0);
               /* Offset the number of fine bits by log2(N)/2 + FINE_OFFSET
                   compared to their "fair share" of total/N */
               offset = ((m->logN[j] + (i<<BITRES))>>1)-FINE_OFFSET;
               /* N=2 is the only point that doesn't match the curve */
               if (N==2)
                  offset += 1<<BITRES>>2;
               /* The number of fine bits we'll allocate if the remainder is
                   to be max_bits. */
               num = max_bits+ndof*offset;
               den = (ndof-1)<<BITRES;
               qb = IMIN((num+(den>>1))/den, MAX_FINE_BITS);
               celt_assert(qb >= 0);
               max_bits += C*qb<<BITRES;
            }
            max_bits = (4*max_bits/(C*((m->eBands[j+1]-m->eBands[j])<<i)))-64;
            celt_assert(max_bits >= 0);
            celt_assert(max_bits < 256);
            *cap++ = (unsigned char)max_bits;
         }
      }
   }
}

#endif /* CUSTOM_MODES */

#define ALLOC_STEPS 6

static OPUS_INLINE int interp_bits2pulses(const CELTMode *m, int start, int end, int skip_start,
      const int *bits1, const int *bits2, const int *thresh, const int *cap, opus_int32 total, opus_int32 *_balance,
      int skip_rsv, int *intensity, int intensity_rsv, int *dual_stereo, int dual_stereo_rsv, int *bits,
      int *ebits, int *fine_priority, int C, int LM, ec_ctx *ec, int encode, int prev, int signalBandwidth)
{
   opus_int32 psum;
   int lo, hi;
   int i, j;
   int logM;
   int stereo;
   int codedBands=-1;
   int alloc_floor;
   opus_int32 left, percoeff;
   int done;
   opus_int32 balance;
   SAVE_STACK;

   alloc_floor = C<<BITRES;
   stereo = C>1;

   logM = LM<<BITRES;
   lo = 0;
   hi = 1<<ALLOC_STEPS;
   for (i=0;i<ALLOC_STEPS;i++)
   {
      int mid = (lo+hi)>>1;
      psum = 0;
      done = 0;
      for (j=end;j-->start;)
      {
         int tmp = bits1[j] + (mid*(opus_int32)bits2[j]>>ALLOC_STEPS);
         if (tmp >= thresh[j] || done)
         {
            done = 1;
            /* Don't allocate more than we can actually use */
            psum += IMIN(tmp, cap[j]);
         } else {
            if (tmp >= alloc_floor)
               psum += alloc_floor;
         }
      }
      if (psum > total)
         hi = mid;
      else
         lo = mid;
   }
   psum = 0;
   /*printf ("interp bisection gave %d\n", lo);*/
   done = 0;
   for (j=end;j-->start;)
   {
      int tmp = bits1[j] + ((opus_int32)lo*bits2[j]>>ALLOC_STEPS);
      if (tmp < thresh[j] && !done)
      {
         if (tmp >= alloc_floor)
            tmp = alloc_floor;
         else
            tmp = 0;
      } else
         done = 1;
      /* Don't allocate more than we can actually use */
      tmp = IMIN(tmp, cap[j]);
      bits[j] = tmp;
      psum += tmp;
   }

   /* Decide which bands to skip, working backwards from the end. */
   for (codedBands=end;;codedBands--)
   {
      int band_width;
      int band_bits;
      int rem;
      j = codedBands-1;
      /* Never skip the first band, nor a band that has been boosted by
          dynalloc.
         In the first case, we'd be coding a bit to signal we're going to waste
          all the other bits.
         In the second case, we'd be coding a bit to redistribute all the bits
          we just signaled should be concentrated in this band. */
      if (j<=skip_start)
      {
         /* Give the bit we reserved to end skipping back. */
         total += skip_rsv;
         break;
      }
      /*Figure out how many left-over bits we would be adding to this band.
        This can include bits we've stolen back from higher, skipped bands.*/
      left = total-psum;
      percoeff = celt_udiv(left, m->eBands[codedBands]-m->eBands[start]);
      left -= (m->eBands[codedBands]-m->eBands[start])*percoeff;
      rem = IMAX(left-(m->eBands[j]-m->eBands[start]),0);
      band_width = m->eBands[codedBands]-m->eBands[j];
      band_bits = (int)(bits[j] + percoeff*band_width + rem);
      /*Only code a skip decision if we're above the threshold for this band.
        Otherwise it is force-skipped.
        This ensures that we have enough bits to code the skip flag.*/
      if (band_bits >= IMAX(thresh[j], alloc_floor+(1<<BITRES)))
      {
         if (encode)
         {
            /*This if() block is the only part of the allocation function that
               is not a mandatory part of the bitstream: any bands we choose to
               skip here must be explicitly signaled.*/
            int depth_threshold;
            /*We choose a threshold with some hysteresis to keep bands from
               fluctuating in and out, but we try not to fold below a certain point. */
            if (codedBands > 17)
               depth_threshold = j<prev ? 7 : 9;
            else
               depth_threshold = 0;
#ifdef FUZZING
            (void)signalBandwidth;
            (void)depth_threshold;
            if ((rand()&0x1) == 0)
#else
            if (codedBands<=start+2 || (band_bits > (depth_threshold*band_width<<LM<<BITRES)>>4 && j<=signalBandwidth))
#endif
            {
               ec_enc_bit_logp(ec, 1, 1);
               break;
            }
            ec_enc_bit_logp(ec, 0, 1);
         } else if (ec_dec_bit_logp(ec, 1)) {
            break;
         }
         /*We used a bit to skip this band.*/
         psum += 1<<BITRES;
         band_bits -= 1<<BITRES;
      }
      /*Reclaim the bits originally allocated to this band.*/
      psum -= bits[j]+intensity_rsv;
      if (intensity_rsv > 0)
         intensity_rsv = LOG2_FRAC_TABLE[j-start];
      psum += intensity_rsv;
      if (band_bits >= alloc_floor)
      {
         /*If we have enough for a fine energy bit per channel, use it.*/
         psum += alloc_floor;
         bits[j] = alloc_floor;
      } else {
         /*Otherwise this band gets nothing at all.*/
         bits[j] = 0;
      }
   }

   celt_assert(codedBands > start);
   /* Code the intensity and dual stereo parameters. */
   if (intensity_rsv > 0)
   {
      if (encode)
      {
         *intensity = IMIN(*intensity, codedBands);
         ec_enc_uint(ec, *intensity-start, codedBands+1-start);
      }
      else
         *intensity = start+ec_dec_uint(ec, codedBands+1-start);
   }
   else
      *intensity = 0;
   if (*intensity <= start)
   {
      total += dual_stereo_rsv;
      dual_stereo_rsv = 0;
   }
   if (dual_stereo_rsv > 0)
   {
      if (encode)
         ec_enc_bit_logp(ec, *dual_stereo, 1);
      else
         *dual_stereo = ec_dec_bit_logp(ec, 1);
   }
   else
      *dual_stereo = 0;

   /* Allocate the remaining bits */
   left = total-psum;
   percoeff = celt_udiv(left, m->eBands[codedBands]-m->eBands[start]);
   left -= (m->eBands[codedBands]-m->eBands[start])*percoeff;
   for (j=start;j<codedBands;j++)
      bits[j] += ((int)percoeff*(m->eBands[j+1]-m->eBands[j]));
   for (j=start;j<codedBands;j++)
   {
      int tmp = (int)IMIN(left, m->eBands[j+1]-m->eBands[j]);
      bits[j] += tmp;
      left -= tmp;
   }
   /*for (j=0;j<end;j++)printf("%d ", bits[j]);printf("\n");*/

   balance = 0;
   for (j=start;j<codedBands;j++)
   {
      int N0, N, den;
      int offset;
      int NClogN;
      opus_int32 excess, bit;

      celt_assert(bits[j] >= 0);
      N0 = m->eBands[j+1]-m->eBands[j];
      N=N0<<LM;
      bit = (opus_int32)bits[j]+balance;

      if (N>1)
      {
         excess = MAX32(bit-cap[j],0);
         bits[j] = bit-excess;

         /* Compensate for the extra DoF in stereo */
         den=(C*N+ ((C==2 && N>2 && !*dual_stereo && j<*intensity) ? 1 : 0));

         NClogN = den*(m->logN[j] + logM);

         /* Offset for the number of fine bits by log2(N)/2 + FINE_OFFSET
            compared to their "fair share" of total/N */
         offset = (NClogN>>1)-den*FINE_OFFSET;

         /* N=2 is the only point that doesn't match the curve */
         if (N==2)
            offset += den<<BITRES>>2;

         /* Changing the offset for allocating the second and third
             fine energy bit */
         if (bits[j] + offset < den*2<<BITRES)
            offset += NClogN>>2;
         else if (bits[j] + offset < den*3<<BITRES)
            offset += NClogN>>3;

         /* Divide with rounding */
         ebits[j] = IMAX(0, (bits[j] + offset + (den<<(BITRES-1))));
         ebits[j] = celt_udiv(ebits[j], den)>>BITRES;

         /* Make sure not to bust */
         if (C*ebits[j] > (bits[j]>>BITRES))
            ebits[j] = bits[j] >> stereo >> BITRES;

         /* More than that is useless because that's about as far as PVQ can go */
         ebits[j] = IMIN(ebits[j], MAX_FINE_BITS);

         /* If we rounded down or capped this band, make it a candidate for the
             final fine energy pass */
         fine_priority[j] = ebits[j]*(den<<BITRES) >= bits[j]+offset;

         /* Remove the allocated fine bits; the rest are assigned to PVQ */
         bits[j] -= C*ebits[j]<<BITRES;

      } else {
         /* For N=1, all bits go to fine energy except for a single sign bit */
         excess = MAX32(0,bit-(C<<BITRES));
         bits[j] = bit-excess;
         ebits[j] = 0;
         fine_priority[j] = 1;
      }

      /* Fine energy can't take advantage of the re-balancing in
          quant_all_bands().
         Instead, do the re-balancing here.*/
      if(excess > 0)
      {
         int extra_fine;
         int extra_bits;
         extra_fine = IMIN(excess>>(stereo+BITRES),MAX_FINE_BITS-ebits[j]);
         ebits[j] += extra_fine;
         extra_bits = extra_fine*C<<BITRES;
         fine_priority[j] = extra_bits >= excess-balance;
         excess -= extra_bits;
      }
      balance = excess;

      celt_assert(bits[j] >= 0);
      celt_assert(ebits[j] >= 0);
   }
   /* Save any remaining bits over the cap for the rebalancing in
       quant_all_bands(). */
   *_balance = balance;

   /* The skipped bands use all their bits for fine energy. */
   for (;j<end;j++)
   {
      ebits[j] = bits[j] >> stereo >> BITRES;
      celt_assert(C*ebits[j]<<BITRES == bits[j]);
      bits[j] = 0;
      fine_priority[j] = ebits[j]<1;
   }
   RESTORE_STACK;
   return codedBands;
}

int clt_compute_allocation(const CELTMode *m, int start, int end, const int *offsets, const int *cap, int alloc_trim, int *intensity, int *dual_stereo,
      opus_int32 total, opus_int32 *balance, int *pulses, int *ebits, int *fine_priority, int C, int LM, ec_ctx *ec, int encode, int prev, int signalBandwidth)
{
   int lo, hi, len, j;
   int codedBands;
   int skip_start;
   int skip_rsv;
   int intensity_rsv;
   int dual_stereo_rsv;
   VARDECL(int, bits1);
   VARDECL(int, bits2);
   VARDECL(int, thresh);
   VARDECL(int, trim_offset);
   SAVE_STACK;

   total = IMAX(total, 0);
   len = m->nbEBands;
   skip_start = start;
   /* Reserve a bit to signal the end of manually skipped bands. */
   skip_rsv = total >= 1<<BITRES ? 1<<BITRES : 0;
   total -= skip_rsv;
   /* Reserve bits for the intensity and dual stereo parameters. */
   intensity_rsv = dual_stereo_rsv = 0;
   if (C==2)
   {
      intensity_rsv = LOG2_FRAC_TABLE[end-start];
      if (intensity_rsv>total)
         intensity_rsv = 0;
      else
      {
         total -= intensity_rsv;
         dual_stereo_rsv = total>=1<<BITRES ? 1<<BITRES : 0;
         total -= dual_stereo_rsv;
      }
   }
   ALLOC(bits1, len, int);
   ALLOC(bits2, len, int);
   ALLOC(thresh, len, int);
   ALLOC(trim_offset, len, int);

   for (j=start;j<end;j++)
   {
      /* Below this threshold, we're sure not to allocate any PVQ bits */
      thresh[j] = IMAX((C)<<BITRES, (3*(m->eBands[j+1]-m->eBands[j])<<LM<<BITRES)>>4);
      /* Tilt of the allocation curve */
      trim_offset[j] = C*(m->eBands[j+1]-m->eBands[j])*(alloc_trim-5-LM)*(end-j-1)
            *(1<<(LM+BITRES))>>6;
      /* Giving less resolution to single-coefficient bands because they get
         more benefit from having one coarse value per coefficient*/
      if ((m->eBands[j+1]-m->eBands[j])<<LM==1)
         trim_offset[j] -= C<<BITRES;
   }
   lo = 1;
   hi = m->nbAllocVectors - 1;
   do
   {
      int done = 0;
      int psum = 0;
      int mid = (lo+hi) >> 1;
      for (j=end;j-->start;)
      {
         int bitsj;
         int N = m->eBands[j+1]-m->eBands[j];
         bitsj = C*N*m->allocVectors[mid*len+j]<<LM>>2;
         if (bitsj > 0)
            bitsj = IMAX(0, bitsj + trim_offset[j]);
         bitsj += offsets[j];
         if (bitsj >= thresh[j] || done)
         {
            done = 1;
            /* Don't allocate more than we can actually use */
            psum += IMIN(bitsj, cap[j]);
         } else {
            if (bitsj >= C<<BITRES)
               psum += C<<BITRES;
         }
      }
      if (psum > total)
         hi = mid - 1;
      else
         lo = mid + 1;
      /*printf ("lo = %d, hi = %d\n", lo, hi);*/
   }
   while (lo <= hi);
   hi = lo--;
   /*printf ("interp between %d and %d\n", lo, hi);*/
   for (j=start;j<end;j++)
   {
      int bits1j, bits2j;
      int N = m->eBands[j+1]-m->eBands[j];
      bits1j = C*N*m->allocVectors[lo*len+j]<<LM>>2;
      bits2j = hi>=m->nbAllocVectors ?
            cap[j] : C*N*m->allocVectors[hi*len+j]<<LM>>2;
      if (bits1j > 0)
         bits1j = IMAX(0, bits1j + trim_offset[j]);
      if (bits2j > 0)
         bits2j = IMAX(0, bits2j + trim_offset[j]);
      if (lo > 0)
         bits1j += offsets[j];
      bits2j += offsets[j];
      if (offsets[j]>0)
         skip_start = j;
      bits2j = IMAX(0,bits2j-bits1j);
      bits1[j] = bits1j;
      bits2[j] = bits2j;
   }
   codedBands = interp_bits2pulses(m, start, end, skip_start, bits1, bits2, thresh, cap,
         total, balance, skip_rsv, intensity, intensity_rsv, dual_stereo, dual_stereo_rsv,
         pulses, ebits, fine_priority, C, LM, ec, encode, prev, signalBandwidth);
   RESTORE_STACK;
   return codedBands;
}
#ifdef ENABLE_QEXT

static const unsigned char last_zero[3] = {64, 50, 0};
static const unsigned char last_cap[3] = {110, 60, 0};
static const unsigned char last_other[4] = {120, 112, 70, 0};

static void ec_enc_depth(ec_enc *enc, opus_int32 depth, opus_int32 cap, opus_int32 *last) {
   int sym = 3;
   if (depth==*last) sym = 2;
   if (depth==cap) sym = 1;
   if (depth==0) sym = 0;
   if (*last == 0) {
      ec_enc_icdf(enc, IMIN(sym, 2), last_zero, 7);
   } else if (*last == cap) {
      ec_enc_icdf(enc, IMIN(sym, 2), last_cap, 7);
   } else {
      ec_enc_icdf(enc, sym, last_other, 7);
   }
   /* We accept some redundancy if depth==last (for last different from 0 and cap). */
   if (sym == 3) ec_enc_uint(enc, depth-1, cap);
   *last = depth;
}

static int ec_dec_depth(ec_dec *dec, opus_int32 cap, opus_int32 *last) {
   int depth, sym;
   if (*last == 0) {
      sym = ec_dec_icdf(dec, last_zero, 7);
      if (sym==2) sym=3;
   } else if (*last == cap) {
      sym = ec_dec_icdf(dec, last_cap, 7);
      if (sym==2) sym=3;
   } else {
      sym = ec_dec_icdf(dec, last_other, 7);
   }
   if (sym==0) depth=0;
   else if (sym==1) depth=cap;
   else if (sym==2) depth=*last;
   else depth = 1 + ec_dec_uint(dec, cap);
   *last = depth;
   return depth;
}

#define MSWAP16(a,b) do {opus_val16 tmp = a;a=b;b=tmp;} while(0)
static opus_val16 median_of_5_val16(const opus_val16 *x)
{
   opus_val16 t0, t1, t2, t3, t4;
   t2 = x[2];
   if (x[0] > x[1])
   {
      t0 = x[1];
      t1 = x[0];
   } else {
      t0 = x[0];
      t1 = x[1];
   }
   if (x[3] > x[4])
   {
      t3 = x[4];
      t4 = x[3];
   } else {
      t3 = x[3];
      t4 = x[4];
   }
   if (t0 > t3)
   {
      MSWAP16(t0, t3);
      MSWAP16(t1, t4);
   }
   if (t2 > t1)
   {
      if (t1 < t3)
         return MIN16(t2, t3);
      else
         return MIN16(t4, t1);
   } else {
      if (t2 < t3)
         return MIN16(t1, t3);
      else
         return MIN16(t2, t4);
   }
}

void clt_compute_extra_allocation(const CELTMode *m, const CELTMode *qext_mode, int start, int end, int qext_end, const celt_glog *bandLogE, const celt_glog *qext_bandLogE,
      opus_int32 total, int *extra_pulses, int *extra_equant, int C, int LM, ec_ctx *ec, int encode, opus_val16 tone_freq, opus_val32 toneishness)
{
   int i;
   opus_int32 last=0;
   opus_val32 sum;
   opus_val32 fill;
   int iter;
   int tot_bands;
   int tot_samples;
   VARDECL(int, depth);
   VARDECL(opus_int32, cap);
#ifdef FUZZING
   float depth_std;
#endif
   SAVE_STACK;
#ifdef FUZZING
   depth_std = -10.f*log(1e-8+(float)rand()/(float)RAND_MAX);
   depth_std = FMAX(0, FMIN(48, depth_std));
#endif
   if (qext_mode != NULL) {
      celt_assert(end==m->nbEBands);
      tot_bands = end + qext_end;
      tot_samples = qext_mode->eBands[qext_end]*C<<LM;
   } else {
      tot_bands = end;
      tot_samples = (m->eBands[end]-m->eBands[start])*C<<LM;
   }
   ALLOC(cap, tot_bands, opus_int32);
   for (i=start;i<end;i++) cap[i] = 12;
   if (qext_mode != NULL) {
      for (i=0;i<qext_end;i++) cap[end+i] = 14;
   }
   if (total <= 0) {
      for (i=start;i<m->nbEBands+qext_end;i++) {
         extra_pulses[i] = extra_equant[i] = 0;
      }
      RESTORE_STACK;
      return;
   }
   ALLOC(depth, tot_bands, int);
   if (encode) {
      VARDECL(opus_val16, flatE);
      VARDECL(int, Ncoef);
      VARDECL(opus_val16, min);
      VARDECL(opus_val16, follower);

      ALLOC(flatE, tot_bands, opus_val16);
      ALLOC(min, tot_bands, opus_val16);
      ALLOC(Ncoef, tot_bands, int);
      for (i=start;i<end;i++) {
         Ncoef[i] = (m->eBands[i+1]-m->eBands[i])*C<<LM;
      }
      /* Remove the effect of band width, eMeans and pre-emphasis to compute the real (flat) spectrum. */
      for (i=start;i<end;i++) {
         flatE[i] = PSHR32(bandLogE[i] - GCONST(0.0625f)*m->logN[i] + SHL32(eMeans[i],DB_SHIFT-4) - GCONST(.0062f)*(i+5)*(i+5), DB_SHIFT-10);
         min[i] = 0;
      }
      if (C==2) {
         for (i=start;i<end;i++) {
            flatE[i] = MAXG(flatE[i], PSHR32(bandLogE[m->nbEBands+i] - GCONST(0.0625f)*m->logN[i] + SHL32(eMeans[i],DB_SHIFT-4) - GCONST(.0062f)*(i+5)*(i+5), DB_SHIFT-10));
         }
      }
      flatE[end-1] += QCONST16(2.f, 10);
      if (qext_mode != NULL) {
         opus_val16 min_depth = 0;
         /* If we have enough bits, give at least 1 bit of depth to all higher bands. */
         if (total >= 3*C*(qext_mode->eBands[qext_end]-qext_mode->eBands[start])<<LM<<BITRES && (toneishness < QCONST32(.98f, 29) || tone_freq > 1.33f))
            min_depth = QCONST16(1.f, 10);
         for (i=0;i<qext_end;i++) {
            Ncoef[end+i] = (qext_mode->eBands[i+1]-qext_mode->eBands[i])*C<<LM;
            min[end+i] = min_depth;
         }
         for (i=0;i<qext_end;i++) {
            flatE[end+i] = PSHR32(qext_bandLogE[i] - GCONST(0.0625f)*qext_mode->logN[i] + SHL32(eMeans[i],DB_SHIFT-4) - GCONST(.0062f)*(end+i+5)*(end+i+5), DB_SHIFT-10);
         }
         if (C==2) {
            for (i=0;i<qext_end;i++) {
               flatE[end+i] = MAXG(flatE[end+i], PSHR32(qext_bandLogE[NB_QEXT_BANDS+i] - GCONST(0.0625f)*qext_mode->logN[i] + SHL32(eMeans[i],DB_SHIFT-4) - GCONST(.0062f)*(end+i+5)*(end+i+5), DB_SHIFT-10));
            }
         }
      }
      ALLOC(follower, tot_bands, opus_val16);
      for (i=start+2;i<tot_bands-2;i++) {
         follower[i] = median_of_5_val16(&flatE[i-2]);
      }
      follower[start] = follower[start+1] = follower[start+2];
      follower[tot_bands-1] = follower[tot_bands-2] = follower[tot_bands-3];
      for (i=start+1;i<tot_bands;i++) {
         follower[i] = MAX16(follower[i], follower[i-1]-QCONST16(1.f, 10));
      }
      for (i=tot_bands-2;i>=start;i--) {
         follower[i] = MAX16(follower[i], follower[i+1]-QCONST16(1.f, 10));
      }
      for (i=start;i<tot_bands;i++) flatE[i] -= MULT16_16_Q15(Q15ONE-PSHR32(toneishness, 14), follower[i]);
      if (qext_mode != NULL) {
         for (i=0;i<qext_end;i++) flatE[end+i] = flatE[end+i] + QCONST16(3.f, 10) + QCONST16(.2f, 10)*i;
      }
      /* Approximate fill level assuming all bands contribute fully. */
      sum = 0;
      for (i=start;i<tot_bands;i++) {
         sum += MULT16_16(Ncoef[i], flatE[i]);
      }
      total >>= BITRES;
      fill = (SHL32(total, 10) + sum)/tot_samples;
      /* Iteratively refine the fill level considering the depth min and cap. */
      for (iter=0;iter<10;iter++) {
         sum = 0;
         for (i=start;i<tot_bands;i++)
            sum += Ncoef[i] * MIN32(SHL32(cap[i], 10), MAX32(min[i], flatE[i]-fill));
         fill -= (SHL32(total, 10) - sum)/tot_samples;
      }
      for (i=start;i<tot_bands;i++) {
#ifdef FIXED_POINT
         depth[i] = PSHR32(MIN32(SHL32(cap[i], 10), MAX32(min[i], flatE[i]-fill)), 10-2);
#else
         depth[i] = (int)floor(.5+4*MIN32(SHL32(cap[i], 10), MAX32(min[i], flatE[i]-fill)));
#endif
#ifdef FUZZING
         depth[i] = (int)-depth_std*log(1e-8+(float)rand()/(float)RAND_MAX);
         depth[i] = IMAX(0, IMIN(cap[i]<<2, depth[i]));
#endif
         if (ec_tell_frac(ec) + 80 < ec->storage*8<<BITRES)
            ec_enc_depth(ec, depth[i], 4*cap[i], &last);
         else
            depth[i] = 0;
      }
   } else {
      for (i=start;i<tot_bands;i++) {
         if (ec_tell_frac(ec) + 80 < ec->storage*8<<BITRES)
            depth[i] = ec_dec_depth(ec, 4*cap[i], &last);
         else
            depth[i] = 0;
      }
   }
   for (i=start;i<end;i++) {
      extra_equant[i] = (depth[i]+3)>>2;
      extra_pulses[i] = ((((m->eBands[i+1]-m->eBands[i])<<LM)-1)*C * depth[i] * (1<<BITRES) + 2)>>2;
   }
   if (qext_mode) {
      for (i=0;i<qext_end;i++) {
         extra_equant[end+i] = (depth[end+i]+3)>>2;
         extra_pulses[end+i] = ((((qext_mode->eBands[i+1]-qext_mode->eBands[i])<<LM)-1)*C * depth[end+i] * (1<<BITRES) + 2)>>2;
      }
   }
   RESTORE_STACK;
}
#endif
