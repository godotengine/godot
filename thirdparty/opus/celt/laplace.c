/* Copyright (c) 2007 CSIRO
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

#include "laplace.h"
#include "mathops.h"

/* The minimum probability of an energy delta (out of 32768). */
#define LAPLACE_LOG_MINP (0)
#define LAPLACE_MINP (1<<LAPLACE_LOG_MINP)
/* The minimum number of guaranteed representable energy deltas (in one
    direction). */
#define LAPLACE_NMIN (16)

/* When called, decay is positive and at most 11456. */
static unsigned ec_laplace_get_freq1(unsigned fs0, int decay)
{
   unsigned ft;
   ft = 32768 - LAPLACE_MINP*(2*LAPLACE_NMIN) - fs0;
   return ft*(opus_int32)(16384-decay)>>15;
}

void ec_laplace_encode(ec_enc *enc, int *value, unsigned fs, int decay)
{
   unsigned fl;
   int val = *value;
   fl = 0;
   if (val)
   {
      int s;
      int i;
      s = -(val<0);
      val = (val+s)^s;
      fl = fs;
      fs = ec_laplace_get_freq1(fs, decay);
      /* Search the decaying part of the PDF.*/
      for (i=1; fs > 0 && i < val; i++)
      {
         fs *= 2;
         fl += fs+2*LAPLACE_MINP;
         fs = (fs*(opus_int32)decay)>>15;
      }
      /* Everything beyond that has probability LAPLACE_MINP. */
      if (!fs)
      {
         int di;
         int ndi_max;
         ndi_max = (32768-fl+LAPLACE_MINP-1)>>LAPLACE_LOG_MINP;
         ndi_max = (ndi_max-s)>>1;
         di = IMIN(val - i, ndi_max - 1);
         fl += (2*di+1+s)*LAPLACE_MINP;
         fs = IMIN(LAPLACE_MINP, 32768-fl);
         *value = (i+di+s)^s;
      }
      else
      {
         fs += LAPLACE_MINP;
         fl += fs&~s;
      }
      celt_assert(fl+fs<=32768);
      celt_assert(fs>0);
   }
   ec_encode_bin(enc, fl, fl+fs, 15);
}

int ec_laplace_decode(ec_dec *dec, unsigned fs, int decay)
{
   int val=0;
   unsigned fl;
   unsigned fm;
   fm = ec_decode_bin(dec, 15);
   fl = 0;
   if (fm >= fs)
   {
      val++;
      fl = fs;
      fs = ec_laplace_get_freq1(fs, decay)+LAPLACE_MINP;
      /* Search the decaying part of the PDF.*/
      while(fs > LAPLACE_MINP && fm >= fl+2*fs)
      {
         fs *= 2;
         fl += fs;
         fs = ((fs-2*LAPLACE_MINP)*(opus_int32)decay)>>15;
         fs += LAPLACE_MINP;
         val++;
      }
      /* Everything beyond that has probability LAPLACE_MINP. */
      if (fs <= LAPLACE_MINP)
      {
         int di;
         di = (fm-fl)>>(LAPLACE_LOG_MINP+1);
         val += di;
         fl += 2*di*LAPLACE_MINP;
      }
      if (fm < fl+fs)
         val = -val;
      else
         fl += fs;
   }
   celt_assert(fl<32768);
   celt_assert(fs>0);
   celt_assert(fl<=fm);
   celt_assert(fm<IMIN(fl+fs,32768));
   ec_dec_update(dec, fl, IMIN(fl+fs,32768), 32768);
   return val;
}

void ec_laplace_encode_p0(ec_enc *enc, int value, opus_uint16 p0, opus_uint16 decay)
{
   int s;
   opus_uint16 sign_icdf[3];
   sign_icdf[0] = 32768-p0;
   sign_icdf[1] = sign_icdf[0]/2;
   sign_icdf[2] = 0;
   s = value == 0 ? 0 : (value > 0 ? 1 : 2);
   ec_enc_icdf16(enc, s, sign_icdf, 15);
   value = abs(value);
   if (value)
   {
      int i;
      opus_uint16 icdf[8];
      icdf[0] = IMAX(7, decay);
      for (i=1;i<7;i++)
      {
         icdf[i] = IMAX(7-i, (icdf[i-1] * (opus_int32)decay) >> 15);
      }
      icdf[7] = 0;
      value--;
      do {
         ec_enc_icdf16(enc, IMIN(value, 7), icdf, 15);
         value -= 7;
      } while (value >= 0);
   }
}

int ec_laplace_decode_p0(ec_dec *dec, opus_uint16 p0, opus_uint16 decay)
{
   int s;
   int value;
   opus_uint16 sign_icdf[3];
   sign_icdf[0] = 32768-p0;
   sign_icdf[1] = sign_icdf[0]/2;
   sign_icdf[2] = 0;
   s = ec_dec_icdf16(dec, sign_icdf, 15);
   if (s==2) s = -1;
   if (s != 0)
   {
      int i;
      int v;
      opus_uint16 icdf[8];
      icdf[0] = IMAX(7, decay);
      for (i=1;i<7;i++)
      {
         icdf[i] = IMAX(7-i, (icdf[i-1] * (opus_int32)decay) >> 15);
      }
      icdf[7] = 0;
      value = 1;
      do {
         v = ec_dec_icdf16(dec, icdf, 15);
         value += v;
      } while (v == 7);
      return s*value;
   } else return 0;
}

#if 0

#include <stdio.h>
#define NB_VALS 10
#define DATA_SIZE 10000
int main() {
   ec_enc enc;
   ec_dec dec;
   unsigned char *ptr;
   int i;
   int decay, p0;
   int val[NB_VALS] = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
   /*for (i=0;i<NB_VALS;i++) {
      val[i] = -log(rand()/(float)RAND_MAX);
      if (rand()%2) val[i] = -val[i];
   }*/
   p0 = 16000;
   decay = 16000;
   ptr = (unsigned char *)malloc(DATA_SIZE);
   ec_enc_init(&enc,ptr,DATA_SIZE);
   for (i=0;i<NB_VALS;i++) {
      printf("%d ", val[i]);
   }
   printf("\n");
   for (i=0;i<NB_VALS;i++) {
      ec_laplace_encode_p0(&enc, val[i], p0, decay);
   }

   ec_enc_done(&enc);

   ec_dec_init(&dec,ec_get_buffer(&enc),ec_range_bytes(&enc));

   for (i=0;i<NB_VALS;i++) {
      val[i] = ec_laplace_decode_p0(&dec, p0, decay);
   }
   for (i=0;i<NB_VALS;i++) {
      printf("%d ", val[i]);
   }
   printf("\n");
}

#endif
