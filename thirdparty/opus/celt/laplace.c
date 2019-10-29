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
