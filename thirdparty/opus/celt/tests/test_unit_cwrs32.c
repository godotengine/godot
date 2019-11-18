/* Copyright (c) 2008-2011 Xiph.Org Foundation, Mozilla Corporation,
                           Gregory Maxwell
   Written by Jean-Marc Valin, Gregory Maxwell, and Timothy B. Terriberry */
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

#include <stdio.h>
#include <string.h>

#ifndef CUSTOM_MODES
#define CUSTOM_MODES
#else
#define TEST_CUSTOM_MODES
#endif

#define CELT_C
#include "stack_alloc.h"
#include "entenc.c"
#include "entdec.c"
#include "entcode.c"
#include "cwrs.c"
#include "mathops.c"
#include "rate.h"

#define NMAX (240)
#define KMAX (128)

#ifdef TEST_CUSTOM_MODES

#define NDIMS (44)
static const int pn[NDIMS]={
   2,   3,   4,   5,   6,   7,   8,   9,  10,
  11,  12,  13,  14,  15,  16,  18,  20,  22,
  24,  26,  28,  30,  32,  36,  40,  44,  48,
  52,  56,  60,  64,  72,  80,  88,  96, 104,
 112, 120, 128, 144, 160, 176, 192, 208
};
static const int pkmax[NDIMS]={
 128, 128, 128, 128,  88,  52,  36,  26,  22,
  18,  16,  15,  13,  12,  12,  11,  10,   9,
   9,   8,   8,   7,   7,   7,   7,   6,   6,
   6,   6,   6,   5,   5,   5,   5,   5,   5,
   4,   4,   4,   4,   4,   4,   4,   4
};

#else /* TEST_CUSTOM_MODES */

#define NDIMS (22)
static const int pn[NDIMS]={
   2,   3,   4,   6,   8,   9,  11,  12,  16,
  18,  22,  24,  32,  36,  44,  48,  64,  72,
  88,  96, 144, 176
};
static const int pkmax[NDIMS]={
 128, 128, 128,  88,  36,  26,  18,  16,  12,
  11,   9,   9,   7,   7,   6,   6,   5,   5,
   5,   5,   4,   4
};

#endif

int main(void){
  int t;
  int n;
  ALLOC_STACK;
  for(t=0;t<NDIMS;t++){
    int pseudo;
    n=pn[t];
    for(pseudo=1;pseudo<41;pseudo++)
    {
      int k;
#if defined(SMALL_FOOTPRINT)
      opus_uint32 uu[KMAX+2U];
#endif
      opus_uint32 inc;
      opus_uint32 nc;
      opus_uint32 i;
      k=get_pulses(pseudo);
      if (k>pkmax[t])break;
      printf("Testing CWRS with N=%i, K=%i...\n",n,k);
#if defined(SMALL_FOOTPRINT)
      nc=ncwrs_urow(n,k,uu);
#else
      nc=CELT_PVQ_V(n,k);
#endif
      inc=nc/20000;
      if(inc<1)inc=1;
      for(i=0;i<nc;i+=inc){
#if defined(SMALL_FOOTPRINT)
        opus_uint32 u[KMAX+2U];
#endif
        int         y[NMAX];
        int         sy;
        opus_uint32 v;
        opus_uint32 ii;
        int         j;
#if defined(SMALL_FOOTPRINT)
        memcpy(u,uu,(k+2U)*sizeof(*u));
        cwrsi(n,k,i,y,u);
#else
        cwrsi(n,k,i,y);
#endif
        sy=0;
        for(j=0;j<n;j++)sy+=abs(y[j]);
        if(sy!=k){
          fprintf(stderr,"N=%d Pulse count mismatch in cwrsi (%d!=%d).\n",
           n,sy,k);
          return 99;
        }
        /*printf("%6u of %u:",i,nc);
        for(j=0;j<n;j++)printf(" %+3i",y[j]);
        printf(" ->");*/
#if defined(SMALL_FOOTPRINT)
        ii=icwrs(n,k,&v,y,u);
#else
        ii=icwrs(n,y);
        v=CELT_PVQ_V(n,k);
#endif
        if(ii!=i){
          fprintf(stderr,"Combination-index mismatch (%lu!=%lu).\n",
           (long)ii,(long)i);
          return 1;
        }
        if(v!=nc){
          fprintf(stderr,"Combination count mismatch (%lu!=%lu).\n",
           (long)v,(long)nc);
          return 2;
        }
        /*printf(" %6u\n",i);*/
      }
      /*printf("\n");*/
    }
  }
  return 0;
}
