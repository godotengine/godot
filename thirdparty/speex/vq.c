/* Copyright (C) 2002 Jean-Marc Valin
   File: vq.c
   Vector quantization

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "config.h"


#include "vq.h"
#include "stack_alloc.h"
#include "arch.h"

#ifdef _USE_SSE
#include <xmmintrin.h>
#include "vq_sse.h"
#elif defined(SHORTCUTS) && (defined(ARM4_ASM) || defined(ARM5E_ASM))
#include "vq_arm4.h"
#elif defined(BFIN_ASM)
#include "vq_bfin.h"
#endif


int scal_quant(spx_word16_t in, const spx_word16_t *boundary, int entries)
{
   int i=0;
   while (i<entries-1 && in>boundary[0])
   {
      boundary++;
      i++;
   }
   return i;
}

int scal_quant32(spx_word32_t in, const spx_word32_t *boundary, int entries)
{
   int i=0;
   while (i<entries-1 && in>boundary[0])
   {
      boundary++;
      i++;
   }
   return i;
}


#ifndef OVERRIDE_VQ_NBEST
/*Finds the indices of the n-best entries in a codebook*/
void vq_nbest(spx_word16_t *in, const spx_word16_t *codebook, int len, int entries, spx_word32_t *E, int N, int *nbest, spx_word32_t *best_dist, char *stack)
{
   int i,j,k,used;
   used = 0;
   for (i=0;i<entries;i++)
   {
      spx_word32_t dist=0;
      for (j=0;j<len;j++)
         dist = MAC16_16(dist,in[j],*codebook++);
#ifdef FIXED_POINT
      dist=SUB32(SHR32(E[i],1),dist);
#else
      dist=.5f*E[i]-dist;
#endif
      if (i<N || dist<best_dist[N-1])
      {
         for (k=N-1; (k >= 1) && (k > used || dist < best_dist[k-1]); k--)
         {
            best_dist[k]=best_dist[k-1];
            nbest[k] = nbest[k-1];
         }
         best_dist[k]=dist;
         nbest[k]=i;
         used++;
      }
   }
}
#endif




#ifndef OVERRIDE_VQ_NBEST_SIGN
/*Finds the indices of the n-best entries in a codebook with sign*/
void vq_nbest_sign(spx_word16_t *in, const spx_word16_t *codebook, int len, int entries, spx_word32_t *E, int N, int *nbest, spx_word32_t *best_dist, char *stack)
{
   int i,j,k, sign, used;
   used=0;
   for (i=0;i<entries;i++)
   {
      spx_word32_t dist=0;
      for (j=0;j<len;j++)
         dist = MAC16_16(dist,in[j],*codebook++);
      if (dist>0)
      {
         sign=0;
         dist=-dist;
      } else
      {
         sign=1;
      }
#ifdef FIXED_POINT
      dist = ADD32(dist,SHR32(E[i],1));
#else
      dist = ADD32(dist,.5f*E[i]);
#endif
      if (i<N || dist<best_dist[N-1])
      {
         for (k=N-1; (k >= 1) && (k > used || dist < best_dist[k-1]); k--)
         {
            best_dist[k]=best_dist[k-1];
            nbest[k] = nbest[k-1];
         }
         best_dist[k]=dist;
         nbest[k]=i;
         used++;
         if (sign)
            nbest[k]+=entries;
      }
   }
}
#endif
