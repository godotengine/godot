/* Copyright (C) 2004 Jean-Marc Valin */
/**
   @file vq_sse.h
   @brief SSE-optimized vq routine
*/
/*
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

#define OVERRIDE_VQ_NBEST
void vq_nbest(spx_word16_t *_in, const __m128 *codebook, int len, int entries, __m128 *E, int N, int *nbest, spx_word32_t *best_dist, char *stack)
{
   int i,j,k,used;
   VARDECL(float *dist);
   VARDECL(__m128 *in);
   __m128 half;
   used = 0;
   ALLOC(dist, entries, float);
   half = _mm_set_ps1(.5f);
   ALLOC(in, len, __m128);
   for (i=0;i<len;i++)
      in[i] = _mm_set_ps1(_in[i]);
   for (i=0;i<entries>>2;i++)
   {
      __m128 d = _mm_mul_ps(E[i], half);
      for (j=0;j<len;j++)
         d = _mm_sub_ps(d, _mm_mul_ps(in[j], *codebook++));
      _mm_storeu_ps(dist+4*i, d);
   }
   for (i=0;i<entries;i++)
   {
      if (i<N || dist[i]<best_dist[N-1])
      {
         for (k=N-1; (k >= 1) && (k > used || dist[i] < best_dist[k-1]); k--)
         {
            best_dist[k]=best_dist[k-1];
            nbest[k] = nbest[k-1];
         }
         best_dist[k]=dist[i];
         nbest[k]=i;
         used++;
      }
   }
}




#define OVERRIDE_VQ_NBEST_SIGN
void vq_nbest_sign(spx_word16_t *_in, const __m128 *codebook, int len, int entries, __m128 *E, int N, int *nbest, spx_word32_t *best_dist, char *stack)
{
   int i,j,k,used;
   VARDECL(float *dist);
   VARDECL(__m128 *in);
   __m128 half;
   used = 0;
   ALLOC(dist, entries, float);
   half = _mm_set_ps1(.5f);
   ALLOC(in, len, __m128);
   for (i=0;i<len;i++)
      in[i] = _mm_set_ps1(_in[i]);
   for (i=0;i<entries>>2;i++)
   {
      __m128 d = _mm_setzero_ps();
      for (j=0;j<len;j++)
         d = _mm_add_ps(d, _mm_mul_ps(in[j], *codebook++));
      _mm_storeu_ps(dist+4*i, d);
   }
   for (i=0;i<entries;i++)
   {
      int sign;
      if (dist[i]>0)
      {
         sign=0;
         dist[i]=-dist[i];
      } else
      {
         sign=1;
      }
      dist[i] += .5f*((float*)E)[i];
      if (i<N || dist[i]<best_dist[N-1])
      {
         for (k=N-1; (k >= 1) && (k > used || dist[i] < best_dist[k-1]); k--)
         {
            best_dist[k]=best_dist[k-1];
            nbest[k] = nbest[k-1];
         }
         best_dist[k]=dist[i];
         nbest[k]=i;
         used++;
         if (sign)
            nbest[k]+=entries;
      }
   }
}
