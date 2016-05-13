/* Copyright (C) 2002 Jean-Marc Valin */
/**
   @file ltp_sse.h
   @brief Long-Term Prediction functions (SSE version)
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

#include <xmmintrin.h>

#define OVERRIDE_INNER_PROD
float inner_prod(const float *a, const float *b, int len)
{
   int i;
   float ret;
   __m128 sum = _mm_setzero_ps();
   for (i=0;i<(len>>2);i+=2)
   {
      sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(a+0), _mm_loadu_ps(b+0)));
      sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(a+4), _mm_loadu_ps(b+4)));
      a += 8;
      b += 8;
   }
   sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
   sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
   _mm_store_ss(&ret, sum);
   return ret;
}

#define OVERRIDE_PITCH_XCORR
void pitch_xcorr(const float *_x, const float *_y, float *corr, int len, int nb_pitch, char *stack)
{
   int i, offset;
   VARDECL(__m128 *x);
   VARDECL(__m128 *y);
   int N, L;
   N = len>>2;
   L = nb_pitch>>2;
   ALLOC(x, N, __m128);
   ALLOC(y, N+L, __m128);
   for (i=0;i<N;i++)
      x[i] = _mm_loadu_ps(_x+(i<<2));
   for (offset=0;offset<4;offset++)
   {
      for (i=0;i<N+L;i++)
         y[i] = _mm_loadu_ps(_y+(i<<2)+offset);
      for (i=0;i<L;i++)
      {
         int j;
         __m128 sum, *xx, *yy;
         sum = _mm_setzero_ps();
         yy = y+i;
         xx = x;
         for (j=0;j<N;j+=2)
         {
            sum = _mm_add_ps(sum, _mm_mul_ps(xx[0], yy[0]));
            sum = _mm_add_ps(sum, _mm_mul_ps(xx[1], yy[1]));
            xx += 2;
            yy += 2;
         }
         sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
         sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
         _mm_store_ss(corr+nb_pitch-1-(i<<2)-offset, sum);
      }
   }
}
