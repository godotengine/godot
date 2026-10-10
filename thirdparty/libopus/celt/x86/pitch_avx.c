/* Copyright (c) 2023 Amazon */
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


#include <immintrin.h>
#include "x86cpu.h"
#include "pitch.h"

#if defined(OPUS_X86_MAY_HAVE_AVX2) && !defined(FIXED_POINT)

/* Like the "regular" xcorr_kernel(), but computes 8 results at a time. */
static void xcorr_kernel_avx(const float *x, const float *y, float sum[8], int len)
{
    __m256 xsum0, xsum1, xsum2, xsum3, xsum4, xsum5, xsum6, xsum7;
    xsum7 = xsum6 = xsum5 = xsum4 = xsum3 = xsum2 = xsum1 = xsum0 = _mm256_setzero_ps();
    int i;
    __m256 x0;
    /* Compute 8 inner products using partial sums. */
    for (i=0;i<len-7;i+=8)
    {
        x0 = _mm256_loadu_ps(x+i);
        xsum0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y+i  ), xsum0);
        xsum1 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y+i+1), xsum1);
        xsum2 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y+i+2), xsum2);
        xsum3 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y+i+3), xsum3);
        xsum4 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y+i+4), xsum4);
        xsum5 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y+i+5), xsum5);
        xsum6 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y+i+6), xsum6);
        xsum7 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y+i+7), xsum7);
    }
    if (i != len) {
        static const int mask[15] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
        __m256i m;
        m = _mm256_loadu_si256((__m256i*)(void*)(mask + 7+i-len));
        x0 = _mm256_maskload_ps(x+i, m);
        xsum0 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y+i  , m), xsum0);
        xsum1 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y+i+1, m), xsum1);
        xsum2 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y+i+2, m), xsum2);
        xsum3 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y+i+3, m), xsum3);
        xsum4 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y+i+4, m), xsum4);
        xsum5 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y+i+5, m), xsum5);
        xsum6 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y+i+6, m), xsum6);
        xsum7 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y+i+7, m), xsum7);
    }
    /* 8 horizontal adds. */
    /* Compute [0 4] [1 5] [2 6] [3 7] */
    xsum0 = _mm256_add_ps(_mm256_permute2f128_ps(xsum0, xsum4, 2<<4), _mm256_permute2f128_ps(xsum0, xsum4, 1 | (3<<4)));
    xsum1 = _mm256_add_ps(_mm256_permute2f128_ps(xsum1, xsum5, 2<<4), _mm256_permute2f128_ps(xsum1, xsum5, 1 | (3<<4)));
    xsum2 = _mm256_add_ps(_mm256_permute2f128_ps(xsum2, xsum6, 2<<4), _mm256_permute2f128_ps(xsum2, xsum6, 1 | (3<<4)));
    xsum3 = _mm256_add_ps(_mm256_permute2f128_ps(xsum3, xsum7, 2<<4), _mm256_permute2f128_ps(xsum3, xsum7, 1 | (3<<4)));
    /* Compute [0 1 4 5] [2 3 6 7] */
    xsum0 = _mm256_hadd_ps(xsum0, xsum1);
    xsum1 = _mm256_hadd_ps(xsum2, xsum3);
    /* Compute [0 1 2 3 4 5 6 7] */
    xsum0 = _mm256_hadd_ps(xsum0, xsum1);
    _mm256_storeu_ps(sum, xsum0);
}

void celt_pitch_xcorr_avx2(const float *_x, const float *_y, float *xcorr, int len, int max_pitch, int arch)
{
   int i;
   celt_assert(max_pitch>0);
   (void)arch;
   for (i=0;i<max_pitch-7;i+=8)
   {
      xcorr_kernel_avx(_x, _y+i, &xcorr[i], len);
   }
   for (;i<max_pitch;i++)
   {
      xcorr[i] = celt_inner_prod(_x, _y+i, len, arch);
   }
}

#endif
