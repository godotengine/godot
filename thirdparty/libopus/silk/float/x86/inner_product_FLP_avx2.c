/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
              2023 Amazon
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "SigProc_FLP.h"
#include <immintrin.h>


/* inner product of two silk_float arrays, with result as double */
double silk_inner_product_FLP_avx2(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
)
{
    opus_int i;
    __m256d accum1, accum2;
    double   result;

    /* 4x unrolled loop */
    result = 0.0;
    accum1 = accum2 = _mm256_setzero_pd();
    for( i = 0; i < dataSize - 7; i += 8 ) {
        __m128  x1f, x2f;
        __m256d x1d, x2d;
        x1f = _mm_loadu_ps( &data1[ i ] );
        x2f = _mm_loadu_ps( &data2[ i ] );
        x1d = _mm256_cvtps_pd( x1f );
        x2d = _mm256_cvtps_pd( x2f );
        accum1 = _mm256_fmadd_pd( x1d, x2d, accum1 );
        x1f = _mm_loadu_ps( &data1[ i + 4 ] );
        x2f = _mm_loadu_ps( &data2[ i + 4 ] );
        x1d = _mm256_cvtps_pd( x1f );
        x2d = _mm256_cvtps_pd( x2f );
        accum2 = _mm256_fmadd_pd( x1d, x2d, accum2 );
    }
    for( ; i < dataSize - 3; i += 4 ) {
        __m128  x1f, x2f;
        __m256d x1d, x2d;
        x1f = _mm_loadu_ps( &data1[ i ] );
        x2f = _mm_loadu_ps( &data2[ i ] );
        x1d = _mm256_cvtps_pd( x1f );
        x2d = _mm256_cvtps_pd( x2f );
        accum1 = _mm256_fmadd_pd( x1d, x2d, accum1 );
    }
    accum1 = _mm256_add_pd(accum1, accum2);
    accum1 = _mm256_add_pd(accum1, _mm256_permute2f128_pd(accum1, accum1, 1));
    accum1 = _mm256_hadd_pd(accum1,accum1);
    result = _mm256_cvtsd_f64(accum1);

    /* add any remaining products */
    for( ; i < dataSize; i++ ) {
        result += data1[ i ] * (double)data2[ i ];
    }

    return result;
}
