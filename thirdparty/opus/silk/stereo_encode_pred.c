/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
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

#include "main.h"

/* Entropy code the mid/side quantization indices */
void silk_stereo_encode_pred(
    ec_enc                      *psRangeEnc,                    /* I/O  Compressor data structure                   */
    opus_int8                   ix[ 2 ][ 3 ]                    /* I    Quantization indices                        */
)
{
    opus_int   n;

    /* Entropy coding */
    n = 5 * ix[ 0 ][ 2 ] + ix[ 1 ][ 2 ];
    silk_assert( n < 25 );
    ec_enc_icdf( psRangeEnc, n, silk_stereo_pred_joint_iCDF, 8 );
    for( n = 0; n < 2; n++ ) {
        silk_assert( ix[ n ][ 0 ] < 3 );
        silk_assert( ix[ n ][ 1 ] < STEREO_QUANT_SUB_STEPS );
        ec_enc_icdf( psRangeEnc, ix[ n ][ 0 ], silk_uniform3_iCDF, 8 );
        ec_enc_icdf( psRangeEnc, ix[ n ][ 1 ], silk_uniform5_iCDF, 8 );
    }
}

/* Entropy code the mid-only flag */
void silk_stereo_encode_mid_only(
    ec_enc                      *psRangeEnc,                    /* I/O  Compressor data structure                   */
    opus_int8                   mid_only_flag
)
{
    /* Encode flag that only mid channel is coded */
    ec_enc_icdf( psRangeEnc, mid_only_flag, silk_stereo_only_code_mid_iCDF, 8 );
}
