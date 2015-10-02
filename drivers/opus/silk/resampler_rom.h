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

#ifndef SILK_FIX_RESAMPLER_ROM_H
#define SILK_FIX_RESAMPLER_ROM_H

#ifdef  __cplusplus
extern "C"
{
#endif

#include "typedef.h"
#include "resampler_structs.h"

#define RESAMPLER_DOWN_ORDER_FIR0               18
#define RESAMPLER_DOWN_ORDER_FIR1               24
#define RESAMPLER_DOWN_ORDER_FIR2               36
#define RESAMPLER_ORDER_FIR_12                  8

/* Tables for 2x downsampler */
static const opus_int16 silk_resampler_down2_0 = 9872;
static const opus_int16 silk_resampler_down2_1 = 39809 - 65536;

/* Tables for 2x upsampler, high quality */
static const opus_int16 silk_resampler_up2_hq_0[ 3 ] = { 1746, 14986, 39083 - 65536 };
static const opus_int16 silk_resampler_up2_hq_1[ 3 ] = { 6854, 25769, 55542 - 65536 };

/* Tables with IIR and FIR coefficients for fractional downsamplers */
extern const opus_int16 silk_Resampler_3_4_COEFS[ 2 + 3 * RESAMPLER_DOWN_ORDER_FIR0 / 2 ];
extern const opus_int16 silk_Resampler_2_3_COEFS[ 2 + 2 * RESAMPLER_DOWN_ORDER_FIR0 / 2 ];
extern const opus_int16 silk_Resampler_1_2_COEFS[ 2 +     RESAMPLER_DOWN_ORDER_FIR1 / 2 ];
extern const opus_int16 silk_Resampler_1_3_COEFS[ 2 +     RESAMPLER_DOWN_ORDER_FIR2 / 2 ];
extern const opus_int16 silk_Resampler_1_4_COEFS[ 2 +     RESAMPLER_DOWN_ORDER_FIR2 / 2 ];
extern const opus_int16 silk_Resampler_1_6_COEFS[ 2 +     RESAMPLER_DOWN_ORDER_FIR2 / 2 ];
extern const opus_int16 silk_Resampler_2_3_COEFS_LQ[ 2 + 2 * 2 ];

/* Table with interplation fractions of 1/24, 3/24, ..., 23/24 */
extern const opus_int16 silk_resampler_frac_FIR_12[ 12 ][ RESAMPLER_ORDER_FIR_12 / 2 ];

#ifdef  __cplusplus
}
#endif

#endif /* SILK_FIX_RESAMPLER_ROM_H */
