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

#include <stdlib.h>
#include "main_FLP.h"

/************************************************/
/* LPC analysis filter                          */
/* NB! State is kept internally and the         */
/* filter always starts with zero state         */
/* first Order output samples are set to zero   */
/************************************************/

/* 16th order LPC analysis filter, does not write first 16 samples */
static OPUS_INLINE void silk_LPC_analysis_filter16_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 16; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ]  * PredCoef[ 0 ]  +
                   s_ptr[ -1 ]  * PredCoef[ 1 ]  +
                   s_ptr[ -2 ]  * PredCoef[ 2 ]  +
                   s_ptr[ -3 ]  * PredCoef[ 3 ]  +
                   s_ptr[ -4 ]  * PredCoef[ 4 ]  +
                   s_ptr[ -5 ]  * PredCoef[ 5 ]  +
                   s_ptr[ -6 ]  * PredCoef[ 6 ]  +
                   s_ptr[ -7 ]  * PredCoef[ 7 ]  +
                   s_ptr[ -8 ]  * PredCoef[ 8 ]  +
                   s_ptr[ -9 ]  * PredCoef[ 9 ]  +
                   s_ptr[ -10 ] * PredCoef[ 10 ] +
                   s_ptr[ -11 ] * PredCoef[ 11 ] +
                   s_ptr[ -12 ] * PredCoef[ 12 ] +
                   s_ptr[ -13 ] * PredCoef[ 13 ] +
                   s_ptr[ -14 ] * PredCoef[ 14 ] +
                   s_ptr[ -15 ] * PredCoef[ 15 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 12th order LPC analysis filter, does not write first 12 samples */
static OPUS_INLINE void silk_LPC_analysis_filter12_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 12; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ]  * PredCoef[ 0 ]  +
                   s_ptr[ -1 ]  * PredCoef[ 1 ]  +
                   s_ptr[ -2 ]  * PredCoef[ 2 ]  +
                   s_ptr[ -3 ]  * PredCoef[ 3 ]  +
                   s_ptr[ -4 ]  * PredCoef[ 4 ]  +
                   s_ptr[ -5 ]  * PredCoef[ 5 ]  +
                   s_ptr[ -6 ]  * PredCoef[ 6 ]  +
                   s_ptr[ -7 ]  * PredCoef[ 7 ]  +
                   s_ptr[ -8 ]  * PredCoef[ 8 ]  +
                   s_ptr[ -9 ]  * PredCoef[ 9 ]  +
                   s_ptr[ -10 ] * PredCoef[ 10 ] +
                   s_ptr[ -11 ] * PredCoef[ 11 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 10th order LPC analysis filter, does not write first 10 samples */
static OPUS_INLINE void silk_LPC_analysis_filter10_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 10; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ]  +
                   s_ptr[ -6 ] * PredCoef[ 6 ]  +
                   s_ptr[ -7 ] * PredCoef[ 7 ]  +
                   s_ptr[ -8 ] * PredCoef[ 8 ]  +
                   s_ptr[ -9 ] * PredCoef[ 9 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 8th order LPC analysis filter, does not write first 8 samples */
static OPUS_INLINE void silk_LPC_analysis_filter8_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 8; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ]  +
                   s_ptr[ -6 ] * PredCoef[ 6 ]  +
                   s_ptr[ -7 ] * PredCoef[ 7 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 6th order LPC analysis filter, does not write first 6 samples */
static OPUS_INLINE void silk_LPC_analysis_filter6_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 6; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/************************************************/
/* LPC analysis filter                          */
/* NB! State is kept internally and the         */
/* filter always starts with zero state         */
/* first Order output samples are set to zero   */
/************************************************/
void silk_LPC_analysis_filter_FLP(
    silk_float                      r_LPC[],                            /* O    LPC residual signal                         */
    const silk_float                PredCoef[],                         /* I    LPC coefficients                            */
    const silk_float                s[],                                /* I    Input signal                                */
    const opus_int                  length,                             /* I    Length of input signal                      */
    const opus_int                  Order                               /* I    LPC order                                   */
)
{
    celt_assert( Order <= length );

    switch( Order ) {
        case 6:
            silk_LPC_analysis_filter6_FLP(  r_LPC, PredCoef, s, length );
        break;

        case 8:
            silk_LPC_analysis_filter8_FLP(  r_LPC, PredCoef, s, length );
        break;

        case 10:
            silk_LPC_analysis_filter10_FLP( r_LPC, PredCoef, s, length );
        break;

        case 12:
            silk_LPC_analysis_filter12_FLP( r_LPC, PredCoef, s, length );
        break;

        case 16:
            silk_LPC_analysis_filter16_FLP( r_LPC, PredCoef, s, length );
        break;

        default:
            celt_assert( 0 );
        break;
    }

    /* Set first Order output samples to zero */
    silk_memset( r_LPC, 0, Order * sizeof( silk_float ) );
}

