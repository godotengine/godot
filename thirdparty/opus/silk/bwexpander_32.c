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

#include "SigProc_FIX.h"

/* Chirp (bandwidth expand) LP AR filter.
   This logic is reused in _celt_lpc(). Any bug fixes should also be applied there. */
void silk_bwexpander_32(
    opus_int32                  *ar,                /* I/O  AR filter to be expanded (without leading 1)                */
    const opus_int              d,                  /* I    Length of ar                                                */
    opus_int32                  chirp_Q16           /* I    Chirp factor in Q16                                         */
)
{
    opus_int   i;
    opus_int32 chirp_minus_one_Q16 = chirp_Q16 - 65536;

    for( i = 0; i < d - 1; i++ ) {
        ar[ i ]    = silk_SMULWW( chirp_Q16, ar[ i ] );
        chirp_Q16 += silk_RSHIFT_ROUND( silk_MUL( chirp_Q16, chirp_minus_one_Q16 ), 16 );
    }
    ar[ d - 1 ] = silk_SMULWW( chirp_Q16, ar[ d - 1 ] );
}

