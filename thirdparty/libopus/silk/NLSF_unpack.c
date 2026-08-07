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

/* Unpack predictor values and indices for entropy coding tables */
void silk_NLSF_unpack(
          opus_int16            ec_ix[],                        /* O    Indices to entropy tables [ LPC_ORDER ]     */
          opus_uint8            pred_Q8[],                      /* O    LSF predictor [ LPC_ORDER ]                 */
    const silk_NLSF_CB_struct   *psNLSF_CB,                     /* I    Codebook object                             */
    const opus_int              CB1_index                       /* I    Index of vector in first LSF codebook       */
)
{
    opus_int   i;
    opus_uint8 entry;
    const opus_uint8 *ec_sel_ptr;

    ec_sel_ptr = &psNLSF_CB->ec_sel[ CB1_index * psNLSF_CB->order / 2 ];
    for( i = 0; i < psNLSF_CB->order; i += 2 ) {
        entry = *ec_sel_ptr++;
        ec_ix  [ i     ] = silk_SMULBB( silk_RSHIFT( entry, 1 ) & 7, 2 * NLSF_QUANT_MAX_AMPLITUDE + 1 );
        pred_Q8[ i     ] = psNLSF_CB->pred_Q8[ i + ( entry & 1 ) * ( psNLSF_CB->order - 1 ) ];
        ec_ix  [ i + 1 ] = silk_SMULBB( silk_RSHIFT( entry, 5 ) & 7, 2 * NLSF_QUANT_MAX_AMPLITUDE + 1 );
        pred_Q8[ i + 1 ] = psNLSF_CB->pred_Q8[ i + ( silk_RSHIFT( entry, 4 ) & 1 ) * ( psNLSF_CB->order - 1 ) + 1 ];
    }
}
