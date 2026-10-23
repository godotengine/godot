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

#ifndef SILK_RESAMPLER_STRUCTS_H
#define SILK_RESAMPLER_STRUCTS_H

#define SILK_RESAMPLER_MAX_FIR_ORDER                 36
#define SILK_RESAMPLER_MAX_IIR_ORDER                 6

typedef struct _silk_resampler_state_struct{
    opus_int32       sIIR[ SILK_RESAMPLER_MAX_IIR_ORDER ]; /* this must be the first element of this struct */
    union{
        opus_int32   i32[ SILK_RESAMPLER_MAX_FIR_ORDER ];
        opus_int16   i16[ SILK_RESAMPLER_MAX_FIR_ORDER ];
    }                sFIR;
    opus_int16       delayBuf[ 96 ];
    opus_int         resampler_function;
    opus_int         batchSize;
    opus_int32       invRatio_Q16;
    opus_int         FIR_Order;
    opus_int         FIR_Fracs;
    opus_int         Fs_in_kHz;
    opus_int         Fs_out_kHz;
    opus_int         inputDelay;
    const opus_int16 *Coefs;
} silk_resampler_state_struct;

#endif /* SILK_RESAMPLER_STRUCTS_H */
