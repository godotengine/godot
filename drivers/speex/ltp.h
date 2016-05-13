/* Copyright (C) 2002 Jean-Marc Valin */
/**
   @file ltp.h
   @brief Long-Term Prediction functions
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

#include <speex/speex_bits.h>
#include "arch.h"

/** LTP parameters. */
typedef struct {
   const signed char *gain_cdbk;
   int     gain_bits;
   int     pitch_bits;
} ltp_params;

#ifdef FIXED_POINT
#define gain_3tap_to_1tap(g) (ABS(g[1]) + (g[0]>0 ? g[0] : -SHR16(g[0],1)) + (g[2]>0 ? g[2] : -SHR16(g[2],1)))
#else
#define gain_3tap_to_1tap(g) (ABS(g[1]) + (g[0]>0 ? g[0] : -.5*g[0]) + (g[2]>0 ? g[2] : -.5*g[2]))
#endif

spx_word32_t inner_prod(const spx_word16_t *x, const spx_word16_t *y, int len);
void pitch_xcorr(const spx_word16_t *_x, const spx_word16_t *_y, spx_word32_t *corr, int len, int nb_pitch, char *stack);

void open_loop_nbest_pitch(spx_word16_t *sw, int start, int end, int len, int *pitch, spx_word16_t *gain, int N, char *stack);


/** Finds the best quantized 3-tap pitch predictor by analysis by synthesis */
int pitch_search_3tap(
spx_word16_t target[],                 /* Target vector */
spx_word16_t *sw,
spx_coef_t ak[],                     /* LPCs for this subframe */
spx_coef_t awk1[],                   /* Weighted LPCs #1 for this subframe */
spx_coef_t awk2[],                   /* Weighted LPCs #2 for this subframe */
spx_sig_t exc[],                    /* Overlapping codebook */
const void *par,
int   start,                    /* Smallest pitch value allowed */
int   end,                      /* Largest pitch value allowed */
spx_word16_t pitch_coef,               /* Voicing (pitch) coefficient */
int   p,                        /* Number of LPC coeffs */
int   nsf,                      /* Number of samples in subframe */
SpeexBits *bits,
char *stack,
spx_word16_t *exc2,
spx_word16_t *r,
int   complexity,
int   cdbk_offset,
int plc_tuning,
spx_word32_t *cumul_gain
);

/*Unquantize adaptive codebook and update pitch contribution*/
void pitch_unquant_3tap(
spx_word16_t exc[],             /* Input excitation */
spx_word32_t exc_out[],         /* Output excitation */
int   start,                    /* Smallest pitch value allowed */
int   end,                      /* Largest pitch value allowed */
spx_word16_t pitch_coef,        /* Voicing (pitch) coefficient */
const void *par,
int   nsf,                      /* Number of samples in subframe */
int *pitch_val,
spx_word16_t *gain_val,
SpeexBits *bits,
char *stack,
int lost,
int subframe_offset,
spx_word16_t last_pitch_gain,
int cdbk_offset
);

/** Forced pitch delay and gain */
int forced_pitch_quant(
spx_word16_t target[],                 /* Target vector */
spx_word16_t *sw,
spx_coef_t ak[],                     /* LPCs for this subframe */
spx_coef_t awk1[],                   /* Weighted LPCs #1 for this subframe */
spx_coef_t awk2[],                   /* Weighted LPCs #2 for this subframe */
spx_sig_t exc[],                    /* Excitation */
const void *par,
int   start,                    /* Smallest pitch value allowed */
int   end,                      /* Largest pitch value allowed */
spx_word16_t pitch_coef,               /* Voicing (pitch) coefficient */
int   p,                        /* Number of LPC coeffs */
int   nsf,                      /* Number of samples in subframe */
SpeexBits *bits,
char *stack,
spx_word16_t *exc2,
spx_word16_t *r,
int complexity,
int cdbk_offset,
int plc_tuning,
spx_word32_t *cumul_gain
);

/** Unquantize forced pitch delay and gain */
void forced_pitch_unquant(
spx_word16_t exc[],             /* Input excitation */
spx_word32_t exc_out[],         /* Output excitation */
int   start,                    /* Smallest pitch value allowed */
int   end,                      /* Largest pitch value allowed */
spx_word16_t pitch_coef,        /* Voicing (pitch) coefficient */
const void *par,
int   nsf,                      /* Number of samples in subframe */
int *pitch_val,
spx_word16_t *gain_val,
SpeexBits *bits,
char *stack,
int lost,
int subframe_offset,
spx_word16_t last_pitch_gain,
int cdbk_offset
);
