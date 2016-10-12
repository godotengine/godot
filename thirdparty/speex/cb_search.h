/* Copyright (C) 2002 Jean-Marc Valin & David Rowe */
/**
   @file cb_search.h
   @brief Overlapped codebook search
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

#ifndef CB_SEARCH_H
#define CB_SEARCH_H

#include <speex/speex_bits.h>
#include "arch.h"

/** Split codebook parameters. */
typedef struct split_cb_params {
   int     subvect_size;
   int     nb_subvect;
   const signed char  *shape_cb;
   int     shape_bits;
   int     have_sign;
} split_cb_params;


void split_cb_search_shape_sign(
spx_word16_t target[],             /* target vector */
spx_coef_t ak[],                /* LPCs for this subframe */
spx_coef_t awk1[],              /* Weighted LPCs for this subframe */
spx_coef_t awk2[],              /* Weighted LPCs for this subframe */
const void *par,                /* Codebook/search parameters */
int   p,                        /* number of LPC coeffs */
int   nsf,                      /* number of samples in subframe */
spx_sig_t *exc,
spx_word16_t *r,
SpeexBits *bits,
char *stack,
int   complexity,
int   update_target
);

void split_cb_shape_sign_unquant(
spx_sig_t *exc,
const void *par,                /* non-overlapping codebook */
int   nsf,                      /* number of samples in subframe */
SpeexBits *bits,
char *stack,
spx_int32_t *seed
);


void noise_codebook_quant(
spx_word16_t target[],             /* target vector */
spx_coef_t ak[],                /* LPCs for this subframe */
spx_coef_t awk1[],              /* Weighted LPCs for this subframe */
spx_coef_t awk2[],              /* Weighted LPCs for this subframe */
const void *par,                /* Codebook/search parameters */
int   p,                        /* number of LPC coeffs */
int   nsf,                      /* number of samples in subframe */
spx_sig_t *exc,
spx_word16_t *r,
SpeexBits *bits,
char *stack,
int   complexity,
int   update_target
);


void noise_codebook_unquant(
spx_sig_t *exc,
const void *par,                /* non-overlapping codebook */
int   nsf,                      /* number of samples in subframe */
SpeexBits *bits,
char *stack,
spx_int32_t *seed
);

#endif
