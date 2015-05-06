/* Copyright (C) 2002 Jean-Marc Valin */
/**
   @file filters.h
   @brief Various analysis/synthesis filters
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

#ifndef FILTERS_H
#define FILTERS_H

#include "arch.h"

spx_word16_t compute_rms(const spx_sig_t *x, int len);
spx_word16_t compute_rms16(const spx_word16_t *x, int len);
void signal_mul(const spx_sig_t *x, spx_sig_t *y, spx_word32_t scale, int len);
void signal_div(const spx_word16_t *x, spx_word16_t *y, spx_word32_t scale, int len);

#ifdef FIXED_POINT

int normalize16(const spx_sig_t *x, spx_word16_t *y, spx_sig_t max_scale, int len);

#endif


#define HIGHPASS_NARROWBAND 0
#define HIGHPASS_WIDEBAND 2
#define HIGHPASS_INPUT 0
#define HIGHPASS_OUTPUT 1
#define HIGHPASS_IRS 4

void highpass(const spx_word16_t *x, spx_word16_t *y, int len, int filtID, spx_mem_t *mem);


void qmf_decomp(const spx_word16_t *xx, const spx_word16_t *aa, spx_word16_t *, spx_word16_t *y2, int N, int M, spx_word16_t *mem, char *stack);
void qmf_synth(const spx_word16_t *x1, const spx_word16_t *x2, const spx_word16_t *a, spx_word16_t *y, int N, int M, spx_word16_t *mem1, spx_word16_t *mem2, char *stack);

void filter_mem16(const spx_word16_t *x, const spx_coef_t *num, const spx_coef_t *den, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack);
void iir_mem16(const spx_word16_t *x, const spx_coef_t *den, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack);
void fir_mem16(const spx_word16_t *x, const spx_coef_t *num, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack);

/* Apply bandwidth expansion on LPC coef */
void bw_lpc(spx_word16_t , const spx_coef_t *lpc_in, spx_coef_t *lpc_out, int order);
void sanitize_values32(spx_word32_t *vec, spx_word32_t min_val, spx_word32_t max_val, int len);


void syn_percep_zero16(const spx_word16_t *xx, const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord, char *stack);
void residue_percep_zero16(const spx_word16_t *xx, const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord, char *stack);

void compute_impulse_response(const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord, char *stack);

void multicomb(
spx_word16_t *exc,          /*decoded excitation*/
spx_word16_t *new_exc,      /*enhanced excitation*/
spx_coef_t *ak,           /*LPC filter coefs*/
int p,               /*LPC order*/
int nsf,             /*sub-frame size*/
int pitch,           /*pitch period*/
int max_pitch,   /*pitch gain (3-tap)*/
spx_word16_t  comb_gain,    /*gain of comb filter*/
char *stack
);

#endif
