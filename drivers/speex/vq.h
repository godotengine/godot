/* Copyright (C) 2002 Jean-Marc Valin */
/**
   @file vq.h
   @brief Vector quantization
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

#ifndef VQ_H
#define VQ_H

#include "arch.h"

int scal_quant(spx_word16_t in, const spx_word16_t *boundary, int entries);
int scal_quant32(spx_word32_t in, const spx_word32_t *boundary, int entries);

#ifdef _USE_SSE
#include <xmmintrin.h>
void vq_nbest(spx_word16_t *in, const __m128 *codebook, int len, int entries, __m128 *E, int N, int *nbest, spx_word32_t *best_dist, char *stack);

void vq_nbest_sign(spx_word16_t *in, const __m128 *codebook, int len, int entries, __m128 *E, int N, int *nbest, spx_word32_t *best_dist, char *stack);
#else
void vq_nbest(spx_word16_t *in, const spx_word16_t *codebook, int len, int entries, spx_word32_t *E, int N, int *nbest, spx_word32_t *best_dist, char *stack);

void vq_nbest_sign(spx_word16_t *in, const spx_word16_t *codebook, int len, int entries, spx_word32_t *E, int N, int *nbest, spx_word32_t *best_dist, char *stack);
#endif

#endif
