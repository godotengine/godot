/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/**
   @file vq.h
   @brief Vector quantisation of the residual
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

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef VQ_H
#define VQ_H

#include "entenc.h"
#include "entdec.h"
#include "modes.h"

#if (defined(OPUS_X86_MAY_HAVE_SSE2) && !defined(FIXED_POINT))
#include "x86/vq_sse.h"
#endif

#if defined(FIXED_POINT)
opus_val32 celt_inner_prod_norm(const celt_norm *x, const celt_norm *y, int len, int arch);
opus_val32 celt_inner_prod_norm_shift(const celt_norm *x, const celt_norm *y, int len, int arch);

void norm_scaleup(celt_norm *X, int N, int shift);
void norm_scaledown(celt_norm *X, int N, int shift);

#else
#define celt_inner_prod_norm celt_inner_prod
#define celt_inner_prod_norm_shift celt_inner_prod
#define norm_scaleup(X, N, shift)
#define norm_scaledown(X, N, shift)
#endif

void exp_rotation(celt_norm *X, int len, int dir, int stride, int K, int spread);

opus_val16 op_pvq_search_c(celt_norm *X, int *iy, int K, int N, int arch);

#if !defined(OVERRIDE_OP_PVQ_SEARCH)
#define op_pvq_search(x, iy, K, N, arch) \
    (op_pvq_search_c(x, iy, K, N, arch))
#endif

/** Algebraic pulse-vector quantiser. The signal x is replaced by the sum of
  * the pitch and a combination of pulses such that its norm is still equal
  * to 1. This is the function that will typically require the most CPU.
 * @param X Residual signal to quantise/encode (returns quantised version)
 * @param N Number of samples to encode
 * @param K Number of pulses to use
 * @param enc Entropy encoder state
 * @ret A mask indicating which blocks in the band received pulses
*/
unsigned alg_quant(celt_norm *X, int N, int K, int spread, int B, ec_enc *enc,
      opus_val32 gain, int resynth
      ARG_QEXT(ec_enc *ext_enc) ARG_QEXT(int extra_bits), int arch);

/** Algebraic pulse decoder
 * @param X Decoded normalised spectrum (returned)
 * @param N Number of samples to decode
 * @param K Number of pulses to use
 * @param dec Entropy decoder state
 * @ret A mask indicating which blocks in the band received pulses
 */
unsigned alg_unquant(celt_norm *X, int N, int K, int spread, int B,
      ec_dec *dec, opus_val32 gain
      ARG_QEXT(ec_enc *ext_dec) ARG_QEXT(int extra_bits));

void renormalise_vector(celt_norm *X, int N, opus_val32 gain, int arch);

opus_int32 stereo_itheta(const celt_norm *X, const celt_norm *Y, int stereo, int N, int arch);

unsigned cubic_quant(celt_norm *X, int N, int K, int B, ec_enc *enc, opus_val32 gain, int resynth);
unsigned cubic_unquant(celt_norm *X, int N, int K, int B, ec_dec *dec, opus_val32 gain);

#endif /* VQ_H */
