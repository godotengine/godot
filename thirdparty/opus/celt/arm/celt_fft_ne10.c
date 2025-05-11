/* Copyright (c) 2015 Xiph.Org Foundation
   Written by Viswanath Puttagunta */
/**
   @file celt_fft_ne10.c
   @brief ARM Neon optimizations for fft using NE10 library
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

#ifndef SKIP_CONFIG_H
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#endif

#include <NE10_dsp.h>
#include "os_support.h"
#include "kiss_fft.h"
#include "stack_alloc.h"

#if !defined(FIXED_POINT)
# define NE10_FFT_ALLOC_C2C_TYPE_NEON ne10_fft_alloc_c2c_float32_neon
# define NE10_FFT_CFG_TYPE_T ne10_fft_cfg_float32_t
# define NE10_FFT_STATE_TYPE_T ne10_fft_state_float32_t
# define NE10_FFT_DESTROY_C2C_TYPE ne10_fft_destroy_c2c_float32
# define NE10_FFT_CPX_TYPE_T ne10_fft_cpx_float32_t
# define NE10_FFT_C2C_1D_TYPE_NEON ne10_fft_c2c_1d_float32_neon
#else
# define NE10_FFT_ALLOC_C2C_TYPE_NEON(nfft) ne10_fft_alloc_c2c_int32_neon(nfft)
# define NE10_FFT_CFG_TYPE_T ne10_fft_cfg_int32_t
# define NE10_FFT_STATE_TYPE_T ne10_fft_state_int32_t
# define NE10_FFT_DESTROY_C2C_TYPE ne10_fft_destroy_c2c_int32
# define NE10_FFT_DESTROY_C2C_TYPE ne10_fft_destroy_c2c_int32
# define NE10_FFT_CPX_TYPE_T ne10_fft_cpx_int32_t
# define NE10_FFT_C2C_1D_TYPE_NEON ne10_fft_c2c_1d_int32_neon
#endif

#if defined(CUSTOM_MODES)

/* nfft lengths in NE10 that support scaled fft */
# define NE10_FFTSCALED_SUPPORT_MAX 4
static const int ne10_fft_scaled_support[NE10_FFTSCALED_SUPPORT_MAX] = {
   480, 240, 120, 60
};

int opus_fft_alloc_arm_neon(kiss_fft_state *st)
{
   int i;
   size_t memneeded = sizeof(struct arch_fft_state);

   st->arch_fft = (arch_fft_state *)opus_alloc(memneeded);
   if (!st->arch_fft)
      return -1;

   for (i = 0; i < NE10_FFTSCALED_SUPPORT_MAX; i++) {
      if(st->nfft == ne10_fft_scaled_support[i])
         break;
   }
   if (i == NE10_FFTSCALED_SUPPORT_MAX) {
      /* This nfft length (scaled fft) is not supported in NE10 */
      st->arch_fft->is_supported = 0;
      st->arch_fft->priv = NULL;
   }
   else {
      st->arch_fft->is_supported = 1;
      st->arch_fft->priv = (void *)NE10_FFT_ALLOC_C2C_TYPE_NEON(st->nfft);
      if (st->arch_fft->priv == NULL) {
         return -1;
      }
   }
   return 0;
}

void opus_fft_free_arm_neon(kiss_fft_state *st)
{
   NE10_FFT_CFG_TYPE_T cfg;

   if (!st->arch_fft)
      return;

   cfg = (NE10_FFT_CFG_TYPE_T)st->arch_fft->priv;
   if (cfg)
      NE10_FFT_DESTROY_C2C_TYPE(cfg);
   opus_free(st->arch_fft);
}
#endif

void opus_fft_neon(const kiss_fft_state *st,
                   const kiss_fft_cpx *fin,
                   kiss_fft_cpx *fout)
{
   NE10_FFT_STATE_TYPE_T state;
   NE10_FFT_CFG_TYPE_T cfg = &state;
   VARDECL(NE10_FFT_CPX_TYPE_T, buffer);
   SAVE_STACK;
   ALLOC(buffer, st->nfft, NE10_FFT_CPX_TYPE_T);

   if (!st->arch_fft->is_supported) {
      /* This nfft length (scaled fft) not supported in NE10 */
      opus_fft_c(st, fin, fout);
   }
   else {
      memcpy((void *)cfg, st->arch_fft->priv, sizeof(NE10_FFT_STATE_TYPE_T));
      state.buffer = (NE10_FFT_CPX_TYPE_T *)&buffer[0];
#if !defined(FIXED_POINT)
      state.is_forward_scaled = 1;

      NE10_FFT_C2C_1D_TYPE_NEON((NE10_FFT_CPX_TYPE_T *)fout,
                                (NE10_FFT_CPX_TYPE_T *)fin,
                                cfg, 0);
#else
      NE10_FFT_C2C_1D_TYPE_NEON((NE10_FFT_CPX_TYPE_T *)fout,
                                (NE10_FFT_CPX_TYPE_T *)fin,
                                cfg, 0, 1);
#endif
   }
   RESTORE_STACK;
}

void opus_ifft_neon(const kiss_fft_state *st,
                    const kiss_fft_cpx *fin,
                    kiss_fft_cpx *fout)
{
   NE10_FFT_STATE_TYPE_T state;
   NE10_FFT_CFG_TYPE_T cfg = &state;
   VARDECL(NE10_FFT_CPX_TYPE_T, buffer);
   SAVE_STACK;
   ALLOC(buffer, st->nfft, NE10_FFT_CPX_TYPE_T);

   if (!st->arch_fft->is_supported) {
      /* This nfft length (scaled fft) not supported in NE10 */
      opus_ifft_c(st, fin, fout);
   }
   else {
      memcpy((void *)cfg, st->arch_fft->priv, sizeof(NE10_FFT_STATE_TYPE_T));
      state.buffer = (NE10_FFT_CPX_TYPE_T *)&buffer[0];
#if !defined(FIXED_POINT)
      state.is_backward_scaled = 0;

      NE10_FFT_C2C_1D_TYPE_NEON((NE10_FFT_CPX_TYPE_T *)fout,
                                (NE10_FFT_CPX_TYPE_T *)fin,
                                cfg, 1);
#else
      NE10_FFT_C2C_1D_TYPE_NEON((NE10_FFT_CPX_TYPE_T *)fout,
                                (NE10_FFT_CPX_TYPE_T *)fin,
                                cfg, 1, 0);
#endif
   }
   RESTORE_STACK;
}
