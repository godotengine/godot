/* Copyright (c) 2010 Xiph.Org Foundation
 * Copyright (c) 2013 Parrot */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "pitch.h"
#include "kiss_fft.h"
#include "mdct.h"

#if defined(OPUS_HAVE_RTCD)

# if defined(FIXED_POINT)
opus_val32 (*const CELT_PITCH_XCORR_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *,
    const opus_val16 *, opus_val32 *, int , int) = {
  celt_pitch_xcorr_c,               /* ARMv4 */
  MAY_HAVE_EDSP(celt_pitch_xcorr),  /* EDSP */
  MAY_HAVE_MEDIA(celt_pitch_xcorr), /* Media */
  MAY_HAVE_NEON(celt_pitch_xcorr)   /* NEON */
};
# else /* !FIXED_POINT */
#  if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void (*const CELT_PITCH_XCORR_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *,
    const opus_val16 *, opus_val32 *, int, int) = {
  celt_pitch_xcorr_c,              /* ARMv4 */
  celt_pitch_xcorr_c,              /* EDSP */
  celt_pitch_xcorr_c,              /* Media */
  celt_pitch_xcorr_float_neon      /* Neon */
};
#  endif
# endif /* FIXED_POINT */

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
#  if defined(HAVE_ARM_NE10)
#   if defined(CUSTOM_MODES)
int (*const OPUS_FFT_ALLOC_ARCH_IMPL[OPUS_ARCHMASK+1])(kiss_fft_state *st) = {
   opus_fft_alloc_arch_c,        /* ARMv4 */
   opus_fft_alloc_arch_c,        /* EDSP */
   opus_fft_alloc_arch_c,        /* Media */
   opus_fft_alloc_arm_neon       /* Neon with NE10 library support */
};

void (*const OPUS_FFT_FREE_ARCH_IMPL[OPUS_ARCHMASK+1])(kiss_fft_state *st) = {
   opus_fft_free_arch_c,         /* ARMv4 */
   opus_fft_free_arch_c,         /* EDSP */
   opus_fft_free_arch_c,         /* Media */
   opus_fft_free_arm_neon        /* Neon with NE10 */
};
#   endif /* CUSTOM_MODES */

void (*const OPUS_FFT[OPUS_ARCHMASK+1])(const kiss_fft_state *cfg,
                                        const kiss_fft_cpx *fin,
                                        kiss_fft_cpx *fout) = {
   opus_fft_c,                   /* ARMv4 */
   opus_fft_c,                   /* EDSP */
   opus_fft_c,                   /* Media */
   opus_fft_neon                 /* Neon with NE10 */
};

void (*const OPUS_IFFT[OPUS_ARCHMASK+1])(const kiss_fft_state *cfg,
                                         const kiss_fft_cpx *fin,
                                         kiss_fft_cpx *fout) = {
   opus_ifft_c,                   /* ARMv4 */
   opus_ifft_c,                   /* EDSP */
   opus_ifft_c,                   /* Media */
   opus_ifft_neon                 /* Neon with NE10 */
};

void (*const CLT_MDCT_FORWARD_IMPL[OPUS_ARCHMASK+1])(const mdct_lookup *l,
                                                     kiss_fft_scalar *in,
                                                     kiss_fft_scalar * OPUS_RESTRICT out,
                                                     const opus_val16 *window,
                                                     int overlap, int shift,
                                                     int stride, int arch) = {
   clt_mdct_forward_c,           /* ARMv4 */
   clt_mdct_forward_c,           /* EDSP */
   clt_mdct_forward_c,           /* Media */
   clt_mdct_forward_neon         /* Neon with NE10 */
};

void (*const CLT_MDCT_BACKWARD_IMPL[OPUS_ARCHMASK+1])(const mdct_lookup *l,
                                                      kiss_fft_scalar *in,
                                                      kiss_fft_scalar * OPUS_RESTRICT out,
                                                      const opus_val16 *window,
                                                      int overlap, int shift,
                                                      int stride, int arch) = {
   clt_mdct_backward_c,           /* ARMv4 */
   clt_mdct_backward_c,           /* EDSP */
   clt_mdct_backward_c,           /* Media */
   clt_mdct_backward_neon         /* Neon with NE10 */
};

#  endif /* HAVE_ARM_NE10 */
# endif /* OPUS_ARM_MAY_HAVE_NEON_INTR */

#endif /* OPUS_HAVE_RTCD */
