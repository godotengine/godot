/* Copyright (c) 2010 Xiph.Org Foundation
 * Copyright (c) 2013 Parrot
 * Copyright (c) 2024 Arm Limited */
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

#include "kiss_fft.h"
#include "mathops.h"
#include "mdct.h"
#include "pitch.h"

#if defined(OPUS_HAVE_RTCD)

# if !defined(DISABLE_FLOAT_API)
#  if defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR)
void (*const CELT_FLOAT2INT16_IMPL[OPUS_ARCHMASK+1])(const float * OPUS_RESTRICT in, short * OPUS_RESTRICT out, int cnt) = {
  celt_float2int16_c,   /* ARMv4 */
  celt_float2int16_c,   /* EDSP */
  celt_float2int16_c,   /* Media */
  celt_float2int16_neon,/* NEON */
  celt_float2int16_neon /* DOTPROD */
};

int (*const OPUS_LIMIT2_CHECKWITHIN1_IMPL[OPUS_ARCHMASK+1])(float * samples, int cnt) = {
  opus_limit2_checkwithin1_c,   /* ARMv4 */
  opus_limit2_checkwithin1_c,   /* EDSP */
  opus_limit2_checkwithin1_c,   /* Media */
  opus_limit2_checkwithin1_neon,/* NEON */
  opus_limit2_checkwithin1_neon /* DOTPROD */
};
#  endif
# endif

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR)
opus_val32 (*const CELT_INNER_PROD_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *x, const opus_val16 *y, int N) = {
  celt_inner_prod_c,   /* ARMv4 */
  celt_inner_prod_c,   /* EDSP */
  celt_inner_prod_c,   /* Media */
  celt_inner_prod_neon,/* NEON */
  celt_inner_prod_neon /* DOTPROD */
};

void (*const DUAL_INNER_PROD_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
      int N, opus_val32 *xy1, opus_val32 *xy2) = {
  dual_inner_prod_c,   /* ARMv4 */
  dual_inner_prod_c,   /* EDSP */
  dual_inner_prod_c,   /* Media */
  dual_inner_prod_neon,/* NEON */
  dual_inner_prod_neon /* DOTPROD */
};
# endif

# if defined(FIXED_POINT)
#  if ((defined(OPUS_ARM_MAY_HAVE_NEON) && !defined(OPUS_ARM_PRESUME_NEON)) || \
    (defined(OPUS_ARM_MAY_HAVE_MEDIA) && !defined(OPUS_ARM_PRESUME_MEDIA)) || \
    (defined(OPUS_ARM_MAY_HAVE_EDSP) && !defined(OPUS_ARM_PRESUME_EDSP)))
opus_val32 (*const CELT_PITCH_XCORR_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *,
    const opus_val16 *, opus_val32 *, int, int, int) = {
  celt_pitch_xcorr_c,               /* ARMv4 */
  MAY_HAVE_EDSP(celt_pitch_xcorr),  /* EDSP */
  MAY_HAVE_MEDIA(celt_pitch_xcorr), /* Media */
  MAY_HAVE_NEON(celt_pitch_xcorr),  /* NEON */
  MAY_HAVE_NEON(celt_pitch_xcorr)   /* DOTPROD */
};

#  endif
# else /* !FIXED_POINT */
#  if defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR)
void (*const CELT_PITCH_XCORR_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *,
    const opus_val16 *, opus_val32 *, int, int, int) = {
  celt_pitch_xcorr_c,              /* ARMv4 */
  celt_pitch_xcorr_c,              /* EDSP */
  celt_pitch_xcorr_c,              /* Media */
  celt_pitch_xcorr_float_neon,     /* Neon */
  celt_pitch_xcorr_float_neon      /* DOTPROD */
};
#  endif
# endif /* FIXED_POINT */

#if defined(FIXED_POINT) && defined(OPUS_HAVE_RTCD) && \
 defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR)

void (*const XCORR_KERNEL_IMPL[OPUS_ARCHMASK + 1])(
         const opus_val16 *x,
         const opus_val16 *y,
         opus_val32       sum[4],
         int              len
) = {
  xcorr_kernel_c,                /* ARMv4 */
  xcorr_kernel_c,                /* EDSP */
  xcorr_kernel_c,                /* Media */
  xcorr_kernel_neon_fixed,       /* Neon */
  xcorr_kernel_neon_fixed        /* DOTPROD */
};

#endif

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
#  if defined(HAVE_ARM_NE10)
#   if defined(CUSTOM_MODES)
int (*const OPUS_FFT_ALLOC_ARCH_IMPL[OPUS_ARCHMASK+1])(kiss_fft_state *st) = {
   opus_fft_alloc_arch_c,        /* ARMv4 */
   opus_fft_alloc_arch_c,        /* EDSP */
   opus_fft_alloc_arch_c,        /* Media */
   opus_fft_alloc_arm_neon,      /* Neon with NE10 library support */
   opus_fft_alloc_arm_neon       /* DOTPROD with NE10 library support */
};

void (*const OPUS_FFT_FREE_ARCH_IMPL[OPUS_ARCHMASK+1])(kiss_fft_state *st) = {
   opus_fft_free_arch_c,         /* ARMv4 */
   opus_fft_free_arch_c,         /* EDSP */
   opus_fft_free_arch_c,         /* Media */
   opus_fft_free_arm_neon,       /* Neon with NE10 */
   opus_fft_free_arm_neon        /* DOTPROD with NE10 */
};
#   endif /* CUSTOM_MODES */

void (*const OPUS_FFT[OPUS_ARCHMASK+1])(const kiss_fft_state *cfg,
                                        const kiss_fft_cpx *fin,
                                        kiss_fft_cpx *fout) = {
   opus_fft_c,                   /* ARMv4 */
   opus_fft_c,                   /* EDSP */
   opus_fft_c,                   /* Media */
   opus_fft_neon,                /* Neon with NE10 */
   opus_fft_neon                 /* DOTPROD with NE10 */
};

void (*const OPUS_IFFT[OPUS_ARCHMASK+1])(const kiss_fft_state *cfg,
                                         const kiss_fft_cpx *fin,
                                         kiss_fft_cpx *fout) = {
   opus_ifft_c,                   /* ARMv4 */
   opus_ifft_c,                   /* EDSP */
   opus_ifft_c,                   /* Media */
   opus_ifft_neon,                /* Neon with NE10 */
   opus_ifft_neon                 /* DOTPROD with NE10 */
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
   clt_mdct_forward_neon,        /* Neon with NE10 */
   clt_mdct_forward_neon         /* DOTPROD with NE10 */
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
   clt_mdct_backward_neon,        /* Neon with NE10 */
   clt_mdct_backward_neon         /* DOTPROD with NE10 */
};

#  endif /* HAVE_ARM_NE10 */
# endif /* OPUS_ARM_MAY_HAVE_NEON_INTR */

#endif /* OPUS_HAVE_RTCD */
