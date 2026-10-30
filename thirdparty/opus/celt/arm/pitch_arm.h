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

#if !defined(PITCH_ARM_H)
# define PITCH_ARM_H

# include "armcpu.h"

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
opus_val32 celt_inner_prod_neon(const opus_val16 *x, const opus_val16 *y, int N);
void dual_inner_prod_neon(const opus_val16 *x, const opus_val16 *y01,
        const opus_val16 *y02, int N, opus_val32 *xy1, opus_val32 *xy2);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_CELT_INNER_PROD (1)
#   define OVERRIDE_DUAL_INNER_PROD (1)
#   define celt_inner_prod(x, y, N, arch) ((void)(arch), PRESUME_NEON(celt_inner_prod)(x, y, N))
#   define dual_inner_prod(x, y01, y02, N, xy1, xy2, arch) ((void)(arch), PRESUME_NEON(dual_inner_prod)(x, y01, y02, N, xy1, xy2))
#  endif
# endif

# if !defined(OVERRIDE_CELT_INNER_PROD)
#  if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern opus_val32 (*const CELT_INNER_PROD_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *x, const opus_val16 *y, int N);
#   define OVERRIDE_CELT_INNER_PROD (1)
#   define celt_inner_prod(x, y, N, arch) ((*CELT_INNER_PROD_IMPL[(arch)&OPUS_ARCHMASK])(x, y, N))
#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_CELT_INNER_PROD (1)
#   define celt_inner_prod(x, y, N, arch) ((void)(arch), celt_inner_prod_neon(x, y, N))
#  endif
# endif

# if !defined(OVERRIDE_DUAL_INNER_PROD)
#  if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void (*const DUAL_INNER_PROD_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *x,
        const opus_val16 *y01, const opus_val16 *y02, int N, opus_val32 *xy1, opus_val32 *xy2);
#   define OVERRIDE_DUAL_INNER_PROD (1)
#   define dual_inner_prod(x, y01, y02, N, xy1, xy2, arch) ((*DUAL_INNER_PROD_IMPL[(arch)&OPUS_ARCHMASK])(x, y01, y02, N, xy1, xy2))
#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_DUAL_INNER_PROD (1)
#   define dual_inner_prod(x, y01, y02, N, xy1, xy2, arch) ((void)(arch), dual_inner_prod_neon(x, y01, y02, N, xy1, xy2))
#  endif
# endif

# if defined(FIXED_POINT)

#  if defined(OPUS_ARM_MAY_HAVE_NEON)
opus_val32 celt_pitch_xcorr_neon(const opus_val16 *_x, const opus_val16 *_y,
    opus_val32 *xcorr, int len, int max_pitch, int arch);
#  endif

#  if defined(OPUS_ARM_MAY_HAVE_MEDIA)
#   define celt_pitch_xcorr_media MAY_HAVE_EDSP(celt_pitch_xcorr)
#  endif

#  if defined(OPUS_ARM_MAY_HAVE_EDSP)
opus_val32 celt_pitch_xcorr_edsp(const opus_val16 *_x, const opus_val16 *_y,
    opus_val32 *xcorr, int len, int max_pitch, int arch);
#  endif

#  if defined(OPUS_HAVE_RTCD) && \
    ((defined(OPUS_ARM_MAY_HAVE_NEON) && !defined(OPUS_ARM_PRESUME_NEON)) || \
     (defined(OPUS_ARM_MAY_HAVE_MEDIA) && !defined(OPUS_ARM_PRESUME_MEDIA)) || \
     (defined(OPUS_ARM_MAY_HAVE_EDSP) && !defined(OPUS_ARM_PRESUME_EDSP)))
extern opus_val32
(*const CELT_PITCH_XCORR_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *,
      const opus_val16 *, opus_val32 *, int, int, int);
#   define OVERRIDE_PITCH_XCORR (1)
#   define celt_pitch_xcorr(_x, _y, xcorr, len, max_pitch, arch) \
  ((*CELT_PITCH_XCORR_IMPL[(arch)&OPUS_ARCHMASK])(_x, _y, \
        xcorr, len, max_pitch, arch))

#  elif defined(OPUS_ARM_PRESUME_EDSP) || \
    defined(OPUS_ARM_PRESUME_MEDIA) || \
    defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_PITCH_XCORR (1)
#   define celt_pitch_xcorr (PRESUME_NEON(celt_pitch_xcorr))

#  endif

#  if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void xcorr_kernel_neon_fixed(
                    const opus_val16 *x,
                    const opus_val16 *y,
                    opus_val32       sum[4],
                    int              len);
#  endif

#  if defined(OPUS_HAVE_RTCD) && \
    (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))

extern void (*const XCORR_KERNEL_IMPL[OPUS_ARCHMASK + 1])(
                    const opus_val16 *x,
                    const opus_val16 *y,
                    opus_val32       sum[4],
                    int              len);

#   define OVERRIDE_XCORR_KERNEL (1)
#   define xcorr_kernel(x, y, sum, len, arch) \
     ((*XCORR_KERNEL_IMPL[(arch) & OPUS_ARCHMASK])(x, y, sum, len))

#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_XCORR_KERNEL (1)
#   define xcorr_kernel(x, y, sum, len, arch) \
      ((void)arch, xcorr_kernel_neon_fixed(x, y, sum, len))

#  endif

#else /* Start !FIXED_POINT */
/* Float case */
#if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void celt_pitch_xcorr_float_neon(const opus_val16 *_x, const opus_val16 *_y,
                                 opus_val32 *xcorr, int len, int max_pitch, int arch);
#endif

#  if defined(OPUS_HAVE_RTCD) && \
    (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void
(*const CELT_PITCH_XCORR_IMPL[OPUS_ARCHMASK+1])(const opus_val16 *,
      const opus_val16 *, opus_val32 *, int, int, int);

#  define OVERRIDE_PITCH_XCORR (1)
#  define celt_pitch_xcorr(_x, _y, xcorr, len, max_pitch, arch) \
  ((*CELT_PITCH_XCORR_IMPL[(arch)&OPUS_ARCHMASK])(_x, _y, \
        xcorr, len, max_pitch, arch))

#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)

#   define OVERRIDE_PITCH_XCORR (1)
#   define celt_pitch_xcorr celt_pitch_xcorr_float_neon

#  endif

#endif /* end !FIXED_POINT */

#endif
