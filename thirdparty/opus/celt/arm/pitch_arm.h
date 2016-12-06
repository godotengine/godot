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

# if defined(FIXED_POINT)

#  if defined(OPUS_ARM_MAY_HAVE_NEON)
opus_val32 celt_pitch_xcorr_neon(const opus_val16 *_x, const opus_val16 *_y,
    opus_val32 *xcorr, int len, int max_pitch);
#  endif

#  if defined(OPUS_ARM_MAY_HAVE_MEDIA)
#   define celt_pitch_xcorr_media MAY_HAVE_EDSP(celt_pitch_xcorr)
#  endif

#  if defined(OPUS_ARM_MAY_HAVE_EDSP)
opus_val32 celt_pitch_xcorr_edsp(const opus_val16 *_x, const opus_val16 *_y,
    opus_val32 *xcorr, int len, int max_pitch);
#  endif

#  if !defined(OPUS_HAVE_RTCD)
#   define OVERRIDE_PITCH_XCORR (1)
#   define celt_pitch_xcorr(_x, _y, xcorr, len, max_pitch, arch) \
  ((void)(arch),PRESUME_NEON(celt_pitch_xcorr)(_x, _y, xcorr, len, max_pitch))
#  endif

#else /* Start !FIXED_POINT */
/* Float case */
#if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void celt_pitch_xcorr_float_neon(const opus_val16 *_x, const opus_val16 *_y,
                                 opus_val32 *xcorr, int len, int max_pitch);
#if !defined(OPUS_HAVE_RTCD) || defined(OPUS_ARM_PRESUME_NEON_INTR)
#define OVERRIDE_PITCH_XCORR (1)
#   define celt_pitch_xcorr(_x, _y, xcorr, len, max_pitch, arch) \
   ((void)(arch),celt_pitch_xcorr_float_neon(_x, _y, xcorr, len, max_pitch))
#endif
#endif

#endif /* end !FIXED_POINT */
#endif
