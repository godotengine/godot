/***********************************************************************
Copyright (c) 2017 Google Inc.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifndef SILK_WARPED_AUTOCORRELATION_FIX_ARM_H
# define SILK_WARPED_AUTOCORRELATION_FIX_ARM_H

# include "celt/arm/armcpu.h"

# if defined(FIXED_POINT)

#  if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void silk_warped_autocorrelation_FIX_neon(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_silk_warped_autocorrelation_FIX (1)
#   define silk_warped_autocorrelation_FIX(corr, scale, input, warping_Q16, length, order, arch) \
    ((void)(arch), PRESUME_NEON(silk_warped_autocorrelation_FIX)(corr, scale, input, warping_Q16, length, order))
#  endif
#  endif

#  if !defined(OVERRIDE_silk_warped_autocorrelation_FIX)
/*Is run-time CPU detection enabled on this platform?*/
#   if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void (*const SILK_WARPED_AUTOCORRELATION_FIX_IMPL[OPUS_ARCHMASK+1])(opus_int32*, opus_int*, const opus_int16*, const opus_int, const opus_int, const opus_int);
#    define OVERRIDE_silk_warped_autocorrelation_FIX (1)
#    define silk_warped_autocorrelation_FIX(corr, scale, input, warping_Q16, length, order, arch) \
    ((*SILK_WARPED_AUTOCORRELATION_FIX_IMPL[(arch)&OPUS_ARCHMASK])(corr, scale, input, warping_Q16, length, order))
#   elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#    define OVERRIDE_silk_warped_autocorrelation_FIX (1)
#    define silk_warped_autocorrelation_FIX(corr, scale, input, warping_Q16, length, order, arch) \
    ((void)(arch), silk_warped_autocorrelation_FIX_neon(corr, scale, input, warping_Q16, length, order))
#   endif
#  endif

# endif /* end FIXED_POINT */

#endif /* end SILK_WARPED_AUTOCORRELATION_FIX_ARM_H */
