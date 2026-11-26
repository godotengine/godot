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

#ifndef SILK_LPC_INV_PRED_GAIN_ARM_H
# define SILK_LPC_INV_PRED_GAIN_ARM_H

# include "celt/arm/armcpu.h"

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
opus_int32 silk_LPC_inverse_pred_gain_neon(         /* O   Returns inverse prediction gain in energy domain, Q30        */
    const opus_int16            *A_Q12,             /* I   Prediction coefficients, Q12 [order]                         */
    const opus_int              order               /* I   Prediction order                                             */
);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_silk_LPC_inverse_pred_gain            (1)
#   define silk_LPC_inverse_pred_gain(A_Q12, order, arch) ((void)(arch), PRESUME_NEON(silk_LPC_inverse_pred_gain)(A_Q12, order))
#  endif
# endif

# if !defined(OVERRIDE_silk_LPC_inverse_pred_gain)
/*Is run-time CPU detection enabled on this platform?*/
#  if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern opus_int32 (*const SILK_LPC_INVERSE_PRED_GAIN_IMPL[OPUS_ARCHMASK+1])(const opus_int16 *A_Q12, const opus_int order);
#   define OVERRIDE_silk_LPC_inverse_pred_gain            (1)
#   define silk_LPC_inverse_pred_gain(A_Q12, order, arch) ((*SILK_LPC_INVERSE_PRED_GAIN_IMPL[(arch)&OPUS_ARCHMASK])(A_Q12, order))
#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_silk_LPC_inverse_pred_gain            (1)
#   define silk_LPC_inverse_pred_gain(A_Q12, order, arch) ((void)(arch), silk_LPC_inverse_pred_gain_neon(A_Q12, order))
#  endif
# endif

#endif /* end SILK_LPC_INV_PRED_GAIN_ARM_H */
