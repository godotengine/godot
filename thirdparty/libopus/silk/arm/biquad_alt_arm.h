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

#ifndef SILK_BIQUAD_ALT_ARM_H
# define SILK_BIQUAD_ALT_ARM_H

# include "celt/arm/armcpu.h"

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void silk_biquad_alt_stride2_neon(
    const opus_int16            *in,                /* I     input signal                                               */
    const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
    const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
    opus_int32                  *S,                 /* I/O   State vector [4]                                           */
    opus_int16                  *out,               /* O     output signal                                              */
    const opus_int32            len                 /* I     signal length (must be even)                               */
);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_silk_biquad_alt_stride2                   (1)
#   define silk_biquad_alt_stride2(in, B_Q28, A_Q28, S, out, len, arch) ((void)(arch), PRESUME_NEON(silk_biquad_alt_stride2)(in, B_Q28, A_Q28, S, out, len))
#  endif
# endif

# if !defined(OVERRIDE_silk_biquad_alt_stride2)
/*Is run-time CPU detection enabled on this platform?*/
#  if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void (*const SILK_BIQUAD_ALT_STRIDE2_IMPL[OPUS_ARCHMASK+1])(
        const opus_int16            *in,                /* I     input signal                                               */
        const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
        const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
        opus_int32                  *S,                 /* I/O   State vector [4]                                           */
        opus_int16                  *out,               /* O     output signal                                              */
        const opus_int32            len                 /* I     signal length (must be even)                               */
    );
#   define OVERRIDE_silk_biquad_alt_stride2                  (1)
#   define silk_biquad_alt_stride2(in, B_Q28, A_Q28, S, out, len, arch) ((*SILK_BIQUAD_ALT_STRIDE2_IMPL[(arch)&OPUS_ARCHMASK])(in, B_Q28, A_Q28, S, out, len))
#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_silk_biquad_alt_stride2                  (1)
#   define silk_biquad_alt_stride2(in, B_Q28, A_Q28, S, out, len, arch) ((void)(arch), silk_biquad_alt_stride2_neon(in, B_Q28, A_Q28, S, out, len))
#  endif
# endif

#endif /* end SILK_BIQUAD_ALT_ARM_H */
