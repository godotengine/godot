/***********************************************************************
Copyright (C) 2014 Vidyo
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
#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "NSQ.h"

#if defined(OPUS_HAVE_RTCD)

# if (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && \
 !defined(OPUS_ARM_PRESUME_NEON_INTR))

/*There is no table for silk_noise_shape_quantizer_short_prediction because the
   NEON version takes different parameters than the C version.
  Instead RTCD is done via if statements at the call sites.
  See NSQ_neon.h for details.*/

opus_int32
 (*const SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL[OPUS_ARCHMASK+1])(
 const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef,
 opus_int order) = {
  silk_NSQ_noise_shape_feedback_loop_c,    /* ARMv4 */
  silk_NSQ_noise_shape_feedback_loop_c,    /* EDSP */
  silk_NSQ_noise_shape_feedback_loop_c,    /* Media */
  silk_NSQ_noise_shape_feedback_loop_neon, /* NEON */
};

# endif

#endif /* OPUS_HAVE_RTCD */
