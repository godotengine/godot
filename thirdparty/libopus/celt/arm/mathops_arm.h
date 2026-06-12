/* Copyright (c) 2024 Arm Limited */
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

#if !defined(MATHOPS_ARM_H)
# define MATHOPS_ARM_H

#include "armcpu.h"
#include "cpu_support.h"
#include "opus_defines.h"

# if !defined(DISABLE_FLOAT_API) && defined(OPUS_ARM_MAY_HAVE_NEON_INTR)

#include <arm_neon.h>

static inline int32x4_t vroundf(float32x4_t x)
{
#  if defined(__aarch64__) || (defined(__ARM_ARCH) && __ARM_ARCH >= 8)
    return vcvtaq_s32_f32(x);
#  else
    uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000));
    uint32x4_t bias = vdupq_n_u32(0x3F000000);
    return vcvtq_s32_f32(vaddq_f32(x, vreinterpretq_f32_u32(vorrq_u32(bias, sign))));
#  endif
}

static inline float vminvf(float32x4_t a)
{
#if defined(__aarch64__)
   return vminvq_f32(a);
#else
    float32x2_t xy = vmin_f32(vget_low_f32(a), vget_high_f32(a));
    float x = vget_lane_f32(xy, 0);
    float y = vget_lane_f32(xy, 1);
    return x < y ? x : y;
#endif
}

static inline float vmaxvf(float32x4_t a)
{
#if defined(__aarch64__)
   return vmaxvq_f32(a);
#else
    float32x2_t xy = vmax_f32(vget_low_f32(a), vget_high_f32(a));
    float x = vget_lane_f32(xy, 0);
    float y = vget_lane_f32(xy, 1);
    return x > y ? x : y;
#endif
}

void celt_float2int16_neon(const float * OPUS_RESTRICT in, short * OPUS_RESTRICT out, int cnt);
#  if defined(OPUS_HAVE_RTCD) && \
    (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void
(*const CELT_FLOAT2INT16_IMPL[OPUS_ARCHMASK+1])(const float * OPUS_RESTRICT in, short * OPUS_RESTRICT out, int cnt);

#   define OVERRIDE_FLOAT2INT16 (1)
#   define celt_float2int16(in, out, cnt, arch) \
      ((*CELT_FLOAT2INT16_IMPL[(arch)&OPUS_ARCHMASK])(in, out, cnt))

#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_FLOAT2INT16 (1)
#   define celt_float2int16(in, out, cnt, arch) ((void)(arch), celt_float2int16_neon(in, out, cnt))
#  endif

int opus_limit2_checkwithin1_neon(float * samples, int cnt);
#  if defined(OPUS_HAVE_RTCD) && \
      (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern int (*const OPUS_LIMIT2_CHECKWITHIN1_IMPL[OPUS_ARCHMASK+1])(float * samples, int cnt);

#   define OVERRIDE_LIMIT2_CHECKWITHIN1 (1)
#   define opus_limit2_checkwithin1(samples, cnt, arch) \
   ((*OPUS_LIMIT2_CHECKWITHIN1_IMPL[(arch)&OPUS_ARCHMASK])(samples, cnt))

#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_LIMIT2_CHECKWITHIN1 (1)
#   define opus_limit2_checkwithin1(samples, cnt, arch) ((void)(arch), opus_limit2_checkwithin1_neon(samples, cnt))
#  endif
# endif

#endif /* MATHOPS_ARM_H */
