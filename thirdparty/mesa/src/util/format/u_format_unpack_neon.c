/*
 * Copyright © 2021 Google LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "util/detect_arch.h"
#include "util/format/u_format.h"

#if (DETECT_ARCH_AARCH64 || DETECT_ARCH_ARM) && !defined(NO_FORMAT_ASM) && !defined(__SOFTFP__)

/* armhf builds default to vfp, not neon, and refuses to compile neon intrinsics
 * unless you tell it "no really".
 */
#if DETECT_ARCH_ARM
#pragma GCC target ("fpu=neon")
#endif

#include <arm_neon.h>
#include "u_format_pack.h"
#include "util/u_cpu_detect.h"

static void
util_format_b8g8r8a8_unorm_unpack_rgba_8unorm_neon(uint8_t *restrict dst, const uint8_t *restrict src, unsigned width)
{
   while (width >= 16) {
      uint8x16x4_t load = vld4q_u8(src);
      uint8x16x4_t swap = { .val = { load.val[2], load.val[1], load.val[0], load.val[3] } };
      vst4q_u8(dst, swap);
      width -= 16;
      dst += 16 * 4;
      src += 16 * 4;
   }
   if (width)
      util_format_b8g8r8a8_unorm_unpack_rgba_8unorm(dst, src, width);
}

static const struct util_format_unpack_description util_format_unpack_descriptions_neon[] = {
   [PIPE_FORMAT_B8G8R8A8_UNORM] = {
      .unpack_rgba_8unorm = &util_format_b8g8r8a8_unorm_unpack_rgba_8unorm_neon,
      .unpack_rgba = &util_format_b8g8r8a8_unorm_unpack_rgba_float,
   },
};

const struct util_format_unpack_description *
util_format_unpack_description_neon(enum pipe_format format)
{
   /* CPU detect for NEON support.  On arm64, it's implied. */
#if DETECT_ARCH_ARM
   if (!util_get_cpu_caps()->has_neon)
      return NULL;
#endif

   if (format >= ARRAY_SIZE(util_format_unpack_descriptions_neon))
      return NULL;

   if (!util_format_unpack_descriptions_neon[format].unpack_rgba)
      return NULL;

   return &util_format_unpack_descriptions_neon[format];
}

#endif /* DETECT_ARCH_AARCH64 | DETECT_ARCH_ARM */
