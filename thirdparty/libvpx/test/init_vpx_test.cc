/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "test/init_vpx_test.h"

#include "./vpx_config.h"

#if !CONFIG_SHARED
#include <string>
#include "gtest/gtest.h"
#if VPX_ARCH_ARM
#include "vpx_ports/arm.h"
#endif
#if VPX_ARCH_X86 || VPX_ARCH_X86_64
#include "vpx_ports/x86.h"
#endif
extern "C" {
#if CONFIG_VP8
extern void vp8_rtcd();
#endif  // CONFIG_VP8
#if CONFIG_VP9
extern void vp9_rtcd();
#endif  // CONFIG_VP9
extern void vpx_dsp_rtcd();
extern void vpx_scale_rtcd();
}

#if VPX_ARCH_ARM || VPX_ARCH_X86 || VPX_ARCH_X86_64
static void append_negative_gtest_filter(const char *str) {
  std::string flag_value = GTEST_FLAG_GET(filter);
  // Negative patterns begin with one '-' followed by a ':' separated list.
  if (flag_value.find('-') == std::string::npos) flag_value += '-';
  // OPT.* matches TEST() functions
  // OPT/* matches TEST_P() functions
  // OPT_* matches tests which have been manually sharded.
  // We do not match OPT* because of SSE/SSE2 collisions.
  const char *search_terminators = "./_";
  for (size_t pos = 0; pos < strlen(search_terminators); ++pos) {
    flag_value += ":";
    flag_value += str;
    flag_value += search_terminators[pos];
    flag_value += "*";
  }
  GTEST_FLAG_SET(filter, flag_value);
}
#endif  // VPX_ARCH_ARM || VPX_ARCH_X86 || VPX_ARCH_X86_64
#endif  // !CONFIG_SHARED

namespace libvpx_test {
void init_vpx_test() {
#if !CONFIG_SHARED
#if VPX_ARCH_AARCH64
  const int caps = arm_cpu_caps();
  if (!(caps & HAS_NEON_DOTPROD)) {
    append_negative_gtest_filter("NEON_DOTPROD");
  }
  if (!(caps & HAS_NEON_I8MM)) {
    append_negative_gtest_filter("NEON_I8MM");
  }
  if (!(caps & HAS_SVE)) {
    append_negative_gtest_filter("SVE");
  }
  if (!(caps & HAS_SVE2)) {
    append_negative_gtest_filter("SVE2");
  }
#elif VPX_ARCH_ARM
  const int caps = arm_cpu_caps();
  if (!(caps & HAS_NEON)) append_negative_gtest_filter("NEON");
#endif  // VPX_ARCH_ARM

#if VPX_ARCH_X86 || VPX_ARCH_X86_64
  const int simd_caps = x86_simd_caps();
  if (!(simd_caps & HAS_MMX)) append_negative_gtest_filter("MMX");
  if (!(simd_caps & HAS_SSE)) append_negative_gtest_filter("SSE");
  if (!(simd_caps & HAS_SSE2)) append_negative_gtest_filter("SSE2");
  if (!(simd_caps & HAS_SSE3)) append_negative_gtest_filter("SSE3");
  if (!(simd_caps & HAS_SSSE3)) {
    append_negative_gtest_filter("SSSE3");
  }
  if (!(simd_caps & HAS_SSE4_1)) {
    append_negative_gtest_filter("SSE4_1");
  }
  if (!(simd_caps & HAS_AVX)) append_negative_gtest_filter("AVX");
  if (!(simd_caps & HAS_AVX2)) append_negative_gtest_filter("AVX2");
  if (!(simd_caps & HAS_AVX512)) {
    append_negative_gtest_filter("AVX512");
  }
#endif  // VPX_ARCH_X86 || VPX_ARCH_X86_64

  // Shared library builds don't support whitebox tests that exercise internal
  // symbols.
#if CONFIG_VP8
  vp8_rtcd();
#endif  // CONFIG_VP8
#if CONFIG_VP9
  vp9_rtcd();
#endif  // CONFIG_VP9
  vpx_dsp_rtcd();
  vpx_scale_rtcd();
#endif  // !CONFIG_SHARED
}
}  // namespace libvpx_test
