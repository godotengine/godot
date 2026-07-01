/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
//  Test and time VPX intra-predictor functions

#include <stdio.h>
#include <string.h>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/init_vpx_test.h"
#include "test/md5_helper.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/vpx_timer.h"

// -----------------------------------------------------------------------------

namespace {

using VpxPredFunc = void (*)(uint8_t *dst, ptrdiff_t y_stride,
                             const uint8_t *above, const uint8_t *left);

const int kBPS = 32;
const int kTotalPixels = 32 * kBPS;
const int kNumVp9IntraPredFuncs = 13;
const char *kVp9IntraPredNames[kNumVp9IntraPredFuncs] = {
  "DC_PRED",   "DC_LEFT_PRED", "DC_TOP_PRED", "DC_128_PRED", "V_PRED",
  "H_PRED",    "D45_PRED",     "D135_PRED",   "D117_PRED",   "D153_PRED",
  "D207_PRED", "D63_PRED",     "TM_PRED"
};

template <typename Pixel>
struct IntraPredTestMem {
  void Init(int block_size, int bd) {
    libvpx_test::ACMRandom rnd(libvpx_test::ACMRandom::DeterministicSeed());
    Pixel *const above = above_mem + 16;
    const int mask = (1 << bd) - 1;
    for (int i = 0; i < kTotalPixels; ++i) ref_src[i] = rnd.Rand16() & mask;
    for (int i = 0; i < kBPS; ++i) left[i] = rnd.Rand16() & mask;
    for (int i = -1; i < kBPS; ++i) above[i] = rnd.Rand16() & mask;

    // d45/d63 require the top row to be extended.
    ASSERT_LE(block_size, kBPS);
    for (int i = block_size; i < 2 * block_size; ++i) {
      above[i] = above[block_size - 1];
    }
  }

  DECLARE_ALIGNED(16, Pixel, src[kTotalPixels]);
  DECLARE_ALIGNED(16, Pixel, ref_src[kTotalPixels]);
  DECLARE_ALIGNED(16, Pixel, left[kBPS]);
  DECLARE_ALIGNED(16, Pixel, above_mem[2 * kBPS + 16]);
};

using Vp9IntraPredTestMem = IntraPredTestMem<uint8_t>;

void CheckMd5Signature(const char name[], const char *const signatures[],
                       const void *data, size_t data_size, int elapsed_time,
                       int idx) {
  libvpx_test::MD5 md5;
  md5.Add(reinterpret_cast<const uint8_t *>(data), data_size);
  printf("Mode %s[%12s]: %5d ms     MD5: %s\n", name, kVp9IntraPredNames[idx],
         elapsed_time, md5.Get());
  EXPECT_STREQ(signatures[idx], md5.Get());
}

void TestIntraPred(const char name[], VpxPredFunc const *pred_funcs,
                   const char *const signatures[], int block_size) {
  const int kNumTests = static_cast<int>(
      2.e10 / (block_size * block_size * kNumVp9IntraPredFuncs));
  Vp9IntraPredTestMem intra_pred_test_mem;
  const uint8_t *const above = intra_pred_test_mem.above_mem + 16;

  intra_pred_test_mem.Init(block_size, 8);

  for (int k = 0; k < kNumVp9IntraPredFuncs; ++k) {
    if (pred_funcs[k] == nullptr) continue;
    memcpy(intra_pred_test_mem.src, intra_pred_test_mem.ref_src,
           sizeof(intra_pred_test_mem.src));
    vpx_usec_timer timer;
    vpx_usec_timer_start(&timer);
    for (int num_tests = 0; num_tests < kNumTests; ++num_tests) {
      pred_funcs[k](intra_pred_test_mem.src, kBPS, above,
                    intra_pred_test_mem.left);
    }
    libvpx_test::ClearSystemState();
    vpx_usec_timer_mark(&timer);
    const int elapsed_time =
        static_cast<int>(vpx_usec_timer_elapsed(&timer) / 1000);
    CheckMd5Signature(name, signatures, intra_pred_test_mem.src,
                      sizeof(intra_pred_test_mem.src), elapsed_time, k);
  }
}

void TestIntraPred4(VpxPredFunc const *pred_funcs) {
  static const char *const kSignatures[kNumVp9IntraPredFuncs] = {
    "e7ed7353c3383fff942e500e9bfe82fe", "2a4a26fcc6ce005eadc08354d196c8a9",
    "269d92eff86f315d9c38fe7640d85b15", "ae2960eea9f71ee3dabe08b282ec1773",
    "6c1abcc44e90148998b51acd11144e9c", "f7bb3186e1ef8a2b326037ff898cad8e",
    "364c1f3fb2f445f935aec2a70a67eaa4", "141624072a4a56773f68fadbdd07c4a7",
    "7be49b08687a5f24df3a2c612fca3876", "459bb5d9fd5b238348179c9a22108cd6",
    "73edb8831bf1bdfce21ae8eaa43b1234", "2e2457f2009c701a355a8b25eb74fcda",
    "52ae4e8bdbe41494c1f43051d4dd7f0b"
  };
  TestIntraPred("Intra4", pred_funcs, kSignatures, 4);
}

void TestIntraPred8(VpxPredFunc const *pred_funcs) {
  static const char *const kSignatures[kNumVp9IntraPredFuncs] = {
    "d8bbae5d6547cfc17e4f5f44c8730e88", "373bab6d931868d41a601d9d88ce9ac3",
    "6fdd5ff4ff79656c14747598ca9e3706", "d9661c2811d6a73674f40ffb2b841847",
    "7c722d10b19ccff0b8c171868e747385", "f81dd986eb2b50f750d3a7da716b7e27",
    "d500f2c8fc78f46a4c74e4dcf51f14fb", "0e3523f9cab2142dd37fd07ec0760bce",
    "79ac4efe907f0a0f1885d43066cfedee", "19ecf2432ac305057de3b6578474eec6",
    "4f985b61acc6dd5d2d2585fa89ea2e2d", "f1bb25a9060dd262f405f15a38f5f674",
    "209ea00801584829e9a0f7be7d4a74ba"
  };
  TestIntraPred("Intra8", pred_funcs, kSignatures, 8);
}

void TestIntraPred16(VpxPredFunc const *pred_funcs) {
  static const char *const kSignatures[kNumVp9IntraPredFuncs] = {
    "50971c07ce26977d30298538fffec619", "527a6b9e0dc5b21b98cf276305432bef",
    "7eff2868f80ebc2c43a4f367281d80f7", "67cd60512b54964ef6aff1bd4816d922",
    "48371c87dc95c08a33b2048f89cf6468", "b0acf2872ee411d7530af6d2625a7084",
    "f32aafed4d8d3776ed58bcb6188756d5", "dae208f3dca583529cff49b73f7c4183",
    "7af66a2f4c8e0b4908e40f047e60c47c", "125e3ab6ab9bc961f183ec366a7afa88",
    "6b90f25b23983c35386b9fd704427622", "f8d6b11d710edc136a7c62c917435f93",
    "ed308f18614a362917f411c218aee532"
  };
  TestIntraPred("Intra16", pred_funcs, kSignatures, 16);
}

void TestIntraPred32(VpxPredFunc const *pred_funcs) {
  static const char *const kSignatures[kNumVp9IntraPredFuncs] = {
    "a0a618c900e65ae521ccc8af789729f2", "985aaa7c72b4a6c2fb431d32100cf13a",
    "10662d09febc3ca13ee4e700120daeb5", "b3b01379ba08916ef6b1b35f7d9ad51c",
    "9f4261755795af97e34679c333ec7004", "bc2c9da91ad97ef0d1610fb0a9041657",
    "75c79b1362ad18abfcdb1aa0aacfc21d", "4039bb7da0f6860090d3c57b5c85468f",
    "b29fff7b61804e68383e3a609b33da58", "e1aa5e49067fd8dba66c2eb8d07b7a89",
    "4e042822909c1c06d3b10a88281df1eb", "72eb9d9e0e67c93f4c66b70348e9fef7",
    "a22d102bcb51ca798aac12ca4ae8f2e8"
  };
  TestIntraPred("Intra32", pred_funcs, kSignatures, 32);
}

}  // namespace

// Defines a test case for |arch| (e.g., C, SSE2, ...) passing the predictors
// to |test_func|. The test name is 'arch.test_func', e.g., C.TestIntraPred4.
#define INTRA_PRED_TEST(arch, test_func, dc, dc_left, dc_top, dc_128, v, h,   \
                        d45, d135, d117, d153, d207, d63, tm)                 \
  TEST(arch, test_func) {                                                     \
    static const VpxPredFunc vpx_intra_pred[] = {                             \
      dc, dc_left, dc_top, dc_128, v, h, d45, d135, d117, d153, d207, d63, tm \
    };                                                                        \
    test_func(vpx_intra_pred);                                                \
  }

// -----------------------------------------------------------------------------

INTRA_PRED_TEST(C, TestIntraPred4, vpx_dc_predictor_4x4_c,
                vpx_dc_left_predictor_4x4_c, vpx_dc_top_predictor_4x4_c,
                vpx_dc_128_predictor_4x4_c, vpx_v_predictor_4x4_c,
                vpx_h_predictor_4x4_c, vpx_d45_predictor_4x4_c,
                vpx_d135_predictor_4x4_c, vpx_d117_predictor_4x4_c,
                vpx_d153_predictor_4x4_c, vpx_d207_predictor_4x4_c,
                vpx_d63_predictor_4x4_c, vpx_tm_predictor_4x4_c)

INTRA_PRED_TEST(C, TestIntraPred8, vpx_dc_predictor_8x8_c,
                vpx_dc_left_predictor_8x8_c, vpx_dc_top_predictor_8x8_c,
                vpx_dc_128_predictor_8x8_c, vpx_v_predictor_8x8_c,
                vpx_h_predictor_8x8_c, vpx_d45_predictor_8x8_c,
                vpx_d135_predictor_8x8_c, vpx_d117_predictor_8x8_c,
                vpx_d153_predictor_8x8_c, vpx_d207_predictor_8x8_c,
                vpx_d63_predictor_8x8_c, vpx_tm_predictor_8x8_c)

INTRA_PRED_TEST(C, TestIntraPred16, vpx_dc_predictor_16x16_c,
                vpx_dc_left_predictor_16x16_c, vpx_dc_top_predictor_16x16_c,
                vpx_dc_128_predictor_16x16_c, vpx_v_predictor_16x16_c,
                vpx_h_predictor_16x16_c, vpx_d45_predictor_16x16_c,
                vpx_d135_predictor_16x16_c, vpx_d117_predictor_16x16_c,
                vpx_d153_predictor_16x16_c, vpx_d207_predictor_16x16_c,
                vpx_d63_predictor_16x16_c, vpx_tm_predictor_16x16_c)

INTRA_PRED_TEST(C, TestIntraPred32, vpx_dc_predictor_32x32_c,
                vpx_dc_left_predictor_32x32_c, vpx_dc_top_predictor_32x32_c,
                vpx_dc_128_predictor_32x32_c, vpx_v_predictor_32x32_c,
                vpx_h_predictor_32x32_c, vpx_d45_predictor_32x32_c,
                vpx_d135_predictor_32x32_c, vpx_d117_predictor_32x32_c,
                vpx_d153_predictor_32x32_c, vpx_d207_predictor_32x32_c,
                vpx_d63_predictor_32x32_c, vpx_tm_predictor_32x32_c)

#if HAVE_SSE2
INTRA_PRED_TEST(SSE2, TestIntraPred4, vpx_dc_predictor_4x4_sse2,
                vpx_dc_left_predictor_4x4_sse2, vpx_dc_top_predictor_4x4_sse2,
                vpx_dc_128_predictor_4x4_sse2, vpx_v_predictor_4x4_sse2,
                vpx_h_predictor_4x4_sse2, vpx_d45_predictor_4x4_sse2, nullptr,
                nullptr, nullptr, vpx_d207_predictor_4x4_sse2, nullptr,
                vpx_tm_predictor_4x4_sse2)

INTRA_PRED_TEST(SSE2, TestIntraPred8, vpx_dc_predictor_8x8_sse2,
                vpx_dc_left_predictor_8x8_sse2, vpx_dc_top_predictor_8x8_sse2,
                vpx_dc_128_predictor_8x8_sse2, vpx_v_predictor_8x8_sse2,
                vpx_h_predictor_8x8_sse2, vpx_d45_predictor_8x8_sse2, nullptr,
                nullptr, nullptr, nullptr, nullptr, vpx_tm_predictor_8x8_sse2)

INTRA_PRED_TEST(SSE2, TestIntraPred16, vpx_dc_predictor_16x16_sse2,
                vpx_dc_left_predictor_16x16_sse2,
                vpx_dc_top_predictor_16x16_sse2,
                vpx_dc_128_predictor_16x16_sse2, vpx_v_predictor_16x16_sse2,
                vpx_h_predictor_16x16_sse2, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, vpx_tm_predictor_16x16_sse2)

INTRA_PRED_TEST(SSE2, TestIntraPred32, vpx_dc_predictor_32x32_sse2,
                vpx_dc_left_predictor_32x32_sse2,
                vpx_dc_top_predictor_32x32_sse2,
                vpx_dc_128_predictor_32x32_sse2, vpx_v_predictor_32x32_sse2,
                vpx_h_predictor_32x32_sse2, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, vpx_tm_predictor_32x32_sse2)
#endif  // HAVE_SSE2

#if HAVE_SSSE3
INTRA_PRED_TEST(SSSE3, TestIntraPred4, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr,
                vpx_d153_predictor_4x4_ssse3, nullptr,
                vpx_d63_predictor_4x4_ssse3, nullptr)
INTRA_PRED_TEST(SSSE3, TestIntraPred8, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr,
                vpx_d153_predictor_8x8_ssse3, vpx_d207_predictor_8x8_ssse3,
                vpx_d63_predictor_8x8_ssse3, nullptr)
INTRA_PRED_TEST(SSSE3, TestIntraPred16, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, vpx_d45_predictor_16x16_ssse3, nullptr,
                nullptr, vpx_d153_predictor_16x16_ssse3,
                vpx_d207_predictor_16x16_ssse3, vpx_d63_predictor_16x16_ssse3,
                nullptr)
INTRA_PRED_TEST(SSSE3, TestIntraPred32, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, vpx_d45_predictor_32x32_ssse3, nullptr,
                nullptr, vpx_d153_predictor_32x32_ssse3,
                vpx_d207_predictor_32x32_ssse3, vpx_d63_predictor_32x32_ssse3,
                nullptr)
#endif  // HAVE_SSSE3

#if HAVE_DSPR2
INTRA_PRED_TEST(DSPR2, TestIntraPred4, vpx_dc_predictor_4x4_dspr2, nullptr,
                nullptr, nullptr, nullptr, vpx_h_predictor_4x4_dspr2, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr,
                vpx_tm_predictor_4x4_dspr2)
INTRA_PRED_TEST(DSPR2, TestIntraPred8, vpx_dc_predictor_8x8_dspr2, nullptr,
                nullptr, nullptr, nullptr, vpx_h_predictor_8x8_dspr2, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr,
                vpx_tm_predictor_8x8_c)
INTRA_PRED_TEST(DSPR2, TestIntraPred16, vpx_dc_predictor_16x16_dspr2, nullptr,
                nullptr, nullptr, nullptr, vpx_h_predictor_16x16_dspr2, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)
#endif  // HAVE_DSPR2

#if HAVE_NEON
INTRA_PRED_TEST(NEON, TestIntraPred4, vpx_dc_predictor_4x4_neon,
                vpx_dc_left_predictor_4x4_neon, vpx_dc_top_predictor_4x4_neon,
                vpx_dc_128_predictor_4x4_neon, vpx_v_predictor_4x4_neon,
                vpx_h_predictor_4x4_neon, vpx_d45_predictor_4x4_neon,
                vpx_d135_predictor_4x4_neon, vpx_d117_predictor_4x4_neon,
                vpx_d153_predictor_4x4_neon, vpx_d207_predictor_4x4_neon,
                vpx_d63_predictor_4x4_neon, vpx_tm_predictor_4x4_neon)
INTRA_PRED_TEST(NEON, TestIntraPred8, vpx_dc_predictor_8x8_neon,
                vpx_dc_left_predictor_8x8_neon, vpx_dc_top_predictor_8x8_neon,
                vpx_dc_128_predictor_8x8_neon, vpx_v_predictor_8x8_neon,
                vpx_h_predictor_8x8_neon, vpx_d45_predictor_8x8_neon,
                vpx_d135_predictor_8x8_neon, vpx_d117_predictor_8x8_neon,
                vpx_d153_predictor_8x8_neon, vpx_d207_predictor_8x8_neon,
                vpx_d63_predictor_8x8_neon, vpx_tm_predictor_8x8_neon)
INTRA_PRED_TEST(NEON, TestIntraPred16, vpx_dc_predictor_16x16_neon,
                vpx_dc_left_predictor_16x16_neon,
                vpx_dc_top_predictor_16x16_neon,
                vpx_dc_128_predictor_16x16_neon, vpx_v_predictor_16x16_neon,
                vpx_h_predictor_16x16_neon, vpx_d45_predictor_16x16_neon,
                vpx_d135_predictor_16x16_neon, vpx_d117_predictor_16x16_neon,
                vpx_d153_predictor_16x16_neon, vpx_d207_predictor_16x16_neon,
                vpx_d63_predictor_16x16_neon, vpx_tm_predictor_16x16_neon)
INTRA_PRED_TEST(NEON, TestIntraPred32, vpx_dc_predictor_32x32_neon,
                vpx_dc_left_predictor_32x32_neon,
                vpx_dc_top_predictor_32x32_neon,
                vpx_dc_128_predictor_32x32_neon, vpx_v_predictor_32x32_neon,
                vpx_h_predictor_32x32_neon, vpx_d45_predictor_32x32_neon,
                vpx_d135_predictor_32x32_neon, vpx_d117_predictor_32x32_neon,
                vpx_d153_predictor_32x32_neon, vpx_d207_predictor_32x32_neon,
                vpx_d63_predictor_32x32_neon, vpx_tm_predictor_32x32_neon)
#endif  // HAVE_NEON

#if HAVE_MSA
INTRA_PRED_TEST(MSA, TestIntraPred4, vpx_dc_predictor_4x4_msa,
                vpx_dc_left_predictor_4x4_msa, vpx_dc_top_predictor_4x4_msa,
                vpx_dc_128_predictor_4x4_msa, vpx_v_predictor_4x4_msa,
                vpx_h_predictor_4x4_msa, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, vpx_tm_predictor_4x4_msa)
INTRA_PRED_TEST(MSA, TestIntraPred8, vpx_dc_predictor_8x8_msa,
                vpx_dc_left_predictor_8x8_msa, vpx_dc_top_predictor_8x8_msa,
                vpx_dc_128_predictor_8x8_msa, vpx_v_predictor_8x8_msa,
                vpx_h_predictor_8x8_msa, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, vpx_tm_predictor_8x8_msa)
INTRA_PRED_TEST(MSA, TestIntraPred16, vpx_dc_predictor_16x16_msa,
                vpx_dc_left_predictor_16x16_msa, vpx_dc_top_predictor_16x16_msa,
                vpx_dc_128_predictor_16x16_msa, vpx_v_predictor_16x16_msa,
                vpx_h_predictor_16x16_msa, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, vpx_tm_predictor_16x16_msa)
INTRA_PRED_TEST(MSA, TestIntraPred32, vpx_dc_predictor_32x32_msa,
                vpx_dc_left_predictor_32x32_msa, vpx_dc_top_predictor_32x32_msa,
                vpx_dc_128_predictor_32x32_msa, vpx_v_predictor_32x32_msa,
                vpx_h_predictor_32x32_msa, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, vpx_tm_predictor_32x32_msa)
#endif  // HAVE_MSA

#if HAVE_VSX
// TODO(crbug.com/webm/1522): Fix test failures.
#if 0
INTRA_PRED_TEST(VSX, TestIntraPred4, nullptr, nullptr, nullptr, nullptr,
                nullptr, vpx_h_predictor_4x4_vsx, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, vpx_tm_predictor_4x4_vsx)

INTRA_PRED_TEST(VSX, TestIntraPred8, vpx_dc_predictor_8x8_vsx, nullptr, nullptr,
                nullptr, nullptr, vpx_h_predictor_8x8_vsx,
                vpx_d45_predictor_8x8_vsx, nullptr, nullptr, nullptr, nullptr,
                vpx_d63_predictor_8x8_vsx, vpx_tm_predictor_8x8_vsx)
#endif

INTRA_PRED_TEST(VSX, TestIntraPred16, vpx_dc_predictor_16x16_vsx,
                vpx_dc_left_predictor_16x16_vsx, vpx_dc_top_predictor_16x16_vsx,
                vpx_dc_128_predictor_16x16_vsx, vpx_v_predictor_16x16_vsx,
                vpx_h_predictor_16x16_vsx, vpx_d45_predictor_16x16_vsx, nullptr,
                nullptr, nullptr, nullptr, vpx_d63_predictor_16x16_vsx,
                vpx_tm_predictor_16x16_vsx)

INTRA_PRED_TEST(VSX, TestIntraPred32, vpx_dc_predictor_32x32_vsx,
                vpx_dc_left_predictor_32x32_vsx, vpx_dc_top_predictor_32x32_vsx,
                vpx_dc_128_predictor_32x32_vsx, vpx_v_predictor_32x32_vsx,
                vpx_h_predictor_32x32_vsx, vpx_d45_predictor_32x32_vsx, nullptr,
                nullptr, nullptr, nullptr, vpx_d63_predictor_32x32_vsx,
                vpx_tm_predictor_32x32_vsx)
#endif  // HAVE_VSX

#if HAVE_LSX
INTRA_PRED_TEST(LSX, TestIntraPred8, vpx_dc_predictor_8x8_lsx, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr)
INTRA_PRED_TEST(LSX, TestIntraPred16, vpx_dc_predictor_16x16_lsx, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr)
#endif  // HAVE_LSX

// -----------------------------------------------------------------------------

#if CONFIG_VP9_HIGHBITDEPTH
namespace {

using VpxHighbdPredFunc = void (*)(uint16_t *dst, ptrdiff_t y_stride,
                                   const uint16_t *above, const uint16_t *left,
                                   int bd);

using Vp9HighbdIntraPredTestMem = IntraPredTestMem<uint16_t>;

void TestHighbdIntraPred(const char name[], VpxHighbdPredFunc const *pred_funcs,
                         const char *const signatures[], int block_size) {
  const int kNumTests = static_cast<int>(
      2.e10 / (block_size * block_size * kNumVp9IntraPredFuncs));
  Vp9HighbdIntraPredTestMem intra_pred_test_mem;
  const uint16_t *const above = intra_pred_test_mem.above_mem + 16;

  intra_pred_test_mem.Init(block_size, 12);

  for (int k = 0; k < kNumVp9IntraPredFuncs; ++k) {
    if (pred_funcs[k] == nullptr) continue;
    memcpy(intra_pred_test_mem.src, intra_pred_test_mem.ref_src,
           sizeof(intra_pred_test_mem.src));
    vpx_usec_timer timer;
    vpx_usec_timer_start(&timer);
    for (int num_tests = 0; num_tests < kNumTests; ++num_tests) {
      pred_funcs[k](intra_pred_test_mem.src, kBPS, above,
                    intra_pred_test_mem.left, 12);
    }
    libvpx_test::ClearSystemState();
    vpx_usec_timer_mark(&timer);
    const int elapsed_time =
        static_cast<int>(vpx_usec_timer_elapsed(&timer) / 1000);
    CheckMd5Signature(name, signatures, intra_pred_test_mem.src,
                      sizeof(intra_pred_test_mem.src), elapsed_time, k);
  }
}

void TestHighbdIntraPred4(VpxHighbdPredFunc const *pred_funcs) {
  static const char *const kSignatures[kNumVp9IntraPredFuncs] = {
    "11f74af6c5737df472f3275cbde062fa", "51bea056b6447c93f6eb8f6b7e8f6f71",
    "27e97f946766331795886f4de04c5594", "53ab15974b049111fb596c5168ec7e3f",
    "f0b640bb176fbe4584cf3d32a9b0320a", "729783ca909e03afd4b47111c80d967b",
    "fbf1c30793d9f32812e4d9f905d53530", "293fc903254a33754133314c6cdba81f",
    "f8074d704233e73dfd35b458c6092374", "aa6363d08544a1ec4da33d7a0be5640d",
    "462abcfdfa3d087bb33c9a88f2aec491", "863eab65d22550dd44a2397277c1ec71",
    "23d61df1574d0fa308f9731811047c4b"
  };
  TestHighbdIntraPred("Intra4", pred_funcs, kSignatures, 4);
}

void TestHighbdIntraPred8(VpxHighbdPredFunc const *pred_funcs) {
  static const char *const kSignatures[kNumVp9IntraPredFuncs] = {
    "03da8829fe94663047fd108c5fcaa71d", "ecdb37b8120a2d3a4c706b016bd1bfd7",
    "1d4543ed8d2b9368cb96898095fe8a75", "f791c9a67b913cbd82d9da8ecede30e2",
    "065c70646f4dbaff913282f55a45a441", "51f87123616662ef7c35691497dfd0ba",
    "2a5b0131ef4716f098ee65e6df01e3dd", "9ffe186a6bc7db95275f1bbddd6f7aba",
    "a3258a2eae2e2bd55cb8f71351b22998", "8d909f0a2066e39b3216092c6289ece4",
    "d183abb30b9f24c886a0517e991b22c7", "702a42fe4c7d665dc561b2aeeb60f311",
    "7b5dbbbe7ae3a4ac2948731600bde5d6"
  };
  TestHighbdIntraPred("Intra8", pred_funcs, kSignatures, 8);
}

void TestHighbdIntraPred16(VpxHighbdPredFunc const *pred_funcs) {
  static const char *const kSignatures[kNumVp9IntraPredFuncs] = {
    "e33cb3f56a878e2fddb1b2fc51cdd275", "c7bff6f04b6052c8ab335d726dbbd52d",
    "d0b0b47b654a9bcc5c6008110a44589b", "78f5da7b10b2b9ab39f114a33b6254e9",
    "c78e31d23831abb40d6271a318fdd6f3", "90d1347f4ec9198a0320daecb6ff90b8",
    "d2c623746cbb64a0c9e29c10f2c57041", "cf28bd387b81ad3e5f1a1c779a4b70a0",
    "24c304330431ddeaf630f6ce94af2eac", "91a329798036bf64e8e00a87b131b8b1",
    "d39111f22885307f920796a42084c872", "e2e702f7250ece98dd8f3f2854c31eeb",
    "e2fb05b01eb8b88549e85641d8ce5b59"
  };
  TestHighbdIntraPred("Intra16", pred_funcs, kSignatures, 16);
}

void TestHighbdIntraPred32(VpxHighbdPredFunc const *pred_funcs) {
  static const char *const kSignatures[kNumVp9IntraPredFuncs] = {
    "a3e8056ba7e36628cce4917cd956fedd", "cc7d3024fe8748b512407edee045377e",
    "2aab0a0f330a1d3e19b8ecb8f06387a3", "a547bc3fb7b06910bf3973122a426661",
    "26f712514da95042f93d6e8dc8e431dc", "bb08c6e16177081daa3d936538dbc2e3",
    "8f031af3e2650e89620d8d2c3a843d8b", "42867c8553285e94ee8e4df7abafbda8",
    "6496bdee96100667833f546e1be3d640", "2ebfa25bf981377e682e580208504300",
    "3e8ae52fd1f607f348aa4cb436c71ab7", "3d4efe797ca82193613696753ea624c4",
    "cb8aab6d372278f3131e8d99efde02d9"
  };
  TestHighbdIntraPred("Intra32", pred_funcs, kSignatures, 32);
}

}  // namespace

// Defines a test case for |arch| (e.g., C, SSE2, ...) passing the predictors
// to |test_func|. The test name is 'arch.test_func', e.g., C.TestIntraPred4.
#define HIGHBD_INTRA_PRED_TEST(arch, test_func, dc, dc_left, dc_top, dc_128,  \
                               v, h, d45, d135, d117, d153, d207, d63, tm)    \
  TEST(arch, test_func) {                                                     \
    static const VpxHighbdPredFunc vpx_intra_pred[] = {                       \
      dc, dc_left, dc_top, dc_128, v, h, d45, d135, d117, d153, d207, d63, tm \
    };                                                                        \
    test_func(vpx_intra_pred);                                                \
  }

// -----------------------------------------------------------------------------

HIGHBD_INTRA_PRED_TEST(
    C, TestHighbdIntraPred4, vpx_highbd_dc_predictor_4x4_c,
    vpx_highbd_dc_left_predictor_4x4_c, vpx_highbd_dc_top_predictor_4x4_c,
    vpx_highbd_dc_128_predictor_4x4_c, vpx_highbd_v_predictor_4x4_c,
    vpx_highbd_h_predictor_4x4_c, vpx_highbd_d45_predictor_4x4_c,
    vpx_highbd_d135_predictor_4x4_c, vpx_highbd_d117_predictor_4x4_c,
    vpx_highbd_d153_predictor_4x4_c, vpx_highbd_d207_predictor_4x4_c,
    vpx_highbd_d63_predictor_4x4_c, vpx_highbd_tm_predictor_4x4_c)

HIGHBD_INTRA_PRED_TEST(
    C, TestHighbdIntraPred8, vpx_highbd_dc_predictor_8x8_c,
    vpx_highbd_dc_left_predictor_8x8_c, vpx_highbd_dc_top_predictor_8x8_c,
    vpx_highbd_dc_128_predictor_8x8_c, vpx_highbd_v_predictor_8x8_c,
    vpx_highbd_h_predictor_8x8_c, vpx_highbd_d45_predictor_8x8_c,
    vpx_highbd_d135_predictor_8x8_c, vpx_highbd_d117_predictor_8x8_c,
    vpx_highbd_d153_predictor_8x8_c, vpx_highbd_d207_predictor_8x8_c,
    vpx_highbd_d63_predictor_8x8_c, vpx_highbd_tm_predictor_8x8_c)

HIGHBD_INTRA_PRED_TEST(
    C, TestHighbdIntraPred16, vpx_highbd_dc_predictor_16x16_c,
    vpx_highbd_dc_left_predictor_16x16_c, vpx_highbd_dc_top_predictor_16x16_c,
    vpx_highbd_dc_128_predictor_16x16_c, vpx_highbd_v_predictor_16x16_c,
    vpx_highbd_h_predictor_16x16_c, vpx_highbd_d45_predictor_16x16_c,
    vpx_highbd_d135_predictor_16x16_c, vpx_highbd_d117_predictor_16x16_c,
    vpx_highbd_d153_predictor_16x16_c, vpx_highbd_d207_predictor_16x16_c,
    vpx_highbd_d63_predictor_16x16_c, vpx_highbd_tm_predictor_16x16_c)

HIGHBD_INTRA_PRED_TEST(
    C, TestHighbdIntraPred32, vpx_highbd_dc_predictor_32x32_c,
    vpx_highbd_dc_left_predictor_32x32_c, vpx_highbd_dc_top_predictor_32x32_c,
    vpx_highbd_dc_128_predictor_32x32_c, vpx_highbd_v_predictor_32x32_c,
    vpx_highbd_h_predictor_32x32_c, vpx_highbd_d45_predictor_32x32_c,
    vpx_highbd_d135_predictor_32x32_c, vpx_highbd_d117_predictor_32x32_c,
    vpx_highbd_d153_predictor_32x32_c, vpx_highbd_d207_predictor_32x32_c,
    vpx_highbd_d63_predictor_32x32_c, vpx_highbd_tm_predictor_32x32_c)

#if HAVE_SSE2
HIGHBD_INTRA_PRED_TEST(
    SSE2, TestHighbdIntraPred4, vpx_highbd_dc_predictor_4x4_sse2,
    vpx_highbd_dc_left_predictor_4x4_sse2, vpx_highbd_dc_top_predictor_4x4_sse2,
    vpx_highbd_dc_128_predictor_4x4_sse2, vpx_highbd_v_predictor_4x4_sse2,
    vpx_highbd_h_predictor_4x4_sse2, nullptr,
    vpx_highbd_d135_predictor_4x4_sse2, vpx_highbd_d117_predictor_4x4_sse2,
    vpx_highbd_d153_predictor_4x4_sse2, vpx_highbd_d207_predictor_4x4_sse2,
    vpx_highbd_d63_predictor_4x4_sse2, vpx_highbd_tm_predictor_4x4_c)

HIGHBD_INTRA_PRED_TEST(
    SSE2, TestHighbdIntraPred8, vpx_highbd_dc_predictor_8x8_sse2,
    vpx_highbd_dc_left_predictor_8x8_sse2, vpx_highbd_dc_top_predictor_8x8_sse2,
    vpx_highbd_dc_128_predictor_8x8_sse2, vpx_highbd_v_predictor_8x8_sse2,
    vpx_highbd_h_predictor_8x8_sse2, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, vpx_highbd_tm_predictor_8x8_sse2)

HIGHBD_INTRA_PRED_TEST(SSE2, TestHighbdIntraPred16,
                       vpx_highbd_dc_predictor_16x16_sse2,
                       vpx_highbd_dc_left_predictor_16x16_sse2,
                       vpx_highbd_dc_top_predictor_16x16_sse2,
                       vpx_highbd_dc_128_predictor_16x16_sse2,
                       vpx_highbd_v_predictor_16x16_sse2,
                       vpx_highbd_h_predictor_16x16_sse2, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr,
                       vpx_highbd_tm_predictor_16x16_sse2)

HIGHBD_INTRA_PRED_TEST(SSE2, TestHighbdIntraPred32,
                       vpx_highbd_dc_predictor_32x32_sse2,
                       vpx_highbd_dc_left_predictor_32x32_sse2,
                       vpx_highbd_dc_top_predictor_32x32_sse2,
                       vpx_highbd_dc_128_predictor_32x32_sse2,
                       vpx_highbd_v_predictor_32x32_sse2,
                       vpx_highbd_h_predictor_32x32_sse2, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr,
                       vpx_highbd_tm_predictor_32x32_sse2)
#endif  // HAVE_SSE2

#if HAVE_SSSE3
HIGHBD_INTRA_PRED_TEST(SSSE3, TestHighbdIntraPred4, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr,
                       vpx_highbd_d45_predictor_4x4_ssse3, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr)
HIGHBD_INTRA_PRED_TEST(SSSE3, TestHighbdIntraPred8, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr,
                       vpx_highbd_d45_predictor_8x8_ssse3,
                       vpx_highbd_d135_predictor_8x8_ssse3,
                       vpx_highbd_d117_predictor_8x8_ssse3,
                       vpx_highbd_d153_predictor_8x8_ssse3,
                       vpx_highbd_d207_predictor_8x8_ssse3,
                       vpx_highbd_d63_predictor_8x8_ssse3, nullptr)
HIGHBD_INTRA_PRED_TEST(SSSE3, TestHighbdIntraPred16, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr,
                       vpx_highbd_d45_predictor_16x16_ssse3,
                       vpx_highbd_d135_predictor_16x16_ssse3,
                       vpx_highbd_d117_predictor_16x16_ssse3,
                       vpx_highbd_d153_predictor_16x16_ssse3,
                       vpx_highbd_d207_predictor_16x16_ssse3,
                       vpx_highbd_d63_predictor_16x16_ssse3, nullptr)
HIGHBD_INTRA_PRED_TEST(SSSE3, TestHighbdIntraPred32, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr,
                       vpx_highbd_d45_predictor_32x32_ssse3,
                       vpx_highbd_d135_predictor_32x32_ssse3,
                       vpx_highbd_d117_predictor_32x32_ssse3,
                       vpx_highbd_d153_predictor_32x32_ssse3,
                       vpx_highbd_d207_predictor_32x32_ssse3,
                       vpx_highbd_d63_predictor_32x32_ssse3, nullptr)
#endif  // HAVE_SSSE3

#if HAVE_NEON
HIGHBD_INTRA_PRED_TEST(
    NEON, TestHighbdIntraPred4, vpx_highbd_dc_predictor_4x4_neon,
    vpx_highbd_dc_left_predictor_4x4_neon, vpx_highbd_dc_top_predictor_4x4_neon,
    vpx_highbd_dc_128_predictor_4x4_neon, vpx_highbd_v_predictor_4x4_neon,
    vpx_highbd_h_predictor_4x4_neon, vpx_highbd_d45_predictor_4x4_neon,
    vpx_highbd_d135_predictor_4x4_neon, vpx_highbd_d117_predictor_4x4_neon,
    vpx_highbd_d153_predictor_4x4_neon, vpx_highbd_d207_predictor_4x4_neon,
    vpx_highbd_d63_predictor_4x4_neon, vpx_highbd_tm_predictor_4x4_neon)
HIGHBD_INTRA_PRED_TEST(
    NEON, TestHighbdIntraPred8, vpx_highbd_dc_predictor_8x8_neon,
    vpx_highbd_dc_left_predictor_8x8_neon, vpx_highbd_dc_top_predictor_8x8_neon,
    vpx_highbd_dc_128_predictor_8x8_neon, vpx_highbd_v_predictor_8x8_neon,
    vpx_highbd_h_predictor_8x8_neon, vpx_highbd_d45_predictor_8x8_neon,
    vpx_highbd_d135_predictor_8x8_neon, vpx_highbd_d117_predictor_8x8_neon,
    vpx_highbd_d153_predictor_8x8_neon, vpx_highbd_d207_predictor_8x8_neon,
    vpx_highbd_d63_predictor_8x8_neon, vpx_highbd_tm_predictor_8x8_neon)
HIGHBD_INTRA_PRED_TEST(
    NEON, TestHighbdIntraPred16, vpx_highbd_dc_predictor_16x16_neon,
    vpx_highbd_dc_left_predictor_16x16_neon,
    vpx_highbd_dc_top_predictor_16x16_neon,
    vpx_highbd_dc_128_predictor_16x16_neon, vpx_highbd_v_predictor_16x16_neon,
    vpx_highbd_h_predictor_16x16_neon, vpx_highbd_d45_predictor_16x16_neon,
    vpx_highbd_d135_predictor_16x16_neon, vpx_highbd_d117_predictor_16x16_neon,
    vpx_highbd_d153_predictor_16x16_neon, vpx_highbd_d207_predictor_16x16_neon,
    vpx_highbd_d63_predictor_16x16_neon, vpx_highbd_tm_predictor_16x16_neon)
HIGHBD_INTRA_PRED_TEST(
    NEON, TestHighbdIntraPred32, vpx_highbd_dc_predictor_32x32_neon,
    vpx_highbd_dc_left_predictor_32x32_neon,
    vpx_highbd_dc_top_predictor_32x32_neon,
    vpx_highbd_dc_128_predictor_32x32_neon, vpx_highbd_v_predictor_32x32_neon,
    vpx_highbd_h_predictor_32x32_neon, vpx_highbd_d45_predictor_32x32_neon,
    vpx_highbd_d135_predictor_32x32_neon, vpx_highbd_d117_predictor_32x32_neon,
    vpx_highbd_d153_predictor_32x32_neon, vpx_highbd_d207_predictor_32x32_neon,
    vpx_highbd_d63_predictor_32x32_neon, vpx_highbd_tm_predictor_32x32_neon)
#endif  // HAVE_NEON

#endif  // CONFIG_VP9_HIGHBITDEPTH

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::libvpx_test::init_vpx_test();
  return RUN_ALL_TESTS();
}
