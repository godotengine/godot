/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>

#include "gtest/gtest.h"

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "vp9/common/vp9_entropy.h"
#include "vp9/common/vp9_scan.h"
#include "vpx_config.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

using libvpx_test::ACMRandom;

namespace {

const int kNumCoeffs = 64;
const double kPi = 3.141592653589793238462643383279502884;

const int kSignBiasMaxDiff255 = 1500;
const int kSignBiasMaxDiff15 = 10000;

using FdctFunc = void (*)(const int16_t *in, tran_low_t *out, int stride);
using IdctFunc = void (*)(const tran_low_t *in, uint8_t *out, int stride);
using FhtFunc = void (*)(const int16_t *in, tran_low_t *out, int stride,
                         int tx_type);
using IhtFunc = void (*)(const tran_low_t *in, uint8_t *out, int stride,
                         int tx_type);

using Dct8x8Param = std::tuple<FdctFunc, IdctFunc, int, vpx_bit_depth_t>;
using Ht8x8Param = std::tuple<FhtFunc, IhtFunc, int, vpx_bit_depth_t>;
using Idct8x8Param = std::tuple<IdctFunc, IdctFunc, int, vpx_bit_depth_t>;

void reference_8x8_dct_1d(const double in[8], double out[8]) {
  const double kInvSqrt2 = 0.707106781186547524400844362104;
  for (int k = 0; k < 8; k++) {
    out[k] = 0.0;
    for (int n = 0; n < 8; n++) {
      out[k] += in[n] * cos(kPi * (2 * n + 1) * k / 16.0);
    }
    if (k == 0) out[k] = out[k] * kInvSqrt2;
  }
}

void reference_8x8_dct_2d(const int16_t input[kNumCoeffs],
                          double output[kNumCoeffs]) {
  // First transform columns
  for (int i = 0; i < 8; ++i) {
    double temp_in[8], temp_out[8];
    for (int j = 0; j < 8; ++j) temp_in[j] = input[j * 8 + i];
    reference_8x8_dct_1d(temp_in, temp_out);
    for (int j = 0; j < 8; ++j) output[j * 8 + i] = temp_out[j];
  }
  // Then transform rows
  for (int i = 0; i < 8; ++i) {
    double temp_in[8], temp_out[8];
    for (int j = 0; j < 8; ++j) temp_in[j] = output[j + i * 8];
    reference_8x8_dct_1d(temp_in, temp_out);
    // Scale by some magic number
    for (int j = 0; j < 8; ++j) output[j + i * 8] = temp_out[j] * 2;
  }
}

void fdct8x8_ref(const int16_t *in, tran_low_t *out, int stride,
                 int /*tx_type*/) {
  vpx_fdct8x8_c(in, out, stride);
}

void fht8x8_ref(const int16_t *in, tran_low_t *out, int stride, int tx_type) {
  vp9_fht8x8_c(in, out, stride, tx_type);
}

#if CONFIG_VP9_HIGHBITDEPTH
void idct8x8_10(const tran_low_t *in, uint8_t *out, int stride) {
  vpx_highbd_idct8x8_64_add_c(in, CAST_TO_SHORTPTR(out), stride, 10);
}

void idct8x8_12(const tran_low_t *in, uint8_t *out, int stride) {
  vpx_highbd_idct8x8_64_add_c(in, CAST_TO_SHORTPTR(out), stride, 12);
}

void iht8x8_10(const tran_low_t *in, uint8_t *out, int stride, int tx_type) {
  vp9_highbd_iht8x8_64_add_c(in, CAST_TO_SHORTPTR(out), stride, tx_type, 10);
}

void iht8x8_12(const tran_low_t *in, uint8_t *out, int stride, int tx_type) {
  vp9_highbd_iht8x8_64_add_c(in, CAST_TO_SHORTPTR(out), stride, tx_type, 12);
}

#if HAVE_SSE2

void idct8x8_12_add_10_c(const tran_low_t *in, uint8_t *out, int stride) {
  vpx_highbd_idct8x8_12_add_c(in, CAST_TO_SHORTPTR(out), stride, 10);
}

void idct8x8_12_add_12_c(const tran_low_t *in, uint8_t *out, int stride) {
  vpx_highbd_idct8x8_12_add_c(in, CAST_TO_SHORTPTR(out), stride, 12);
}

void idct8x8_12_add_10_sse2(const tran_low_t *in, uint8_t *out, int stride) {
  vpx_highbd_idct8x8_12_add_sse2(in, CAST_TO_SHORTPTR(out), stride, 10);
}

void idct8x8_12_add_12_sse2(const tran_low_t *in, uint8_t *out, int stride) {
  vpx_highbd_idct8x8_12_add_sse2(in, CAST_TO_SHORTPTR(out), stride, 12);
}

void idct8x8_64_add_10_sse2(const tran_low_t *in, uint8_t *out, int stride) {
  vpx_highbd_idct8x8_64_add_sse2(in, CAST_TO_SHORTPTR(out), stride, 10);
}

void idct8x8_64_add_12_sse2(const tran_low_t *in, uint8_t *out, int stride) {
  vpx_highbd_idct8x8_64_add_sse2(in, CAST_TO_SHORTPTR(out), stride, 12);
}
#endif  // HAVE_SSE2
#endif  // CONFIG_VP9_HIGHBITDEPTH

// Visual Studio 2022 (cl.exe) < 17.12.3 targeting AArch64 with optimizations
// enabled produces invalid code in RunExtremalCheck() and
// RunInvAccuracyCheck(). See:
// https://developercommunity.visualstudio.com/t/1770-preview-1:-Misoptimization-for-AR/10369786
#if defined(_MSC_FULL_VER) && _MSC_FULL_VER < 194234435 && \
    defined(_M_ARM64) && !defined(__clang__)
#define AOM_WORK_AROUND_MSVC_BUG_10369786
#endif

#ifdef AOM_WORK_AROUND_MSVC_BUG_10369786
#pragma optimize("", off)
#endif
class FwdTrans8x8TestBase {
 public:
  virtual ~FwdTrans8x8TestBase() = default;

 protected:
  virtual void RunFwdTxfm(int16_t *in, tran_low_t *out, int stride) = 0;
  virtual void RunInvTxfm(tran_low_t *out, uint8_t *dst, int stride) = 0;

  void RunSignBiasCheck() {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    DECLARE_ALIGNED(16, int16_t, test_input_block[64]);
    DECLARE_ALIGNED(16, tran_low_t, test_output_block[64]);
    int count_sign_block[64][2];
    const int count_test_block = 100000;

    memset(count_sign_block, 0, sizeof(count_sign_block));

    for (int i = 0; i < count_test_block; ++i) {
      // Initialize a test block with input range [-255, 255].
      for (int j = 0; j < 64; ++j) {
        test_input_block[j] = ((rnd.Rand16() >> (16 - bit_depth_)) & mask_) -
                              ((rnd.Rand16() >> (16 - bit_depth_)) & mask_);
      }
      ASM_REGISTER_STATE_CHECK(
          RunFwdTxfm(test_input_block, test_output_block, pitch_));

      for (int j = 0; j < 64; ++j) {
        if (test_output_block[j] < 0) {
          ++count_sign_block[j][0];
        } else if (test_output_block[j] > 0) {
          ++count_sign_block[j][1];
        }
      }
    }

    for (int j = 0; j < 64; ++j) {
      const int diff = abs(count_sign_block[j][0] - count_sign_block[j][1]);
      const int max_diff = kSignBiasMaxDiff255;
      ASSERT_LT(diff, max_diff << (bit_depth_ - 8))
          << "Error: 8x8 FDCT/FHT has a sign bias > "
          << 1. * max_diff / count_test_block * 100 << "%"
          << " for input range [-255, 255] at index " << j
          << " count0: " << count_sign_block[j][0]
          << " count1: " << count_sign_block[j][1] << " diff: " << diff;
    }

    memset(count_sign_block, 0, sizeof(count_sign_block));

    for (int i = 0; i < count_test_block; ++i) {
      // Initialize a test block with input range [-mask_ / 16, mask_ / 16].
      for (int j = 0; j < 64; ++j) {
        test_input_block[j] =
            ((rnd.Rand16() & mask_) >> 4) - ((rnd.Rand16() & mask_) >> 4);
      }
      ASM_REGISTER_STATE_CHECK(
          RunFwdTxfm(test_input_block, test_output_block, pitch_));

      for (int j = 0; j < 64; ++j) {
        if (test_output_block[j] < 0) {
          ++count_sign_block[j][0];
        } else if (test_output_block[j] > 0) {
          ++count_sign_block[j][1];
        }
      }
    }

    for (int j = 0; j < 64; ++j) {
      const int diff = abs(count_sign_block[j][0] - count_sign_block[j][1]);
      const int max_diff = kSignBiasMaxDiff15;
      ASSERT_LT(diff, max_diff << (bit_depth_ - 8))
          << "Error: 8x8 FDCT/FHT has a sign bias > "
          << 1. * max_diff / count_test_block * 100 << "%"
          << " for input range [-15, 15] at index " << j
          << " count0: " << count_sign_block[j][0]
          << " count1: " << count_sign_block[j][1] << " diff: " << diff;
    }
  }

  void RunRoundTripErrorCheck() {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    int max_error = 0;
    int total_error = 0;
    const int count_test_block = 100000;
    DECLARE_ALIGNED(16, int16_t, test_input_block[64]);
    DECLARE_ALIGNED(16, tran_low_t, test_temp_block[64]);
    DECLARE_ALIGNED(16, uint8_t, dst[64]);
    DECLARE_ALIGNED(16, uint8_t, src[64]);
#if CONFIG_VP9_HIGHBITDEPTH
    DECLARE_ALIGNED(16, uint16_t, dst16[64]);
    DECLARE_ALIGNED(16, uint16_t, src16[64]);
#endif

    for (int i = 0; i < count_test_block; ++i) {
      // Initialize a test block with input range [-mask_, mask_].
      for (int j = 0; j < 64; ++j) {
        if (bit_depth_ == VPX_BITS_8) {
          src[j] = rnd.Rand8();
          dst[j] = rnd.Rand8();
          test_input_block[j] = src[j] - dst[j];
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          src16[j] = rnd.Rand16() & mask_;
          dst16[j] = rnd.Rand16() & mask_;
          test_input_block[j] = src16[j] - dst16[j];
#endif
        }
      }

      ASM_REGISTER_STATE_CHECK(
          RunFwdTxfm(test_input_block, test_temp_block, pitch_));
      for (int j = 0; j < 64; ++j) {
        if (test_temp_block[j] > 0) {
          test_temp_block[j] += 2;
          test_temp_block[j] /= 4;
          test_temp_block[j] *= 4;
        } else {
          test_temp_block[j] -= 2;
          test_temp_block[j] /= 4;
          test_temp_block[j] *= 4;
        }
      }
      if (bit_depth_ == VPX_BITS_8) {
        ASM_REGISTER_STATE_CHECK(RunInvTxfm(test_temp_block, dst, pitch_));
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        ASM_REGISTER_STATE_CHECK(
            RunInvTxfm(test_temp_block, CAST_TO_BYTEPTR(dst16), pitch_));
#endif
      }

      for (int j = 0; j < 64; ++j) {
#if CONFIG_VP9_HIGHBITDEPTH
        const int diff =
            bit_depth_ == VPX_BITS_8 ? dst[j] - src[j] : dst16[j] - src16[j];
#else
        const int diff = dst[j] - src[j];
#endif
        const int error = diff * diff;
        if (max_error < error) max_error = error;
        total_error += error;
      }
    }

    ASSERT_GE(1 << 2 * (bit_depth_ - 8), max_error)
        << "Error: 8x8 FDCT/IDCT or FHT/IHT has an individual"
        << " roundtrip error > 1";

    ASSERT_GE((count_test_block << 2 * (bit_depth_ - 8)) / 5, total_error)
        << "Error: 8x8 FDCT/IDCT or FHT/IHT has average roundtrip "
        << "error > 1/5 per block";
  }

  void RunExtremalCheck() {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    int max_error = 0;
    int total_error = 0;
    int total_coeff_error = 0;
    const int count_test_block = 100000;
    DECLARE_ALIGNED(16, int16_t, test_input_block[64]);
    DECLARE_ALIGNED(16, tran_low_t, test_temp_block[64]);
    DECLARE_ALIGNED(16, tran_low_t, ref_temp_block[64]);
    DECLARE_ALIGNED(16, uint8_t, dst[64]);
    DECLARE_ALIGNED(16, uint8_t, src[64]);
#if CONFIG_VP9_HIGHBITDEPTH
    DECLARE_ALIGNED(16, uint16_t, dst16[64]);
    DECLARE_ALIGNED(16, uint16_t, src16[64]);
#endif

    for (int i = 0; i < count_test_block; ++i) {
      // Initialize a test block with input range [-mask_, mask_].
      for (int j = 0; j < 64; ++j) {
        if (bit_depth_ == VPX_BITS_8) {
          if (i == 0) {
            src[j] = 255;
            dst[j] = 0;
          } else if (i == 1) {
            src[j] = 0;
            dst[j] = 255;
          } else {
            src[j] = rnd.Rand8() % 2 ? 255 : 0;
            dst[j] = rnd.Rand8() % 2 ? 255 : 0;
          }
          test_input_block[j] = src[j] - dst[j];
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          if (i == 0) {
            src16[j] = mask_;
            dst16[j] = 0;
          } else if (i == 1) {
            src16[j] = 0;
            dst16[j] = mask_;
          } else {
            src16[j] = rnd.Rand8() % 2 ? mask_ : 0;
            dst16[j] = rnd.Rand8() % 2 ? mask_ : 0;
          }
          test_input_block[j] = src16[j] - dst16[j];
#endif
        }
      }

      ASM_REGISTER_STATE_CHECK(
          RunFwdTxfm(test_input_block, test_temp_block, pitch_));
      ASM_REGISTER_STATE_CHECK(
          fwd_txfm_ref(test_input_block, ref_temp_block, pitch_, tx_type_));
      if (bit_depth_ == VPX_BITS_8) {
        ASM_REGISTER_STATE_CHECK(RunInvTxfm(test_temp_block, dst, pitch_));
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        ASM_REGISTER_STATE_CHECK(
            RunInvTxfm(test_temp_block, CAST_TO_BYTEPTR(dst16), pitch_));
#endif
      }

      for (int j = 0; j < 64; ++j) {
#if CONFIG_VP9_HIGHBITDEPTH
        const int diff =
            bit_depth_ == VPX_BITS_8 ? dst[j] - src[j] : dst16[j] - src16[j];
#else
        const int diff = dst[j] - src[j];
#endif
        const int error = diff * diff;
        if (max_error < error) max_error = error;
        total_error += error;

        const int coeff_diff = test_temp_block[j] - ref_temp_block[j];
        total_coeff_error += abs(coeff_diff);
      }

      ASSERT_GE(1 << 2 * (bit_depth_ - 8), max_error)
          << "Error: Extremal 8x8 FDCT/IDCT or FHT/IHT has"
          << " an individual roundtrip error > 1";

      ASSERT_GE((count_test_block << 2 * (bit_depth_ - 8)) / 5, total_error)
          << "Error: Extremal 8x8 FDCT/IDCT or FHT/IHT has average"
          << " roundtrip error > 1/5 per block";

      ASSERT_EQ(0, total_coeff_error)
          << "Error: Extremal 8x8 FDCT/FHT has"
          << " overflow issues in the intermediate steps > 1";
    }
  }

  void RunInvAccuracyCheck() {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int count_test_block = 1000;
    DECLARE_ALIGNED(16, int16_t, in[kNumCoeffs]);
    DECLARE_ALIGNED(16, tran_low_t, coeff[kNumCoeffs]);
    DECLARE_ALIGNED(16, uint8_t, dst[kNumCoeffs]);
    DECLARE_ALIGNED(16, uint8_t, src[kNumCoeffs]);
#if CONFIG_VP9_HIGHBITDEPTH
    DECLARE_ALIGNED(16, uint16_t, src16[kNumCoeffs]);
    DECLARE_ALIGNED(16, uint16_t, dst16[kNumCoeffs]);
#endif

    for (int i = 0; i < count_test_block; ++i) {
      double out_r[kNumCoeffs];

      // Initialize a test block with input range [-255, 255].
      for (int j = 0; j < kNumCoeffs; ++j) {
        if (bit_depth_ == VPX_BITS_8) {
          src[j] = rnd.Rand8() % 2 ? 255 : 0;
          dst[j] = src[j] > 0 ? 0 : 255;
          in[j] = src[j] - dst[j];
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          src16[j] = rnd.Rand8() % 2 ? mask_ : 0;
          dst16[j] = src16[j] > 0 ? 0 : mask_;
          in[j] = src16[j] - dst16[j];
#endif
        }
      }

      reference_8x8_dct_2d(in, out_r);
      for (int j = 0; j < kNumCoeffs; ++j) {
        coeff[j] = static_cast<tran_low_t>(round(out_r[j]));
      }

      if (bit_depth_ == VPX_BITS_8) {
        ASM_REGISTER_STATE_CHECK(RunInvTxfm(coeff, dst, pitch_));
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        ASM_REGISTER_STATE_CHECK(
            RunInvTxfm(coeff, CAST_TO_BYTEPTR(dst16), pitch_));
#endif
      }

      for (int j = 0; j < kNumCoeffs; ++j) {
#if CONFIG_VP9_HIGHBITDEPTH
        const int diff =
            bit_depth_ == VPX_BITS_8 ? dst[j] - src[j] : dst16[j] - src16[j];
#else
        const int diff = dst[j] - src[j];
#endif
        const uint32_t error = diff * diff;
        ASSERT_GE(1u << 2 * (bit_depth_ - 8), error)
            << "Error: 8x8 IDCT has error " << error << " at index " << j;
      }
    }
  }

  void RunFwdAccuracyCheck() {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int count_test_block = 1000;
    DECLARE_ALIGNED(16, int16_t, in[kNumCoeffs]);
    DECLARE_ALIGNED(16, tran_low_t, coeff_r[kNumCoeffs]);
    DECLARE_ALIGNED(16, tran_low_t, coeff[kNumCoeffs]);

    for (int i = 0; i < count_test_block; ++i) {
      double out_r[kNumCoeffs];

      // Initialize a test block with input range [-mask_, mask_].
      for (int j = 0; j < kNumCoeffs; ++j) {
        in[j] = rnd.Rand8() % 2 == 0 ? mask_ : -mask_;
      }

      RunFwdTxfm(in, coeff, pitch_);
      reference_8x8_dct_2d(in, out_r);
      for (int j = 0; j < kNumCoeffs; ++j) {
        coeff_r[j] = static_cast<tran_low_t>(round(out_r[j]));
      }

      for (int j = 0; j < kNumCoeffs; ++j) {
        const int32_t diff = coeff[j] - coeff_r[j];
        const uint32_t error = diff * diff;
        ASSERT_GE(9u << 2 * (bit_depth_ - 8), error)
            << "Error: 8x8 DCT has error " << error << " at index " << j;
      }
    }
  }

  void CompareInvReference(IdctFunc ref_txfm, int thresh) {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    const int count_test_block = 10000;
    const int eob = 12;
    DECLARE_ALIGNED(16, tran_low_t, coeff[kNumCoeffs]);
    DECLARE_ALIGNED(16, uint8_t, dst[kNumCoeffs]);
    DECLARE_ALIGNED(16, uint8_t, ref[kNumCoeffs]);
#if CONFIG_VP9_HIGHBITDEPTH
    DECLARE_ALIGNED(16, uint16_t, dst16[kNumCoeffs]);
    DECLARE_ALIGNED(16, uint16_t, ref16[kNumCoeffs]);
#endif
    const int16_t *scan = vp9_default_scan_orders[TX_8X8].scan;

    for (int i = 0; i < count_test_block; ++i) {
      for (int j = 0; j < kNumCoeffs; ++j) {
        if (j < eob) {
          // Random values less than the threshold, either positive or negative
          coeff[scan[j]] = rnd(thresh) * (1 - 2 * (i % 2));
        } else {
          coeff[scan[j]] = 0;
        }
        if (bit_depth_ == VPX_BITS_8) {
          dst[j] = 0;
          ref[j] = 0;
#if CONFIG_VP9_HIGHBITDEPTH
        } else {
          dst16[j] = 0;
          ref16[j] = 0;
#endif
        }
      }
      if (bit_depth_ == VPX_BITS_8) {
        ref_txfm(coeff, ref, pitch_);
        ASM_REGISTER_STATE_CHECK(RunInvTxfm(coeff, dst, pitch_));
#if CONFIG_VP9_HIGHBITDEPTH
      } else {
        ref_txfm(coeff, CAST_TO_BYTEPTR(ref16), pitch_);
        ASM_REGISTER_STATE_CHECK(
            RunInvTxfm(coeff, CAST_TO_BYTEPTR(dst16), pitch_));
#endif
      }

      for (int j = 0; j < kNumCoeffs; ++j) {
#if CONFIG_VP9_HIGHBITDEPTH
        const int diff =
            bit_depth_ == VPX_BITS_8 ? dst[j] - ref[j] : dst16[j] - ref16[j];
#else
        const int diff = dst[j] - ref[j];
#endif
        const uint32_t error = diff * diff;
        ASSERT_EQ(0u, error)
            << "Error: 8x8 IDCT has error " << error << " at index " << j;
      }
    }
  }
  int pitch_;
  int tx_type_;
  FhtFunc fwd_txfm_ref;
  vpx_bit_depth_t bit_depth_;
  int mask_;
};
#ifdef AOM_WORK_AROUND_MSVC_BUG_10369786
#pragma optimize("", on)
#endif

class FwdTrans8x8DCT : public FwdTrans8x8TestBase,
                       public ::testing::TestWithParam<Dct8x8Param> {
 public:
  ~FwdTrans8x8DCT() override = default;

  void SetUp() override {
    fwd_txfm_ = GET_PARAM(0);
    inv_txfm_ = GET_PARAM(1);
    tx_type_ = GET_PARAM(2);
    pitch_ = 8;
    fwd_txfm_ref = fdct8x8_ref;
    bit_depth_ = GET_PARAM(3);
    mask_ = (1 << bit_depth_) - 1;
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  void RunFwdTxfm(int16_t *in, tran_low_t *out, int stride) override {
    fwd_txfm_(in, out, stride);
  }
  void RunInvTxfm(tran_low_t *out, uint8_t *dst, int stride) override {
    inv_txfm_(out, dst, stride);
  }

  FdctFunc fwd_txfm_;
  IdctFunc inv_txfm_;
};

TEST_P(FwdTrans8x8DCT, SignBiasCheck) { RunSignBiasCheck(); }

TEST_P(FwdTrans8x8DCT, RoundTripErrorCheck) { RunRoundTripErrorCheck(); }

TEST_P(FwdTrans8x8DCT, ExtremalCheck) { RunExtremalCheck(); }

TEST_P(FwdTrans8x8DCT, FwdAccuracyCheck) { RunFwdAccuracyCheck(); }

TEST_P(FwdTrans8x8DCT, InvAccuracyCheck) { RunInvAccuracyCheck(); }

class FwdTrans8x8HT : public FwdTrans8x8TestBase,
                      public ::testing::TestWithParam<Ht8x8Param> {
 public:
  ~FwdTrans8x8HT() override = default;

  void SetUp() override {
    fwd_txfm_ = GET_PARAM(0);
    inv_txfm_ = GET_PARAM(1);
    tx_type_ = GET_PARAM(2);
    pitch_ = 8;
    fwd_txfm_ref = fht8x8_ref;
    bit_depth_ = GET_PARAM(3);
    mask_ = (1 << bit_depth_) - 1;
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  void RunFwdTxfm(int16_t *in, tran_low_t *out, int stride) override {
    fwd_txfm_(in, out, stride, tx_type_);
  }
  void RunInvTxfm(tran_low_t *out, uint8_t *dst, int stride) override {
    inv_txfm_(out, dst, stride, tx_type_);
  }

  FhtFunc fwd_txfm_;
  IhtFunc inv_txfm_;
};

TEST_P(FwdTrans8x8HT, SignBiasCheck) { RunSignBiasCheck(); }

TEST_P(FwdTrans8x8HT, RoundTripErrorCheck) { RunRoundTripErrorCheck(); }

TEST_P(FwdTrans8x8HT, ExtremalCheck) { RunExtremalCheck(); }

#if HAVE_SSE2 && CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE
class InvTrans8x8DCT : public FwdTrans8x8TestBase,
                       public ::testing::TestWithParam<Idct8x8Param> {
 public:
  ~InvTrans8x8DCT() override = default;

  void SetUp() override {
    ref_txfm_ = GET_PARAM(0);
    inv_txfm_ = GET_PARAM(1);
    thresh_ = GET_PARAM(2);
    pitch_ = 8;
    bit_depth_ = GET_PARAM(3);
    mask_ = (1 << bit_depth_) - 1;
  }

  void TearDown() override { libvpx_test::ClearSystemState(); }

 protected:
  void RunInvTxfm(tran_low_t *out, uint8_t *dst, int stride) override {
    inv_txfm_(out, dst, stride);
  }
  void RunFwdTxfm(int16_t * /*out*/, tran_low_t * /*dst*/,
                  int /*stride*/) override {}

  IdctFunc ref_txfm_;
  IdctFunc inv_txfm_;
  int thresh_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(InvTrans8x8DCT);

TEST_P(InvTrans8x8DCT, CompareReference) {
  CompareInvReference(ref_txfm_, thresh_);
}
#endif  // HAVE_SSE2 && CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE

using std::make_tuple;

#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    C, FwdTrans8x8DCT,
    ::testing::Values(
        make_tuple(&vpx_fdct8x8_c, &vpx_idct8x8_64_add_c, 0, VPX_BITS_8),
        make_tuple(&vpx_highbd_fdct8x8_c, &idct8x8_10, 0, VPX_BITS_10),
        make_tuple(&vpx_highbd_fdct8x8_c, &idct8x8_12, 0, VPX_BITS_12)));
#else
INSTANTIATE_TEST_SUITE_P(C, FwdTrans8x8DCT,
                         ::testing::Values(make_tuple(&vpx_fdct8x8_c,
                                                      &vpx_idct8x8_64_add_c, 0,
                                                      VPX_BITS_8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH

#if CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    C, FwdTrans8x8HT,
    ::testing::Values(
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_c, 0, VPX_BITS_8),
        make_tuple(&vp9_highbd_fht8x8_c, &iht8x8_10, 0, VPX_BITS_10),
        make_tuple(&vp9_highbd_fht8x8_c, &iht8x8_10, 1, VPX_BITS_10),
        make_tuple(&vp9_highbd_fht8x8_c, &iht8x8_10, 2, VPX_BITS_10),
        make_tuple(&vp9_highbd_fht8x8_c, &iht8x8_10, 3, VPX_BITS_10),
        make_tuple(&vp9_highbd_fht8x8_c, &iht8x8_12, 0, VPX_BITS_12),
        make_tuple(&vp9_highbd_fht8x8_c, &iht8x8_12, 1, VPX_BITS_12),
        make_tuple(&vp9_highbd_fht8x8_c, &iht8x8_12, 2, VPX_BITS_12),
        make_tuple(&vp9_highbd_fht8x8_c, &iht8x8_12, 3, VPX_BITS_12),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_c, 1, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_c, 2, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_c, 3, VPX_BITS_8)));
#else
INSTANTIATE_TEST_SUITE_P(
    C, FwdTrans8x8HT,
    ::testing::Values(
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_c, 0, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_c, 1, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_c, 2, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_c, 3, VPX_BITS_8)));
#endif  // CONFIG_VP9_HIGHBITDEPTH

#if HAVE_NEON && !CONFIG_EMULATE_HARDWARE
INSTANTIATE_TEST_SUITE_P(NEON, FwdTrans8x8DCT,
                         ::testing::Values(make_tuple(&vpx_fdct8x8_neon,
                                                      &vpx_idct8x8_64_add_neon,
                                                      0, VPX_BITS_8)));

#if !CONFIG_VP9_HIGHBITDEPTH
INSTANTIATE_TEST_SUITE_P(
    NEON, FwdTrans8x8HT,
    ::testing::Values(
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_neon, 0, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_neon, 1, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_neon, 2, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_c, &vp9_iht8x8_64_add_neon, 3, VPX_BITS_8)));
#endif  // !CONFIG_VP9_HIGHBITDEPTH
#endif  // HAVE_NEON && !CONFIG_EMULATE_HARDWARE

#if HAVE_SSE2 && !CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE
INSTANTIATE_TEST_SUITE_P(SSE2, FwdTrans8x8DCT,
                         ::testing::Values(make_tuple(&vpx_fdct8x8_sse2,
                                                      &vpx_idct8x8_64_add_sse2,
                                                      0, VPX_BITS_8)));
INSTANTIATE_TEST_SUITE_P(
    SSE2, FwdTrans8x8HT,
    ::testing::Values(
        make_tuple(&vp9_fht8x8_sse2, &vp9_iht8x8_64_add_sse2, 0, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_sse2, &vp9_iht8x8_64_add_sse2, 1, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_sse2, &vp9_iht8x8_64_add_sse2, 2, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_sse2, &vp9_iht8x8_64_add_sse2, 3, VPX_BITS_8)));
#endif  // HAVE_SSE2 && !CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE

#if HAVE_SSE2 && CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE
INSTANTIATE_TEST_SUITE_P(
    SSE2, FwdTrans8x8DCT,
    ::testing::Values(make_tuple(&vpx_fdct8x8_sse2, &vpx_idct8x8_64_add_c, 0,
                                 VPX_BITS_8),
                      make_tuple(&vpx_highbd_fdct8x8_c, &idct8x8_64_add_10_sse2,
                                 12, VPX_BITS_10),
                      make_tuple(&vpx_highbd_fdct8x8_sse2,
                                 &idct8x8_64_add_10_sse2, 12, VPX_BITS_10),
                      make_tuple(&vpx_highbd_fdct8x8_c, &idct8x8_64_add_12_sse2,
                                 12, VPX_BITS_12),
                      make_tuple(&vpx_highbd_fdct8x8_sse2,
                                 &idct8x8_64_add_12_sse2, 12, VPX_BITS_12)));

INSTANTIATE_TEST_SUITE_P(
    SSE2, FwdTrans8x8HT,
    ::testing::Values(
        make_tuple(&vp9_fht8x8_sse2, &vp9_iht8x8_64_add_c, 0, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_sse2, &vp9_iht8x8_64_add_c, 1, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_sse2, &vp9_iht8x8_64_add_c, 2, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_sse2, &vp9_iht8x8_64_add_c, 3, VPX_BITS_8)));

// Optimizations take effect at a threshold of 6201, so we use a value close to
// that to test both branches.
INSTANTIATE_TEST_SUITE_P(
    SSE2, InvTrans8x8DCT,
    ::testing::Values(
        make_tuple(&idct8x8_12_add_10_c, &idct8x8_12_add_10_sse2, 6225,
                   VPX_BITS_10),
        make_tuple(&idct8x8_10, &idct8x8_64_add_10_sse2, 6225, VPX_BITS_10),
        make_tuple(&idct8x8_12_add_12_c, &idct8x8_12_add_12_sse2, 6225,
                   VPX_BITS_12),
        make_tuple(&idct8x8_12, &idct8x8_64_add_12_sse2, 6225, VPX_BITS_12)));
#endif  // HAVE_SSE2 && CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE

#if HAVE_SSSE3 && VPX_ARCH_X86_64 && !CONFIG_VP9_HIGHBITDEPTH && \
    !CONFIG_EMULATE_HARDWARE
INSTANTIATE_TEST_SUITE_P(SSSE3, FwdTrans8x8DCT,
                         ::testing::Values(make_tuple(&vpx_fdct8x8_ssse3,
                                                      &vpx_idct8x8_64_add_sse2,
                                                      0, VPX_BITS_8)));
#endif

#if HAVE_MSA && !CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE
INSTANTIATE_TEST_SUITE_P(MSA, FwdTrans8x8DCT,
                         ::testing::Values(make_tuple(&vpx_fdct8x8_msa,
                                                      &vpx_idct8x8_64_add_msa,
                                                      0, VPX_BITS_8)));
INSTANTIATE_TEST_SUITE_P(
    MSA, FwdTrans8x8HT,
    ::testing::Values(
        make_tuple(&vp9_fht8x8_msa, &vp9_iht8x8_64_add_msa, 0, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_msa, &vp9_iht8x8_64_add_msa, 1, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_msa, &vp9_iht8x8_64_add_msa, 2, VPX_BITS_8),
        make_tuple(&vp9_fht8x8_msa, &vp9_iht8x8_64_add_msa, 3, VPX_BITS_8)));
#endif  // HAVE_MSA && !CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE

#if HAVE_VSX && !CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE
INSTANTIATE_TEST_SUITE_P(VSX, FwdTrans8x8DCT,
                         ::testing::Values(make_tuple(&vpx_fdct8x8_c,
                                                      &vpx_idct8x8_64_add_vsx,
                                                      0, VPX_BITS_8)));
#endif  // HAVE_VSX && !CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE

#if HAVE_LSX && !CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE
INSTANTIATE_TEST_SUITE_P(LSX, FwdTrans8x8DCT,
                         ::testing::Values(make_tuple(&vpx_fdct8x8_lsx,
                                                      &vpx_idct8x8_64_add_c, 0,
                                                      VPX_BITS_8)));
#endif  // HAVE_LSX && !CONFIG_VP9_HIGHBITDEPTH && !CONFIG_EMULATE_HARDWARE
}  // namespace
