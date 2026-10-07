/*
 *  Copyright (c) 2026 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "gtest/gtest.h"
#include "vp9/common/vp9_entropy.h"

namespace {

TEST(VP9Entropy, PtEnergyClassValuesInBound) {
  for (int i = 0; i < ENTROPY_TOKENS; ++i) {
    EXPECT_LE(vp9_pt_energy_class[i], MAX_ENERGY_CLASS)
        << "vp9_pt_energy_class[" << i << "]";
    EXPECT_GE(vp9_pt_energy_class[i], 0) << "vp9_pt_energy_class[" << i << "]";
  }
}

}  // namespace
