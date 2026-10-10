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
#include "vp9/encoder/vp9_tokenize.h"

namespace {

TEST(VP9Tokenize, Vp9GetTokenInBounds) {
  EXPECT_LE(EOB_TOKEN, MAX_TOKEN);
  for (int i = -CAT6_MIN_VAL; i <= CAT6_MIN_VAL; ++i) {
    auto v = vp9_get_token(i);
    EXPECT_LE(v, MAX_TOKEN) << "vp9_get_token(" << i << ")";
    EXPECT_GE(v, 0) << "vp9_get_token(" << i << ")";
  }
}

}  // namespace
