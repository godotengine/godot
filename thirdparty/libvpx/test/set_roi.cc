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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "gtest/gtest.h"
#include "test/acm_random.h"
#include "vp8/encoder/onyx_int.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"

using libvpx_test::ACMRandom;

namespace {

TEST(VP8RoiMapTest, ParameterCheck) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  int delta_q[MAX_MB_SEGMENTS] = { -2, -25, 0, 31 };
  int delta_lf[MAX_MB_SEGMENTS] = { -2, -25, 0, 31 };
  unsigned int threshold[MAX_MB_SEGMENTS] = { 0, 100, 200, 300 };

  const int internalq_trans[] = {
    0,  1,  2,  3,  4,  5,  7,   8,   9,   10,  12,  13,  15,  17,  18,  19,
    20, 21, 23, 24, 25, 26, 27,  28,  29,  30,  31,  33,  35,  37,  39,  41,
    43, 45, 47, 49, 51, 53, 55,  57,  59,  61,  64,  67,  70,  73,  76,  79,
    82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127,
  };

  // Initialize elements of cpi with valid defaults.
  VP8_COMP cpi;
  cpi.mb.e_mbd.mb_segment_abs_delta = SEGMENT_DELTADATA;
  cpi.cyclic_refresh_mode_enabled = 0;
  cpi.mb.e_mbd.segmentation_enabled = 0;
  cpi.mb.e_mbd.update_mb_segmentation_map = 0;
  cpi.mb.e_mbd.update_mb_segmentation_data = 0;
  cpi.common.mb_rows = 240 >> 4;
  cpi.common.mb_cols = 320 >> 4;
  const int mbs = (cpi.common.mb_rows * cpi.common.mb_cols);
  memset(cpi.segment_feature_data, 0, sizeof(cpi.segment_feature_data));

  // Segment map
  cpi.segmentation_map = reinterpret_cast<unsigned char *>(vpx_calloc(mbs, 1));

  // Allocate memory for the source memory map.
  unsigned char *roi_map =
      reinterpret_cast<unsigned char *>(vpx_calloc(mbs, 1));
  memset(&roi_map[mbs >> 2], 1, (mbs >> 2));
  memset(&roi_map[mbs >> 1], 2, (mbs >> 2));
  memset(&roi_map[mbs - (mbs >> 2)], 3, (mbs >> 2));

  // Do a test call with valid parameters.
  int roi_retval =
      vp8_set_roimap(&cpi, roi_map, cpi.common.mb_rows, cpi.common.mb_cols,
                     delta_q, delta_lf, threshold);
  EXPECT_EQ(0, roi_retval)
      << "vp8_set_roimap roi failed with default test parameters";

  // Check that the values in the cpi structure get set as expected.
  if (roi_retval == 0) {
    // Check that the segment map got set.
    const int mapcompare = memcmp(roi_map, cpi.segmentation_map, mbs);
    EXPECT_EQ(0, mapcompare) << "segment map error";

    // Check the q deltas (note the need to translate into
    // the interanl range of 0-127.
    for (int i = 0; i < MAX_MB_SEGMENTS; ++i) {
      const int transq = internalq_trans[abs(delta_q[i])];
      if (abs(cpi.segment_feature_data[MB_LVL_ALT_Q][i]) != transq) {
        EXPECT_EQ(transq, cpi.segment_feature_data[MB_LVL_ALT_Q][i])
            << "segment delta_q  error";
        break;
      }
    }

    // Check the loop filter deltas
    for (int i = 0; i < MAX_MB_SEGMENTS; ++i) {
      if (cpi.segment_feature_data[MB_LVL_ALT_LF][i] != delta_lf[i]) {
        EXPECT_EQ(delta_lf[i], cpi.segment_feature_data[MB_LVL_ALT_LF][i])
            << "segment delta_lf error";
        break;
      }
    }

    // Check the breakout thresholds
    for (int i = 0; i < MAX_MB_SEGMENTS; ++i) {
      unsigned int breakout =
          static_cast<unsigned int>(cpi.segment_encode_breakout[i]);

      if (threshold[i] != breakout) {
        EXPECT_EQ(threshold[i], breakout) << "breakout threshold error";
        break;
      }
    }

    // Segmentation, and segmentation update flages should be set.
    EXPECT_EQ(1, cpi.mb.e_mbd.segmentation_enabled)
        << "segmentation_enabled error";
    EXPECT_EQ(1, cpi.mb.e_mbd.update_mb_segmentation_map)
        << "update_mb_segmentation_map error";
    EXPECT_EQ(1, cpi.mb.e_mbd.update_mb_segmentation_data)
        << "update_mb_segmentation_data error";

    // Try a range of delta q and lf parameters (some legal, some not)
    for (int i = 0; i < 1000; ++i) {
      int rand_deltas[4];
      int deltas_valid;
      rand_deltas[0] = rnd(160) - 80;
      rand_deltas[1] = rnd(160) - 80;
      rand_deltas[2] = rnd(160) - 80;
      rand_deltas[3] = rnd(160) - 80;

      deltas_valid =
          ((abs(rand_deltas[0]) <= 63) && (abs(rand_deltas[1]) <= 63) &&
           (abs(rand_deltas[2]) <= 63) && (abs(rand_deltas[3]) <= 63))
              ? 0
              : -1;

      // Test with random delta q values.
      roi_retval =
          vp8_set_roimap(&cpi, roi_map, cpi.common.mb_rows, cpi.common.mb_cols,
                         rand_deltas, delta_lf, threshold);
      EXPECT_EQ(deltas_valid, roi_retval) << "dq range check error";

      // One delta_q error shown at a time
      if (deltas_valid != roi_retval) break;

      // Test with random loop filter values.
      roi_retval =
          vp8_set_roimap(&cpi, roi_map, cpi.common.mb_rows, cpi.common.mb_cols,
                         delta_q, rand_deltas, threshold);
      EXPECT_EQ(deltas_valid, roi_retval) << "dlf range check error";

      // One delta loop filter error shown at a time
      if (deltas_valid != roi_retval) break;
    }

    // Test invalid number of rows or colums.
    roi_retval =
        vp8_set_roimap(&cpi, roi_map, cpi.common.mb_rows + 1,
                       cpi.common.mb_cols, delta_q, delta_lf, threshold);
    EXPECT_EQ(-1, roi_retval) << "MB rows bounds check error";

    roi_retval =
        vp8_set_roimap(&cpi, roi_map, cpi.common.mb_rows,
                       cpi.common.mb_cols - 1, delta_q, delta_lf, threshold);
    EXPECT_EQ(-1, roi_retval) << "MB cols bounds check error";
  }

  // Free allocated memory
  if (cpi.segmentation_map) vpx_free(cpi.segmentation_map);
  if (roi_map) vpx_free(roi_map);
}

}  // namespace
