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
#include "vp9/common/vp9_scan.h"

namespace {

constexpr int kNumValues[] = { 4 * 4, 8 * 8, 16 * 16, 32 * 32 };

TEST(VP9Scan, DefaultScanOrderValuesInBound) {
  for (int tx_size = 0; tx_size < TX_SIZES; ++tx_size) {
    const ScanOrder &scan_order = vp9_default_scan_orders[tx_size];
    for (int i = 0; i < kNumValues[tx_size]; ++i) {
      EXPECT_LE(scan_order.scan[i], MAX_SCAN_VALUE)
          << "vp9_default_scan_orders[" << tx_size << "].scan[" << i << "]";
      EXPECT_GE(scan_order.scan[i], 0)
          << "vp9_default_scan_orders[" << tx_size << "].scan[" << i << "]";
      EXPECT_LE(scan_order.iscan[i], MAX_SCAN_VALUE)
          << "vp9_default_scan_orders[" << tx_size << "].iscan[" << i << "]";
      EXPECT_GE(scan_order.iscan[i], 0)
          << "vp9_default_scan_orders[" << tx_size << "].iscan[" << i << "]";
    }
  }
}

TEST(VP9Scan, ScanOrderValuesInBound) {
  for (int tx_size = 0; tx_size < TX_SIZES; ++tx_size) {
    for (int tx_type = 0; tx_type < TX_TYPES; ++tx_type) {
      const ScanOrder &scan_order = vp9_scan_orders[tx_size][tx_type];
      for (int i = 0; i < kNumValues[tx_size]; ++i) {
        EXPECT_LE(scan_order.scan[i], MAX_SCAN_VALUE)
            << "vp9_scan_orders[" << tx_size << "][" << tx_type << "].scan["
            << i << "]";
        EXPECT_GE(scan_order.scan[i], 0)
            << "vp9_scan_orders[" << tx_size << "][" << tx_type << "].scan["
            << i << "]";
        EXPECT_LE(scan_order.iscan[i], MAX_SCAN_VALUE)
            << "vp9_scan_orders[" << tx_size << "][" << tx_type << "].iscan["
            << i << "]";
        EXPECT_GE(scan_order.iscan[i], 0)
            << "vp9_scan_orders[" << tx_size << "][" << tx_type << "].iscan["
            << i << "]";
      }
    }
  }
}

}  // namespace
