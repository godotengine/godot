// Copyright 2013 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// TODO(skal): implement gradient smoothing.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./quant_levels_dec.h"

int DequantizeLevels(uint8_t* const data, int width, int height,
                     int row, int num_rows) {
  if (data == NULL || width <= 0 || height <= 0 || row < 0 || num_rows < 0 ||
      row + num_rows > height) {
    return 0;
  }
  return 1;
}

