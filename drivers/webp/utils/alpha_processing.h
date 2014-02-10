// Copyright 2013 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Utilities for processing transparent channel.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_UTILS_ALPHA_PROCESSING_H_
#define WEBP_UTILS_ALPHA_PROCESSING_H_

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pre-Multiply operation transforms x into x * A / 255  (where x=Y,R,G or B).
// Un-Multiply operation transforms x into x * 255 / A.

// Pre-Multiply or Un-Multiply (if 'inverse' is true) argb values in a row.
void WebPMultARGBRow(uint32_t* const ptr, int width, int inverse);

// Same a WebPMultARGBRow(), but for several rows.
void WebPMultARGBRows(uint8_t* ptr, int stride, int width, int num_rows,
                      int inverse);

// Same for a row of single values, with side alpha values.
void WebPMultRow(uint8_t* const ptr, const uint8_t* const alpha,
                 int width, int inverse);

// Same a WebPMultRow(), but for several 'num_rows' rows.
void WebPMultRows(uint8_t* ptr, int stride,
                  const uint8_t* alpha, int alpha_stride,
                  int width, int num_rows, int inverse);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif    // WEBP_UTILS_ALPHA_PROCESSING_H_
