/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "./vpx_config.h"
#define RTCD_C
#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/vpx_once.h"

void vpx_dsp_rtcd() {
  once(setup_rtcd_internal);
}
