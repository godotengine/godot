/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "./vpx_config.h"
#define RTCD_C
#include "./vp8_rtcd.h"
#include "vpx_ports/vpx_once.h"


void vp8_rtcd()
{
    once(setup_rtcd_internal);
}
