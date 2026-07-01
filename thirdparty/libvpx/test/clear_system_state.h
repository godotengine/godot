/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_TEST_CLEAR_SYSTEM_STATE_H_
#define VPX_TEST_CLEAR_SYSTEM_STATE_H_

#include "./vpx_config.h"
#include "vpx_ports/system_state.h"

namespace libvpx_test {

// Reset system to a known state. This function should be used for all non-API
// test cases.
inline void ClearSystemState() { vpx_clear_system_state(); }

}  // namespace libvpx_test
#endif  // VPX_TEST_CLEAR_SYSTEM_STATE_H_
