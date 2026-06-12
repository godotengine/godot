/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_STATIC_ASSERT_H_
#define VPX_VPX_PORTS_STATIC_ASSERT_H_

#if defined(_MSC_VER)
#define VPX_STATIC_ASSERT(boolexp)              \
  do {                                          \
    char vpx_static_assert[(boolexp) ? 1 : -1]; \
    (void)vpx_static_assert;                    \
  } while (0)
#else  // !_MSC_VER
#define VPX_STATIC_ASSERT(boolexp)                         \
  do {                                                     \
    struct {                                               \
      unsigned int vpx_static_assert : (boolexp) ? 1 : -1; \
    } vpx_static_assert;                                   \
    (void)vpx_static_assert;                               \
  } while (0)
#endif  // _MSC_VER

#endif  // VPX_VPX_PORTS_STATIC_ASSERT_H_
