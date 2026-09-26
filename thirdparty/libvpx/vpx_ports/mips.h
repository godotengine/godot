/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_MIPS_H_
#define VPX_VPX_PORTS_MIPS_H_

#ifdef __cplusplus
extern "C" {
#endif

#define HAS_MMI 0x01
#define HAS_MSA 0x02

int mips_cpu_caps(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_PORTS_MIPS_H_
