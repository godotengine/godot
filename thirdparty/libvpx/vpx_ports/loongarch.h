/*
 * Copyright (c) 2021 Loongson Technology Corporation Limited
 * Contributed by Jin Bo  <jinbo@loongson.cn>
 * Contributed by Lu Wang <wanglu@loongson.cn>
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_LOONGARCH_H_
#define VPX_VPX_PORTS_LOONGARCH_H_

#ifdef __cplusplus
extern "C" {
#endif

#define HAS_LSX 0x01
#define HAS_LASX 0x02

int loongarch_cpu_caps(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_PORTS_LOONGARCH_H_
