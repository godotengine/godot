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

#include "./vpx_config.h"
#include "vpx_ports/loongarch.h"

#define LOONGARCH_CFG2 0x02
#define LOONGARCH_CFG2_LSX (1 << 6)
#define LOONGARCH_CFG2_LASX (1 << 7)

#if CONFIG_RUNTIME_CPU_DETECT
#if defined(__loongarch__) && defined(__linux__)
int loongarch_cpu_caps(void) {
  int reg = 0;
  int flag = 0;

  __asm__ volatile("cpucfg %0, %1 \n\t" : "+&r"(reg) : "r"(LOONGARCH_CFG2));
  if (reg & LOONGARCH_CFG2_LSX) flag |= HAS_LSX;

  if (reg & LOONGARCH_CFG2_LASX) flag |= HAS_LASX;

  return flag;
}
#else /* end __loongarch__ && __linux__ */
#error \
    "--enable-runtime-cpu-detect selected, but no CPU detection method " \
"available for your platform. Reconfigure with --disable-runtime-cpu-detect."
#endif
#else /* end CONFIG_RUNTIME_CPU_DETECT */
int loongarch_cpu_caps(void) { return 0; }
#endif
