/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <stdio.h>
#include <string.h>
#include "./vpx_config.h"
#include "vpx_ports/mips.h"

#if CONFIG_RUNTIME_CPU_DETECT
#if defined(__mips__) && defined(__linux__)
int mips_cpu_caps(void) {
  char cpuinfo_line[512];
  int flag = 0x0;
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (!f) {
    // Assume nothing if /proc/cpuinfo is unavailable.
    // This will occur for Chrome sandbox for Pepper or Render process.
    return 0;
  }
  while (fgets(cpuinfo_line, sizeof(cpuinfo_line) - 1, f)) {
    if (memcmp(cpuinfo_line, "cpu model", 9) == 0) {
      // Workaround early kernel without mmi in ASEs line.
      if (strstr(cpuinfo_line, "Loongson-3")) {
        flag |= HAS_MMI;
      } else if (strstr(cpuinfo_line, "Loongson-2K")) {
        flag |= HAS_MMI | HAS_MSA;
      }
    }
    if (memcmp(cpuinfo_line, "ASEs implemented", 16) == 0) {
      if (strstr(cpuinfo_line, "loongson-mmi") &&
          strstr(cpuinfo_line, "loongson-ext")) {
        flag |= HAS_MMI;
      }
      if (strstr(cpuinfo_line, "msa")) {
        flag |= HAS_MSA;
      }
      // ASEs is the last line, so we can break here.
      break;
    }
  }
  fclose(f);
  return flag;
}
#else /* end __mips__ && __linux__ */
#error \
    "--enable-runtime-cpu-detect selected, but no CPU detection method " \
"available for your platform. Reconfigure with --disable-runtime-cpu-detect."
#endif
#else /* end CONFIG_RUNTIME_CPU_DETECT */
int mips_cpu_caps(void) { return 0; }
#endif
