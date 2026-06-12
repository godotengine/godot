/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_config.h"
#include "vp8_rtcd.h"
#if VPX_ARCH_ARM
#include "vpx_ports/arm.h"
#elif VPX_ARCH_X86 || VPX_ARCH_X86_64
#include "vpx_ports/x86.h"
#elif VPX_ARCH_PPC
#include "vpx_ports/ppc.h"
#elif VPX_ARCH_MIPS
#include "vpx_ports/mips.h"
#elif VPX_ARCH_LOONGARCH
#include "vpx_ports/loongarch.h"
#endif
#include "vp8/common/onyxc_int.h"
#include "vp8/common/systemdependent.h"

#if CONFIG_MULTITHREAD
#if HAVE_UNISTD_H
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
typedef void(WINAPI *PGNSI)(LPSYSTEM_INFO);
#endif
#endif

#if CONFIG_MULTITHREAD
static int get_cpu_count(void) {
  int core_count = 16;

#if HAVE_UNISTD_H
#if defined(_SC_NPROCESSORS_ONLN)
  core_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(_SC_NPROC_ONLN)
  core_count = (int)sysconf(_SC_NPROC_ONLN);
#endif
#elif defined(_WIN32)
  {
#if _WIN32_WINNT < 0x0501
#error _WIN32_WINNT must target Windows XP or newer.
#endif
    SYSTEM_INFO sysinfo;
    GetNativeSystemInfo(&sysinfo);
    core_count = (int)sysinfo.dwNumberOfProcessors;
  }
#else
/* other platforms */
#endif

  return core_count > 0 ? core_count : 1;
}
#endif

void vp8_machine_specific_config(VP8_COMMON *ctx) {
#if CONFIG_MULTITHREAD
  ctx->processor_core_count = get_cpu_count();
#else
  (void)ctx;
#endif /* CONFIG_MULTITHREAD */
}
