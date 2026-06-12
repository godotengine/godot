/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <asm/cputable.h>
#include <linux/auxvec.h>

#include "./vpx_config.h"
#include "vpx_ports/ppc.h"

#if CONFIG_RUNTIME_CPU_DETECT
static int cpu_env_flags(int *flags) {
  char *env;
  env = getenv("VPX_SIMD_CAPS");
  if (env && *env) {
    *flags = (int)strtol(env, NULL, 0);
    return 0;
  }
  *flags = 0;
  return -1;
}

static int cpu_env_mask(void) {
  char *env;
  env = getenv("VPX_SIMD_CAPS_MASK");
  return env && *env ? (int)strtol(env, NULL, 0) : ~0;
}

int ppc_simd_caps(void) {
  int flags;
  int mask;
  int fd;
  ssize_t count;
  unsigned int i;
  uint64_t buf[64];

  // If VPX_SIMD_CAPS is set then allow only those capabilities.
  if (!cpu_env_flags(&flags)) {
    return flags;
  }

  mask = cpu_env_mask();

  fd = open("/proc/self/auxv", O_RDONLY);
  if (fd < 0) {
    return 0;
  }

  while ((count = read(fd, buf, sizeof(buf))) > 0) {
    for (i = 0; i < (count / sizeof(*buf)); i += 2) {
      if (buf[i] == AT_HWCAP) {
#if HAVE_VSX
        if (buf[i + 1] & PPC_FEATURE_HAS_VSX) {
          flags |= HAS_VSX;
        }
#endif  // HAVE_VSX
        goto out_close;
      } else if (buf[i] == AT_NULL) {
        goto out_close;
      }
    }
  }
out_close:
  close(fd);
  return flags & mask;
}
#else
// If there is no RTCD the function pointers are not used and can not be
// changed.
int ppc_simd_caps(void) { return 0; }
#endif  // CONFIG_RUNTIME_CPU_DETECT
