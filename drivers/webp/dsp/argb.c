// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//   ARGB making functions.
//
// Author: Djordje Pesut (djordje.pesut@imgtec.com)

#include "./dsp.h"

static WEBP_INLINE uint32_t MakeARGB32(int a, int r, int g, int b) {
  return (((uint32_t)a << 24) | (r << 16) | (g << 8) | b);
}

static void PackARGB(const uint8_t* a, const uint8_t* r, const uint8_t* g,
                     const uint8_t* b, int len, uint32_t* out) {
  int i;
  for (i = 0; i < len; ++i) {
    out[i] = MakeARGB32(a[4 * i], r[4 * i], g[4 * i], b[4 * i]);
  }
}

static void PackRGB(const uint8_t* r, const uint8_t* g, const uint8_t* b,
                    int len, int step, uint32_t* out) {
  int i, offset = 0;
  for (i = 0; i < len; ++i) {
    out[i] = MakeARGB32(0xff, r[offset], g[offset], b[offset]);
    offset += step;
  }
}

void (*VP8PackARGB)(const uint8_t*, const uint8_t*, const uint8_t*,
                    const uint8_t*, int, uint32_t*);
void (*VP8PackRGB)(const uint8_t*, const uint8_t*, const uint8_t*,
                   int, int, uint32_t*);

extern void VP8EncDspARGBInitMIPSdspR2(void);
extern void VP8EncDspARGBInitSSE2(void);

static volatile VP8CPUInfo argb_last_cpuinfo_used =
    (VP8CPUInfo)&argb_last_cpuinfo_used;

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspARGBInit(void) {
  if (argb_last_cpuinfo_used == VP8GetCPUInfo) return;

  VP8PackARGB = PackARGB;
  VP8PackRGB = PackRGB;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_USE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8EncDspARGBInitSSE2();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      VP8EncDspARGBInitMIPSdspR2();
    }
#endif
  }
  argb_last_cpuinfo_used = VP8GetCPUInfo;
}
