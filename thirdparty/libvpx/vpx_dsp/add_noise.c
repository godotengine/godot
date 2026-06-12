/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <stdlib.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vpx_dsp/postproc.h"
#include "vpx_ports/mem.h"

void vpx_plane_add_noise_c(uint8_t *start, const int8_t *noise, int blackclamp,
                           int whiteclamp, int width, int height, int pitch) {
  int i, j;
  int bothclamp = blackclamp + whiteclamp;
  for (i = 0; i < height; ++i) {
    uint8_t *pos = start + i * pitch;
    const int8_t *ref = (const int8_t *)(noise + (rand() & 0xff));  // NOLINT

    for (j = 0; j < width; ++j) {
      int v = pos[j];

      v = clamp(v - blackclamp, 0, 255);
      v = clamp(v + bothclamp, 0, 255);
      v = clamp(v - whiteclamp, 0, 255);

      pos[j] = v + ref[j];
    }
  }
}

static double gaussian(double sigma, double mu, double x) {
  return 1 / (sigma * sqrt(2.0 * 3.14159265)) *
         (exp(-(x - mu) * (x - mu) / (2 * sigma * sigma)));
}

int vpx_setup_noise(double sigma, int8_t *noise, int size) {
  int8_t char_dist[256];
  int next = 0, i, j;

  // set up a 256 entry lookup that matches gaussian distribution
  for (i = -32; i < 32; ++i) {
    const int a_i = (int)(0.5 + 256 * gaussian(sigma, 0, i));
    if (a_i) {
      for (j = 0; j < a_i; ++j) {
        if (next + j >= 256) goto set_noise;
        char_dist[next + j] = (int8_t)i;
      }
      next = next + j;
    }
  }

  // Rounding error - might mean we have less than 256.
  for (; next < 256; ++next) {
    char_dist[next] = 0;
  }

set_noise:
  for (i = 0; i < size; ++i) {
    noise[i] = char_dist[rand() & 0xff];  // NOLINT
  }

  // Returns the highest non 0 value used in distribution.
  return -char_dist[0];
}
