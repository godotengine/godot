/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vpx/vpx_codec.h"
#include "vpx/vpx_integer.h"
#include "./y4minput.h"
#include "vpx_dsp/ssim.h"
#include "vpx_ports/mem.h"

static const int64_t cc1 = 26634;        // (64^2*(.01*255)^2
static const int64_t cc2 = 239708;       // (64^2*(.03*255)^2
static const int64_t cc1_10 = 428658;    // (64^2*(.01*1023)^2
static const int64_t cc2_10 = 3857925;   // (64^2*(.03*1023)^2
static const int64_t cc1_12 = 6868593;   // (64^2*(.01*4095)^2
static const int64_t cc2_12 = 61817334;  // (64^2*(.03*4095)^2

#if CONFIG_VP9_HIGHBITDEPTH
static uint64_t calc_plane_error16(uint16_t *orig, int orig_stride,
                                   uint16_t *recon, int recon_stride,
                                   unsigned int cols, unsigned int rows) {
  unsigned int row, col;
  uint64_t total_sse = 0;
  int diff;
  if (orig == NULL || recon == NULL) {
    assert(0);
    return 0;
  }

  for (row = 0; row < rows; row++) {
    for (col = 0; col < cols; col++) {
      diff = orig[col] - recon[col];
      total_sse += diff * diff;
    }

    orig += orig_stride;
    recon += recon_stride;
  }
  return total_sse;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static uint64_t calc_plane_error(uint8_t *orig, int orig_stride, uint8_t *recon,
                                 int recon_stride, unsigned int cols,
                                 unsigned int rows) {
  unsigned int row, col;
  uint64_t total_sse = 0;
  int diff;
  if (orig == NULL || recon == NULL) {
    assert(0);
    return 0;
  }

  for (row = 0; row < rows; row++) {
    for (col = 0; col < cols; col++) {
      diff = orig[col] - recon[col];
      total_sse += diff * diff;
    }

    orig += orig_stride;
    recon += recon_stride;
  }
  return total_sse;
}

#define MAX_PSNR 100
static double mse2psnr(double samples, double peak, double mse) {
  double psnr;

  if (mse > 0.0)
    psnr = 10.0 * log10(peak * peak * samples / mse);
  else
    psnr = MAX_PSNR;  // Limit to prevent / 0

  if (psnr > MAX_PSNR) psnr = MAX_PSNR;

  return psnr;
}

typedef enum { RAW_YUV, Y4M } input_file_type;

typedef struct input_file {
  FILE *file;
  input_file_type type;
  unsigned char *buf;
  y4m_input y4m;
  vpx_image_t img;
  int w;
  int h;
  int bit_depth;
  int frame_size;
} input_file_t;

// Open a file and determine if its y4m or raw.  If y4m get the header.
static int open_input_file(const char *file_name, input_file_t *input, int w,
                           int h, int bit_depth) {
  char y4m_buf[4];
  input->w = w;
  input->h = h;
  input->bit_depth = bit_depth;
  input->type = RAW_YUV;
  input->buf = NULL;
  input->file = strcmp(file_name, "-") ? fopen(file_name, "rb") : stdin;
  if (input->file == NULL) return -1;
  if (fread(y4m_buf, 1, 4, input->file) != 4) return -1;
  if (memcmp(y4m_buf, "YUV4", 4) == 0) input->type = Y4M;
  switch (input->type) {
    case Y4M:
      y4m_input_open(&input->y4m, input->file, y4m_buf, 4, 0);
      input->w = input->y4m.pic_w;
      input->h = input->y4m.pic_h;
      input->bit_depth = input->y4m.bit_depth;
      // Y4M alloc's its own buf. Init this to avoid problems if we never
      // read frames.
      memset(&input->img, 0, sizeof(input->img));
      break;
    case RAW_YUV:
      fseek(input->file, 0, SEEK_SET);
      input->w = w;
      input->h = h;
      // handle odd frame sizes
      input->frame_size = w * h + ((w + 1) / 2) * ((h + 1) / 2) * 2;
      if (bit_depth > 8) {
        input->frame_size *= 2;
      }
      input->buf = malloc(input->frame_size);
      break;
  }
  return 0;
}

static void close_input_file(input_file_t *in) {
  if (in->file) fclose(in->file);
  if (in->type == Y4M) {
    vpx_img_free(&in->img);
  } else {
    free(in->buf);
  }
}

// Returns 1 on success, 0 on failure due to a read error or eof (or format
// error in the case of y4m).
static int read_input_file(input_file_t *in, unsigned char **y,
                           unsigned char **u, unsigned char **v, int bd) {
  size_t r1 = 0;
  switch (in->type) {
    case Y4M:
      r1 = y4m_input_fetch_frame(&in->y4m, in->file, &in->img);
      if (r1 == (size_t)-1) return 0;
      *y = in->img.planes[0];
      *u = in->img.planes[1];
      *v = in->img.planes[2];
      break;
    case RAW_YUV:
      if (bd < 9) {
        r1 = fread(in->buf, in->frame_size, 1, in->file);
        *y = in->buf;
        *u = in->buf + in->w * in->h;
        *v = *u + ((1 + in->w) / 2) * ((1 + in->h) / 2);
      } else {
        r1 = fread(in->buf, in->frame_size, 1, in->file);
        *y = in->buf;
        *u = in->buf + (in->w * in->h) * 2;
        *v = *u + 2 * ((1 + in->w) / 2) * ((1 + in->h) / 2);
      }
      break;
  }

  return r1 != 0;
}

static void ssim_parms_8x8(const uint8_t *s, int sp, const uint8_t *r, int rp,
                           uint32_t *sum_s, uint32_t *sum_r, uint32_t *sum_sq_s,
                           uint32_t *sum_sq_r, uint32_t *sum_sxr) {
  int i, j;
  if (s == NULL || r == NULL || sum_s == NULL || sum_r == NULL ||
      sum_sq_s == NULL || sum_sq_r == NULL || sum_sxr == NULL) {
    assert(0);
    return;
  }
  for (i = 0; i < 8; i++, s += sp, r += rp) {
    for (j = 0; j < 8; j++) {
      *sum_s += s[j];
      *sum_r += r[j];
      *sum_sq_s += s[j] * s[j];
      *sum_sq_r += r[j] * r[j];
      *sum_sxr += s[j] * r[j];
    }
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
static void highbd_ssim_parms_8x8(const uint16_t *s, int sp, const uint16_t *r,
                                  int rp, uint32_t *sum_s, uint32_t *sum_r,
                                  uint32_t *sum_sq_s, uint32_t *sum_sq_r,
                                  uint32_t *sum_sxr) {
  int i, j;
  if (s == NULL || r == NULL || sum_s == NULL || sum_r == NULL ||
      sum_sq_s == NULL || sum_sq_r == NULL || sum_sxr == NULL) {
    assert(0);
    return;
  }
  for (i = 0; i < 8; i++, s += sp, r += rp) {
    for (j = 0; j < 8; j++) {
      *sum_s += s[j];
      *sum_r += r[j];
      *sum_sq_s += s[j] * s[j];
      *sum_sq_r += r[j] * r[j];
      *sum_sxr += s[j] * r[j];
    }
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static double similarity(uint32_t sum_s, uint32_t sum_r, uint32_t sum_sq_s,
                         uint32_t sum_sq_r, uint32_t sum_sxr, int count,
                         uint32_t bd) {
  double ssim_n, ssim_d;
  int64_t c1 = 0, c2 = 0;
  if (bd == 8) {
    // scale the constants by number of pixels
    c1 = (cc1 * count * count) >> 12;
    c2 = (cc2 * count * count) >> 12;
  } else if (bd == 10) {
    c1 = (cc1_10 * count * count) >> 12;
    c2 = (cc2_10 * count * count) >> 12;
  } else if (bd == 12) {
    c1 = (cc1_12 * count * count) >> 12;
    c2 = (cc2_12 * count * count) >> 12;
  } else {
    assert(0);
  }

  ssim_n = (2.0 * sum_s * sum_r + c1) *
           (2.0 * count * sum_sxr - 2.0 * sum_s * sum_r + c2);

  ssim_d = ((double)sum_s * sum_s + (double)sum_r * sum_r + c1) *
           ((double)count * sum_sq_s - (double)sum_s * sum_s +
            (double)count * sum_sq_r - (double)sum_r * sum_r + c2);

  return ssim_n / ssim_d;
}

static double ssim_8x8(const uint8_t *s, int sp, const uint8_t *r, int rp) {
  uint32_t sum_s = 0, sum_r = 0, sum_sq_s = 0, sum_sq_r = 0, sum_sxr = 0;
  ssim_parms_8x8(s, sp, r, rp, &sum_s, &sum_r, &sum_sq_s, &sum_sq_r, &sum_sxr);
  return similarity(sum_s, sum_r, sum_sq_s, sum_sq_r, sum_sxr, 64, 8);
}

#if CONFIG_VP9_HIGHBITDEPTH
static double highbd_ssim_8x8(const uint16_t *s, int sp, const uint16_t *r,
                              int rp, uint32_t bd) {
  uint32_t sum_s = 0, sum_r = 0, sum_sq_s = 0, sum_sq_r = 0, sum_sxr = 0;
  highbd_ssim_parms_8x8(s, sp, r, rp, &sum_s, &sum_r, &sum_sq_s, &sum_sq_r,
                        &sum_sxr);
  return similarity(sum_s, sum_r, sum_sq_s, sum_sq_r, sum_sxr, 64, bd);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

// We are using a 8x8 moving window with starting location of each 8x8 window
// on the 4x4 pixel grid. Such arrangement allows the windows to overlap
// block boundaries to penalize blocking artifacts.
static double ssim2(const uint8_t *img1, const uint8_t *img2, int stride_img1,
                    int stride_img2, int width, int height) {
  int i, j;
  int samples = 0;
  double ssim_total = 0;

  // sample point start with each 4x4 location
  for (i = 0; i <= height - 8;
       i += 4, img1 += stride_img1 * 4, img2 += stride_img2 * 4) {
    for (j = 0; j <= width - 8; j += 4) {
      double v = ssim_8x8(img1 + j, stride_img1, img2 + j, stride_img2);
      ssim_total += v;
      samples++;
    }
  }
  ssim_total /= samples;
  return ssim_total;
}

#if CONFIG_VP9_HIGHBITDEPTH
static double highbd_ssim2(const uint8_t *img1, const uint8_t *img2,
                           int stride_img1, int stride_img2, int width,
                           int height, uint32_t bd) {
  int i, j;
  int samples = 0;
  double ssim_total = 0;

  // sample point start with each 4x4 location
  for (i = 0; i <= height - 8;
       i += 4, img1 += stride_img1 * 4, img2 += stride_img2 * 4) {
    for (j = 0; j <= width - 8; j += 4) {
      double v =
          highbd_ssim_8x8(CONVERT_TO_SHORTPTR(img1 + j), stride_img1,
                          CONVERT_TO_SHORTPTR(img2 + j), stride_img2, bd);
      ssim_total += v;
      samples++;
    }
  }
  ssim_total /= samples;
  return ssim_total;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

int main(int argc, char *argv[]) {
  FILE *framestats = NULL;
  int bit_depth = 8;
  int w = 0, h = 0, tl_skip = 0, tl_skips_remaining = 0;
  double ssimavg = 0, ssimyavg = 0, ssimuavg = 0, ssimvavg = 0;
  double psnrglb = 0, psnryglb = 0, psnruglb = 0, psnrvglb = 0;
  double psnravg = 0, psnryavg = 0, psnruavg = 0, psnrvavg = 0;
  double *ssimy = NULL, *ssimu = NULL, *ssimv = NULL;
  uint64_t *psnry = NULL, *psnru = NULL, *psnrv = NULL;
  size_t i, n_frames = 0, allocated_frames = 0;
  int return_value = 0;
  input_file_t in[2];
  double peak = 255.0;

  memset(in, 0, sizeof(in));

  if (argc < 3) {
    fprintf(stderr,
            "Usage: %s file1.{yuv|y4m} file2.{yuv|y4m}"
            " [WxH tl_skip={0,1,3} frame_stats_file bits]\n",
            argv[0]);
    return 1;
  }

  if (argc > 3) {
    if (sscanf(argv[3], "%dx%d", &w, &h) != 2) {
      fprintf(stderr, "arguments for w/h not assigned!\n");
      goto clean_up;
    }
    // Limit width/height to 4K. The frame_size set in the function
    // open_input_file() will still be within range of int.
    if (w < 1 || w > 4096 || h < 1 || h > 4096) {
      fprintf(stderr,
              "width or height is too large (above 4096) or below 1!\n");
      goto clean_up;
    }
  }

  if (argc > 6) {
    if (sscanf(argv[6], "%d", &bit_depth) != 1) {
      fprintf(stderr, "argument for bit_depth not assigned!\n");
      goto clean_up;
    }
  }

  if (open_input_file(argv[1], &in[0], w, h, bit_depth) < 0) {
    fprintf(stderr, "File %s can't be opened or parsed!\n", argv[1]);
    goto clean_up;
  }

  if (w == 0 && h == 0) {
    // If a y4m is the first file and w, h is not set grab from first file.
    w = in[0].w;
    h = in[0].h;
    bit_depth = in[0].bit_depth;
  }
  if (bit_depth == 10) peak = 1023.0;

  if (bit_depth == 12) peak = 4095.0;

  if (open_input_file(argv[2], &in[1], w, h, bit_depth) < 0) {
    fprintf(stderr, "File %s can't be opened or parsed!\n", argv[2]);
    goto clean_up;
  }

  if (in[0].w != in[1].w || in[0].h != in[1].h || in[0].w != w ||
      in[0].h != h || w == 0 || h == 0) {
    fprintf(stderr,
            "Failing: Image dimensions don't match or are unspecified!\n");
    return_value = 1;
    goto clean_up;
  }

  if (in[0].bit_depth != in[1].bit_depth) {
    fprintf(stderr,
            "Failing: Image bit depths don't match or are unspecified!\n");
    return_value = 1;
    goto clean_up;
  }

  bit_depth = in[0].bit_depth;

  // Number of frames to skip from file1.yuv for every frame used. Normal
  // values 0, 1 and 3 correspond to TL2, TL1 and TL0 respectively for a 3TL
  // encoding in mode 10. 7 would be reasonable for comparing TL0 of a 4-layer
  // encoding.
  if (argc > 4) {
    if (sscanf(argv[4], "%d", &tl_skip) != 1) {
      fprintf(stderr, "argument for tl_skip not assigned!\n");
      goto clean_up;
    }
    if (argc > 5) {
      framestats = fopen(argv[5], "w");
      if (!framestats) {
        fprintf(stderr, "Could not open \"%s\" for writing: %s\n", argv[5],
                strerror(errno));
        return_value = 1;
        goto clean_up;
      }
    }
  }

  while (1) {
    int r1, r2;
    unsigned char *y[2], *u[2], *v[2];

    r1 = read_input_file(&in[0], &y[0], &u[0], &v[0], bit_depth);
    if (r1 == 0) {
      if (ferror(in[0].file)) {
        fprintf(stderr, "Failed to read data from '%s'\n", argv[1]);
        return_value = 1;
        goto clean_up;
      }
      break;
    }

    // Reading parts of file1.yuv that were not used in temporal layer.
    if (tl_skips_remaining > 0) {
      --tl_skips_remaining;
      continue;
    }
    // Use frame, but skip |tl_skip| after it.
    tl_skips_remaining = tl_skip;

    r2 = read_input_file(&in[1], &y[1], &u[1], &v[1], bit_depth);
    if (r2 == 0) {
      if (ferror(in[1].file)) {
        fprintf(stderr, "Failed to read data from '%s'\n", argv[2]);
        return_value = 1;
        goto clean_up;
      }
      break;
    }

#if CONFIG_VP9_HIGHBITDEPTH
#define psnr_and_ssim(ssim, psnr, buf0, buf1, w, h)                           \
  do {                                                                        \
    if (bit_depth < 9) {                                                      \
      ssim = ssim2(buf0, buf1, w, w, w, h);                                   \
      psnr = calc_plane_error(buf0, w, buf1, w, w, h);                        \
    } else {                                                                  \
      ssim = highbd_ssim2(CONVERT_TO_BYTEPTR(buf0), CONVERT_TO_BYTEPTR(buf1), \
                          w, w, w, h, bit_depth);                             \
      psnr = calc_plane_error16(CAST_TO_SHORTPTR(buf0), w,                    \
                                CAST_TO_SHORTPTR(buf1), w, w, h);             \
    }                                                                         \
  } while (0)
#else
#define psnr_and_ssim(ssim, psnr, buf0, buf1, w, h)  \
  do {                                               \
    ssim = ssim2(buf0, buf1, w, w, w, h);            \
    psnr = calc_plane_error(buf0, w, buf1, w, w, h); \
  } while (0)
#endif  // CONFIG_VP9_HIGHBITDEPTH

    if (n_frames == allocated_frames) {
      allocated_frames = allocated_frames == 0 ? 1024 : allocated_frames * 2;
      ssimy = realloc(ssimy, allocated_frames * sizeof(*ssimy));
      ssimu = realloc(ssimu, allocated_frames * sizeof(*ssimu));
      ssimv = realloc(ssimv, allocated_frames * sizeof(*ssimv));
      psnry = realloc(psnry, allocated_frames * sizeof(*psnry));
      psnru = realloc(psnru, allocated_frames * sizeof(*psnru));
      psnrv = realloc(psnrv, allocated_frames * sizeof(*psnrv));
      if (!(ssimy && ssimu && ssimv && psnry && psnru && psnrv)) {
        fprintf(stderr, "Error allocating SSIM/PSNR data.\n");
        exit(EXIT_FAILURE);
      }
    }
    psnr_and_ssim(ssimy[n_frames], psnry[n_frames], y[0], y[1], w, h);
    psnr_and_ssim(ssimu[n_frames], psnru[n_frames], u[0], u[1], (w + 1) / 2,
                  (h + 1) / 2);
    psnr_and_ssim(ssimv[n_frames], psnrv[n_frames], v[0], v[1], (w + 1) / 2,
                  (h + 1) / 2);

    n_frames++;
  }

  if (framestats) {
    fprintf(framestats,
            "ssim,ssim-y,ssim-u,ssim-v,psnr,psnr-y,psnr-u,psnr-v\n");
  }

  for (i = 0; i < n_frames; ++i) {
    double frame_ssim;
    double frame_psnr, frame_psnry, frame_psnru, frame_psnrv;

    frame_ssim = 0.8 * ssimy[i] + 0.1 * (ssimu[i] + ssimv[i]);
    ssimavg += frame_ssim;
    ssimyavg += ssimy[i];
    ssimuavg += ssimu[i];
    ssimvavg += ssimv[i];

    frame_psnr =
        mse2psnr(w * h * 6 / 4, peak, (double)psnry[i] + psnru[i] + psnrv[i]);
    frame_psnry = mse2psnr(w * h * 4 / 4, peak, (double)psnry[i]);
    frame_psnru = mse2psnr(w * h * 1 / 4, peak, (double)psnru[i]);
    frame_psnrv = mse2psnr(w * h * 1 / 4, peak, (double)psnrv[i]);

    psnravg += frame_psnr;
    psnryavg += frame_psnry;
    psnruavg += frame_psnru;
    psnrvavg += frame_psnrv;

    psnryglb += psnry[i];
    psnruglb += psnru[i];
    psnrvglb += psnrv[i];

    if (framestats) {
      fprintf(framestats, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", frame_ssim,
              ssimy[i], ssimu[i], ssimv[i], frame_psnr, frame_psnry,
              frame_psnru, frame_psnrv);
    }
  }

  ssimavg /= n_frames;
  ssimyavg /= n_frames;
  ssimuavg /= n_frames;
  ssimvavg /= n_frames;

  printf("VpxSSIM: %lf\n", 100 * pow(ssimavg, 8.0));
  printf("SSIM: %lf\n", ssimavg);
  printf("SSIM-Y: %lf\n", ssimyavg);
  printf("SSIM-U: %lf\n", ssimuavg);
  printf("SSIM-V: %lf\n", ssimvavg);
  puts("");

  psnravg /= n_frames;
  psnryavg /= n_frames;
  psnruavg /= n_frames;
  psnrvavg /= n_frames;

  printf("AvgPSNR: %lf\n", psnravg);
  printf("AvgPSNR-Y: %lf\n", psnryavg);
  printf("AvgPSNR-U: %lf\n", psnruavg);
  printf("AvgPSNR-V: %lf\n", psnrvavg);
  puts("");

  psnrglb = psnryglb + psnruglb + psnrvglb;
  psnrglb = mse2psnr((double)n_frames * w * h * 6 / 4, peak, psnrglb);
  psnryglb = mse2psnr((double)n_frames * w * h * 4 / 4, peak, psnryglb);
  psnruglb = mse2psnr((double)n_frames * w * h * 1 / 4, peak, psnruglb);
  psnrvglb = mse2psnr((double)n_frames * w * h * 1 / 4, peak, psnrvglb);

  printf("GlbPSNR: %lf\n", psnrglb);
  printf("GlbPSNR-Y: %lf\n", psnryglb);
  printf("GlbPSNR-U: %lf\n", psnruglb);
  printf("GlbPSNR-V: %lf\n", psnrvglb);
  puts("");

  printf("Nframes: %d\n", (int)n_frames);

clean_up:

  close_input_file(&in[0]);
  close_input_file(&in[1]);

  if (framestats) fclose(framestats);

  free(ssimy);
  free(ssimu);
  free(ssimv);

  free(psnry);
  free(psnru);
  free(psnrv);

  return return_value;
}
