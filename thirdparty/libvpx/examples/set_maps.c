/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

// VP8 Set Active and ROI Maps
// ===========================
//
// This is an example demonstrating how to control the VP8 encoder's
// ROI and Active maps.
//
// ROI (Reigon of Interest) maps are a way for the application to assign
// each macroblock in the image to a region, and then set quantizer and
// filtering parameters on that image.
//
// Active maps are a way for the application to specify on a
// macroblock-by-macroblock basis whether there is any activity in that
// macroblock.
//
//
// Configuration
// -------------
// An ROI map is set on frame 22. If the width of the image in macroblocks
// is evenly divisble by 4, then the output will appear to have distinct
// columns, where the quantizer, loopfilter, and static threshold differ
// from column to column.
//
// An active map is set on frame 33. If the width of the image in macroblocks
// is evenly divisble by 4, then the output will appear to have distinct
// columns, where one column will have motion and the next will not.
//
// The active map is cleared on frame 44.
//
// Observing The Effects
// ---------------------
// Use the `simple_decoder` example to decode this sample, and observe
// the change in the image at frames 22, 33, and 44.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vp8cx.h"
#include "vpx/vpx_encoder.h"

#include "../tools_common.h"
#include "../video_writer.h"

static const char *exec_name;

void usage_exit(void) {
  fprintf(stderr, "Usage: %s <codec> <width> <height> <infile> <outfile>\n",
          exec_name);
  exit(EXIT_FAILURE);
}

static void set_roi_map(const vpx_codec_enc_cfg_t *cfg,
                        vpx_codec_ctx_t *codec) {
  unsigned int i;
  vpx_roi_map_t roi;
  memset(&roi, 0, sizeof(roi));

  roi.rows = (cfg->g_h + 15) / 16;
  roi.cols = (cfg->g_w + 15) / 16;

  roi.delta_q[0] = 0;
  roi.delta_q[1] = -2;
  roi.delta_q[2] = -4;
  roi.delta_q[3] = -6;

  roi.delta_lf[0] = 0;
  roi.delta_lf[1] = 1;
  roi.delta_lf[2] = 2;
  roi.delta_lf[3] = 3;

  roi.static_threshold[0] = 1500;
  roi.static_threshold[1] = 1000;
  roi.static_threshold[2] = 500;
  roi.static_threshold[3] = 0;

  roi.roi_map = (uint8_t *)malloc(roi.rows * roi.cols);
  for (i = 0; i < roi.rows * roi.cols; ++i) roi.roi_map[i] = i % 4;

  if (vpx_codec_control(codec, VP8E_SET_ROI_MAP, &roi))
    die_codec(codec, "Failed to set ROI map");

  free(roi.roi_map);
}

static void set_active_map(const vpx_codec_enc_cfg_t *cfg,
                           vpx_codec_ctx_t *codec) {
  unsigned int i;
  vpx_active_map_t map = { 0, 0, 0 };

  map.rows = (cfg->g_h + 15) / 16;
  map.cols = (cfg->g_w + 15) / 16;

  map.active_map = (uint8_t *)malloc(map.rows * map.cols);
  for (i = 0; i < map.rows * map.cols; ++i) map.active_map[i] = i % 2;

  if (vpx_codec_control(codec, VP8E_SET_ACTIVEMAP, &map))
    die_codec(codec, "Failed to set active map");

  free(map.active_map);
}

static void unset_active_map(const vpx_codec_enc_cfg_t *cfg,
                             vpx_codec_ctx_t *codec) {
  vpx_active_map_t map = { 0, 0, 0 };

  map.rows = (cfg->g_h + 15) / 16;
  map.cols = (cfg->g_w + 15) / 16;
  map.active_map = NULL;

  if (vpx_codec_control(codec, VP8E_SET_ACTIVEMAP, &map))
    die_codec(codec, "Failed to set active map");
}

static int encode_frame(vpx_codec_ctx_t *codec, vpx_image_t *img,
                        int frame_index, VpxVideoWriter *writer) {
  int got_pkts = 0;
  vpx_codec_iter_t iter = NULL;
  const vpx_codec_cx_pkt_t *pkt = NULL;
  const vpx_codec_err_t res =
      vpx_codec_encode(codec, img, frame_index, 1, 0, VPX_DL_GOOD_QUALITY);
  if (res != VPX_CODEC_OK) die_codec(codec, "Failed to encode frame");

  while ((pkt = vpx_codec_get_cx_data(codec, &iter)) != NULL) {
    got_pkts = 1;

    if (pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
      const int keyframe = (pkt->data.frame.flags & VPX_FRAME_IS_KEY) != 0;
      if (!vpx_video_writer_write_frame(writer, pkt->data.frame.buf,
                                        pkt->data.frame.sz,
                                        pkt->data.frame.pts)) {
        die_codec(codec, "Failed to write compressed frame");
      }

      printf(keyframe ? "K" : ".");
      fflush(stdout);
    }
  }

  return got_pkts;
}

int main(int argc, char **argv) {
  FILE *infile = NULL;
  vpx_codec_ctx_t codec;
  vpx_codec_enc_cfg_t cfg;
  int frame_count = 0;
  vpx_image_t raw;
  vpx_codec_err_t res;
  VpxVideoInfo info;
  VpxVideoWriter *writer = NULL;
  const VpxInterface *encoder = NULL;
  const int fps = 2;  // TODO(dkovalev) add command line argument
  const double bits_per_pixel_per_frame = 0.067;

  exec_name = argv[0];
  if (argc != 6) die("Invalid number of arguments");

  memset(&info, 0, sizeof(info));

  encoder = get_vpx_encoder_by_name(argv[1]);
  if (encoder == NULL) {
    die("Unsupported codec.");
  }
  assert(encoder != NULL);
  info.codec_fourcc = encoder->fourcc;
  info.frame_width = (int)strtol(argv[2], NULL, 0);
  info.frame_height = (int)strtol(argv[3], NULL, 0);
  info.time_base.numerator = 1;
  info.time_base.denominator = fps;

  if (info.frame_width <= 0 || info.frame_height <= 0 ||
      (info.frame_width % 2) != 0 || (info.frame_height % 2) != 0) {
    die("Invalid frame size: %dx%d", info.frame_width, info.frame_height);
  }

  if (!vpx_img_alloc(&raw, VPX_IMG_FMT_I420, info.frame_width,
                     info.frame_height, 1)) {
    die("Failed to allocate image.");
  }

  printf("Using %s\n", vpx_codec_iface_name(encoder->codec_interface()));

  res = vpx_codec_enc_config_default(encoder->codec_interface(), &cfg, 0);
  if (res) die_codec(&codec, "Failed to get default codec config.");

  cfg.g_w = info.frame_width;
  cfg.g_h = info.frame_height;
  cfg.g_timebase.num = info.time_base.numerator;
  cfg.g_timebase.den = info.time_base.denominator;
  cfg.rc_target_bitrate =
      (unsigned int)(bits_per_pixel_per_frame * cfg.g_w * cfg.g_h * fps / 1000);
  cfg.g_lag_in_frames = 0;

  writer = vpx_video_writer_open(argv[5], kContainerIVF, &info);
  if (!writer) die("Failed to open %s for writing.", argv[5]);

  if (!(infile = fopen(argv[4], "rb")))
    die("Failed to open %s for reading.", argv[4]);

  if (vpx_codec_enc_init(&codec, encoder->codec_interface(), &cfg, 0))
    die("Failed to initialize encoder");

  // Encode frames.
  while (vpx_img_read(&raw, infile)) {
    ++frame_count;

    if (frame_count == 22 && encoder->fourcc == VP8_FOURCC) {
      set_roi_map(&cfg, &codec);
    } else if (frame_count == 33) {
      set_active_map(&cfg, &codec);
    } else if (frame_count == 44) {
      unset_active_map(&cfg, &codec);
    }

    encode_frame(&codec, &raw, frame_count, writer);
  }

  // Flush encoder.
  while (encode_frame(&codec, NULL, -1, writer)) {
  }

  printf("\n");
  fclose(infile);
  printf("Processed %d frames.\n", frame_count);

  vpx_img_free(&raw);
  if (vpx_codec_destroy(&codec)) die_codec(&codec, "Failed to destroy codec.");

  vpx_video_writer_close(writer);

  return EXIT_SUCCESS;
}
