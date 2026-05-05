/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

// Two Pass Encoder
// ================
//
// This is an example of a two pass encoder loop. It takes an input file in
// YV12 format, passes it through the encoder twice, and writes the compressed
// frames to disk in IVF format. It builds upon the simple_encoder example.
//
// Twopass Variables
// -----------------
// Twopass mode needs to track the current pass number and the buffer of
// statistics packets.
//
// Updating The Configuration
// ---------------------------------
// In two pass mode, the configuration has to be updated on each pass. The
// statistics buffer is passed on the last pass.
//
// Encoding A Frame
// ----------------
// Encoding a frame in two pass mode is identical to the simple encoder
// example. To increase the quality while sacrificing encoding speed,
// VPX_DL_BEST_QUALITY can be used in place of VPX_DL_GOOD_QUALITY.
//
// Processing Statistics Packets
// -----------------------------
// Each packet of type `VPX_CODEC_CX_FRAME_PKT` contains the encoded data
// for this frame. We write a IVF frame header, followed by the raw data.
//
//
// Pass Progress Reporting
// -----------------------------
// It's sometimes helpful to see when each pass completes.
//
//
// Clean-up
// -----------------------------
// Destruction of the encoder instance must be done on each pass. The
// raw image should be destroyed at the end as usual.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vpx_encoder.h"

#include "../tools_common.h"
#include "../video_writer.h"

static const char *exec_name;

void usage_exit(void) {
  fprintf(stderr,
          "Usage: %s <codec> <width> <height> <infile> <outfile> "
          "<frame limit>\n",
          exec_name);
  exit(EXIT_FAILURE);
}

static int get_frame_stats(vpx_codec_ctx_t *ctx, const vpx_image_t *img,
                           vpx_codec_pts_t pts, unsigned int duration,
                           vpx_enc_frame_flags_t flags, unsigned int deadline,
                           vpx_fixed_buf_t *stats) {
  int got_pkts = 0;
  vpx_codec_iter_t iter = NULL;
  const vpx_codec_cx_pkt_t *pkt = NULL;
  const vpx_codec_err_t res =
      vpx_codec_encode(ctx, img, pts, duration, flags, deadline);
  if (res != VPX_CODEC_OK) die_codec(ctx, "Failed to get frame stats.");

  while ((pkt = vpx_codec_get_cx_data(ctx, &iter)) != NULL) {
    got_pkts = 1;

    if (pkt->kind == VPX_CODEC_STATS_PKT) {
      const uint8_t *const pkt_buf = pkt->data.twopass_stats.buf;
      const size_t pkt_size = pkt->data.twopass_stats.sz;
      stats->buf = realloc(stats->buf, stats->sz + pkt_size);
      if (!stats->buf) die("Failed to reallocate stats buffer.");
      memcpy((uint8_t *)stats->buf + stats->sz, pkt_buf, pkt_size);
      stats->sz += pkt_size;
    }
  }

  return got_pkts;
}

static int encode_frame(vpx_codec_ctx_t *ctx, const vpx_image_t *img,
                        vpx_codec_pts_t pts, unsigned int duration,
                        vpx_enc_frame_flags_t flags, unsigned int deadline,
                        VpxVideoWriter *writer) {
  int got_pkts = 0;
  vpx_codec_iter_t iter = NULL;
  const vpx_codec_cx_pkt_t *pkt = NULL;
  const vpx_codec_err_t res =
      vpx_codec_encode(ctx, img, pts, duration, flags, deadline);
  if (res != VPX_CODEC_OK) die_codec(ctx, "Failed to encode frame.");

  while ((pkt = vpx_codec_get_cx_data(ctx, &iter)) != NULL) {
    got_pkts = 1;
    if (pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
      const int keyframe = (pkt->data.frame.flags & VPX_FRAME_IS_KEY) != 0;

      if (!vpx_video_writer_write_frame(writer, pkt->data.frame.buf,
                                        pkt->data.frame.sz,
                                        pkt->data.frame.pts))
        die_codec(ctx, "Failed to write compressed frame.");
      printf(keyframe ? "K" : ".");
      fflush(stdout);
    }
  }

  return got_pkts;
}

static vpx_fixed_buf_t pass0(vpx_image_t *raw, FILE *infile,
                             const VpxInterface *encoder,
                             const vpx_codec_enc_cfg_t *cfg, int max_frames) {
  vpx_codec_ctx_t codec;
  int frame_count = 0;
  vpx_fixed_buf_t stats = { NULL, 0 };

  if (vpx_codec_enc_init(&codec, encoder->codec_interface(), cfg, 0))
    die("Failed to initialize encoder");

  // Calculate frame statistics.
  while (vpx_img_read(raw, infile)) {
    ++frame_count;
    get_frame_stats(&codec, raw, frame_count, 1, 0, VPX_DL_GOOD_QUALITY,
                    &stats);
    if (max_frames > 0 && frame_count >= max_frames) break;
  }

  // Flush encoder.
  while (get_frame_stats(&codec, NULL, frame_count, 1, 0, VPX_DL_GOOD_QUALITY,
                         &stats)) {
  }

  printf("Pass 0 complete. Processed %d frames.\n", frame_count);
  if (vpx_codec_destroy(&codec)) die_codec(&codec, "Failed to destroy codec.");

  return stats;
}

static void pass1(vpx_image_t *raw, FILE *infile, const char *outfile_name,
                  const VpxInterface *encoder, const vpx_codec_enc_cfg_t *cfg,
                  int max_frames) {
  VpxVideoInfo info = { encoder->fourcc,
                        cfg->g_w,
                        cfg->g_h,
                        { cfg->g_timebase.num, cfg->g_timebase.den } };
  VpxVideoWriter *writer = NULL;
  vpx_codec_ctx_t codec;
  int frame_count = 0;

  writer = vpx_video_writer_open(outfile_name, kContainerIVF, &info);
  if (!writer) die("Failed to open %s for writing", outfile_name);

  if (vpx_codec_enc_init(&codec, encoder->codec_interface(), cfg, 0))
    die("Failed to initialize encoder");

  // Encode frames.
  while (vpx_img_read(raw, infile)) {
    ++frame_count;
    encode_frame(&codec, raw, frame_count, 1, 0, VPX_DL_GOOD_QUALITY, writer);

    if (max_frames > 0 && frame_count >= max_frames) break;
  }

  // Flush encoder.
  while (encode_frame(&codec, NULL, -1, 1, 0, VPX_DL_GOOD_QUALITY, writer)) {
  }

  printf("\n");

  if (vpx_codec_destroy(&codec)) die_codec(&codec, "Failed to destroy codec.");

  vpx_video_writer_close(writer);

  printf("Pass 1 complete. Processed %d frames.\n", frame_count);
}

int main(int argc, char **argv) {
  FILE *infile = NULL;
  int w, h;
  vpx_codec_ctx_t codec;
  vpx_codec_enc_cfg_t cfg;
  vpx_image_t raw;
  vpx_codec_err_t res;
  vpx_fixed_buf_t stats;

  const VpxInterface *encoder = NULL;
  const int fps = 30;       // TODO(dkovalev) add command line argument
  const int bitrate = 200;  // kbit/s TODO(dkovalev) add command line argument
  const char *const codec_arg = argv[1];
  const char *const width_arg = argv[2];
  const char *const height_arg = argv[3];
  const char *const infile_arg = argv[4];
  const char *const outfile_arg = argv[5];
  int max_frames = 0;
  exec_name = argv[0];

  if (argc != 7) die("Invalid number of arguments.");

  max_frames = (int)strtol(argv[6], NULL, 0);

  encoder = get_vpx_encoder_by_name(codec_arg);
  if (!encoder) die("Unsupported codec.");

  w = (int)strtol(width_arg, NULL, 0);
  h = (int)strtol(height_arg, NULL, 0);

  if (w <= 0 || h <= 0 || (w % 2) != 0 || (h % 2) != 0)
    die("Invalid frame size: %dx%d", w, h);

  if (!vpx_img_alloc(&raw, VPX_IMG_FMT_I420, w, h, 1))
    die("Failed to allocate image (%dx%d)", w, h);

  printf("Using %s\n", vpx_codec_iface_name(encoder->codec_interface()));

  // Configuration
  res = vpx_codec_enc_config_default(encoder->codec_interface(), &cfg, 0);
  if (res) die_codec(&codec, "Failed to get default codec config.");

  cfg.g_w = w;
  cfg.g_h = h;
  cfg.g_timebase.num = 1;
  cfg.g_timebase.den = fps;
  cfg.rc_target_bitrate = bitrate;

  if (!(infile = fopen(infile_arg, "rb")))
    die("Failed to open %s for reading", infile_arg);

  // Pass 0
  cfg.g_pass = VPX_RC_FIRST_PASS;
  stats = pass0(&raw, infile, encoder, &cfg, max_frames);

  // Pass 1
  rewind(infile);
  cfg.g_pass = VPX_RC_LAST_PASS;
  cfg.rc_twopass_stats_in = stats;
  pass1(&raw, infile, outfile_arg, encoder, &cfg, max_frames);
  free(stats.buf);

  vpx_img_free(&raw);
  fclose(infile);

  return EXIT_SUCCESS;
}
