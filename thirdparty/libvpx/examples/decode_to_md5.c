/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

// Frame-by-frame MD5 Checksum
// ===========================
//
// This example builds upon the simple decoder loop to show how checksums
// of the decoded output can be generated. These are used for validating
// decoder implementations against the reference implementation, for example.
//
// MD5 algorithm
// -------------
// The Message-Digest 5 (MD5) is a well known hash function. We have provided
// an implementation derived from the RSA Data Security, Inc. MD5 Message-Digest
// Algorithm for your use. Our implmentation only changes the interface of this
// reference code. You must include the `md5_utils.h` header for access to these
// functions.
//
// Processing The Decoded Data
// ---------------------------
// Each row of the image is passed to the MD5 accumulator. First the Y plane
// is processed, then U, then V. It is important to honor the image's `stride`
// values.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vp8dx.h"
#include "vpx/vpx_decoder.h"

#include "../md5_utils.h"
#include "../tools_common.h"
#include "../video_reader.h"
#include "./vpx_config.h"

static void get_image_md5(const vpx_image_t *img, unsigned char digest[16]) {
  int plane, y;
  MD5Context md5;

  MD5Init(&md5);

  for (plane = 0; plane < 3; ++plane) {
    const unsigned char *buf = img->planes[plane];
    const int stride = img->stride[plane];
    const int w = plane ? (img->d_w + 1) >> 1 : img->d_w;
    const int h = plane ? (img->d_h + 1) >> 1 : img->d_h;

    for (y = 0; y < h; ++y) {
      MD5Update(&md5, buf, w);
      buf += stride;
    }
  }

  MD5Final(digest, &md5);
}

static void print_md5(FILE *stream, unsigned char digest[16]) {
  int i;

  for (i = 0; i < 16; ++i) fprintf(stream, "%02x", digest[i]);
}

static const char *exec_name;

void usage_exit(void) {
  fprintf(stderr, "Usage: %s <infile> <outfile>\n", exec_name);
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  int frame_cnt = 0;
  FILE *outfile = NULL;
  vpx_codec_ctx_t codec;
  VpxVideoReader *reader = NULL;
  const VpxVideoInfo *info = NULL;
  const VpxInterface *decoder = NULL;

  exec_name = argv[0];

  if (argc != 3) die("Invalid number of arguments.");

  reader = vpx_video_reader_open(argv[1]);
  if (!reader) die("Failed to open %s for reading.", argv[1]);

  if (!(outfile = fopen(argv[2], "wb")))
    die("Failed to open %s for writing.", argv[2]);

  info = vpx_video_reader_get_info(reader);

  decoder = get_vpx_decoder_by_fourcc(info->codec_fourcc);
  if (!decoder) die("Unknown input codec.");

  printf("Using %s\n", vpx_codec_iface_name(decoder->codec_interface()));

  if (vpx_codec_dec_init(&codec, decoder->codec_interface(), NULL, 0))
    die_codec(&codec, "Failed to initialize decoder");

  while (vpx_video_reader_read_frame(reader)) {
    vpx_codec_iter_t iter = NULL;
    vpx_image_t *img = NULL;
    size_t frame_size = 0;
    const unsigned char *frame =
        vpx_video_reader_get_frame(reader, &frame_size);
    if (vpx_codec_decode(&codec, frame, (unsigned int)frame_size, NULL, 0))
      die_codec(&codec, "Failed to decode frame");

    while ((img = vpx_codec_get_frame(&codec, &iter)) != NULL) {
      unsigned char digest[16];

      get_image_md5(img, digest);
      print_md5(outfile, digest);
      fprintf(outfile, "  img-%dx%d-%04d.i420\n", img->d_w, img->d_h,
              ++frame_cnt);
    }
  }

  printf("Processed %d frames.\n", frame_cnt);
  if (vpx_codec_destroy(&codec)) die_codec(&codec, "Failed to destroy codec.");

  vpx_video_reader_close(reader);

  fclose(outfile);
  return EXIT_SUCCESS;
}
