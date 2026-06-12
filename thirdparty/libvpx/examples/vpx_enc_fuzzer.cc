/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/*
 * Fuzzer for libvpx encoders
 * ==========================
 * Requirements
 * --------------
 * Requires Clang 6.0 or above as -fsanitize=fuzzer is used as a linker
 * option.

 * Steps to build
 * --------------
 * Clone libvpx repository
   $git clone https://chromium.googlesource.com/webm/libvpx

 * Create a directory in parallel to libvpx and change directory
   $mkdir vpx_enc_fuzzer
   $cd vpx_enc_fuzzer/

 * Enable sanitizers (Supported: address integer memory thread undefined)
   $source ../libvpx/tools/set_analyzer_env.sh address

 * Configure libvpx.
 * Note --size-limit and VPX_MAX_ALLOCABLE_MEMORY are defined to avoid
 * Out of memory errors when running generated fuzzer binary
   $../libvpx/configure --disable-unit-tests --size-limit=12288x12288 \
   --extra-cflags="-fsanitize=fuzzer-no-link \
   -DVPX_MAX_ALLOCABLE_MEMORY=1073741824" \
   --disable-webm-io --enable-debug --enable-vp8-encoder \
   --enable-vp9-encoder --disable-examples

 * Build libvpx
   $make -j32

 * Build vp9 fuzzer
   $ $CXX $CXXFLAGS -std=gnu++17 -DENCODER=vp9 \
   -fsanitize=fuzzer -I../libvpx -I. -Wl,--start-group \
   ../libvpx/examples/vpx_enc_fuzzer.cc -o ./vpx_enc_fuzzer_vp9 \
   ./libvpx.a -Wl,--end-group

 * ENCODER should be defined as vp9 or vp8 to enable vp9/vp8
 *
 * create a corpus directory and copy some ivf files there.
 * Based on which codec (vp8/vp9) is being tested, it is recommended to
 * have corresponding ivf files in corpus directory
 * Empty corpus directory also is acceptable, though not recommended
   $mkdir CORPUS && cp some-files CORPUS

 * Run fuzzing:
   $./vpx_enc_fuzzer_vp9 CORPUS

 * References:
 * http://llvm.org/docs/LibFuzzer.html
 * https://github.com/google/oss-fuzz
 */

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vp8cx.h"
#include "vpx/vpx_encoder.h"
#include "vpx_ports/mem_ops.h"
#include "third_party/nalloc/nalloc.h"

// fuzz header to have config options, before raw image data
#define FUZZ_HDR_SZ 32

#define VPXC_INTERFACE(name) VPXC_INTERFACE_(name)
#define VPXC_INTERFACE_(name) vpx_codec_##name##_cx()

extern "C" void usage_exit(void) { exit(EXIT_FAILURE); }

static int vpx_img_plane_width(const vpx_image_t *img, int plane) {
  if (plane > 0 && img->x_chroma_shift > 0)
    return (img->d_w + 1) >> img->x_chroma_shift;
  else
    return img->d_w;
}

static int vpx_img_plane_height(const vpx_image_t *img, int plane) {
  if (plane > 0 && img->y_chroma_shift > 0)
    return (img->d_h + 1) >> img->y_chroma_shift;
  else
    return img->d_h;
}

static int fuzz_vpx_img_read(vpx_image_t *img, const uint8_t *data,
                             size_t size) {
  int plane;
  // TODO: wtc - Need to clamp the sample values so that they are in range
  // For example, if the bit depth is 10, the sample values must be <= 1023.
  assert(img->bit_depth == 8);
  const size_t bytespp = (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;

  if (size == 0) return 0;
  size_t used = 0;
  for (plane = 0; plane < 3; ++plane) {
    unsigned char *buf = img->planes[plane];
    const int stride = img->stride[plane];
    int w = vpx_img_plane_width(img, plane);
    const int h = vpx_img_plane_height(img, plane);
    int y;

    // Assuming that for nv12 we read all chroma data at once
    if (img->fmt == VPX_IMG_FMT_NV12 && plane > 1) break;
    // Fixing NV12 chroma width if it is odd
    if (img->fmt == VPX_IMG_FMT_NV12 && plane == 1) w = (w + 1) & ~1;

    for (y = 0; y < h; ++y) {
      size_t nb = bytespp * w;
      if (nb > size - used) {
        nb = size - used;
      }
      memcpy(buf, data, nb);
      memset(buf + nb, 0, bytespp * w - nb);
      buf += stride;
      data += nb;
      used += nb;
    }
  }

  return used;
}

static int encode_frame(vpx_codec_ctx_t *codec, vpx_image_t *img,
                        int frame_index, int flags, FILE *out,
                        vpx_enc_deadline_t quality) {
  int got_pkts = 0;
  vpx_codec_iter_t iter = NULL;
  const vpx_codec_cx_pkt_t *pkt = NULL;
  const vpx_codec_err_t res =
      vpx_codec_encode(codec, img, frame_index, 1, flags, quality);
  if (res != VPX_CODEC_OK) return 0;

  while ((pkt = vpx_codec_get_cx_data(codec, &iter)) != NULL) {
    got_pkts = 1;

    if (pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
      if (fwrite(pkt->data.frame.buf, 1, pkt->data.frame.sz, out) !=
          pkt->data.frame.sz)
        return 0;
    }
  }

  return got_pkts;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size <= FUZZ_HDR_SZ) {
    return 0;
  }
  nalloc_init(nullptr);

  int keyframe_interval = 0;
  int frame_count = 0;
  vpx_codec_ctx_t codec;
  vpx_image_t raw;
  vpx_codec_enc_cfg_t cfg;
  vpx_enc_deadline_t quality = VPX_DL_GOOD_QUALITY;

  if ((data[0] & 0x80) != 0) {
    keyframe_interval = 8;
  }
  if ((data[0] & 0x40) != 0) {
    quality = VPX_DL_REALTIME;
  } else if ((data[0] & 0x20) != 0) {
    quality = VPX_DL_BEST_QUALITY;
  }

  if (vpx_codec_enc_config_default(VPXC_INTERFACE(ENCODER), &cfg, 0)) abort();
  FILE *out = fopen("/dev/null", "wb");

  switch (data[0] & 0x1F) {
    case 0: cfg.g_w = 64; cfg.g_h = 1;
    case 1: cfg.g_w = 1; cfg.g_h = 48;
    case 2: cfg.g_w = 1; cfg.g_h = 1;
    case 3: cfg.g_w = 4; cfg.g_h = 4;
    case 4: cfg.g_w = 16; cfg.g_h = 16;
    default: cfg.g_w = 64; cfg.g_h = 48;
  }
  cfg.g_timebase.num = 1;
  cfg.g_timebase.den = 30;  // fps
  cfg.rc_target_bitrate = 200;
  cfg.g_error_resilient = 1;

  if (vpx_codec_enc_init(&codec, VPXC_INTERFACE(ENCODER), &cfg, 0)) {
    return 0;
  }

  if (!vpx_img_alloc(&raw, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h, 1)) {
    goto fail;
  }

  nalloc_start(data, size);
  // We may want to add more config options (for more complex encoders as seen
  // in the examples) in the future while still maintaining the same format (so
  // that generated corpus is still valid). So we reserve FUZZ_HDR_SZ=32 bytes
  // for this even if we just use one byte so far.
  data += FUZZ_HDR_SZ;
  size -= FUZZ_HDR_SZ;

  // Encode frames.
  while (1) {
    int flags = 0;
    size_t size_read = fuzz_vpx_img_read(&raw, data, size);
    if (size_read == 0) break;
    data += size_read;
    size -= size_read;
    if (keyframe_interval > 0 && frame_count % keyframe_interval == 0)
      flags |= VPX_EFLAG_FORCE_KF;
    encode_frame(&codec, &raw, frame_count++, flags, out, quality);
  }

  // Flush encoder.
  while (encode_frame(&codec, NULL, -1, 0, out, quality)) {
  }

fail:
  nalloc_end();
  vpx_img_free(&raw);
  vpx_codec_destroy(&codec);
  fclose(out);
  return 0;
}
