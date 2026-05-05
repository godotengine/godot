/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/*
 * Fuzzer for libvpx decoders
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
   $mkdir vpx_dec_fuzzer
   $cd vpx_dec_fuzzer/

 * Enable sanitizers (Supported: address integer memory thread undefined)
   $source ../libvpx/tools/set_analyzer_env.sh address

 * Configure libvpx.
 * Note --size-limit and VPX_MAX_ALLOCABLE_MEMORY are defined to avoid
 * Out of memory errors when running generated fuzzer binary
   $../libvpx/configure --disable-unit-tests --size-limit=12288x12288 \
   --extra-cflags="-fsanitize=fuzzer-no-link \
   -DVPX_MAX_ALLOCABLE_MEMORY=1073741824" \
   --disable-webm-io --enable-debug --disable-vp8-encoder \
   --disable-vp9-encoder --disable-examples

 * Build libvpx
   $make -j32

 * Build vp9 fuzzer
   $ $CXX $CXXFLAGS -std=gnu++17 -DDECODER=vp9 \
   -fsanitize=fuzzer -I../libvpx -I. -Wl,--start-group \
   ../libvpx/examples/vpx_dec_fuzzer.cc -o ./vpx_dec_fuzzer_vp9 \
   ./libvpx.a -Wl,--end-group

 * DECODER should be defined as vp9 or vp8 to enable vp9/vp8
 *
 * create a corpus directory and copy some ivf files there.
 * Based on which codec (vp8/vp9) is being tested, it is recommended to
 * have corresponding ivf files in corpus directory
 * Empty corpus directoy also is acceptable, though not recommended
   $mkdir CORPUS && cp some-files CORPUS

 * Run fuzzing:
   $./vpx_dec_fuzzer_vp9 CORPUS

 * References:
 * http://llvm.org/docs/LibFuzzer.html
 * https://github.com/google/oss-fuzz
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <memory>

#include "third_party/nalloc/nalloc.h"
#include "vpx/vp8dx.h"
#include "vpx/vpx_decoder.h"
#include "vpx_ports/mem_ops.h"

#define IVF_FRAME_HDR_SZ (4 + 8) /* 4 byte size + 8 byte timestamp */
#define IVF_FILE_HDR_SZ 32

#define VPXD_INTERFACE(name) VPXD_INTERFACE_(name)
#define VPXD_INTERFACE_(name) vpx_codec_##name##_dx()

extern "C" void usage_exit(void) { exit(EXIT_FAILURE); }

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size <= IVF_FILE_HDR_SZ) {
    return 0;
  }
  nalloc_init(nullptr);

  vpx_codec_ctx_t codec;
  // Set thread count in the range [1, 64].
  const unsigned int threads = (data[IVF_FILE_HDR_SZ] & 0x3f) + 1;
  vpx_codec_dec_cfg_t cfg = { threads, 0, 0 };
  vpx_codec_flags_t flags = 0;
  if ((data[IVF_FILE_HDR_SZ] & 0x40) != 0) {
    flags |= VPX_CODEC_USE_POSTPROC;
  }
  vpx_codec_err_t err =
      vpx_codec_dec_init(&codec, VPXD_INTERFACE(DECODER), &cfg, flags);
  if (err == VPX_CODEC_INCAPABLE) {
    // vpx_codec_dec_init may fail with VPX_CODEC_USE_POSTPROC
    // if the library is configured with --disable-postproc.
    flags = 0;
    if (vpx_codec_dec_init(&codec, VPXD_INTERFACE(DECODER), &cfg, flags)) {
      return 0;
    }
  } else if (err != 0) {
    return 0;
  }

  nalloc_start(data, size);

  if (threads > 1) {
    const int enable = (data[IVF_FILE_HDR_SZ] & 0xa0) != 0;
    err = vpx_codec_control(&codec, VP9D_SET_LOOP_FILTER_OPT, enable);
  }

  data += IVF_FILE_HDR_SZ;
  size -= IVF_FILE_HDR_SZ;

  int frame_cnt = 0;
  while (size > IVF_FRAME_HDR_SZ) {
    size_t frame_size = mem_get_le32(data);
    size -= IVF_FRAME_HDR_SZ;
    data += IVF_FRAME_HDR_SZ;
    frame_size = std::min(size, frame_size);

    vpx_codec_stream_info_t stream_info;
    stream_info.sz = sizeof(stream_info);
    err = vpx_codec_peek_stream_info(VPXD_INTERFACE(DECODER), data, size,
                                     &stream_info);

    ++frame_cnt;
    if (flags & VPX_CODEC_USE_POSTPROC) {
      if (frame_cnt % 16 == 4) {
        vp8_postproc_cfg_t pp = { 0, 0, 0 };
        if (vpx_codec_control(&codec, VP8_SET_POSTPROC, &pp)) goto fail;
      } else if (frame_cnt % 16 == 12) {
        vp8_postproc_cfg_t pp = { VP8_DEBLOCK | VP8_DEMACROBLOCK | VP8_MFQE, 4,
                                  0 };
        if (vpx_codec_control(&codec, VP8_SET_POSTPROC, &pp)) goto fail;
      }
    }

    err = vpx_codec_decode(&codec, data, frame_size, nullptr, 0);
    static_cast<void>(err);
    vpx_codec_iter_t iter = nullptr;
    vpx_image_t *img = nullptr;
    while ((img = vpx_codec_get_frame(&codec, &iter)) != nullptr) {
    }
    data += frame_size;
    size -= frame_size;
  }
fail:
  vpx_codec_destroy(&codec);
  nalloc_end();
  return 0;
}
