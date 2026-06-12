/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_TEST_MD5_HELPER_H_
#define VPX_TEST_MD5_HELPER_H_

#include "./md5_utils.h"
#include "vpx/vpx_decoder.h"

namespace libvpx_test {
class MD5 {
 public:
  MD5() { MD5Init(&md5_); }

  void Add(const vpx_image_t *img) {
    for (int plane = 0; plane < 3; ++plane) {
      const uint8_t *buf = img->planes[plane];
      // Calculate the width and height to do the md5 check. For the chroma
      // plane, we never want to round down and thus skip a pixel so if
      // we are shifting by 1 (chroma_shift) we add 1 before doing the shift.
      // This works only for chroma_shift of 0 and 1.
      const int bytes_per_sample =
          (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;
      const int h =
          plane ? (img->d_h + img->y_chroma_shift) >> img->y_chroma_shift
                : img->d_h;
      const int w =
          (plane ? (img->d_w + img->x_chroma_shift) >> img->x_chroma_shift
                 : img->d_w) *
          bytes_per_sample;

      for (int y = 0; y < h; ++y) {
        MD5Update(&md5_, buf, w);
        buf += img->stride[plane];
      }
    }
  }

  void Add(const uint8_t *data, size_t size) {
    MD5Update(&md5_, data, static_cast<uint32_t>(size));
  }

  const char *Get() {
    static const char hex[16] = {
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    };
    uint8_t tmp[16];
    MD5Context ctx_tmp = md5_;

    MD5Final(tmp, &ctx_tmp);
    for (int i = 0; i < 16; i++) {
      res_[i * 2 + 0] = hex[tmp[i] >> 4];
      res_[i * 2 + 1] = hex[tmp[i] & 0xf];
    }
    res_[32] = 0;

    return res_;
  }

 protected:
  char res_[33];
  MD5Context md5_;
};

}  // namespace libvpx_test

#endif  // VPX_TEST_MD5_HELPER_H_
