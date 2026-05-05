/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_TEST_VPX_SCALE_TEST_H_
#define VPX_TEST_VPX_SCALE_TEST_H_

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./vpx_scale_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_scale/yv12config.h"

using libvpx_test::ACMRandom;

namespace libvpx_test {

class VpxScaleBase {
 public:
  virtual ~VpxScaleBase() { libvpx_test::ClearSystemState(); }

  void ResetImage(YV12_BUFFER_CONFIG *const img, const int width,
                  const int height) {
    memset(img, 0, sizeof(*img));
    ASSERT_EQ(
        0, vp8_yv12_alloc_frame_buffer(img, width, height, VP8BORDERINPIXELS))
        << "for width: " << width << " height: " << height;
    memset(img->buffer_alloc, kBufFiller, img->frame_size);
  }

  void ResetImages(const int width, const int height) {
    ResetImage(&img_, width, height);
    ResetImage(&ref_img_, width, height);
    ResetImage(&dst_img_, width, height);

    FillPlane(img_.y_buffer, img_.y_crop_width, img_.y_crop_height,
              img_.y_stride);
    FillPlane(img_.u_buffer, img_.uv_crop_width, img_.uv_crop_height,
              img_.uv_stride);
    FillPlane(img_.v_buffer, img_.uv_crop_width, img_.uv_crop_height,
              img_.uv_stride);
  }

  void ResetScaleImage(YV12_BUFFER_CONFIG *const img, const int width,
                       const int height) {
    memset(img, 0, sizeof(*img));
#if CONFIG_VP9_HIGHBITDEPTH
    ASSERT_EQ(0, vpx_alloc_frame_buffer(img, width, height, 1, 1, 0,
                                        VP9_ENC_BORDER_IN_PIXELS, 0));
#else
    ASSERT_EQ(0, vpx_alloc_frame_buffer(img, width, height, 1, 1,
                                        VP9_ENC_BORDER_IN_PIXELS, 0));
#endif
    memset(img->buffer_alloc, kBufFiller, img->frame_size);
  }

  void ResetScaleImages(const int src_width, const int src_height,
                        const int dst_width, const int dst_height) {
    ResetScaleImage(&img_, src_width, src_height);
    ResetScaleImage(&ref_img_, dst_width, dst_height);
    ResetScaleImage(&dst_img_, dst_width, dst_height);
    FillPlaneExtreme(img_.y_buffer, img_.y_crop_width, img_.y_crop_height,
                     img_.y_stride);
    FillPlaneExtreme(img_.u_buffer, img_.uv_crop_width, img_.uv_crop_height,
                     img_.uv_stride);
    FillPlaneExtreme(img_.v_buffer, img_.uv_crop_width, img_.uv_crop_height,
                     img_.uv_stride);
  }

  void DeallocImages() {
    vp8_yv12_de_alloc_frame_buffer(&img_);
    vp8_yv12_de_alloc_frame_buffer(&ref_img_);
    vp8_yv12_de_alloc_frame_buffer(&dst_img_);
  }

  void DeallocScaleImages() {
    vpx_free_frame_buffer(&img_);
    vpx_free_frame_buffer(&ref_img_);
    vpx_free_frame_buffer(&dst_img_);
  }

 protected:
  static const int kBufFiller = 123;
  static const int kBufMax = kBufFiller - 1;

  static void FillPlane(uint8_t *const buf, const int width, const int height,
                        const int stride) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        buf[x + (y * stride)] = (x + (width * y)) % kBufMax;
      }
    }
  }

  static void FillPlaneExtreme(uint8_t *const buf, const int width,
                               const int height, const int stride) {
    ACMRandom rnd;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        buf[x + (y * stride)] = rnd.Rand8() % 2 ? 255 : 0;
      }
    }
  }

  static void ExtendPlane(uint8_t *buf, int crop_width, int crop_height,
                          int width, int height, int stride, int padding) {
    // Copy the outermost visible pixel to a distance of at least 'padding.'
    // The buffers are allocated such that there may be excess space outside the
    // padding. As long as the minimum amount of padding is achieved it is not
    // necessary to fill this space as well.
    uint8_t *left = buf - padding;
    uint8_t *right = buf + crop_width;
    const int right_extend = padding + (width - crop_width);
    const int bottom_extend = padding + (height - crop_height);

    // Fill the border pixels from the nearest image pixel.
    for (int y = 0; y < crop_height; ++y) {
      memset(left, left[padding], padding);
      memset(right, right[-1], right_extend);
      left += stride;
      right += stride;
    }

    left = buf - padding;
    uint8_t *top = left - (stride * padding);
    // The buffer does not always extend as far as the stride.
    // Equivalent to padding + width + padding.
    const int extend_width = padding + crop_width + right_extend;

    // The first row was already extended to the left and right. Copy it up.
    for (int y = 0; y < padding; ++y) {
      memcpy(top, left, extend_width);
      top += stride;
    }

    uint8_t *bottom = left + (crop_height * stride);
    for (int y = 0; y < bottom_extend; ++y) {
      memcpy(bottom, left + (crop_height - 1) * stride, extend_width);
      bottom += stride;
    }
  }

  void ReferenceExtendBorder() {
    ExtendPlane(ref_img_.y_buffer, ref_img_.y_crop_width,
                ref_img_.y_crop_height, ref_img_.y_width, ref_img_.y_height,
                ref_img_.y_stride, ref_img_.border);
    ExtendPlane(ref_img_.u_buffer, ref_img_.uv_crop_width,
                ref_img_.uv_crop_height, ref_img_.uv_width, ref_img_.uv_height,
                ref_img_.uv_stride, ref_img_.border / 2);
    ExtendPlane(ref_img_.v_buffer, ref_img_.uv_crop_width,
                ref_img_.uv_crop_height, ref_img_.uv_width, ref_img_.uv_height,
                ref_img_.uv_stride, ref_img_.border / 2);
  }

  void ReferenceCopyFrame() {
    // Copy img_ to ref_img_ and extend frame borders. This will be used for
    // verifying extend_fn_ as well as copy_frame_fn_.
    EXPECT_EQ(ref_img_.frame_size, img_.frame_size);
    for (int y = 0; y < img_.y_crop_height; ++y) {
      for (int x = 0; x < img_.y_crop_width; ++x) {
        ref_img_.y_buffer[x + y * ref_img_.y_stride] =
            img_.y_buffer[x + y * img_.y_stride];
      }
    }

    for (int y = 0; y < img_.uv_crop_height; ++y) {
      for (int x = 0; x < img_.uv_crop_width; ++x) {
        ref_img_.u_buffer[x + y * ref_img_.uv_stride] =
            img_.u_buffer[x + y * img_.uv_stride];
        ref_img_.v_buffer[x + y * ref_img_.uv_stride] =
            img_.v_buffer[x + y * img_.uv_stride];
      }
    }

    ReferenceExtendBorder();
  }

  void CompareImages(const YV12_BUFFER_CONFIG actual) {
    EXPECT_EQ(ref_img_.frame_size, actual.frame_size);
    EXPECT_EQ(0, memcmp(ref_img_.buffer_alloc, actual.buffer_alloc,
                        ref_img_.frame_size));
  }

  YV12_BUFFER_CONFIG img_;
  YV12_BUFFER_CONFIG ref_img_;
  YV12_BUFFER_CONFIG dst_img_;
};

}  // namespace libvpx_test

#endif  // VPX_TEST_VPX_SCALE_TEST_H_
