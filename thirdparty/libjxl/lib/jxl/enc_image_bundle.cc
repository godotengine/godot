// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_image_bundle.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

Status ApplyColorTransform(const ColorEncoding& c_current,
                           float intensity_target, const Image3F& color,
                           const ImageF* black, const Rect& rect,
                           const ColorEncoding& c_desired,
                           const JxlCmsInterface& cms, ThreadPool* pool,
                           Image3F* out) {
  ColorSpaceTransform c_transform(cms);
  // Changing IsGray is probably a bug.
  JXL_ENSURE(c_current.IsGray() == c_desired.IsGray());
  bool is_gray = c_current.IsGray();
  JxlMemoryManager* memory_amanger = color.memory_manager();
  if (out->xsize() < rect.xsize() || out->ysize() < rect.ysize()) {
    JXL_ASSIGN_OR_RETURN(
        *out, Image3F::Create(memory_amanger, rect.xsize(), rect.ysize()));
  } else {
    JXL_RETURN_IF_ERROR(out->ShrinkTo(rect.xsize(), rect.ysize()));
  }
  const auto init = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(c_transform.Init(c_current, c_desired, intensity_target,
                                         rect.xsize(), num_threads));
    return true;
  };
  const auto transform_row = [&](const uint32_t y,
                                 const size_t thread) -> Status {
    float* mutable_src_buf = c_transform.BufSrc(thread);
    const float* src_buf = mutable_src_buf;
    // Interleave input.
    if (is_gray) {
      src_buf = rect.ConstPlaneRow(color, 0, y);
    } else if (c_current.IsCMYK()) {
      if (!black)
        return JXL_FAILURE("Black plane is missing for CMYK transform");
      const float* JXL_RESTRICT row_in0 = rect.ConstPlaneRow(color, 0, y);
      const float* JXL_RESTRICT row_in1 = rect.ConstPlaneRow(color, 1, y);
      const float* JXL_RESTRICT row_in2 = rect.ConstPlaneRow(color, 2, y);
      const float* JXL_RESTRICT row_in3 = rect.ConstRow(*black, y);
      for (size_t x = 0; x < rect.xsize(); x++) {
        // CMYK convention in JXL: 0 = max ink, 1 = white
        mutable_src_buf[4 * x + 0] = row_in0[x];
        mutable_src_buf[4 * x + 1] = row_in1[x];
        mutable_src_buf[4 * x + 2] = row_in2[x];
        mutable_src_buf[4 * x + 3] = row_in3[x];
      }
    } else {
      const float* JXL_RESTRICT row_in0 = rect.ConstPlaneRow(color, 0, y);
      const float* JXL_RESTRICT row_in1 = rect.ConstPlaneRow(color, 1, y);
      const float* JXL_RESTRICT row_in2 = rect.ConstPlaneRow(color, 2, y);
      for (size_t x = 0; x < rect.xsize(); x++) {
        mutable_src_buf[3 * x + 0] = row_in0[x];
        mutable_src_buf[3 * x + 1] = row_in1[x];
        mutable_src_buf[3 * x + 2] = row_in2[x];
      }
    }
    float* JXL_RESTRICT dst_buf = c_transform.BufDst(thread);
    JXL_RETURN_IF_ERROR(
        c_transform.Run(thread, src_buf, dst_buf, rect.xsize()));
    float* JXL_RESTRICT row_out0 = out->PlaneRow(0, y);
    float* JXL_RESTRICT row_out1 = out->PlaneRow(1, y);
    float* JXL_RESTRICT row_out2 = out->PlaneRow(2, y);
    // De-interleave output and convert type.
    if (is_gray) {
      for (size_t x = 0; x < rect.xsize(); x++) {
        row_out0[x] = dst_buf[x];
        row_out1[x] = dst_buf[x];
        row_out2[x] = dst_buf[x];
      }
    } else {
      for (size_t x = 0; x < rect.xsize(); x++) {
        row_out0[x] = dst_buf[3 * x + 0];
        row_out1[x] = dst_buf[3 * x + 1];
        row_out2[x] = dst_buf[3 * x + 2];
      }
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, rect.ysize(), init, transform_row,
                                "Colorspace transform"));
  return true;
}

namespace {

// Copies ib:rect, converts, and copies into out.
Status CopyToT(const ImageMetadata* metadata, const ImageBundle* ib,
               const Rect& rect, const ColorEncoding& c_desired,
               const JxlCmsInterface& cms, ThreadPool* pool, Image3F* out) {
  return ApplyColorTransform(ib->c_current(), metadata->IntensityTarget(),
                             ib->color(), ib->black(), rect, c_desired, cms,
                             pool, out);
}

}  // namespace

Status ImageBundle::TransformTo(const ColorEncoding& c_desired,
                                const JxlCmsInterface& cms, ThreadPool* pool) {
  JXL_RETURN_IF_ERROR(CopyTo(Rect(color_), c_desired, cms, &color_, pool));
  c_current_ = c_desired;
  return true;
}
Status ImageBundle::CopyTo(const Rect& rect, const ColorEncoding& c_desired,
                           const JxlCmsInterface& cms, Image3F* out,
                           ThreadPool* pool) const {
  return CopyToT(metadata_, this, rect, c_desired, cms, pool, out);
}
Status TransformIfNeeded(const ImageBundle& in, const ColorEncoding& c_desired,
                         const JxlCmsInterface& cms, ThreadPool* pool,
                         ImageBundle* store, const ImageBundle** out) {
  if (in.c_current().SameColorEncoding(c_desired) && !in.HasBlack()) {
    *out = &in;
    return true;
  }
  JxlMemoryManager* memory_manager = in.memory_manager();
  // TODO(janwas): avoid copying via createExternal+copyBackToIO
  // instead of copy+createExternal+copyBackToIO
  JXL_ASSIGN_OR_RETURN(
      Image3F color,
      Image3F::Create(memory_manager, in.color().xsize(), in.color().ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(in.color(), &color));
  JXL_RETURN_IF_ERROR(store->SetFromImage(std::move(color), in.c_current()));

  // Must at least copy the alpha channel for use by external_image.
  if (in.HasExtraChannels()) {
    std::vector<ImageF> extra_channels;
    for (const ImageF& extra_channel : in.extra_channels()) {
      JXL_ASSIGN_OR_RETURN(ImageF ec,
                           ImageF::Create(memory_manager, extra_channel.xsize(),
                                          extra_channel.ysize()));
      JXL_RETURN_IF_ERROR(CopyImageTo(extra_channel, &ec));
      extra_channels.emplace_back(std::move(ec));
    }
    JXL_RETURN_IF_ERROR(store->SetExtraChannels(std::move(extra_channels)));
  }

  if (!store->TransformTo(c_desired, cms, pool)) {
    return false;
  }
  *out = store;
  return true;
}

}  // namespace jxl
