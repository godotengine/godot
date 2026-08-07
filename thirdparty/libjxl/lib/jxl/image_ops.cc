// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/image_ops.h"

#include <cstddef>
#include <cstring>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"

namespace jxl {

Status PadImageToBlockMultipleInPlace(Image3F* JXL_RESTRICT in,
                                      size_t block_dim) {
  const size_t xsize_orig = in->xsize();
  const size_t ysize_orig = in->ysize();
  const size_t xsize = RoundUpTo(xsize_orig, block_dim);
  const size_t ysize = RoundUpTo(ysize_orig, block_dim);
  // Expands image size to the originally-allocated size.
  JXL_RETURN_IF_ERROR(in->ShrinkTo(xsize, ysize));
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize_orig; y++) {
      float* JXL_RESTRICT row = in->PlaneRow(c, y);
      for (size_t x = xsize_orig; x < xsize; x++) {
        row[x] = row[xsize_orig - 1];
      }
    }
    const float* JXL_RESTRICT row_src = in->ConstPlaneRow(c, ysize_orig - 1);
    for (size_t y = ysize_orig; y < ysize; y++) {
      memcpy(in->PlaneRow(c, y), row_src, xsize * sizeof(float));
    }
  }
  return true;
}

static Status DoDownsampleImage(const ImageF& input, size_t factor,
                                ImageF* output) {
  JXL_ENSURE(factor != 1);
  JXL_RETURN_IF_ERROR(output->ShrinkTo(DivCeil(input.xsize(), factor),
                                       DivCeil(input.ysize(), factor)));
  size_t in_stride = input.PixelsPerRow();
  for (size_t y = 0; y < output->ysize(); y++) {
    float* row_out = output->Row(y);
    const float* row_in = input.Row(factor * y);
    for (size_t x = 0; x < output->xsize(); x++) {
      size_t cnt = 0;
      float sum = 0;
      for (size_t iy = 0; iy < factor && iy + factor * y < input.ysize();
           iy++) {
        for (size_t ix = 0; ix < factor && ix + factor * x < input.xsize();
             ix++) {
          sum += row_in[iy * in_stride + x * factor + ix];
          cnt++;
        }
      }
      row_out[x] = sum / cnt;
    }
  }
  return true;
}

StatusOr<ImageF> DownsampleImage(const ImageF& image, size_t factor) {
  ImageF downsampled;
  // Allocate extra space to avoid a reallocation when padding.
  JxlMemoryManager* memory_manager = image.memory_manager();
  JXL_ASSIGN_OR_RETURN(
      downsampled,
      ImageF::Create(memory_manager, DivCeil(image.xsize(), factor) + kBlockDim,
                     DivCeil(image.ysize(), factor) + kBlockDim));
  JXL_RETURN_IF_ERROR(DoDownsampleImage(image, factor, &downsampled));
  return downsampled;
}

StatusOr<Image3F> DownsampleImage(const Image3F& opsin, size_t factor) {
  JXL_ENSURE(factor != 1);
  // Allocate extra space to avoid a reallocation when padding.
  Image3F downsampled;
  JxlMemoryManager* memory_manager = opsin.memory_manager();
  JXL_ASSIGN_OR_RETURN(
      downsampled, Image3F::Create(memory_manager,
                                   DivCeil(opsin.xsize(), factor) + kBlockDim,
                                   DivCeil(opsin.ysize(), factor) + kBlockDim));
  JXL_RETURN_IF_ERROR(downsampled.ShrinkTo(downsampled.xsize() - kBlockDim,
                                           downsampled.ysize() - kBlockDim));
  for (size_t c = 0; c < 3; c++) {
    JXL_RETURN_IF_ERROR(
        DoDownsampleImage(opsin.Plane(c), factor, &downsampled.Plane(c)));
  }
  return downsampled;
}

}  // namespace jxl
