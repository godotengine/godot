// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_comparator.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <memory>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_gamma_correct.h"
#include "lib/jxl/enc_image_bundle.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {
namespace {

// color is linear, but blending happens in gamma-compressed space using
// (gamma-compressed) grayscale background color, alpha image represents
// weights of the sRGB colors in the [0 .. (1 << bit_depth) - 1] interval,
// output image is in linear space.
void AlphaBlend(const Image3F& in, const size_t c, float background_linear,
                const ImageF& alpha, Image3F* out) {
  const float background = LinearToSrgb8Direct(background_linear);

  for (size_t y = 0; y < out->ysize(); ++y) {
    const float* JXL_RESTRICT row_a = alpha.ConstRow(y);
    const float* JXL_RESTRICT row_i = in.ConstPlaneRow(c, y);
    float* JXL_RESTRICT row_o = out->PlaneRow(c, y);
    for (size_t x = 0; x < out->xsize(); ++x) {
      const float a = row_a[x];
      if (a <= 0.f) {
        row_o[x] = background_linear;
      } else if (a >= 1.f) {
        row_o[x] = row_i[x];
      } else {
        const float w_fg = a;
        const float w_bg = 1.0f - w_fg;
        const float fg = w_fg * LinearToSrgb8Direct(row_i[x]);
        const float bg = w_bg * background;
        row_o[x] = Srgb8ToLinearDirect(fg + bg);
      }
    }
  }
}

void AlphaBlend(float background_linear, ImageBundle* io_linear_srgb) {
  // No alpha => all opaque.
  if (!io_linear_srgb->HasAlpha()) return;

  for (size_t c = 0; c < 3; ++c) {
    AlphaBlend(*io_linear_srgb->color(), c, background_linear,
               *io_linear_srgb->alpha(), io_linear_srgb->color());
  }
}

Status ComputeScoreImpl(const ImageBundle& rgb0, const ImageBundle& rgb1,
                        Comparator* comparator, ImageF* distmap, float& score) {
  JXL_RETURN_IF_ERROR(comparator->SetReferenceImage(rgb0));
  JXL_RETURN_IF_ERROR(comparator->CompareWith(rgb1, distmap, &score));
  return true;
}

}  // namespace

Status ComputeScore(const ImageBundle& rgb0, const ImageBundle& rgb1,
                    Comparator* comparator, const JxlCmsInterface& cms,
                    float* score, ImageF* diffmap, ThreadPool* pool,
                    bool ignore_alpha) {
  JxlMemoryManager* memory_manager = rgb0.memory_manager();
  // Convert to linear sRGB (unless already in that space)
  auto metadata0 = jxl::make_unique<ImageMetadata>();
  *metadata0 = *rgb0.metadata();
  auto store0 = jxl::make_unique<ImageBundle>(memory_manager, metadata0.get());
  const ImageBundle* linear_srgb0;
  JXL_RETURN_IF_ERROR(
      TransformIfNeeded(rgb0, ColorEncoding::LinearSRGB(rgb0.IsGray()), cms,
                        pool, store0.get(), &linear_srgb0));
  auto metadata1 = jxl::make_unique<ImageMetadata>();
  *metadata1 = *rgb1.metadata();
  auto store1 = jxl::make_unique<ImageBundle>(memory_manager, metadata1.get());
  const ImageBundle* linear_srgb1;
  JXL_RETURN_IF_ERROR(
      TransformIfNeeded(rgb1, ColorEncoding::LinearSRGB(rgb1.IsGray()), cms,
                        pool, store1.get(), &linear_srgb1));

  // No alpha: skip blending, only need a single call to Butteraugli.
  if (ignore_alpha || (!rgb0.HasAlpha() && !rgb1.HasAlpha())) {
    JXL_RETURN_IF_ERROR(ComputeScoreImpl(*linear_srgb0, *linear_srgb1,
                                         comparator, diffmap, *score));
    return true;
  }

  // Blend on black and white backgrounds

  ImageF diffmap_black;
  float dist_black;
  {
    const float black = 0.0f;
    JXL_ASSIGN_OR_RETURN(ImageBundle blended_black0, linear_srgb0->Copy());
    JXL_ASSIGN_OR_RETURN(ImageBundle blended_black1, linear_srgb1->Copy());
    AlphaBlend(black, &blended_black0);
    AlphaBlend(black, &blended_black1);
    JXL_RETURN_IF_ERROR(ComputeScoreImpl(blended_black0, blended_black1,
                                         comparator, &diffmap_black,
                                         dist_black));
  }

  ImageF diffmap_white;
  float dist_white;
  {
    const float white = 1.0f;
    JXL_ASSIGN_OR_RETURN(ImageBundle blended_white0, linear_srgb0->Copy());
    JXL_ASSIGN_OR_RETURN(ImageBundle blended_white1, linear_srgb1->Copy());
    AlphaBlend(white, &blended_white0);
    AlphaBlend(white, &blended_white1);
    JXL_RETURN_IF_ERROR(ComputeScoreImpl(blended_white0, blended_white1,
                                         comparator, &diffmap_white,
                                         dist_white));
  }

  // diffmap and return values are the max of diffmap_black/white.
  if (diffmap != nullptr) {
    const size_t xsize = rgb0.xsize();
    const size_t ysize = rgb0.ysize();
    JXL_ASSIGN_OR_RETURN(*diffmap,
                         ImageF::Create(memory_manager, xsize, ysize));
    for (size_t y = 0; y < ysize; ++y) {
      const float* JXL_RESTRICT row_black = diffmap_black.ConstRow(y);
      const float* JXL_RESTRICT row_white = diffmap_white.ConstRow(y);
      float* JXL_RESTRICT row_out = diffmap->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = std::max(row_black[x], row_white[x]);
      }
    }
  }
  *score = std::max(dist_black, dist_white);
  return true;
}

}  // namespace jxl
