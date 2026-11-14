// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_butteraugli_comparator.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <cstddef>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_image_bundle.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {

JxlButteraugliComparator::JxlButteraugliComparator(
    const ButteraugliParams& params, const JxlCmsInterface& cms)
    : params_(params), cms_(cms) {}

Status JxlButteraugliComparator::SetReferenceImage(const ImageBundle& ref) {
  const ImageBundle* ref_linear_srgb;
  JxlMemoryManager* memory_manager = ref.memory_manager();
  ImageMetadata metadata = *ref.metadata();
  ImageBundle store(memory_manager, &metadata);
  if (!TransformIfNeeded(ref, ColorEncoding::LinearSRGB(ref.IsGray()), cms_,
                         /*pool=*/nullptr, &store, &ref_linear_srgb)) {
    return false;
  }
  JXL_ASSIGN_OR_RETURN(comparator_, ButteraugliComparator::Make(
                                        ref_linear_srgb->color(), params_));
  xsize_ = ref.xsize();
  ysize_ = ref.ysize();
  intensity_target_ = ref.metadata()->IntensityTarget();
  return true;
}

Status JxlButteraugliComparator::SetLinearReferenceImage(
    const Image3F& linear) {
  JXL_ASSIGN_OR_RETURN(comparator_,
                       ButteraugliComparator::Make(linear, params_));
  xsize_ = linear.xsize();
  ysize_ = linear.ysize();
  return true;
}

Status JxlButteraugliComparator::CompareWith(const ImageBundle& actual,
                                             ImageF* diffmap, float* score) {
  if (!comparator_) {
    return JXL_FAILURE("Must set reference image first");
  }
  if (xsize_ != actual.xsize() || ysize_ != actual.ysize()) {
    return JXL_FAILURE("Images must have same size");
  }
  JxlMemoryManager* memory_manager = actual.memory_manager();

  const ImageBundle* actual_linear_srgb;
  ImageMetadata metadata = *actual.metadata();
  ImageBundle store(memory_manager, &metadata);
  if (!TransformIfNeeded(actual, ColorEncoding::LinearSRGB(actual.IsGray()),
                         cms_,
                         /*pool=*/nullptr, &store, &actual_linear_srgb)) {
    return false;
  }

  JXL_ASSIGN_OR_RETURN(ImageF temp_diffmap,
                       ImageF::Create(memory_manager, xsize_, ysize_));
  const Image3F* scaled_actual_linear_srgb = &actual_linear_srgb->color();
  Image3F scaled_actual_linear_srgb_store;
  if (intensity_target_ != 0 &&
      actual.metadata()->IntensityTarget() != intensity_target_) {
    scaled_actual_linear_srgb = &scaled_actual_linear_srgb_store;
    JXL_ASSIGN_OR_RETURN(scaled_actual_linear_srgb_store,
                         Image3F::Create(memory_manager, xsize_, ysize_));
    const float scale =
        actual.metadata()->IntensityTarget() / intensity_target_;
    for (size_t c = 0; c < 3; ++c) {
      for (size_t y = 0; y < ysize_; ++y) {
        const float* JXL_RESTRICT source_row =
            actual_linear_srgb->color().ConstPlaneRow(c, y);
        float* JXL_RESTRICT scaled_row =
            scaled_actual_linear_srgb_store.PlaneRow(c, y);
        for (size_t x = 0; x < xsize_; ++x) {
          scaled_row[x] = scale * source_row[x];
        }
      }
    }
  }
  JXL_RETURN_IF_ERROR(
      comparator_->Diffmap(*scaled_actual_linear_srgb, temp_diffmap));

  if (score != nullptr) {
    *score = ButteraugliScoreFromDiffmap(temp_diffmap, &params_);
  }
  if (diffmap != nullptr) {
    diffmap->Swap(temp_diffmap);
  }

  return true;
}

float JxlButteraugliComparator::GoodQualityScore() const {
  return ButteraugliFuzzyInverse(1.5);
}

float JxlButteraugliComparator::BadQualityScore() const {
  return ButteraugliFuzzyInverse(0.5);
}

}  // namespace jxl
