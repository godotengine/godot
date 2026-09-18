// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_COMPARATOR_H_
#define LIB_JXL_ENC_COMPARATOR_H_

#include <jxl/cms_interface.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {

class Comparator {
 public:
  virtual ~Comparator() = default;

  // Sets the reference image, the first to compare
  // Image must be in linear sRGB (gamma expanded) in range 0.0f-1.0f as
  // the range from standard black point to standard white point, but values
  // outside permitted.
  virtual Status SetReferenceImage(const ImageBundle& ref) = 0;

  // Sets the actual image (with loss), the second to compare
  // Image must be in linear sRGB (gamma expanded) in range 0.0f-1.0f as
  // the range from standard black point to standard white point, but values
  // outside permitted.
  // In diffmap it outputs the local score per pixel, while in score it outputs
  // a single score. Any one may be set to nullptr to not compute it.
  virtual Status CompareWith(const ImageBundle& actual, ImageF* diffmap,
                             float* score) = 0;

  // Quality thresholds for diffmap and score values.
  // The good score must represent a value where the images are considered to
  // be perceptually indistinguishable (but not identical)
  // The bad value must be larger than good to indicate "lower means better"
  // and smaller than good to indicate "higher means better"
  virtual float GoodQualityScore() const = 0;
  virtual float BadQualityScore() const = 0;
};

// Computes the score given images in any RGB color model, optionally with
// alpha channel.
Status ComputeScore(const ImageBundle& rgb0, const ImageBundle& rgb1,
                    Comparator* comparator, const JxlCmsInterface& cms,
                    float* score, ImageF* diffmap = nullptr,
                    ThreadPool* pool = nullptr, bool ignore_alpha = false);

}  // namespace jxl

#endif  // LIB_JXL_ENC_COMPARATOR_H_
