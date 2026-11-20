// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_IMAGE_BUNDLE_H_
#define LIB_JXL_ENC_IMAGE_BUNDLE_H_

#include <jxl/cms_interface.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {

Status ApplyColorTransform(const ColorEncoding& c_current,
                           float intensity_target, const Image3F& color,
                           const ImageF* black, const Rect& rect,
                           const ColorEncoding& c_desired,
                           const JxlCmsInterface& cms, ThreadPool* pool,
                           Image3F* out);

// Does color transformation from in.c_current() to c_desired if the color
// encodings are different, or nothing if they are already the same.
// If color transformation is done, stores the transformed values into store and
// sets the out pointer to store, else leaves store untouched and sets the out
// pointer to &in.
// Returns false if color transform fails.
Status TransformIfNeeded(const ImageBundle& in, const ColorEncoding& c_desired,
                         const JxlCmsInterface& cms, ThreadPool* pool,
                         ImageBundle* store, const ImageBundle** out);

}  // namespace jxl

#endif  // LIB_JXL_ENC_IMAGE_BUNDLE_H_
