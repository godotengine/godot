// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_JPEG_DEC_JPEG_DATA_H_
#define LIB_JXL_JPEG_DEC_JPEG_DATA_H_

#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {
Status DecodeJPEGData(Span<const uint8_t> encoded, JPEGData* jpeg_data);
}
}  // namespace jxl

#endif  // LIB_JXL_JPEG_DEC_JPEG_DATA_H_
