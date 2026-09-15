// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Functions for writing a JPEGData object into a jpeg byte stream.

#ifndef LIB_JXL_JPEG_DEC_JPEG_DATA_WRITER_H_
#define LIB_JXL_JPEG_DEC_JPEG_DATA_WRITER_H_

#include <stddef.h>
#include <stdint.h>

#include <functional>

#include "lib/jxl/jpeg/dec_jpeg_serialization_state.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {

// Function type used to write len bytes into buf. Returns the number of bytes
// written.
using JPEGOutput = std::function<size_t(const uint8_t* buf, size_t len)>;

Status WriteJpeg(const JPEGData& jpg, const JPEGOutput& out);

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_DEC_JPEG_DATA_WRITER_H_
