// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Functions for reading a jpeg byte stream into a JPEGData object.

#ifndef LIB_JXL_JPEG_ENC_JPEG_DATA_READER_H_
#define LIB_JXL_JPEG_ENC_JPEG_DATA_READER_H_

#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {

enum class JpegReadMode {
  kReadHeader,  // only basic headers
  kReadTables,  // headers and tables (quant, Huffman, ...)
  kReadAll,     // everything
};

// Parses the JPEG stream contained in data[*pos ... len) and fills in *jpg with
// the parsed information.
// If mode is kReadHeader, it fills in only the image dimensions in *jpg.
// Returns false if the data is not valid JPEG, or if it contains an unsupported
// JPEG feature.
bool ReadJpeg(const uint8_t* data, size_t len, JpegReadMode mode,
              JPEGData* jpg);

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_ENC_JPEG_DATA_READER_H_
