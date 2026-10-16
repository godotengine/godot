// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_JPEG_ENC_JPEG_DATA_H_
#define LIB_JXL_JPEG_ENC_JPEG_DATA_H_

#include <jxl/memory_manager.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {

namespace jpeg {

// Optional text/EXIF metadata.
struct Blobs {
  std::vector<uint8_t> exif;
  std::vector<uint8_t> iptc;
  std::vector<uint8_t> jhgm;
  std::vector<uint8_t> jumbf;
  std::vector<uint8_t> xmp;
};

Status EncodeJPEGData(JxlMemoryManager* memory_manager, JPEGData& jpeg_data,
                      std::vector<uint8_t>* bytes,
                      const CompressParams& cparams);

Status SetColorEncodingFromJpegData(const jpeg::JPEGData& jpg,
                                    ColorEncoding* color_encoding);
Status SetChromaSubsamplingFromJpegData(const JPEGData& jpg,
                                        YCbCrChromaSubsampling* cs);
Status SetColorTransformFromJpegData(const JPEGData& jpg,
                                     ColorTransform* color_transform);

/**
 * Decodes bytes containing JPEG codestream as coefficients only,
 * for lossless JPEG transcoding.
 */
StatusOr<std::unique_ptr<JPEGData>> ParseJPG(JxlMemoryManager* memory_manager,
                                             Bytes bytes);
Status SetBlobsFromJpegData(const jpeg::JPEGData& jpeg_data, Blobs* blobs);

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_ENC_JPEG_DATA_H_
