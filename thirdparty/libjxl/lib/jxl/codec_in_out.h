// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_CODEC_IN_OUT_H_
#define LIB_JXL_CODEC_IN_OUT_H_

// Holds inputs/outputs for decoding/encoding images.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/luminance.h"

namespace jxl {

// Optional text/EXIF metadata.
struct Blobs {
  std::vector<uint8_t> exif;
  std::vector<uint8_t> iptc;
  std::vector<uint8_t> jhgm;
  std::vector<uint8_t> jumbf;
  std::vector<uint8_t> xmp;
};

// Holds a preview, a main image or one or more frames, plus the inputs/outputs
// to/from decoding/encoding.
class CodecInOut {
 public:
  explicit CodecInOut(JxlMemoryManager* memory_manager)
      : memory_manager(memory_manager),
        preview_frame(memory_manager, &metadata.m) {
    frames.reserve(1);
    frames.emplace_back(memory_manager, &metadata.m);
  }

  // Move-only.
  CodecInOut(CodecInOut&&) = default;
  CodecInOut& operator=(CodecInOut&&) = default;

  size_t LastStillFrame() const {
    JXL_DASSERT(!frames.empty());
    size_t last = 0;
    for (size_t i = 0; i < frames.size(); i++) {
      last = i;
      if (frames[i].duration > 0) break;
    }
    return last;
  }

  ImageBundle& Main() { return frames[LastStillFrame()]; }
  const ImageBundle& Main() const { return frames[LastStillFrame()]; }

  // If c_current.IsGray(), all planes must be identical.
  Status SetFromImage(Image3F&& color, const ColorEncoding& c_current) {
    JXL_RETURN_IF_ERROR(Main().SetFromImage(std::move(color), c_current));
    SetIntensityTarget(&this->metadata.m);
    JXL_RETURN_IF_ERROR(SetSize(Main().xsize(), Main().ysize()));
    return true;
  }

  Status SetSize(size_t xsize, size_t ysize) {
    JXL_RETURN_IF_ERROR(metadata.size.Set(xsize, ysize));
    return true;
  }

  Status CheckMetadata() const {
    JXL_ENSURE(metadata.m.bit_depth.bits_per_sample != 0);
    JXL_ENSURE(!metadata.m.color_encoding.ICC().empty());

    if (preview_frame.xsize() != 0) {
      JXL_RETURN_IF_ERROR(preview_frame.VerifyMetadata());
    }
    JXL_ENSURE(preview_frame.metadata() == &metadata.m);

    for (const ImageBundle& ib : frames) {
      JXL_RETURN_IF_ERROR(ib.VerifyMetadata());
      JXL_ENSURE(ib.metadata() == &metadata.m);
    }
    return true;
  }

  size_t xsize() const { return metadata.size.xsize(); }
  size_t ysize() const { return metadata.size.ysize(); }
  Status ShrinkTo(size_t xsize, size_t ysize) {
    // preview is unaffected.
    for (ImageBundle& ib : frames) {
      JXL_RETURN_IF_ERROR(ib.ShrinkTo(xsize, ysize));
    }
    JXL_RETURN_IF_ERROR(SetSize(xsize, ysize));
    return true;
  }

  // -- DECODER OUTPUT, ENCODER INPUT:

  // Metadata stored into / retrieved from bitstreams.

  JxlMemoryManager* memory_manager;

  Blobs blobs;

  CodecMetadata metadata;  // applies to preview and all frames

  // If metadata.have_preview:
  ImageBundle preview_frame;

  std::vector<ImageBundle> frames;  // size=1 if !metadata.have_animation

  // If the image should be written to a JPEG, use this quality for encoding.
  size_t jpeg_quality;
};

}  // namespace jxl

#endif  // LIB_JXL_CODEC_IN_OUT_H_
