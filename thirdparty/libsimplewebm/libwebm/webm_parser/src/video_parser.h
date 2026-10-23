// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_VIDEO_PARSER_H_
#define SRC_VIDEO_PARSER_H_

#include <cassert>
#include <cstdint>

#include "src/bool_parser.h"
#include "src/colour_parser.h"
#include "src/float_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "src/projection_parser.h"
#include "webm/callback.h"
#include "webm/dom_types.h"
#include "webm/id.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Video
// http://www.webmproject.org/docs/container/#Video
class VideoParser : public MasterValueParser<Video> {
 public:
  VideoParser()
      : MasterValueParser<Video>(
            MakeChild<IntParser<FlagInterlaced>>(Id::kFlagInterlaced,
                                                 &Video::interlaced),
            MakeChild<IntParser<StereoMode>>(Id::kStereoMode,
                                             &Video::stereo_mode),
            MakeChild<UnsignedIntParser>(Id::kAlphaMode, &Video::alpha_mode),
            MakeChild<UnsignedIntParser>(Id::kPixelWidth, &Video::pixel_width),
            MakeChild<UnsignedIntParser>(Id::kPixelHeight,
                                         &Video::pixel_height),
            MakeChild<UnsignedIntParser>(Id::kPixelCropBottom,
                                         &Video::pixel_crop_bottom),
            MakeChild<UnsignedIntParser>(Id::kPixelCropTop,
                                         &Video::pixel_crop_top),
            MakeChild<UnsignedIntParser>(Id::kPixelCropLeft,
                                         &Video::pixel_crop_left),
            MakeChild<UnsignedIntParser>(Id::kPixelCropRight,
                                         &Video::pixel_crop_right),
            MakeChild<UnsignedIntParser>(Id::kDisplayWidth,
                                         &Video::display_width)
                .NotifyOnParseComplete(),
            MakeChild<UnsignedIntParser>(Id::kDisplayHeight,
                                         &Video::display_height)
                .NotifyOnParseComplete(),
            MakeChild<IntParser<DisplayUnit>>(Id::kDisplayUnit,
                                              &Video::display_unit),
            MakeChild<IntParser<AspectRatioType>>(Id::kAspectRatioType,
                                                  &Video::aspect_ratio_type),
            MakeChild<FloatParser>(Id::kFrameRate, &Video::frame_rate),
            MakeChild<ColourParser>(Id::kColour, &Video::colour),
            MakeChild<ProjectionParser>(Id::kProjection, &Video::projection)) {}

  Status Init(const ElementMetadata& metadata,
              std::uint64_t max_size) override {
    display_width_has_value_ = false;
    display_height_has_value_ = false;

    return MasterValueParser::Init(metadata, max_size);
  }

  void InitAfterSeek(const Ancestory& child_ancestory,
                     const ElementMetadata& child_metadata) override {
    display_width_has_value_ = false;
    display_height_has_value_ = false;

    return MasterValueParser::InitAfterSeek(child_ancestory, child_metadata);
  }

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override {
    const Status status =
        MasterValueParser::Feed(callback, reader, num_bytes_read);
    if (status.completed_ok()) {
      FixMissingDisplaySize();
    }
    return status;
  }

 protected:
  void OnChildParsed(const ElementMetadata& metadata) override {
    assert(metadata.id == Id::kDisplayWidth ||
           metadata.id == Id::kDisplayHeight);

    if (metadata.id == Id::kDisplayWidth) {
      display_width_has_value_ = metadata.size > 0;
    } else {
      display_height_has_value_ = metadata.size > 0;
    }
  }

 private:
  bool display_width_has_value_;
  bool display_height_has_value_;

  void FixMissingDisplaySize() {
    if (!display_width_has_value_) {
      *mutable_value()->display_width.mutable_value() =
          value().pixel_width.value();
    }

    if (!display_height_has_value_) {
      *mutable_value()->display_height.mutable_value() =
          value().pixel_height.value();
    }
  }
};

}  // namespace webm

#endif  // SRC_VIDEO_PARSER_H_
