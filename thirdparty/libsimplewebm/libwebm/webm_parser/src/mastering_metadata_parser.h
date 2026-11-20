// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_MASTERING_METADATA_PARSER_H_
#define SRC_MASTERING_METADATA_PARSER_H_

#include "src/float_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#MasteringMetadata
// http://www.webmproject.org/docs/container/#MasteringMetadata
class MasteringMetadataParser : public MasterValueParser<MasteringMetadata> {
 public:
  MasteringMetadataParser()
      : MasterValueParser<MasteringMetadata>(
            MakeChild<FloatParser>(
                Id::kPrimaryRChromaticityX,
                &MasteringMetadata::primary_r_chromaticity_x),
            MakeChild<FloatParser>(
                Id::kPrimaryRChromaticityY,
                &MasteringMetadata::primary_r_chromaticity_y),
            MakeChild<FloatParser>(
                Id::kPrimaryGChromaticityX,
                &MasteringMetadata::primary_g_chromaticity_x),
            MakeChild<FloatParser>(
                Id::kPrimaryGChromaticityY,
                &MasteringMetadata::primary_g_chromaticity_y),
            MakeChild<FloatParser>(
                Id::kPrimaryBChromaticityX,
                &MasteringMetadata::primary_b_chromaticity_x),
            MakeChild<FloatParser>(
                Id::kPrimaryBChromaticityY,
                &MasteringMetadata::primary_b_chromaticity_y),
            MakeChild<FloatParser>(
                Id::kWhitePointChromaticityX,
                &MasteringMetadata::white_point_chromaticity_x),
            MakeChild<FloatParser>(
                Id::kWhitePointChromaticityY,
                &MasteringMetadata::white_point_chromaticity_y),
            MakeChild<FloatParser>(Id::kLuminanceMax,
                                   &MasteringMetadata::luminance_max),
            MakeChild<FloatParser>(Id::kLuminanceMin,
                                   &MasteringMetadata::luminance_min)) {}
};

}  // namespace webm

#endif  // SRC_MASTERING_METADATA_PARSER_H_
