// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "hdr_util.h"

#include <climits>
#include <cstddef>
#include <new>

#include "mkvparser/mkvparser.h"

namespace libwebm {
const int Vp9CodecFeatures::kValueNotPresent = INT_MAX;

bool CopyPrimaryChromaticity(const mkvparser::PrimaryChromaticity& parser_pc,
                             PrimaryChromaticityPtr* muxer_pc) {
  muxer_pc->reset(new (std::nothrow)
                      mkvmuxer::PrimaryChromaticity(parser_pc.x, parser_pc.y));
  if (!muxer_pc->get())
    return false;
  return true;
}

bool MasteringMetadataValuePresent(double value) {
  return value != mkvparser::MasteringMetadata::kValueNotPresent;
}

bool CopyMasteringMetadata(const mkvparser::MasteringMetadata& parser_mm,
                           mkvmuxer::MasteringMetadata* muxer_mm) {
  if (MasteringMetadataValuePresent(parser_mm.luminance_max))
    muxer_mm->set_luminance_max(parser_mm.luminance_max);
  if (MasteringMetadataValuePresent(parser_mm.luminance_min))
    muxer_mm->set_luminance_min(parser_mm.luminance_min);

  PrimaryChromaticityPtr r_ptr(nullptr);
  PrimaryChromaticityPtr g_ptr(nullptr);
  PrimaryChromaticityPtr b_ptr(nullptr);
  PrimaryChromaticityPtr wp_ptr(nullptr);

  if (parser_mm.r) {
    if (!CopyPrimaryChromaticity(*parser_mm.r, &r_ptr))
      return false;
  }
  if (parser_mm.g) {
    if (!CopyPrimaryChromaticity(*parser_mm.g, &g_ptr))
      return false;
  }
  if (parser_mm.b) {
    if (!CopyPrimaryChromaticity(*parser_mm.b, &b_ptr))
      return false;
  }
  if (parser_mm.white_point) {
    if (!CopyPrimaryChromaticity(*parser_mm.white_point, &wp_ptr))
      return false;
  }

  if (!muxer_mm->SetChromaticity(r_ptr.get(), g_ptr.get(), b_ptr.get(),
                                 wp_ptr.get())) {
    return false;
  }

  return true;
}

bool ColourValuePresent(long long value) {
  return value != mkvparser::Colour::kValueNotPresent;
}

bool CopyColour(const mkvparser::Colour& parser_colour,
                mkvmuxer::Colour* muxer_colour) {
  if (!muxer_colour)
    return false;

  if (ColourValuePresent(parser_colour.matrix_coefficients))
    muxer_colour->set_matrix_coefficients(parser_colour.matrix_coefficients);
  if (ColourValuePresent(parser_colour.bits_per_channel))
    muxer_colour->set_bits_per_channel(parser_colour.bits_per_channel);
  if (ColourValuePresent(parser_colour.chroma_subsampling_horz)) {
    muxer_colour->set_chroma_subsampling_horz(
        parser_colour.chroma_subsampling_horz);
  }
  if (ColourValuePresent(parser_colour.chroma_subsampling_vert)) {
    muxer_colour->set_chroma_subsampling_vert(
        parser_colour.chroma_subsampling_vert);
  }
  if (ColourValuePresent(parser_colour.cb_subsampling_horz))
    muxer_colour->set_cb_subsampling_horz(parser_colour.cb_subsampling_horz);
  if (ColourValuePresent(parser_colour.cb_subsampling_vert))
    muxer_colour->set_cb_subsampling_vert(parser_colour.cb_subsampling_vert);
  if (ColourValuePresent(parser_colour.chroma_siting_horz))
    muxer_colour->set_chroma_siting_horz(parser_colour.chroma_siting_horz);
  if (ColourValuePresent(parser_colour.chroma_siting_vert))
    muxer_colour->set_chroma_siting_vert(parser_colour.chroma_siting_vert);
  if (ColourValuePresent(parser_colour.range))
    muxer_colour->set_range(parser_colour.range);
  if (ColourValuePresent(parser_colour.transfer_characteristics)) {
    muxer_colour->set_transfer_characteristics(
        parser_colour.transfer_characteristics);
  }
  if (ColourValuePresent(parser_colour.primaries))
    muxer_colour->set_primaries(parser_colour.primaries);
  if (ColourValuePresent(parser_colour.max_cll))
    muxer_colour->set_max_cll(parser_colour.max_cll);
  if (ColourValuePresent(parser_colour.max_fall))
    muxer_colour->set_max_fall(parser_colour.max_fall);

  if (parser_colour.mastering_metadata) {
    mkvmuxer::MasteringMetadata muxer_mm;
    if (!CopyMasteringMetadata(*parser_colour.mastering_metadata, &muxer_mm))
      return false;
    if (!muxer_colour->SetMasteringMetadata(muxer_mm))
      return false;
  }
  return true;
}

// Format of VPx private data:
//
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |    ID Byte    |   Length      |                               |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               |
//  |                                                               |
//  :               Bytes 1..Length of Codec Feature                :
//  |                                                               |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
// ID Byte Format
// ID byte is an unsigned byte.
//   0 1 2 3 4 5 6 7
//  +-+-+-+-+-+-+-+-+
//  |X|    ID       |
//  +-+-+-+-+-+-+-+-+
//
// The X bit is reserved.
//
// See the following link for more information:
// http://www.webmproject.org/vp9/profiles/
bool ParseVpxCodecPrivate(const uint8_t* private_data, int32_t length,
                          Vp9CodecFeatures* features) {
  const int kVpxCodecPrivateMinLength = 3;
  if (!private_data || !features || length < kVpxCodecPrivateMinLength)
    return false;

  const uint8_t kVp9ProfileId = 1;
  const uint8_t kVp9LevelId = 2;
  const uint8_t kVp9BitDepthId = 3;
  const uint8_t kVp9ChromaSubsamplingId = 4;
  const int kVpxFeatureLength = 1;
  int offset = 0;

  // Set features to not set.
  features->profile = Vp9CodecFeatures::kValueNotPresent;
  features->level = Vp9CodecFeatures::kValueNotPresent;
  features->bit_depth = Vp9CodecFeatures::kValueNotPresent;
  features->chroma_subsampling = Vp9CodecFeatures::kValueNotPresent;
  do {
    const uint8_t id_byte = private_data[offset++];
    const uint8_t length_byte = private_data[offset++];
    if (length_byte != kVpxFeatureLength)
      return false;
    if (id_byte == kVp9ProfileId) {
      const int priv_profile = static_cast<int>(private_data[offset++]);
      if (priv_profile < 0 || priv_profile > 3)
        return false;
      if (features->profile != Vp9CodecFeatures::kValueNotPresent &&
          features->profile != priv_profile) {
        return false;
      }
      features->profile = priv_profile;
    } else if (id_byte == kVp9LevelId) {
      const int priv_level = static_cast<int>(private_data[offset++]);

      const int kNumLevels = 14;
      const int levels[kNumLevels] = {10, 11, 20, 21, 30, 31, 40,
                                      41, 50, 51, 52, 60, 61, 62};

      for (int i = 0; i < kNumLevels; ++i) {
        if (priv_level == levels[i]) {
          if (features->level != Vp9CodecFeatures::kValueNotPresent &&
              features->level != priv_level) {
            return false;
          }
          features->level = priv_level;
          break;
        }
      }
      if (features->level == Vp9CodecFeatures::kValueNotPresent)
        return false;
    } else if (id_byte == kVp9BitDepthId) {
      const int priv_profile = static_cast<int>(private_data[offset++]);
      if (priv_profile != 8 && priv_profile != 10 && priv_profile != 12)
        return false;
      if (features->bit_depth != Vp9CodecFeatures::kValueNotPresent &&
          features->bit_depth != priv_profile) {
        return false;
      }
      features->bit_depth = priv_profile;
    } else if (id_byte == kVp9ChromaSubsamplingId) {
      const int priv_profile = static_cast<int>(private_data[offset++]);
      if (priv_profile != 0 && priv_profile != 1 && priv_profile != 2 &&
          priv_profile != 3)
        return false;
      if (features->chroma_subsampling != Vp9CodecFeatures::kValueNotPresent &&
          features->chroma_subsampling != priv_profile) {
        return false;
      }
      features->chroma_subsampling = priv_profile;
    } else {
      // Invalid ID.
      return false;
    }
  } while (offset + kVpxCodecPrivateMinLength <= length);

  return true;
}
}  // namespace libwebm
