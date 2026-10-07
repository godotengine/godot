// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "common/vp9_header_parser.h"

#include <stdio.h>

namespace vp9_parser {

bool Vp9HeaderParser::SetFrame(const uint8_t* frame, size_t length) {
  if (!frame || length == 0)
    return false;

  frame_ = frame;
  frame_size_ = length;
  bit_offset_ = 0;
  profile_ = -1;
  show_existing_frame_ = 0;
  key_ = 0;
  altref_ = 0;
  error_resilient_mode_ = 0;
  intra_only_ = 0;
  reset_frame_context_ = 0;
  color_space_ = 0;
  color_range_ = 0;
  subsampling_x_ = 0;
  subsampling_y_ = 0;
  refresh_frame_flags_ = 0;
  return true;
}

bool Vp9HeaderParser::ParseUncompressedHeader(const uint8_t* frame,
                                              size_t length) {
  if (!SetFrame(frame, length))
    return false;
  const int frame_marker = VpxReadLiteral(2);
  if (frame_marker != kVp9FrameMarker) {
    fprintf(stderr, "Invalid VP9 frame_marker:%d\n", frame_marker);
    return false;
  }

  profile_ = ReadBit();
  profile_ |= ReadBit() << 1;
  if (profile_ > 2)
    profile_ += ReadBit();

  // TODO(fgalligan): Decide how to handle show existing frames.
  show_existing_frame_ = ReadBit();
  if (show_existing_frame_)
    return true;

  key_ = !ReadBit();
  altref_ = !ReadBit();
  error_resilient_mode_ = ReadBit();
  if (key_) {
    if (!ValidateVp9SyncCode()) {
      fprintf(stderr, "Invalid Sync code!\n");
      return false;
    }

    ParseColorSpace();
    ParseFrameResolution();
    ParseFrameParallelMode();
    ParseTileInfo();
  } else {
    intra_only_ = altref_ ? ReadBit() : 0;
    reset_frame_context_ = error_resilient_mode_ ? 0 : VpxReadLiteral(2);
    if (intra_only_) {
      if (!ValidateVp9SyncCode()) {
        fprintf(stderr, "Invalid Sync code!\n");
        return false;
      }

      if (profile_ > 0) {
        ParseColorSpace();
      } else {
        // NOTE: The intra-only frame header does not include the specification
        // of either the color format or color sub-sampling in profile 0. VP9
        // specifies that the default color format should be YUV 4:2:0 in this
        // case (normative).
        color_space_ = kVpxCsBt601;
        color_range_ = kVpxCrStudioRange;
        subsampling_y_ = subsampling_x_ = 1;
        bit_depth_ = 8;
      }

      refresh_frame_flags_ = VpxReadLiteral(kRefFrames);
      ParseFrameResolution();
    } else {
      refresh_frame_flags_ = VpxReadLiteral(kRefFrames);
      for (int i = 0; i < kRefsPerFrame; ++i) {
        VpxReadLiteral(kRefFrames_LOG2);  // Consume ref.
        ReadBit();  // Consume ref sign bias.
      }

      bool found = false;
      for (int i = 0; i < kRefsPerFrame; ++i) {
        if (ReadBit()) {
          // Found previous reference, width and height did not change since
          // last frame.
          found = true;
          break;
        }
      }

      if (!found)
        ParseFrameResolution();
    }
  }
  return true;
}

int Vp9HeaderParser::ReadBit() {
  const size_t off = bit_offset_;
  const size_t byte_offset = off >> 3;
  const int bit_shift = 7 - static_cast<int>(off & 0x7);
  if (byte_offset < frame_size_) {
    const int bit = (frame_[byte_offset] >> bit_shift) & 1;
    bit_offset_++;
    return bit;
  } else {
    return 0;
  }
}

int Vp9HeaderParser::VpxReadLiteral(int bits) {
  int value = 0;
  for (int bit = bits - 1; bit >= 0; --bit)
    value |= ReadBit() << bit;
  return value;
}

bool Vp9HeaderParser::ValidateVp9SyncCode() {
  const int sync_code_0 = VpxReadLiteral(8);
  const int sync_code_1 = VpxReadLiteral(8);
  const int sync_code_2 = VpxReadLiteral(8);
  return (sync_code_0 == 0x49 && sync_code_1 == 0x83 && sync_code_2 == 0x42);
}

void Vp9HeaderParser::ParseColorSpace() {
  bit_depth_ = 0;
  if (profile_ >= 2)
    bit_depth_ = ReadBit() ? 12 : 10;
  else
    bit_depth_ = 8;
  color_space_ = VpxReadLiteral(3);
  if (color_space_ != kVpxCsSrgb) {
    color_range_ = ReadBit();
    if (profile_ == 1 || profile_ == 3) {
      subsampling_x_ = ReadBit();
      subsampling_y_ = ReadBit();
      ReadBit();
    } else {
      subsampling_y_ = subsampling_x_ = 1;
    }
  } else {
    color_range_ = kVpxCrFullRange;
    if (profile_ == 1 || profile_ == 3) {
      subsampling_y_ = subsampling_x_ = 0;
      ReadBit();
    }
  }
}

void Vp9HeaderParser::ParseFrameResolution() {
  width_ = VpxReadLiteral(16) + 1;
  height_ = VpxReadLiteral(16) + 1;
  if (ReadBit()) {
    display_width_ = VpxReadLiteral(16) + 1;
    display_height_ = VpxReadLiteral(16) + 1;
  } else {
    display_width_ = width_;
    display_height_ = height_;
  }
}

void Vp9HeaderParser::ParseFrameParallelMode() {
  if (!error_resilient_mode_) {
    ReadBit();  // Consume refresh frame context
    frame_parallel_mode_ = ReadBit();
  } else {
    frame_parallel_mode_ = 1;
  }
}

void Vp9HeaderParser::ParseTileInfo() {
  VpxReadLiteral(2);  // Consume frame context index

  // loopfilter
  VpxReadLiteral(6);  // Consume filter level
  VpxReadLiteral(3);  // Consume sharpness level

  const bool mode_ref_delta_enabled = ReadBit();
  if (mode_ref_delta_enabled) {
    const bool mode_ref_delta_update = ReadBit();
    if (mode_ref_delta_update) {
      const int kMaxRefLFDeltas = 4;
      for (int i = 0; i < kMaxRefLFDeltas; ++i) {
        if (ReadBit())
          VpxReadLiteral(7);  // Consume ref_deltas + sign
      }

      const int kMaxModeDeltas = 2;
      for (int i = 0; i < kMaxModeDeltas; ++i) {
        if (ReadBit())
          VpxReadLiteral(7);  // Consume mode_delta + sign
      }
    }
  }

  // quantization
  VpxReadLiteral(8);  // Consume base_q
  SkipDeltaQ();  // y dc
  SkipDeltaQ();  // uv ac
  SkipDeltaQ();  // uv dc

  // segmentation
  const bool segmentation_enabled = ReadBit();
  if (!segmentation_enabled) {
    const int aligned_width = AlignPowerOfTwo(width_, kMiSizeLog2);
    const int mi_cols = aligned_width >> kMiSizeLog2;
    const int aligned_mi_cols = AlignPowerOfTwo(mi_cols, kMiSizeLog2);
    const int sb_cols = aligned_mi_cols >> 3;  // to_sbs(mi_cols);
    int min_log2_n_tiles, max_log2_n_tiles;

    for (max_log2_n_tiles = 0;
         (sb_cols >> max_log2_n_tiles) >= kMinTileWidthB64;
         max_log2_n_tiles++) {
    }
    max_log2_n_tiles--;
    if (max_log2_n_tiles < 0)
      max_log2_n_tiles = 0;

    for (min_log2_n_tiles = 0; (kMaxTileWidthB64 << min_log2_n_tiles) < sb_cols;
         min_log2_n_tiles++) {
    }

    // columns
    const int max_log2_tile_cols = max_log2_n_tiles;
    const int min_log2_tile_cols = min_log2_n_tiles;
    int max_ones = max_log2_tile_cols - min_log2_tile_cols;
    int log2_tile_cols = min_log2_tile_cols;
    while (max_ones-- && ReadBit())
      log2_tile_cols++;

    // rows
    int log2_tile_rows = ReadBit();
    if (log2_tile_rows)
      log2_tile_rows += ReadBit();

    row_tiles_ = 1 << log2_tile_rows;
    column_tiles_ = 1 << log2_tile_cols;
  }
}

void Vp9HeaderParser::SkipDeltaQ() {
  if (ReadBit())
    VpxReadLiteral(4);
}

int Vp9HeaderParser::AlignPowerOfTwo(int value, int n) {
  return (((value) + ((1 << (n)) - 1)) & ~((1 << (n)) - 1));
}

}  // namespace vp9_parser
