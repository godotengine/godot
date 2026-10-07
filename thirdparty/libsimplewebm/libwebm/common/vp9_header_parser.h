// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef LIBWEBM_COMMON_VP9_HEADER_PARSER_H_
#define LIBWEBM_COMMON_VP9_HEADER_PARSER_H_

#include <stddef.h>
#include <stdint.h>

namespace vp9_parser {

const int kVp9FrameMarker = 2;
const int kMinTileWidthB64 = 4;
const int kMaxTileWidthB64 = 64;
const int kRefFrames = 8;
const int kRefsPerFrame = 3;
const int kRefFrames_LOG2 = 3;
const int kVpxCsBt601 = 1;
const int kVpxCsSrgb = 7;
const int kVpxCrStudioRange = 0;
const int kVpxCrFullRange = 1;
const int kMiSizeLog2 = 3;

// Class to parse the header of a VP9 frame.
class Vp9HeaderParser {
 public:
  Vp9HeaderParser()
      : frame_(NULL),
        frame_size_(0),
        bit_offset_(0),
        profile_(-1),
        show_existing_frame_(0),
        key_(0),
        altref_(0),
        error_resilient_mode_(0),
        intra_only_(0),
        reset_frame_context_(0),
        bit_depth_(0),
        color_space_(0),
        color_range_(0),
        subsampling_x_(0),
        subsampling_y_(0),
        refresh_frame_flags_(0),
        width_(0),
        height_(0),
        row_tiles_(0),
        column_tiles_(0),
        frame_parallel_mode_(0) {}

  // Parse the VP9 uncompressed header. The return values of the remaining
  // functions are only valid on success.
  bool ParseUncompressedHeader(const uint8_t* frame, size_t length);

  size_t frame_size() const { return frame_size_; }
  int profile() const { return profile_; }
  int key() const { return key_; }
  int altref() const { return altref_; }
  int error_resilient_mode() const { return error_resilient_mode_; }
  int bit_depth() const { return bit_depth_; }
  int color_space() const { return color_space_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int display_width() const { return display_width_; }
  int display_height() const { return display_height_; }
  int refresh_frame_flags() const { return refresh_frame_flags_; }
  int row_tiles() const { return row_tiles_; }
  int column_tiles() const { return column_tiles_; }
  int frame_parallel_mode() const { return frame_parallel_mode_; }

 private:
  // Set the compressed VP9 frame.
  bool SetFrame(const uint8_t* frame, size_t length);

  // Returns the next bit of the frame.
  int ReadBit();

  // Returns the next |bits| of the frame.
  int VpxReadLiteral(int bits);

  // Returns true if the vp9 sync code is valid.
  bool ValidateVp9SyncCode();

  // Parses bit_depth_, color_space_, subsampling_x_, subsampling_y_, and
  // color_range_.
  void ParseColorSpace();

  // Parses width and height of the frame.
  void ParseFrameResolution();

  // Parses frame_parallel_mode_. This function skips over some features.
  void ParseFrameParallelMode();

  // Parses row and column tiles. This function skips over some features.
  void ParseTileInfo();
  void SkipDeltaQ();
  int AlignPowerOfTwo(int value, int n);

  const uint8_t* frame_;
  size_t frame_size_;
  size_t bit_offset_;
  int profile_;
  int show_existing_frame_;
  int key_;
  int altref_;
  int error_resilient_mode_;
  int intra_only_;
  int reset_frame_context_;
  int bit_depth_;
  int color_space_;
  int color_range_;
  int subsampling_x_;
  int subsampling_y_;
  int refresh_frame_flags_;
  int width_;
  int height_;
  int display_width_;
  int display_height_;
  int row_tiles_;
  int column_tiles_;
  int frame_parallel_mode_;
};

}  // namespace vp9_parser

#endif  // LIBWEBM_COMMON_VP9_HEADER_PARSER_H_
