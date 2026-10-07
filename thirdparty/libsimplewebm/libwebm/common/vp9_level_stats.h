// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef LIBWEBM_COMMON_VP9_LEVEL_STATS_H_
#define LIBWEBM_COMMON_VP9_LEVEL_STATS_H_

#include <limits>
#include <queue>
#include <utility>

#include "common/vp9_header_parser.h"

namespace vp9_parser {

const int kMaxVp9RefFrames = 8;

// Defined VP9 levels. See http://www.webmproject.org/vp9/profiles/ for
// detailed information on VP9 levels.
const int kNumVp9Levels = 14;
enum Vp9Level {
  LEVEL_UNKNOWN = 0,
  LEVEL_1 = 10,
  LEVEL_1_1 = 11,
  LEVEL_2 = 20,
  LEVEL_2_1 = 21,
  LEVEL_3 = 30,
  LEVEL_3_1 = 31,
  LEVEL_4 = 40,
  LEVEL_4_1 = 41,
  LEVEL_5 = 50,
  LEVEL_5_1 = 51,
  LEVEL_5_2 = 52,
  LEVEL_6 = 60,
  LEVEL_6_1 = 61,
  LEVEL_6_2 = 62
};

struct Vp9LevelRow {
  Vp9Level level;
  int64_t max_luma_sample_rate;
  int64_t max_luma_picture_size;
  int64_t max_luma_picture_breadth;
  double average_bitrate;
  double max_cpb_size;
  double compression_ratio;
  int max_tiles;
  int min_altref_distance;
  int max_ref_frames;
};

// Class to determine the VP9 level of a VP9 bitstream.
class Vp9LevelStats {
 public:
  static const Vp9LevelRow Vp9LevelTable[kNumVp9Levels];

  Vp9LevelStats()
      : frames(0),
        displayed_frames(0),
        start_ns_(-1),
        end_ns_(-1),
        duration_ns_(-1),
        max_luma_picture_size_(0),
        max_luma_picture_breadth_(0),
        current_luma_size_(0),
        max_luma_size_(0),
        max_luma_end_ns_(0),
        max_luma_sample_rate_grace_percent_(1.5),
        first_altref(true),
        frames_since_last_altref(0),
        minimum_altref_distance(std::numeric_limits<int>::max()),
        min_altref_end_ns(0),
        max_cpb_window_size_(0),
        max_cpb_window_end_ns_(0),
        current_cpb_size_(0),
        max_cpb_size_(0),
        max_cpb_start_ns_(0),
        max_cpb_end_ns_(0),
        total_compressed_size_(0),
        total_uncompressed_bits_(0),
        frames_refreshed_(0),
        max_frames_refreshed_(0),
        max_column_tiles_(0),
        estimate_last_frame_duration_(true) {}

  ~Vp9LevelStats() = default;
  Vp9LevelStats(Vp9LevelStats&& other) = delete;
  Vp9LevelStats(const Vp9LevelStats& other) = delete;
  Vp9LevelStats& operator=(Vp9LevelStats&& other) = delete;
  Vp9LevelStats& operator=(const Vp9LevelStats& other) = delete;

  // Collects stats on a VP9 frame. The frame must already be parsed by
  // |parser|. |time_ns| is the start time of the frame in nanoseconds.
  void AddFrame(const Vp9HeaderParser& parser, int64_t time_ns);

  // Returns the current VP9 level. All of the video frames should have been
  // processed with AddFrame before calling this function.
  Vp9Level GetLevel() const;

  // Returns the maximum luma samples (pixels) per second. The Alt-Ref frames
  // are taken into account, therefore this number may be larger than the
  // display luma samples per second
  int64_t GetMaxLumaSampleRate() const;

  // The maximum frame size (width * height) in samples.
  int64_t GetMaxLumaPictureSize() const;

  // The maximum frame breadth (max of width and height) in samples.
  int64_t GetMaxLumaPictureBreadth() const;

  // The average bitrate of the video in kbps.
  double GetAverageBitRate() const;

  // The largest data size for any 4 consecutive frames in kilobits.
  double GetMaxCpbSize() const;

  // The ratio of total bytes decompressed over total bytes compressed.
  double GetCompressionRatio() const;

  // The maximum number of VP9 column tiles.
  int GetMaxColumnTiles() const;

  // The minimum distance in frames between two consecutive alternate reference
  // frames.
  int GetMinimumAltrefDistance() const;

  // The maximum number of reference frames that had to be stored.
  int GetMaxReferenceFrames() const;

  // Sets the duration of the video stream in nanoseconds. If the duration is
  // not explictly set by this function then this class will use end - start
  // as the duration.
  void set_duration(int64_t time_ns) { duration_ns_ = time_ns; }
  double max_luma_sample_rate_grace_percent() const {
    return max_luma_sample_rate_grace_percent_;
  }
  void set_max_luma_sample_rate_grace_percent(double percent) {
    max_luma_sample_rate_grace_percent_ = percent;
  }
  bool estimate_last_frame_duration() const {
    return estimate_last_frame_duration_;
  }

  // If true try to estimate the last frame's duration if the stream's duration
  // is not set or the stream's duration equals the last frame's timestamp.
  void set_estimate_last_frame_duration(bool flag) {
    estimate_last_frame_duration_ = flag;
  }

 private:
  int frames;
  int displayed_frames;

  int64_t start_ns_;
  int64_t end_ns_;
  int64_t duration_ns_;

  int64_t max_luma_picture_size_;
  int64_t max_luma_picture_breadth_;

  // This is used to calculate the maximum number of luma samples per second.
  // The first value is the luma picture size and the second value is the time
  // in nanoseconds of one frame.
  std::queue<std::pair<int64_t, int64_t>> luma_window_;
  int64_t current_luma_size_;
  int64_t max_luma_size_;
  int64_t max_luma_end_ns_;

  // MaxLumaSampleRate = (ExampleFrameRate + ExampleFrameRate /
  // MinimumAltrefDistance) * MaxLumaPictureSize. For levels 1-4
  // ExampleFrameRate / MinimumAltrefDistance is non-integer, so using a sliding
  // window of one frame to calculate MaxLumaSampleRate may have frames >
  // (ExampleFrameRate + ExampleFrameRate / MinimumAltrefDistance) in the
  // window. In order to address this issue, a grace percent of 1.5 was added.
  double max_luma_sample_rate_grace_percent_;

  bool first_altref;
  int frames_since_last_altref;
  int minimum_altref_distance;
  int64_t min_altref_end_ns;

  // This is used to calculate the maximum number of compressed bytes for four
  // consecutive frames. The first value is the compressed frame size and the
  // second value is the time in nanoseconds of one frame.
  std::queue<std::pair<int64_t, int64_t>> cpb_window_;
  int64_t max_cpb_window_size_;
  int64_t max_cpb_window_end_ns_;
  int64_t current_cpb_size_;
  int64_t max_cpb_size_;
  int64_t max_cpb_start_ns_;
  int64_t max_cpb_end_ns_;

  int64_t total_compressed_size_;
  int64_t total_uncompressed_bits_;
  int frames_refreshed_;
  int max_frames_refreshed_;

  int max_column_tiles_;

  bool estimate_last_frame_duration_;
};

}  // namespace vp9_parser

#endif  // LIBWEBM_COMMON_VP9_LEVEL_STATS_H_
