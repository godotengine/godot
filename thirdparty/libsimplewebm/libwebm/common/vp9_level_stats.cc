// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "common/vp9_level_stats.h"

#include <inttypes.h>

#include <limits>
#include <utility>

#include "common/webm_constants.h"

namespace vp9_parser {

const Vp9LevelRow Vp9LevelStats::Vp9LevelTable[kNumVp9Levels] = {
    {LEVEL_1, 829440, 36864, 512, 200, 400, 2, 1, 4, 8},
    {LEVEL_1_1, 2764800, 73728, 768, 800, 1000, 2, 1, 4, 8},
    {LEVEL_2, 4608000, 122880, 960, 1800, 1500, 2, 1, 4, 8},
    {LEVEL_2_1, 9216000, 245760, 1344, 3600, 2800, 2, 2, 4, 8},
    {LEVEL_3, 20736000, 552960, 2048, 7200, 6000, 2, 4, 4, 8},
    {LEVEL_3_1, 36864000, 983040, 2752, 12000, 10000, 2, 4, 4, 8},
    {LEVEL_4, 83558400, 2228224, 4160, 18000, 16000, 4, 4, 4, 8},
    {LEVEL_4_1, 160432128, 2228224, 4160, 30000, 18000, 4, 4, 5, 6},
    {LEVEL_5, 311951360, 8912896, 8384, 60000, 36000, 6, 8, 6, 4},
    {LEVEL_5_1, 588251136, 8912896, 8384, 120000, 46000, 8, 8, 10, 4},
    // CPB Size = 0 for levels 5_2 to 6_2
    {LEVEL_5_2, 1176502272, 8912896, 8384, 180000, 0, 8, 8, 10, 4},
    {LEVEL_6, 1176502272, 35651584, 16832, 180000, 0, 8, 16, 10, 4},
    {LEVEL_6_1, 2353004544, 35651584, 16832, 240000, 0, 8, 16, 10, 4},
    {LEVEL_6_2, 4706009088, 35651584, 16832, 480000, 0, 8, 16, 10, 4}};

void Vp9LevelStats::AddFrame(const Vp9HeaderParser& parser, int64_t time_ns) {
  ++frames;
  if (start_ns_ == -1)
    start_ns_ = time_ns;
  end_ns_ = time_ns;

  const int width = parser.width();
  const int height = parser.height();
  const int64_t luma_picture_size = width * height;
  const int64_t luma_picture_breadth = (width > height) ? width : height;
  if (luma_picture_size > max_luma_picture_size_)
    max_luma_picture_size_ = luma_picture_size;
  if (luma_picture_breadth > max_luma_picture_breadth_)
    max_luma_picture_breadth_ = luma_picture_breadth;

  total_compressed_size_ += parser.frame_size();

  while (!luma_window_.empty() &&
         luma_window_.front().first <
             (time_ns - (libwebm::kNanosecondsPerSecondi - 1))) {
    current_luma_size_ -= luma_window_.front().second;
    luma_window_.pop();
  }
  current_luma_size_ += luma_picture_size;
  luma_window_.push(std::make_pair(time_ns, luma_picture_size));
  if (current_luma_size_ > max_luma_size_) {
    max_luma_size_ = current_luma_size_;
    max_luma_end_ns_ = luma_window_.back().first;
  }

  // Record CPB stats.
  // Remove all frames that are less than window size.
  while (cpb_window_.size() > 3) {
    current_cpb_size_ -= cpb_window_.front().second;
    cpb_window_.pop();
  }
  cpb_window_.push(std::make_pair(time_ns, parser.frame_size()));

  current_cpb_size_ += parser.frame_size();
  if (current_cpb_size_ > max_cpb_size_) {
    max_cpb_size_ = current_cpb_size_;
    max_cpb_start_ns_ = cpb_window_.front().first;
    max_cpb_end_ns_ = cpb_window_.back().first;
  }

  if (max_cpb_window_size_ < static_cast<int64_t>(cpb_window_.size())) {
    max_cpb_window_size_ = cpb_window_.size();
    max_cpb_window_end_ns_ = time_ns;
  }

  // Record altref stats.
  if (parser.altref()) {
    const int delta_altref = frames_since_last_altref;
    if (first_altref) {
      first_altref = false;
    } else if (delta_altref < minimum_altref_distance) {
      minimum_altref_distance = delta_altref;
      min_altref_end_ns = time_ns;
    }
    frames_since_last_altref = 0;
  } else {
    ++frames_since_last_altref;
    ++displayed_frames;
    // TODO(fgalligan): Add support for other color formats. Currently assuming
    // 420.
    total_uncompressed_bits_ +=
        (luma_picture_size * parser.bit_depth() * 3) / 2;
  }

  // Count max reference frames.
  if (parser.key() == 1) {
    frames_refreshed_ = 0;
  } else {
    frames_refreshed_ |= parser.refresh_frame_flags();

    int ref_frame_count = frames_refreshed_ & 1;
    for (int i = 1; i < kMaxVp9RefFrames; ++i) {
      ref_frame_count += (frames_refreshed_ >> i) & 1;
    }

    if (ref_frame_count > max_frames_refreshed_)
      max_frames_refreshed_ = ref_frame_count;
  }

  // Count max tiles.
  const int tiles = parser.column_tiles();
  if (tiles > max_column_tiles_)
    max_column_tiles_ = tiles;
}

Vp9Level Vp9LevelStats::GetLevel() const {
  const int64_t max_luma_sample_rate = GetMaxLumaSampleRate();
  const int64_t max_luma_picture_size = GetMaxLumaPictureSize();
  const int64_t max_luma_picture_breadth = GetMaxLumaPictureBreadth();
  const double average_bitrate = GetAverageBitRate();
  const double max_cpb_size = GetMaxCpbSize();
  const double compression_ratio = GetCompressionRatio();
  const int max_column_tiles = GetMaxColumnTiles();
  const int min_altref_distance = GetMinimumAltrefDistance();
  const int max_ref_frames = GetMaxReferenceFrames();

  int level_index = 0;
  Vp9Level max_level = LEVEL_UNKNOWN;
  const double grace_multiplier =
      max_luma_sample_rate_grace_percent_ / 100.0 + 1.0;
  for (int i = 0; i < kNumVp9Levels; ++i) {
    if (max_luma_sample_rate <=
        Vp9LevelTable[i].max_luma_sample_rate * grace_multiplier) {
      if (max_level < Vp9LevelTable[i].level) {
        max_level = Vp9LevelTable[i].level;
        level_index = i;
      }
      break;
    }
  }
  for (int i = 0; i < kNumVp9Levels; ++i) {
    if (max_luma_picture_size <= Vp9LevelTable[i].max_luma_picture_size) {
      if (max_level < Vp9LevelTable[i].level) {
        max_level = Vp9LevelTable[i].level;
        level_index = i;
      }
      break;
    }
  }
  for (int i = 0; i < kNumVp9Levels; ++i) {
    if (max_luma_picture_breadth <= Vp9LevelTable[i].max_luma_picture_breadth) {
      if (max_level < Vp9LevelTable[i].level) {
        max_level = Vp9LevelTable[i].level;
        level_index = i;
      }
      break;
    }
  }
  for (int i = 0; i < kNumVp9Levels; ++i) {
    if (average_bitrate <= Vp9LevelTable[i].average_bitrate) {
      if (max_level < Vp9LevelTable[i].level) {
        max_level = Vp9LevelTable[i].level;
        level_index = i;
      }
      break;
    }
  }
  for (int i = 0; i < kNumVp9Levels; ++i) {
    // Only check CPB size for levels that are defined.
    if (Vp9LevelTable[i].max_cpb_size > 0 &&
        max_cpb_size <= Vp9LevelTable[i].max_cpb_size) {
      if (max_level < Vp9LevelTable[i].level) {
        max_level = Vp9LevelTable[i].level;
        level_index = i;
      }
      break;
    }
  }
  for (int i = 0; i < kNumVp9Levels; ++i) {
    if (max_column_tiles <= Vp9LevelTable[i].max_tiles) {
      if (max_level < Vp9LevelTable[i].level) {
        max_level = Vp9LevelTable[i].level;
        level_index = i;
      }
      break;
    }
  }

  for (int i = 0; i < kNumVp9Levels; ++i) {
    if (max_ref_frames <= Vp9LevelTable[i].max_ref_frames) {
      if (max_level < Vp9LevelTable[i].level) {
        max_level = Vp9LevelTable[i].level;
        level_index = i;
      }
      break;
    }
  }

  // Check if the current level meets the minimum altref distance requirement.
  // If not, set to unknown level as we can't move up a level as the minimum
  // altref distance get farther apart and we can't move down a level as we are
  // already at the minimum level for all the other requirements.
  if (min_altref_distance < Vp9LevelTable[level_index].min_altref_distance)
    max_level = LEVEL_UNKNOWN;

  // The minimum compression ratio has the same behavior as minimum altref
  // distance.
  if (compression_ratio < Vp9LevelTable[level_index].compression_ratio)
    max_level = LEVEL_UNKNOWN;
  return max_level;
}

int64_t Vp9LevelStats::GetMaxLumaSampleRate() const { return max_luma_size_; }

int64_t Vp9LevelStats::GetMaxLumaPictureSize() const {
  return max_luma_picture_size_;
}

int64_t Vp9LevelStats::GetMaxLumaPictureBreadth() const {
  return max_luma_picture_breadth_;
}

double Vp9LevelStats::GetAverageBitRate() const {
  const int64_t frame_duration_ns = end_ns_ - start_ns_;
  double duration_seconds =
      ((duration_ns_ == -1) ? frame_duration_ns : duration_ns_) /
      libwebm::kNanosecondsPerSecond;
  if (estimate_last_frame_duration_ &&
      (duration_ns_ == -1 || duration_ns_ <= frame_duration_ns)) {
    const double sec_per_frame = frame_duration_ns /
                                 libwebm::kNanosecondsPerSecond /
                                 (displayed_frames - 1);
    duration_seconds += sec_per_frame;
  }

  return total_compressed_size_ / duration_seconds / 125.0;
}

double Vp9LevelStats::GetMaxCpbSize() const { return max_cpb_size_ / 125.0; }

double Vp9LevelStats::GetCompressionRatio() const {
  return total_uncompressed_bits_ /
         static_cast<double>(total_compressed_size_ * 8);
}

int Vp9LevelStats::GetMaxColumnTiles() const { return max_column_tiles_; }

int Vp9LevelStats::GetMinimumAltrefDistance() const {
  if (minimum_altref_distance != std::numeric_limits<int>::max())
    return minimum_altref_distance;
  else
    return -1;
}

int Vp9LevelStats::GetMaxReferenceFrames() const {
  return max_frames_refreshed_;
}

}  // namespace vp9_parser
