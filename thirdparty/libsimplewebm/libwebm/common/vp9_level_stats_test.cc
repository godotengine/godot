// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "common/vp9_level_stats.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "common/hdr_util.h"
#include "common/vp9_header_parser.h"
#include "mkvparser/mkvparser.h"
#include "mkvparser/mkvreader.h"
#include "testing/test_util.h"

namespace {

// TODO(fgalligan): Refactor this test with other test files in this directory.
class Vp9LevelStatsTests : public ::testing::Test {
 public:
  Vp9LevelStatsTests() : is_reader_open_(false) {}

  ~Vp9LevelStatsTests() override { CloseReader(); }

  void CloseReader() {
    if (is_reader_open_) {
      reader_.Close();
    }
    is_reader_open_ = false;
  }

  void CreateAndLoadSegment(const std::string& filename,
                            int expected_doc_type_ver) {
    ASSERT_NE(0u, filename.length());
    filename_ = test::GetTestFilePath(filename);
    ASSERT_EQ(0, reader_.Open(filename_.c_str()));
    is_reader_open_ = true;
    pos_ = 0;
    mkvparser::EBMLHeader ebml_header;
    ebml_header.Parse(&reader_, pos_);
    ASSERT_EQ(1, ebml_header.m_version);
    ASSERT_EQ(1, ebml_header.m_readVersion);
    ASSERT_STREQ("webm", ebml_header.m_docType);
    ASSERT_EQ(expected_doc_type_ver, ebml_header.m_docTypeVersion);
    ASSERT_EQ(2, ebml_header.m_docTypeReadVersion);
    mkvparser::Segment* temp;
    ASSERT_EQ(0, mkvparser::Segment::CreateInstance(&reader_, pos_, temp));
    segment_.reset(temp);
    ASSERT_FALSE(HasFailure());
    ASSERT_GE(0, segment_->Load());
  }

  void CreateAndLoadSegment(const std::string& filename) {
    CreateAndLoadSegment(filename, 4);
  }

  void ProcessTheFrames() {
    std::vector<uint8_t> data;
    size_t data_len = 0;
    const mkvparser::Tracks* const parser_tracks = segment_->GetTracks();
    ASSERT_TRUE(parser_tracks != NULL);
    const mkvparser::Cluster* cluster = segment_->GetFirst();
    ASSERT_TRUE(cluster);

    while ((cluster != NULL) && !cluster->EOS()) {
      const mkvparser::BlockEntry* block_entry;
      long status = cluster->GetFirst(block_entry);  // NOLINT
      ASSERT_EQ(0, status);

      while ((block_entry != NULL) && !block_entry->EOS()) {
        const mkvparser::Block* const block = block_entry->GetBlock();
        ASSERT_TRUE(block != NULL);
        const long long trackNum = block->GetTrackNumber();  // NOLINT
        const mkvparser::Track* const parser_track =
            parser_tracks->GetTrackByNumber(
                static_cast<unsigned long>(trackNum));  // NOLINT
        ASSERT_TRUE(parser_track != NULL);
        const long long track_type = parser_track->GetType();  // NOLINT

        if (track_type == mkvparser::Track::kVideo) {
          const int frame_count = block->GetFrameCount();
          const long long time_ns = block->GetTime(cluster);  // NOLINT

          for (int i = 0; i < frame_count; ++i) {
            const mkvparser::Block::Frame& frame = block->GetFrame(i);
            if (static_cast<size_t>(frame.len) > data.size()) {
              data.resize(frame.len);
              data_len = static_cast<size_t>(frame.len);
            }
            ASSERT_FALSE(frame.Read(&reader_, &data[0]));
            parser_.ParseUncompressedHeader(&data[0], data_len);
            stats_.AddFrame(parser_, time_ns);
          }
        }

        status = cluster->GetNext(block_entry, block_entry);
        ASSERT_EQ(0, status);
      }

      cluster = segment_->GetNext(cluster);
    }
  }

 protected:
  mkvparser::MkvReader reader_;
  bool is_reader_open_;
  std::unique_ptr<mkvparser::Segment> segment_;
  std::string filename_;
  long long pos_;  // NOLINT
  vp9_parser::Vp9HeaderParser parser_;
  vp9_parser::Vp9LevelStats stats_;
};

TEST_F(Vp9LevelStatsTests, VideoOnlyFile) {
  ASSERT_NO_FATAL_FAILURE(CreateAndLoadSegment("test_stereo_left_right.webm"));
  ProcessTheFrames();
  EXPECT_EQ(256, parser_.width());
  EXPECT_EQ(144, parser_.height());
  EXPECT_EQ(1, parser_.column_tiles());
  EXPECT_EQ(0, parser_.frame_parallel_mode());

  EXPECT_EQ(11, stats_.GetLevel());
  EXPECT_EQ(479232, stats_.GetMaxLumaSampleRate());
  EXPECT_EQ(36864, stats_.GetMaxLumaPictureSize());
  EXPECT_DOUBLE_EQ(264.03233333333333, stats_.GetAverageBitRate());
  EXPECT_DOUBLE_EQ(147.136, stats_.GetMaxCpbSize());
  EXPECT_DOUBLE_EQ(19.267458404715583, stats_.GetCompressionRatio());
  EXPECT_EQ(1, stats_.GetMaxColumnTiles());
  EXPECT_EQ(11, stats_.GetMinimumAltrefDistance());
  EXPECT_EQ(3, stats_.GetMaxReferenceFrames());

  EXPECT_TRUE(stats_.estimate_last_frame_duration());
  stats_.set_estimate_last_frame_duration(false);
  EXPECT_DOUBLE_EQ(275.512, stats_.GetAverageBitRate());
}

TEST_F(Vp9LevelStatsTests, Muxed) {
  ASSERT_NO_FATAL_FAILURE(
      CreateAndLoadSegment("bbb_480p_vp9_opus_1second.webm", 4));
  ProcessTheFrames();
  EXPECT_EQ(854, parser_.width());
  EXPECT_EQ(480, parser_.height());
  EXPECT_EQ(2, parser_.column_tiles());
  EXPECT_EQ(1, parser_.frame_parallel_mode());

  EXPECT_EQ(30, stats_.GetLevel());
  EXPECT_EQ(9838080, stats_.GetMaxLumaSampleRate());
  EXPECT_EQ(409920, stats_.GetMaxLumaPictureSize());
  EXPECT_DOUBLE_EQ(447.09394572025053, stats_.GetAverageBitRate());
  EXPECT_DOUBLE_EQ(118.464, stats_.GetMaxCpbSize());
  EXPECT_DOUBLE_EQ(241.17670131398313, stats_.GetCompressionRatio());
  EXPECT_EQ(2, stats_.GetMaxColumnTiles());
  EXPECT_EQ(9, stats_.GetMinimumAltrefDistance());
  EXPECT_EQ(3, stats_.GetMaxReferenceFrames());

  stats_.set_estimate_last_frame_duration(false);
  EXPECT_DOUBLE_EQ(468.38413361169108, stats_.GetAverageBitRate());
}

TEST_F(Vp9LevelStatsTests, SetDuration) {
  ASSERT_NO_FATAL_FAILURE(CreateAndLoadSegment("test_stereo_left_right.webm"));
  ProcessTheFrames();
  const int64_t kDurationNano = 2080000000;  // 2.08 seconds
  stats_.set_duration(kDurationNano);
  EXPECT_EQ(256, parser_.width());
  EXPECT_EQ(144, parser_.height());
  EXPECT_EQ(1, parser_.column_tiles());
  EXPECT_EQ(0, parser_.frame_parallel_mode());

  EXPECT_EQ(11, stats_.GetLevel());
  EXPECT_EQ(479232, stats_.GetMaxLumaSampleRate());
  EXPECT_EQ(36864, stats_.GetMaxLumaPictureSize());
  EXPECT_DOUBLE_EQ(264.9153846153846, stats_.GetAverageBitRate());
  EXPECT_DOUBLE_EQ(147.136, stats_.GetMaxCpbSize());
  EXPECT_DOUBLE_EQ(19.267458404715583, stats_.GetCompressionRatio());
  EXPECT_EQ(1, stats_.GetMaxColumnTiles());
  EXPECT_EQ(11, stats_.GetMinimumAltrefDistance());
  EXPECT_EQ(3, stats_.GetMaxReferenceFrames());
}

}  // namespace
