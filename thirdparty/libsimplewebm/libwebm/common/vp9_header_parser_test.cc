// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "common/vp9_header_parser.h"

#include <new>
#include <string>

#include "gtest/gtest.h"

#include "common/hdr_util.h"
#include "mkvparser/mkvparser.h"
#include "mkvparser/mkvreader.h"
#include "testing/test_util.h"

namespace {

class Vp9HeaderParserTests : public ::testing::Test {
 public:
  Vp9HeaderParserTests() : is_reader_open_(false), segment_(NULL) {}

  ~Vp9HeaderParserTests() override {
    CloseReader();
    if (segment_ != NULL) {
      delete segment_;
      segment_ = NULL;
    }
  }

  void CloseReader() {
    if (is_reader_open_) {
      reader_.Close();
    }
    is_reader_open_ = false;
  }

  void CreateAndLoadSegment(const std::string& filename,
                            int expected_doc_type_ver) {
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
    ASSERT_EQ(0, mkvparser::Segment::CreateInstance(&reader_, pos_, segment_));
    ASSERT_FALSE(HasFailure());
    ASSERT_GE(0, segment_->Load());
  }

  void CreateAndLoadSegment(const std::string& filename) {
    CreateAndLoadSegment(filename, 4);
  }

  // Load a corrupted segment with no expectation of correctness.
  void CreateAndLoadInvalidSegment(const std::string& filename) {
    filename_ = test::GetTestFilePath(filename);
    ASSERT_EQ(0, reader_.Open(filename_.c_str()));
    is_reader_open_ = true;
    pos_ = 0;
    mkvparser::EBMLHeader ebml_header;
    ebml_header.Parse(&reader_, pos_);
    ASSERT_EQ(0, mkvparser::Segment::CreateInstance(&reader_, pos_, segment_));
    ASSERT_GE(0, segment_->Load());
  }

  void ProcessTheFrames(bool invalid_bitstream) {
    unsigned char* data = NULL;
    size_t data_len = 0;
    const mkvparser::Tracks* const parser_tracks = segment_->GetTracks();
    ASSERT_TRUE(parser_tracks != NULL);
    const mkvparser::Cluster* cluster = segment_->GetFirst();
    ASSERT_TRUE(cluster != NULL);

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

          for (int i = 0; i < frame_count; ++i) {
            const mkvparser::Block::Frame& frame = block->GetFrame(i);

            if (static_cast<size_t>(frame.len) > data_len) {
              delete[] data;
              data = new (std::nothrow) unsigned char[frame.len];
              ASSERT_TRUE(data != NULL);
              data_len = static_cast<size_t>(frame.len);
            }
            ASSERT_FALSE(frame.Read(&reader_, data));
            ASSERT_EQ(parser_.ParseUncompressedHeader(data, data_len),
                      !invalid_bitstream);
          }
        }

        status = cluster->GetNext(block_entry, block_entry);
        ASSERT_EQ(0, status);
      }

      cluster = segment_->GetNext(cluster);
    }
    delete[] data;
  }

 protected:
  mkvparser::MkvReader reader_;
  bool is_reader_open_;
  mkvparser::Segment* segment_;
  std::string filename_;
  long long pos_;  // NOLINT
  vp9_parser::Vp9HeaderParser parser_;
};

TEST_F(Vp9HeaderParserTests, VideoOnlyFile) {
  ASSERT_NO_FATAL_FAILURE(CreateAndLoadSegment("test_stereo_left_right.webm"));
  ProcessTheFrames(false);
  EXPECT_EQ(256, parser_.width());
  EXPECT_EQ(144, parser_.height());
  EXPECT_EQ(1, parser_.column_tiles());
  EXPECT_EQ(0, parser_.frame_parallel_mode());
}

TEST_F(Vp9HeaderParserTests, Muxed) {
  ASSERT_NO_FATAL_FAILURE(
      CreateAndLoadSegment("bbb_480p_vp9_opus_1second.webm", 4));
  ProcessTheFrames(false);
  EXPECT_EQ(854, parser_.width());
  EXPECT_EQ(480, parser_.height());
  EXPECT_EQ(2, parser_.column_tiles());
  EXPECT_EQ(1, parser_.frame_parallel_mode());
}

TEST_F(Vp9HeaderParserTests, Invalid) {
  const char* files[] = {
      "invalid/invalid_vp9_bitstream-bug_1416.webm",
      "invalid/invalid_vp9_bitstream-bug_1417.webm",
  };

  for (int i = 0; i < static_cast<int>(sizeof(files) / sizeof(files[0])); ++i) {
    SCOPED_TRACE(files[i]);
    ASSERT_NO_FATAL_FAILURE(CreateAndLoadInvalidSegment(files[i]));
    ProcessTheFrames(true);
    CloseReader();
    delete segment_;
    segment_ = NULL;
  }
}

TEST_F(Vp9HeaderParserTests, API) {
  vp9_parser::Vp9HeaderParser parser;
  uint8_t data;
  EXPECT_FALSE(parser.ParseUncompressedHeader(NULL, 0));
  EXPECT_FALSE(parser.ParseUncompressedHeader(NULL, 10));
  EXPECT_FALSE(parser.ParseUncompressedHeader(&data, 0));
}

}  // namespace
