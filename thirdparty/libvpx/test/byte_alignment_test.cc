/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string>

#include "./vpx_config.h"
#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/md5_helper.h"
#include "test/util.h"
#if CONFIG_WEBM_IO
#include "test/webm_video_source.h"
#endif

namespace {

#if CONFIG_WEBM_IO

const int kLegacyByteAlignment = 0;
const int kLegacyYPlaneByteAlignment = 32;
const int kNumPlanesToCheck = 3;
const char kVP9TestFile[] = "vp90-2-02-size-lf-1920x1080.webm";
const char kVP9Md5File[] = "vp90-2-02-size-lf-1920x1080.webm.md5";

struct ByteAlignmentTestParam {
  int byte_alignment;
  vpx_codec_err_t expected_value;
  bool decode_remaining;
};

const ByteAlignmentTestParam kBaTestParams[] = {
  { kLegacyByteAlignment, VPX_CODEC_OK, true },
  { 32, VPX_CODEC_OK, true },
  { 64, VPX_CODEC_OK, true },
  { 128, VPX_CODEC_OK, true },
  { 256, VPX_CODEC_OK, true },
  { 512, VPX_CODEC_OK, true },
  { 1024, VPX_CODEC_OK, true },
  { 1, VPX_CODEC_INVALID_PARAM, false },
  { -2, VPX_CODEC_INVALID_PARAM, false },
  { 4, VPX_CODEC_INVALID_PARAM, false },
  { 16, VPX_CODEC_INVALID_PARAM, false },
  { 255, VPX_CODEC_INVALID_PARAM, false },
  { 2048, VPX_CODEC_INVALID_PARAM, false },
};

// Class for testing byte alignment of reference buffers.
class ByteAlignmentTest
    : public ::testing::TestWithParam<ByteAlignmentTestParam> {
 protected:
  ByteAlignmentTest()
      : video_(nullptr), decoder_(nullptr), md5_file_(nullptr) {}

  void SetUp() override {
    video_ = new libvpx_test::WebMVideoSource(kVP9TestFile);
    ASSERT_NE(video_, nullptr);
    video_->Init();
    video_->Begin();

    const vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
    decoder_ = new libvpx_test::VP9Decoder(cfg, 0);
    ASSERT_NE(decoder_, nullptr);

    OpenMd5File(kVP9Md5File);
  }

  void TearDown() override {
    if (md5_file_ != nullptr) fclose(md5_file_);

    delete decoder_;
    delete video_;
  }

  void SetByteAlignment(int byte_alignment, vpx_codec_err_t expected_value) {
    decoder_->Control(VP9_SET_BYTE_ALIGNMENT, byte_alignment, expected_value);
  }

  vpx_codec_err_t DecodeOneFrame(int byte_alignment_to_check) {
    const vpx_codec_err_t res =
        decoder_->DecodeFrame(video_->cxdata(), video_->frame_size());
    CheckDecodedFrames(byte_alignment_to_check);
    if (res == VPX_CODEC_OK) video_->Next();
    return res;
  }

  vpx_codec_err_t DecodeRemainingFrames(int byte_alignment_to_check) {
    for (; video_->cxdata() != nullptr; video_->Next()) {
      const vpx_codec_err_t res =
          decoder_->DecodeFrame(video_->cxdata(), video_->frame_size());
      if (res != VPX_CODEC_OK) return res;
      CheckDecodedFrames(byte_alignment_to_check);
    }
    return VPX_CODEC_OK;
  }

 private:
  // Check if |data| is aligned to |byte_alignment_to_check|.
  // |byte_alignment_to_check| must be a power of 2.
  void CheckByteAlignment(const uint8_t *data, int byte_alignment_to_check) {
    ASSERT_EQ(0u, reinterpret_cast<size_t>(data) % byte_alignment_to_check);
  }

  // Iterate through the planes of the decoded frames and check for
  // alignment based off |byte_alignment_to_check|.
  void CheckDecodedFrames(int byte_alignment_to_check) {
    libvpx_test::DxDataIterator dec_iter = decoder_->GetDxData();
    const vpx_image_t *img;

    // Get decompressed data
    while ((img = dec_iter.Next()) != nullptr) {
      if (byte_alignment_to_check == kLegacyByteAlignment) {
        CheckByteAlignment(img->planes[0], kLegacyYPlaneByteAlignment);
      } else {
        for (int i = 0; i < kNumPlanesToCheck; ++i) {
          CheckByteAlignment(img->planes[i], byte_alignment_to_check);
        }
      }
      CheckMd5(*img);
    }
  }

  // TODO(fgalligan): Move the MD5 testing code into another class.
  void OpenMd5File(const std::string &md5_file_name_) {
    md5_file_ = libvpx_test::OpenTestDataFile(md5_file_name_);
    ASSERT_NE(md5_file_, nullptr)
        << "MD5 file open failed. Filename: " << md5_file_name_;
  }

  void CheckMd5(const vpx_image_t &img) {
    ASSERT_NE(md5_file_, nullptr);
    char expected_md5[33];
    char junk[128];

    // Read correct md5 checksums.
    const int res = fscanf(md5_file_, "%s  %s", expected_md5, junk);
    ASSERT_NE(EOF, res) << "Read md5 data failed";
    expected_md5[32] = '\0';

    ::libvpx_test::MD5 md5_res;
    md5_res.Add(&img);
    const char *const actual_md5 = md5_res.Get();

    // Check md5 match.
    ASSERT_STREQ(expected_md5, actual_md5) << "MD5 checksums don't match";
  }

  libvpx_test::WebMVideoSource *video_;
  libvpx_test::VP9Decoder *decoder_;
  FILE *md5_file_;
};

TEST_F(ByteAlignmentTest, SwitchByteAlignment) {
  const int num_elements = 14;
  const int byte_alignments[] = { 0, 32,   64, 128, 256, 512, 1024,
                                  0, 1024, 32, 512, 64,  256, 128 };

  for (int i = 0; i < num_elements; ++i) {
    SetByteAlignment(byte_alignments[i], VPX_CODEC_OK);
    ASSERT_EQ(VPX_CODEC_OK, DecodeOneFrame(byte_alignments[i]));
  }
  SetByteAlignment(byte_alignments[0], VPX_CODEC_OK);
  ASSERT_EQ(VPX_CODEC_OK, DecodeRemainingFrames(byte_alignments[0]));
}

TEST_P(ByteAlignmentTest, TestAlignment) {
  const ByteAlignmentTestParam t = GetParam();
  SetByteAlignment(t.byte_alignment, t.expected_value);
  if (t.decode_remaining) {
    ASSERT_EQ(VPX_CODEC_OK, DecodeRemainingFrames(t.byte_alignment));
  }
}

INSTANTIATE_TEST_SUITE_P(Alignments, ByteAlignmentTest,
                         ::testing::ValuesIn(kBaTestParams));

#endif  // CONFIG_WEBM_IO

}  // namespace
