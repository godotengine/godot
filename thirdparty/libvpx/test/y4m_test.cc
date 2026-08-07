/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "./y4menc.h"
#include "test/md5_helper.h"
#include "test/util.h"
#include "test/y4m_video_source.h"

namespace {

using std::string;

static const unsigned int kWidth = 160;
static const unsigned int kHeight = 90;
static const unsigned int kFrames = 10;

struct Y4mTestParam {
  const char *filename;
  unsigned int bit_depth;
  vpx_img_fmt format;
  const char *md5raw;
};

const Y4mTestParam kY4mTestVectors[] = {
  { "park_joy_90p_8_420.y4m", 8, VPX_IMG_FMT_I420,
    "e5406275b9fc6bb3436c31d4a05c1cab" },
  { "park_joy_90p_8_422.y4m", 8, VPX_IMG_FMT_I422,
    "284a47a47133b12884ec3a14e959a0b6" },
  { "park_joy_90p_8_444.y4m", 8, VPX_IMG_FMT_I444,
    "90517ff33843d85de712fd4fe60dbed0" },
  { "park_joy_90p_10_420_20f.y4m", 10, VPX_IMG_FMT_I42016,
    "2f56ab9809269f074df7e3daf1ce0be6" },
  { "park_joy_90p_10_422_20f.y4m", 10, VPX_IMG_FMT_I42216,
    "1b5c73d2e8e8c4e02dc4889ecac41c83" },
  { "park_joy_90p_10_444_20f.y4m", 10, VPX_IMG_FMT_I44416,
    "ec4ab5be53195c5b838d1d19e1bc2674" },
  { "park_joy_90p_12_420_20f.y4m", 12, VPX_IMG_FMT_I42016,
    "3370856c8ddebbd1f9bb2e66f97677f4" },
  { "park_joy_90p_12_422_20f.y4m", 12, VPX_IMG_FMT_I42216,
    "4eab364318dd8201acbb182e43bd4966" },
  { "park_joy_90p_12_444_20f.y4m", 12, VPX_IMG_FMT_I44416,
    "f189dfbbd92119fc8e5f211a550166be" },
};

static void write_image_file(const vpx_image_t *img, FILE *file) {
  int plane, y;
  for (plane = 0; plane < 3; ++plane) {
    const unsigned char *buf = img->planes[plane];
    const int stride = img->stride[plane];
    const int bytes_per_sample = (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;
    const int h =
        (plane ? (img->d_h + img->y_chroma_shift) >> img->y_chroma_shift
               : img->d_h);
    const int w =
        (plane ? (img->d_w + img->x_chroma_shift) >> img->x_chroma_shift
               : img->d_w);
    for (y = 0; y < h; ++y) {
      fwrite(buf, bytes_per_sample, w, file);
      buf += stride;
    }
  }
}

class Y4mVideoSourceTest : public ::testing::TestWithParam<Y4mTestParam>,
                           public ::libvpx_test::Y4mVideoSource {
 protected:
  Y4mVideoSourceTest() : Y4mVideoSource("", 0, 0) {}

  ~Y4mVideoSourceTest() override { CloseSource(); }

  virtual void Init(const std::string &file_name, int limit) {
    file_name_ = file_name;
    start_ = 0;
    limit_ = limit;
    frame_ = 0;
    Begin();
  }

  // Checks y4m header information
  void HeaderChecks(unsigned int bit_depth, vpx_img_fmt_t fmt) {
    ASSERT_NE(input_file_, nullptr);
    ASSERT_EQ(y4m_.pic_w, (int)kWidth);
    ASSERT_EQ(y4m_.pic_h, (int)kHeight);
    ASSERT_EQ(img()->d_w, kWidth);
    ASSERT_EQ(img()->d_h, kHeight);
    ASSERT_EQ(y4m_.bit_depth, bit_depth);
    ASSERT_EQ(y4m_.vpx_fmt, fmt);
    if (fmt == VPX_IMG_FMT_I420 || fmt == VPX_IMG_FMT_I42016) {
      ASSERT_EQ(y4m_.bps, (int)y4m_.bit_depth * 3 / 2);
      ASSERT_EQ(img()->x_chroma_shift, 1U);
      ASSERT_EQ(img()->y_chroma_shift, 1U);
    }
    if (fmt == VPX_IMG_FMT_I422 || fmt == VPX_IMG_FMT_I42216) {
      ASSERT_EQ(y4m_.bps, (int)y4m_.bit_depth * 2);
      ASSERT_EQ(img()->x_chroma_shift, 1U);
      ASSERT_EQ(img()->y_chroma_shift, 0U);
    }
    if (fmt == VPX_IMG_FMT_I444 || fmt == VPX_IMG_FMT_I44416) {
      ASSERT_EQ(y4m_.bps, (int)y4m_.bit_depth * 3);
      ASSERT_EQ(img()->x_chroma_shift, 0U);
      ASSERT_EQ(img()->y_chroma_shift, 0U);
    }
  }

  // Checks MD5 of the raw frame data
  void Md5Check(const string &expected_md5) {
    ASSERT_NE(input_file_, nullptr);
    libvpx_test::MD5 md5;
    for (unsigned int i = start_; i < limit_; i++) {
      md5.Add(img());
      Next();
    }
    ASSERT_EQ(string(md5.Get()), expected_md5);
  }
};

TEST_P(Y4mVideoSourceTest, SourceTest) {
  const Y4mTestParam t = GetParam();
  Init(t.filename, kFrames);
  HeaderChecks(t.bit_depth, t.format);
  Md5Check(t.md5raw);
}

INSTANTIATE_TEST_SUITE_P(C, Y4mVideoSourceTest,
                         ::testing::ValuesIn(kY4mTestVectors));

class Y4mVideoWriteTest : public Y4mVideoSourceTest {
 protected:
  Y4mVideoWriteTest() : tmpfile_(nullptr) {}

  ~Y4mVideoWriteTest() override {
    delete tmpfile_;
    input_file_ = nullptr;
  }

  void ReplaceInputFile(FILE *input_file) {
    CloseSource();
    frame_ = 0;
    input_file_ = input_file;
    rewind(input_file_);
    ReadSourceToStart();
  }

  // Writes out a y4m file and then reads it back
  void WriteY4mAndReadBack() {
    ASSERT_NE(input_file_, nullptr);
    char buf[Y4M_BUFFER_SIZE] = { 0 };
    const struct VpxRational framerate = { y4m_.fps_n, y4m_.fps_d };
    tmpfile_ = new libvpx_test::TempOutFile;
    ASSERT_NE(tmpfile_->file(), nullptr);
    y4m_write_file_header(buf, sizeof(buf), kWidth, kHeight, &framerate,
                          y4m_.vpx_fmt, y4m_.bit_depth);
    fputs(buf, tmpfile_->file());
    for (unsigned int i = start_; i < limit_; i++) {
      y4m_write_frame_header(buf, sizeof(buf));
      fputs(buf, tmpfile_->file());
      write_image_file(img(), tmpfile_->file());
      Next();
    }
    ReplaceInputFile(tmpfile_->file());
  }

  void Init(const std::string &file_name, int limit) override {
    Y4mVideoSourceTest::Init(file_name, limit);
    WriteY4mAndReadBack();
  }
  libvpx_test::TempOutFile *tmpfile_;
};

TEST_P(Y4mVideoWriteTest, WriteTest) {
  const Y4mTestParam t = GetParam();
  Init(t.filename, kFrames);
  HeaderChecks(t.bit_depth, t.format);
  Md5Check(t.md5raw);
}

INSTANTIATE_TEST_SUITE_P(C, Y4mVideoWriteTest,
                         ::testing::ValuesIn(kY4mTestVectors));

static const char kY4MRegularHeader[] =
    "YUV4MPEG2 W4 H4 F30:1 Ip A0:0 C420jpeg XYSCSS=420JPEG\n"
    "FRAME\n"
    "012345678912345601230123";

TEST(Y4MHeaderTest, RegularHeader) {
  libvpx_test::TempOutFile f;
  ASSERT_NE(f.file(), nullptr);
  fwrite(kY4MRegularHeader, 1, sizeof(kY4MRegularHeader), f.file());
  fflush(f.file());
  EXPECT_EQ(0, fseek(f.file(), 0, 0));

  y4m_input y4m;
  EXPECT_EQ(y4m_input_open(&y4m, f.file(), /*skip_buffer=*/nullptr,
                           /*num_skip=*/0, /*only_420=*/0),
            0);
  EXPECT_EQ(y4m.pic_w, 4);
  EXPECT_EQ(y4m.pic_h, 4);
  EXPECT_EQ(y4m.fps_n, 30);
  EXPECT_EQ(y4m.fps_d, 1);
  EXPECT_EQ(y4m.interlace, 'p');
  EXPECT_EQ(strcmp("420jpeg", y4m.chroma_type), 0);
  y4m_input_close(&y4m);
}

// Testing that headers over 100 characters can be parsed.
static const char kY4MLongHeader[] =
    "YUV4MPEG2 W4 H4 F30:1 Ip A0:0 C420jpeg XYSCSS=420JPEG "
    "XCOLORRANGE=LIMITED XSOME_UNKNOWN_METADATA XOTHER_UNKNOWN_METADATA\n"
    "FRAME\n"
    "012345678912345601230123";

TEST(Y4MHeaderTest, LongHeader) {
  libvpx_test::TempOutFile f;
  ASSERT_NE(f.file(), nullptr);
  fwrite(kY4MLongHeader, 1, sizeof(kY4MLongHeader), f.file());
  fflush(f.file());
  EXPECT_EQ(fseek(f.file(), 0, 0), 0);

  y4m_input y4m;
  EXPECT_EQ(y4m_input_open(&y4m, f.file(), /*skip_buffer=*/nullptr,
                           /*num_skip=*/0, /*only_420=*/0),
            0);
  EXPECT_EQ(y4m.pic_w, 4);
  EXPECT_EQ(y4m.pic_h, 4);
  EXPECT_EQ(y4m.fps_n, 30);
  EXPECT_EQ(y4m.fps_d, 1);
  EXPECT_EQ(y4m.interlace, 'p');
  EXPECT_EQ(strcmp("420jpeg", y4m.chroma_type), 0);
  y4m_input_close(&y4m);
}

}  // namespace
