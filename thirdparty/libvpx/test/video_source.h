/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_TEST_VIDEO_SOURCE_H_
#define VPX_TEST_VIDEO_SOURCE_H_

#if defined(_WIN32)
#undef NOMINMAX
#define NOMINMAX
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#if !defined(_WIN32)
#include "gtest/gtest.h"
#endif
#include "test/acm_random.h"
#include "vpx/vpx_encoder.h"

namespace libvpx_test {

// Helper macros to ensure LIBVPX_TEST_DATA_PATH is a quoted string.
// These are undefined right below GetDataPath
// NOTE: LIBVPX_TEST_DATA_PATH MUST NOT be a quoted string before
// Stringification or the GetDataPath will fail at runtime
#define TO_STRING(S) #S
#define STRINGIFY(S) TO_STRING(S)

// A simple function to encapsulate cross platform retrieval of test data path
static std::string GetDataPath() {
  const char *const data_path = getenv("LIBVPX_TEST_DATA_PATH");
  if (data_path == nullptr) {
#ifdef LIBVPX_TEST_DATA_PATH
    // In some environments, we cannot set environment variables
    // Instead, we set the data path by using a preprocessor symbol
    // which can be set from make files
    return STRINGIFY(LIBVPX_TEST_DATA_PATH);
#else
    return ".";
#endif
  }
  return data_path;
}

// Undefining stringification macros because they are not used elsewhere
#undef TO_STRING
#undef STRINGIFY

inline FILE *OpenTestDataFile(const std::string &file_name) {
  const std::string path_to_source = GetDataPath() + "/" + file_name;
  return fopen(path_to_source.c_str(), "rb");
}

static FILE *GetTempOutFile(std::string *file_name, const char *io_mode) {
  file_name->clear();
#if defined(_WIN32)
  char fname[MAX_PATH];
  char tmppath[MAX_PATH];
  if (GetTempPathA(MAX_PATH, tmppath)) {
    // Assume for now that the filename generated is unique per process
    if (GetTempFileNameA(tmppath, "lvx", 0, fname)) {
      file_name->assign(fname);
      return fopen(fname, io_mode);
    }
  }
  return nullptr;
#else
  std::string temp_dir = testing::TempDir();
  if (temp_dir.empty()) return nullptr;
  // Versions of testing::TempDir() prior to release-1.11.0-214-g5e6a5336 may
  // use the value of an environment variable without checking for a trailing
  // path delimiter.
  if (temp_dir[temp_dir.size() - 1] != '/') temp_dir += '/';
  const char name_template[] = "libvpxtest.XXXXXX";
  std::unique_ptr<char[]> temp_file_name(
      new char[temp_dir.size() + sizeof(name_template)]);
  if (temp_file_name == nullptr) return nullptr;
  memcpy(temp_file_name.get(), temp_dir.data(), temp_dir.size());
  memcpy(temp_file_name.get() + temp_dir.size(), name_template,
         sizeof(name_template));
  const int fd = mkstemp(temp_file_name.get());
  if (fd == -1) return nullptr;
  *file_name = temp_file_name.get();
  return fdopen(fd, io_mode);
#endif
}

class TempOutFile {
 public:
  TempOutFile() { file_ = GetTempOutFile(&file_name_, "wb+"); }
  TempOutFile(const char *io_mode) {
    file_ = GetTempOutFile(&file_name_, io_mode);
  }
  ~TempOutFile() {
    CloseFile();
    if (!file_name_.empty()) {
      EXPECT_EQ(0, remove(file_name_.c_str()));
    }
  }
  FILE *file() { return file_; }
  const std::string &file_name() { return file_name_; }

 protected:
  void CloseFile() {
    if (file_) {
      fclose(file_);
      file_ = nullptr;
    }
  }
  FILE *file_;
  std::string file_name_;
};

// Abstract base class for test video sources, which provide a stream of
// vpx_image_t images with associated timestamps and duration.
class VideoSource {
 public:
  virtual ~VideoSource() {}

  // Prepare the stream for reading, rewind/open as necessary.
  virtual void Begin() = 0;

  // Advance the cursor to the next frame
  virtual void Next() = 0;

  // Get the current video frame, or nullptr on End-Of-Stream.
  virtual vpx_image_t *img() const = 0;

  // Get the presentation timestamp of the current frame.
  virtual vpx_codec_pts_t pts() const = 0;

  // Get the current frame's duration
  virtual unsigned long duration() const = 0;

  // Get the timebase for the stream
  virtual vpx_rational_t timebase() const = 0;

  // Get the current frame counter, starting at 0.
  virtual unsigned int frame() const = 0;

  // Get the current file limit.
  virtual unsigned int limit() const = 0;
};

class DummyVideoSource : public VideoSource {
 public:
  DummyVideoSource()
      : img_(nullptr), limit_(100), width_(80), height_(64),
        format_(VPX_IMG_FMT_I420) {
    ReallocImage();
  }

  ~DummyVideoSource() override { vpx_img_free(img_); }

  void Begin() override {
    frame_ = 0;
    FillFrame();
  }

  void Next() override {
    ++frame_;
    FillFrame();
  }

  vpx_image_t *img() const override {
    return (frame_ < limit_) ? img_ : nullptr;
  }

  // Models a stream where Timebase = 1/FPS, so pts == frame.
  vpx_codec_pts_t pts() const override { return frame_; }

  unsigned long duration() const override { return 1; }

  vpx_rational_t timebase() const override {
    const vpx_rational_t t = { 1, 30 };
    return t;
  }

  unsigned int frame() const override { return frame_; }

  unsigned int limit() const override { return limit_; }

  void set_limit(unsigned int limit) { limit_ = limit; }

  void SetSize(unsigned int width, unsigned int height) {
    if (width != width_ || height != height_) {
      width_ = width;
      height_ = height;
      ReallocImage();
    }
  }

  void SetImageFormat(vpx_img_fmt_t format) {
    if (format_ != format) {
      format_ = format;
      ReallocImage();
    }
  }

 protected:
  virtual void FillFrame() {
    if (img_) memset(img_->img_data, 0, raw_sz_);
  }

  void ReallocImage() {
    vpx_img_free(img_);
    img_ = vpx_img_alloc(nullptr, format_, width_, height_, 32);
    ASSERT_NE(img_, nullptr);
    raw_sz_ = ((img_->w + 31) & ~31u) * img_->h * img_->bps / 8;
  }

  vpx_image_t *img_;
  size_t raw_sz_;
  unsigned int limit_;
  unsigned int frame_;
  unsigned int width_;
  unsigned int height_;
  vpx_img_fmt_t format_;
};

class RandomVideoSource : public DummyVideoSource {
 public:
  RandomVideoSource(int seed = ACMRandom::DeterministicSeed())
      : rnd_(seed), seed_(seed) {}

  // Reset the RNG to get a matching stream for the second pass
  void Begin() override {
    frame_ = 0;
    rnd_.Reset(seed_);
    FillFrame();
  }

 protected:
  // 15 frames of noise, followed by 15 static frames. Reset to 0 rather
  // than holding previous frames to encourage keyframes to be thrown.
  void FillFrame() override {
    if (img_) {
      if (frame_ % 30 < 15) {
        for (size_t i = 0; i < raw_sz_; ++i) img_->img_data[i] = rnd_.Rand8();
      } else {
        memset(img_->img_data, 0, raw_sz_);
      }
    }
  }

  ACMRandom rnd_;
  int seed_;
};

// Abstract base class for test video sources, which provide a stream of
// decompressed images to the decoder.
class CompressedVideoSource {
 public:
  virtual ~CompressedVideoSource() {}

  virtual void Init() = 0;

  // Prepare the stream for reading, rewind/open as necessary.
  virtual void Begin() = 0;

  // Advance the cursor to the next frame
  virtual void Next() = 0;

  virtual const uint8_t *cxdata() const = 0;

  virtual size_t frame_size() const = 0;

  virtual unsigned int frame_number() const = 0;
};

}  // namespace libvpx_test

#endif  // VPX_TEST_VIDEO_SOURCE_H_
