/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <memory>
#include <string>

#include "./vpx_config.h"
#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/ivf_video_source.h"
#include "test/md5_helper.h"
#include "test/test_vectors.h"
#include "test/util.h"
#if CONFIG_WEBM_IO
#include "test/webm_video_source.h"
#endif

namespace {

const int kVideoNameParam = 1;

struct ExternalFrameBuffer {
  uint8_t *data;
  size_t size;
  int in_use;
};

// Class to manipulate a list of external frame buffers.
class ExternalFrameBufferList {
 public:
  ExternalFrameBufferList()
      : num_buffers_(0), num_used_buffers_(0), ext_fb_list_(nullptr) {}

  virtual ~ExternalFrameBufferList() {
    for (int i = 0; i < num_buffers_; ++i) {
      delete[] ext_fb_list_[i].data;
    }
    delete[] ext_fb_list_;
  }

  // Creates the list to hold the external buffers. Returns true on success.
  bool CreateBufferList(int num_buffers) {
    if (num_buffers < 0) return false;

    num_buffers_ = num_buffers;
    ext_fb_list_ = new ExternalFrameBuffer[num_buffers_];
    EXPECT_NE(ext_fb_list_, nullptr);
    memset(ext_fb_list_, 0, sizeof(ext_fb_list_[0]) * num_buffers_);
    return true;
  }

  // Searches the frame buffer list for a free frame buffer. Makes sure
  // that the frame buffer is at least |min_size| in bytes. Marks that the
  // frame buffer is in use by libvpx. Finally sets |fb| to point to the
  // external frame buffer. Returns < 0 on an error.
  int GetFreeFrameBuffer(size_t min_size, vpx_codec_frame_buffer_t *fb) {
    EXPECT_NE(fb, nullptr);
    const int idx = FindFreeBufferIndex();
    if (idx == num_buffers_) return -1;

    if (ext_fb_list_[idx].size < min_size) {
      delete[] ext_fb_list_[idx].data;
      ext_fb_list_[idx].data = new uint8_t[min_size];
      memset(ext_fb_list_[idx].data, 0, min_size);
      ext_fb_list_[idx].size = min_size;
    }

    SetFrameBuffer(idx, fb);

    num_used_buffers_++;
    return 0;
  }

  // Test function that will not allocate any data for the frame buffer.
  // Returns < 0 on an error.
  int GetZeroFrameBuffer(size_t min_size, vpx_codec_frame_buffer_t *fb) {
    EXPECT_NE(fb, nullptr);
    const int idx = FindFreeBufferIndex();
    if (idx == num_buffers_) return -1;

    if (ext_fb_list_[idx].size < min_size) {
      delete[] ext_fb_list_[idx].data;
      ext_fb_list_[idx].data = nullptr;
      ext_fb_list_[idx].size = min_size;
    }

    SetFrameBuffer(idx, fb);
    return 0;
  }

  // Marks the external frame buffer that |fb| is pointing to as free.
  // Returns < 0 on an error.
  int ReturnFrameBuffer(vpx_codec_frame_buffer_t *fb) {
    if (fb == nullptr) {
      EXPECT_NE(fb, nullptr);
      return -1;
    }
    ExternalFrameBuffer *const ext_fb =
        reinterpret_cast<ExternalFrameBuffer *>(fb->priv);
    if (ext_fb == nullptr) {
      EXPECT_NE(ext_fb, nullptr);
      return -1;
    }
    EXPECT_EQ(1, ext_fb->in_use);
    ext_fb->in_use = 0;
    num_used_buffers_--;
    return 0;
  }

  // Checks that the vpx_image_t data is contained within the external frame
  // buffer private data passed back in the vpx_image_t.
  void CheckImageFrameBuffer(const vpx_image_t *img) {
    if (img->fb_priv != nullptr) {
      const struct ExternalFrameBuffer *const ext_fb =
          reinterpret_cast<ExternalFrameBuffer *>(img->fb_priv);

      ASSERT_TRUE(img->planes[0] >= ext_fb->data &&
                  img->planes[0] < (ext_fb->data + ext_fb->size));
    }
  }

  int num_used_buffers() const { return num_used_buffers_; }

 private:
  // Returns the index of the first free frame buffer. Returns |num_buffers_|
  // if there are no free frame buffers.
  int FindFreeBufferIndex() {
    int i;
    // Find a free frame buffer.
    for (i = 0; i < num_buffers_; ++i) {
      if (!ext_fb_list_[i].in_use) break;
    }
    return i;
  }

  // Sets |fb| to an external frame buffer. idx is the index into the frame
  // buffer list.
  void SetFrameBuffer(int idx, vpx_codec_frame_buffer_t *fb) {
    ASSERT_NE(fb, nullptr);
    fb->data = ext_fb_list_[idx].data;
    fb->size = ext_fb_list_[idx].size;
    ASSERT_EQ(0, ext_fb_list_[idx].in_use);
    ext_fb_list_[idx].in_use = 1;
    fb->priv = &ext_fb_list_[idx];
  }

  int num_buffers_;
  int num_used_buffers_;
  ExternalFrameBuffer *ext_fb_list_;
};

#if CONFIG_WEBM_IO

// Callback used by libvpx to request the application to return a frame
// buffer of at least |min_size| in bytes.
int get_vp9_frame_buffer(void *user_priv, size_t min_size,
                         vpx_codec_frame_buffer_t *fb) {
  ExternalFrameBufferList *const fb_list =
      reinterpret_cast<ExternalFrameBufferList *>(user_priv);
  return fb_list->GetFreeFrameBuffer(min_size, fb);
}

// Callback used by libvpx to tell the application that |fb| is not needed
// anymore.
int release_vp9_frame_buffer(void *user_priv, vpx_codec_frame_buffer_t *fb) {
  ExternalFrameBufferList *const fb_list =
      reinterpret_cast<ExternalFrameBufferList *>(user_priv);
  return fb_list->ReturnFrameBuffer(fb);
}

// Callback will not allocate data for frame buffer.
int get_vp9_zero_frame_buffer(void *user_priv, size_t min_size,
                              vpx_codec_frame_buffer_t *fb) {
  ExternalFrameBufferList *const fb_list =
      reinterpret_cast<ExternalFrameBufferList *>(user_priv);
  return fb_list->GetZeroFrameBuffer(min_size, fb);
}

// Callback will allocate one less byte than |min_size|.
int get_vp9_one_less_byte_frame_buffer(void *user_priv, size_t min_size,
                                       vpx_codec_frame_buffer_t *fb) {
  ExternalFrameBufferList *const fb_list =
      reinterpret_cast<ExternalFrameBufferList *>(user_priv);
  return fb_list->GetFreeFrameBuffer(min_size - 1, fb);
}

// Callback will not release the external frame buffer.
int do_not_release_vp9_frame_buffer(void *user_priv,
                                    vpx_codec_frame_buffer_t *fb) {
  (void)user_priv;
  (void)fb;
  return 0;
}

#endif  // CONFIG_WEBM_IO

// Class for testing passing in external frame buffers to libvpx.
class ExternalFrameBufferMD5Test
    : public ::libvpx_test::DecoderTest,
      public ::libvpx_test::CodecTestWithParam<const char *> {
 protected:
  ExternalFrameBufferMD5Test()
      : DecoderTest(GET_PARAM(::libvpx_test::kCodecFactoryParam)),
        md5_file_(nullptr), num_buffers_(0) {}

  ~ExternalFrameBufferMD5Test() override {
    if (md5_file_ != nullptr) fclose(md5_file_);
  }

  void PreDecodeFrameHook(const libvpx_test::CompressedVideoSource &video,
                          libvpx_test::Decoder *decoder) override {
    if (num_buffers_ > 0 && video.frame_number() == 0) {
      // Have libvpx use frame buffers we create.
      ASSERT_TRUE(fb_list_.CreateBufferList(num_buffers_));
      ASSERT_EQ(VPX_CODEC_OK,
                decoder->SetFrameBufferFunctions(GetVP9FrameBuffer,
                                                 ReleaseVP9FrameBuffer, this));
    }
  }

  void OpenMD5File(const std::string &md5_file_name_) {
    md5_file_ = libvpx_test::OpenTestDataFile(md5_file_name_);
    ASSERT_NE(md5_file_, nullptr)
        << "Md5 file open failed. Filename: " << md5_file_name_;
  }

  void DecompressedFrameHook(const vpx_image_t &img,
                             const unsigned int frame_number) override {
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
    ASSERT_STREQ(expected_md5, actual_md5)
        << "Md5 checksums don't match: frame number = " << frame_number;
  }

  // Callback to get a free external frame buffer. Return value < 0 is an
  // error.
  static int GetVP9FrameBuffer(void *user_priv, size_t min_size,
                               vpx_codec_frame_buffer_t *fb) {
    ExternalFrameBufferMD5Test *const md5Test =
        reinterpret_cast<ExternalFrameBufferMD5Test *>(user_priv);
    return md5Test->fb_list_.GetFreeFrameBuffer(min_size, fb);
  }

  // Callback to release an external frame buffer. Return value < 0 is an
  // error.
  static int ReleaseVP9FrameBuffer(void *user_priv,
                                   vpx_codec_frame_buffer_t *fb) {
    ExternalFrameBufferMD5Test *const md5Test =
        reinterpret_cast<ExternalFrameBufferMD5Test *>(user_priv);
    return md5Test->fb_list_.ReturnFrameBuffer(fb);
  }

  void set_num_buffers(int num_buffers) { num_buffers_ = num_buffers; }
  int num_buffers() const { return num_buffers_; }

 private:
  FILE *md5_file_;
  int num_buffers_;
  ExternalFrameBufferList fb_list_;
};

#if CONFIG_WEBM_IO
const char kVP9TestFile[] = "vp90-2-02-size-lf-1920x1080.webm";
const char kVP9NonRefTestFile[] = "vp90-2-22-svc_1280x720_1.webm";

// Class for testing passing in external frame buffers to libvpx.
class ExternalFrameBufferTest : public ::testing::Test {
 protected:
  ExternalFrameBufferTest()
      : video_(nullptr), decoder_(nullptr), num_buffers_(0) {}

  void SetUp() override {
    video_ = new libvpx_test::WebMVideoSource(kVP9TestFile);
    ASSERT_NE(video_, nullptr);
    video_->Init();
    video_->Begin();

    vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
    decoder_ = new libvpx_test::VP9Decoder(cfg, 0);
    ASSERT_NE(decoder_, nullptr);
  }

  void TearDown() override {
    delete decoder_;
    decoder_ = nullptr;
    delete video_;
    video_ = nullptr;
  }

  // Passes the external frame buffer information to libvpx.
  vpx_codec_err_t SetFrameBufferFunctions(
      int num_buffers, vpx_get_frame_buffer_cb_fn_t cb_get,
      vpx_release_frame_buffer_cb_fn_t cb_release) {
    if (num_buffers > 0) {
      num_buffers_ = num_buffers;
      EXPECT_TRUE(fb_list_.CreateBufferList(num_buffers_));
    }

    return decoder_->SetFrameBufferFunctions(cb_get, cb_release, &fb_list_);
  }

  vpx_codec_err_t DecodeOneFrame() {
    const vpx_codec_err_t res =
        decoder_->DecodeFrame(video_->cxdata(), video_->frame_size());
    CheckDecodedFrames();
    if (res == VPX_CODEC_OK) video_->Next();
    return res;
  }

  vpx_codec_err_t DecodeRemainingFrames() {
    for (; video_->cxdata() != nullptr; video_->Next()) {
      const vpx_codec_err_t res =
          decoder_->DecodeFrame(video_->cxdata(), video_->frame_size());
      if (res != VPX_CODEC_OK) return res;
      CheckDecodedFrames();
    }
    return VPX_CODEC_OK;
  }

  void CheckDecodedFrames() {
    libvpx_test::DxDataIterator dec_iter = decoder_->GetDxData();
    const vpx_image_t *img = nullptr;

    // Get decompressed data
    while ((img = dec_iter.Next()) != nullptr) {
      fb_list_.CheckImageFrameBuffer(img);
    }
  }

  libvpx_test::WebMVideoSource *video_;
  libvpx_test::VP9Decoder *decoder_;
  int num_buffers_;
  ExternalFrameBufferList fb_list_;
};

class ExternalFrameBufferNonRefTest : public ExternalFrameBufferTest {
 protected:
  void SetUp() override {
    video_ = new libvpx_test::WebMVideoSource(kVP9NonRefTestFile);
    ASSERT_NE(video_, nullptr);
    video_->Init();
    video_->Begin();

    vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
    decoder_ = new libvpx_test::VP9Decoder(cfg, 0);
    ASSERT_NE(decoder_, nullptr);
  }

  virtual void CheckFrameBufferRelease() {
    TearDown();
    ASSERT_EQ(0, fb_list_.num_used_buffers());
  }
};
#endif  // CONFIG_WEBM_IO

// This test runs through the set of test vectors, and decodes them.
// Libvpx will call into the application to allocate a frame buffer when
// needed. The md5 checksums are computed for each frame in the video file.
// If md5 checksums match the correct md5 data, then the test is passed.
// Otherwise, the test failed.
TEST_P(ExternalFrameBufferMD5Test, ExtFBMD5Match) {
  const std::string filename = GET_PARAM(kVideoNameParam);

  // Number of buffers equals #VP9_MAXIMUM_REF_BUFFERS +
  // #VPX_MAXIMUM_WORK_BUFFERS + four jitter buffers.
  const int jitter_buffers = 4;
  const int num_buffers =
      VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS + jitter_buffers;
  set_num_buffers(num_buffers);

#if CONFIG_VP8_DECODER
  // Tell compiler we are not using kVP8TestVectors.
  (void)libvpx_test::kVP8TestVectors;
#endif

  // Open compressed video file.
  std::unique_ptr<libvpx_test::CompressedVideoSource> video;
  if (filename.substr(filename.length() - 3, 3) == "ivf") {
    video.reset(new libvpx_test::IVFVideoSource(filename));
  } else {
#if CONFIG_WEBM_IO
    video.reset(new libvpx_test::WebMVideoSource(filename));
#else
    fprintf(stderr, "WebM IO is disabled, skipping test vector %s\n",
            filename.c_str());
    return;
#endif
  }
  ASSERT_NE(video.get(), nullptr);
  video->Init();

  // Construct md5 file name.
  const std::string md5_filename = filename + ".md5";
  OpenMD5File(md5_filename);

  // Decode frame, and check the md5 matching.
  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
}

#if CONFIG_WEBM_IO
TEST_F(ExternalFrameBufferTest, MinFrameBuffers) {
  // Minimum number of external frame buffers for VP9 is
  // #VP9_MAXIMUM_REF_BUFFERS + #VPX_MAXIMUM_WORK_BUFFERS.
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  ASSERT_EQ(VPX_CODEC_OK,
            SetFrameBufferFunctions(num_buffers, get_vp9_frame_buffer,
                                    release_vp9_frame_buffer));
  ASSERT_EQ(VPX_CODEC_OK, DecodeRemainingFrames());
}

TEST_F(ExternalFrameBufferTest, EightJitterBuffers) {
  // Number of buffers equals #VP9_MAXIMUM_REF_BUFFERS +
  // #VPX_MAXIMUM_WORK_BUFFERS + eight jitter buffers.
  const int jitter_buffers = 8;
  const int num_buffers =
      VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS + jitter_buffers;
  ASSERT_EQ(VPX_CODEC_OK,
            SetFrameBufferFunctions(num_buffers, get_vp9_frame_buffer,
                                    release_vp9_frame_buffer));
  ASSERT_EQ(VPX_CODEC_OK, DecodeRemainingFrames());
}

TEST_F(ExternalFrameBufferTest, NotEnoughBuffers) {
  // Minimum number of external frame buffers for VP9 is
  // #VP9_MAXIMUM_REF_BUFFERS + #VPX_MAXIMUM_WORK_BUFFERS. Most files will
  // only use 5 frame buffers at one time.
  const int num_buffers = 2;
  ASSERT_EQ(VPX_CODEC_OK,
            SetFrameBufferFunctions(num_buffers, get_vp9_frame_buffer,
                                    release_vp9_frame_buffer));
  ASSERT_EQ(VPX_CODEC_OK, DecodeOneFrame());
  // Only run this on long clips. Decoding a very short clip will return
  // VPX_CODEC_OK even with only 2 buffers.
  ASSERT_EQ(VPX_CODEC_MEM_ERROR, DecodeRemainingFrames());
}

TEST_F(ExternalFrameBufferTest, NoRelease) {
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  ASSERT_EQ(VPX_CODEC_OK,
            SetFrameBufferFunctions(num_buffers, get_vp9_frame_buffer,
                                    do_not_release_vp9_frame_buffer));
  ASSERT_EQ(VPX_CODEC_OK, DecodeOneFrame());
  ASSERT_EQ(VPX_CODEC_MEM_ERROR, DecodeRemainingFrames());
}

TEST_F(ExternalFrameBufferTest, NullRealloc) {
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  ASSERT_EQ(VPX_CODEC_OK,
            SetFrameBufferFunctions(num_buffers, get_vp9_zero_frame_buffer,
                                    release_vp9_frame_buffer));
  ASSERT_EQ(VPX_CODEC_MEM_ERROR, DecodeOneFrame());
}

TEST_F(ExternalFrameBufferTest, ReallocOneLessByte) {
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  ASSERT_EQ(VPX_CODEC_OK, SetFrameBufferFunctions(
                              num_buffers, get_vp9_one_less_byte_frame_buffer,
                              release_vp9_frame_buffer));
  ASSERT_EQ(VPX_CODEC_MEM_ERROR, DecodeOneFrame());
}

TEST_F(ExternalFrameBufferTest, NullGetFunction) {
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  ASSERT_EQ(
      VPX_CODEC_INVALID_PARAM,
      SetFrameBufferFunctions(num_buffers, nullptr, release_vp9_frame_buffer));
}

TEST_F(ExternalFrameBufferTest, NullReleaseFunction) {
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  ASSERT_EQ(
      VPX_CODEC_INVALID_PARAM,
      SetFrameBufferFunctions(num_buffers, get_vp9_frame_buffer, nullptr));
}

TEST_F(ExternalFrameBufferTest, SetAfterDecode) {
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  ASSERT_EQ(VPX_CODEC_OK, DecodeOneFrame());
  ASSERT_EQ(VPX_CODEC_ERROR,
            SetFrameBufferFunctions(num_buffers, get_vp9_frame_buffer,
                                    release_vp9_frame_buffer));
}

TEST_F(ExternalFrameBufferNonRefTest, ReleaseNonRefFrameBuffer) {
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  ASSERT_EQ(VPX_CODEC_OK,
            SetFrameBufferFunctions(num_buffers, get_vp9_frame_buffer,
                                    release_vp9_frame_buffer));
  ASSERT_EQ(VPX_CODEC_OK, DecodeRemainingFrames());
  CheckFrameBufferRelease();
}
#endif  // CONFIG_WEBM_IO

VP9_INSTANTIATE_TEST_SUITE(
    ExternalFrameBufferMD5Test,
    ::testing::ValuesIn(libvpx_test::kVP9TestVectors,
                        libvpx_test::kVP9TestVectors +
                            libvpx_test::kNumVP9TestVectors));
}  // namespace
