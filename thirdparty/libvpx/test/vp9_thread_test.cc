/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
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
#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/md5_helper.h"
#if CONFIG_WEBM_IO
#include "test/webm_video_source.h"
#endif
#include "vpx_util/vpx_thread.h"

namespace {

using std::string;

class VPxWorkerThreadTest : public ::testing::TestWithParam<bool> {
 protected:
  ~VPxWorkerThreadTest() override = default;
  void SetUp() override { vpx_get_worker_interface()->init(&worker_); }

  void TearDown() override { vpx_get_worker_interface()->end(&worker_); }

  void Run(VPxWorker *worker) {
    const bool synchronous = GetParam();
    if (synchronous) {
      vpx_get_worker_interface()->execute(worker);
    } else {
      vpx_get_worker_interface()->launch(worker);
    }
  }

  VPxWorker worker_;
};

int ThreadHook(void *data, void *return_value) {
  int *const hook_data = reinterpret_cast<int *>(data);
  *hook_data = 5;
  return *reinterpret_cast<int *>(return_value);
}

TEST_P(VPxWorkerThreadTest, HookSuccess) {
  // should be a no-op.
  EXPECT_NE(vpx_get_worker_interface()->sync(&worker_), 0);

  for (int i = 0; i < 2; ++i) {
    EXPECT_NE(vpx_get_worker_interface()->reset(&worker_), 0);

    int hook_data = 0;
    int return_value = 1;  // return successfully from the hook
    worker_.hook = ThreadHook;
    worker_.data1 = &hook_data;
    worker_.data2 = &return_value;

    Run(&worker_);
    EXPECT_NE(vpx_get_worker_interface()->sync(&worker_), 0);
    EXPECT_FALSE(worker_.had_error);
    EXPECT_EQ(5, hook_data);

    // should be a no-op.
    EXPECT_NE(vpx_get_worker_interface()->sync(&worker_), 0);
  }
}

TEST_P(VPxWorkerThreadTest, HookFailure) {
  EXPECT_NE(vpx_get_worker_interface()->reset(&worker_), 0);

  int hook_data = 0;
  int return_value = 0;  // return failure from the hook
  worker_.hook = ThreadHook;
  worker_.data1 = &hook_data;
  worker_.data2 = &return_value;

  Run(&worker_);
  EXPECT_FALSE(vpx_get_worker_interface()->sync(&worker_));
  EXPECT_EQ(1, worker_.had_error);

  // Ensure _reset() clears the error and _launch() can be called again.
  return_value = 1;
  EXPECT_NE(vpx_get_worker_interface()->reset(&worker_), 0);
  EXPECT_FALSE(worker_.had_error);
  vpx_get_worker_interface()->launch(&worker_);
  EXPECT_NE(vpx_get_worker_interface()->sync(&worker_), 0);
  EXPECT_FALSE(worker_.had_error);
}

TEST_P(VPxWorkerThreadTest, EndWithoutSync) {
  // Create a large number of threads to increase the chances of detecting a
  // race. Doing more work in the hook is no guarantee as any race would occur
  // post hook execution in the main thread loop driver.
  static const int kNumWorkers = 64;
  VPxWorker workers[kNumWorkers];
  int hook_data[kNumWorkers];
  int return_value[kNumWorkers];

  for (int n = 0; n < kNumWorkers; ++n) {
    vpx_get_worker_interface()->init(&workers[n]);
    return_value[n] = 1;  // return successfully from the hook
    workers[n].hook = ThreadHook;
    workers[n].data1 = &hook_data[n];
    workers[n].data2 = &return_value[n];
  }

  for (int i = 0; i < 2; ++i) {
    for (int n = 0; n < kNumWorkers; ++n) {
      EXPECT_NE(vpx_get_worker_interface()->reset(&workers[n]), 0);
      hook_data[n] = 0;
    }

    for (int n = 0; n < kNumWorkers; ++n) {
      Run(&workers[n]);
    }

    for (int n = kNumWorkers - 1; n >= 0; --n) {
      vpx_get_worker_interface()->end(&workers[n]);
    }
  }
}

TEST(VPxWorkerThreadTest, TestInterfaceAPI) {
  EXPECT_EQ(0, vpx_set_worker_interface(nullptr));
  EXPECT_NE(vpx_get_worker_interface(), nullptr);
  for (int i = 0; i < 6; ++i) {
    VPxWorkerInterface winterface = *vpx_get_worker_interface();
    switch (i) {
      default:
      case 0: winterface.init = nullptr; break;
      case 1: winterface.reset = nullptr; break;
      case 2: winterface.sync = nullptr; break;
      case 3: winterface.launch = nullptr; break;
      case 4: winterface.execute = nullptr; break;
      case 5: winterface.end = nullptr; break;
    }
    EXPECT_EQ(0, vpx_set_worker_interface(&winterface));
  }
}

// -----------------------------------------------------------------------------
// Multi-threaded decode tests
#if CONFIG_WEBM_IO
// Decodes |filename| with |num_threads|. Returns the md5 of the decoded frames.
string DecodeFile(const string &filename, int num_threads) {
  libvpx_test::WebMVideoSource video(filename);
  video.Init();

  vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
  cfg.threads = num_threads;
  libvpx_test::VP9Decoder decoder(cfg, 0);

  libvpx_test::MD5 md5;
  for (video.Begin(); video.cxdata(); video.Next()) {
    const vpx_codec_err_t res =
        decoder.DecodeFrame(video.cxdata(), video.frame_size());
    if (res != VPX_CODEC_OK) {
      EXPECT_EQ(VPX_CODEC_OK, res) << decoder.DecodeError();
      break;
    }

    libvpx_test::DxDataIterator dec_iter = decoder.GetDxData();
    const vpx_image_t *img = nullptr;

    // Get decompressed data
    while ((img = dec_iter.Next())) {
      md5.Add(img);
    }
  }
  return string(md5.Get());
}

// Trivial serialized thread worker interface implementation.
// Note any worker that requires synchronization between other workers will
// hang.
namespace impl {
namespace {

void Init(VPxWorker *const worker) { memset(worker, 0, sizeof(*worker)); }
int Reset(VPxWorker *const /*worker*/) { return 1; }
int Sync(VPxWorker *const worker) { return !worker->had_error; }

void Execute(VPxWorker *const worker) {
  worker->had_error |= !worker->hook(worker->data1, worker->data2);
}

void Launch(VPxWorker *const worker) { Execute(worker); }
void End(VPxWorker *const /*worker*/) {}

}  // namespace
}  // namespace impl

TEST(VPxWorkerThreadTest, TestSerialInterface) {
  static const VPxWorkerInterface serial_interface = {
    impl::Init, impl::Reset, impl::Sync, impl::Launch, impl::Execute, impl::End
  };
  static const char expected_md5[] = "b35a1b707b28e82be025d960aba039bc";
  static const char filename[] = "vp90-2-03-size-226x226.webm";
  VPxWorkerInterface default_interface = *vpx_get_worker_interface();

  EXPECT_NE(vpx_set_worker_interface(&serial_interface), 0);
  EXPECT_EQ(expected_md5, DecodeFile(filename, 2));

  // Reset the interface.
  EXPECT_NE(vpx_set_worker_interface(&default_interface), 0);
  EXPECT_EQ(expected_md5, DecodeFile(filename, 2));
}

struct FileParam {
  const char *name;
  const char *expected_md5;
  friend std::ostream &operator<<(std::ostream &os, const FileParam &param) {
    return os << "file name: " << param.name
              << " digest: " << param.expected_md5;
  }
};

class VP9DecodeMultiThreadedTest : public ::testing::TestWithParam<FileParam> {
};

TEST_P(VP9DecodeMultiThreadedTest, Decode) {
  for (int t = 1; t <= 8; ++t) {
    EXPECT_EQ(GetParam().expected_md5, DecodeFile(GetParam().name, t))
        << "threads = " << t;
  }
}

const FileParam kNoTilesNonFrameParallelFiles[] = {
  { "vp90-2-03-size-226x226.webm", "b35a1b707b28e82be025d960aba039bc" }
};

const FileParam kFrameParallelFiles[] = {
  { "vp90-2-08-tile_1x2_frame_parallel.webm",
    "68ede6abd66bae0a2edf2eb9232241b6" },
  { "vp90-2-08-tile_1x4_frame_parallel.webm",
    "368ebc6ebf3a5e478d85b2c3149b2848" },
  { "vp90-2-08-tile_1x8_frame_parallel.webm",
    "17e439da2388aff3a0f69cb22579c6c1" },
};

const FileParam kFrameParallelResizeFiles[] = {
  { "vp90-2-14-resize-fp-tiles-1-16.webm", "0cd5e632c326297e975f38949c31ea94" },
  { "vp90-2-14-resize-fp-tiles-1-2-4-8-16.webm",
    "5c78a96a42e7f4a4f6b2edcdb791e44c" },
  { "vp90-2-14-resize-fp-tiles-1-2.webm", "e030450ae85c3277be2a418769df98e2" },
  { "vp90-2-14-resize-fp-tiles-1-4.webm", "312eed4e2b64eb7a4e7f18916606a430" },
  { "vp90-2-14-resize-fp-tiles-16-1.webm", "1755c16d8af16a9cb3fe7338d90abe52" },
  { "vp90-2-14-resize-fp-tiles-16-2.webm", "500300592d3fcb6f12fab25e48aaf4df" },
  { "vp90-2-14-resize-fp-tiles-16-4.webm", "47c48379fa6331215d91c67648e1af6e" },
  { "vp90-2-14-resize-fp-tiles-16-8-4-2-1.webm",
    "eecf17290739bc708506fa4827665989" },
  { "vp90-2-14-resize-fp-tiles-16-8.webm", "29b6bb54e4c26b5ca85d5de5fed94e76" },
  { "vp90-2-14-resize-fp-tiles-1-8.webm", "1b6f175e08cd82cf84bb800ac6d1caa3" },
  { "vp90-2-14-resize-fp-tiles-2-16.webm", "ca3b03e4197995d8d5444ede7a6c0804" },
  { "vp90-2-14-resize-fp-tiles-2-1.webm", "99aec065369d70bbb78ccdff65afed3f" },
  { "vp90-2-14-resize-fp-tiles-2-4.webm", "22d0ebdb49b87d2920a85aea32e1afd5" },
  { "vp90-2-14-resize-fp-tiles-2-8.webm", "c2115cf051c62e0f7db1d4a783831541" },
  { "vp90-2-14-resize-fp-tiles-4-16.webm", "c690d7e1719b31367564cac0af0939cb" },
  { "vp90-2-14-resize-fp-tiles-4-1.webm", "a926020b2cc3e15ad4cc271853a0ff26" },
  { "vp90-2-14-resize-fp-tiles-4-2.webm", "42699063d9e581f1993d0cf890c2be78" },
  { "vp90-2-14-resize-fp-tiles-4-8.webm", "7f76d96036382f45121e3d5aa6f8ec52" },
  { "vp90-2-14-resize-fp-tiles-8-16.webm", "76a43fcdd7e658542913ea43216ec55d" },
  { "vp90-2-14-resize-fp-tiles-8-1.webm", "8e3fbe89486ca60a59299dea9da91378" },
  { "vp90-2-14-resize-fp-tiles-8-2.webm", "ae96f21f21b6370cc0125621b441fc52" },
  { "vp90-2-14-resize-fp-tiles-8-4.webm", "3eb4f24f10640d42218f7fd7b9fd30d4" },
};

const FileParam kNonFrameParallelFiles[] = {
  { "vp90-2-08-tile_1x2.webm", "570b4a5d5a70d58b5359671668328a16" },
  { "vp90-2-08-tile_1x4.webm", "988d86049e884c66909d2d163a09841a" },
  { "vp90-2-08-tile_1x8.webm", "0941902a52e9092cb010905eab16364c" },
  { "vp90-2-08-tile-4x1.webm", "06505aade6647c583c8e00a2f582266f" },
  { "vp90-2-08-tile-4x4.webm", "85c2299892460d76e2c600502d52bfe2" },
};

INSTANTIATE_TEST_SUITE_P(NoTilesNonFrameParallel, VP9DecodeMultiThreadedTest,
                         ::testing::ValuesIn(kNoTilesNonFrameParallelFiles));
INSTANTIATE_TEST_SUITE_P(FrameParallel, VP9DecodeMultiThreadedTest,
                         ::testing::ValuesIn(kFrameParallelFiles));
INSTANTIATE_TEST_SUITE_P(FrameParallelResize, VP9DecodeMultiThreadedTest,
                         ::testing::ValuesIn(kFrameParallelResizeFiles));
INSTANTIATE_TEST_SUITE_P(NonFrameParallel, VP9DecodeMultiThreadedTest,
                         ::testing::ValuesIn(kNonFrameParallelFiles));
#endif  // CONFIG_WEBM_IO

INSTANTIATE_TEST_SUITE_P(Synchronous, VPxWorkerThreadTest, ::testing::Bool());

}  // namespace
