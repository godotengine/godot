// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_upsampling.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_upsampling.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/simd_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Clamp;
using hwy::HWY_NAMESPACE::LoadU;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Store;

namespace {
constexpr static size_t kChunkSize = 1024;
}  // namespace

class UpsamplingStage : public RenderPipelineStage {
 public:
  explicit UpsamplingStage(JxlMemoryManager* memory_manager,
                           const CustomTransformData& ups_factors, size_t c,
                           size_t shift)
      : RenderPipelineStage(RenderPipelineStage::Settings::Symmetric(
            /*shift=*/shift, /*border=*/2)),
        c_(c),
        memory_manager_(memory_manager) {
    const float* weights = shift == 1   ? ups_factors.upsampling2_weights
                           : shift == 2 ? ups_factors.upsampling4_weights
                                        : ups_factors.upsampling8_weights;
    size_t N = 1 << shift;
    size_t H = N / 2;
    for (size_t ky = 0; ky < H; ++ky) {
      for (size_t kx = 0; kx < H; ++kx) {
        size_t offset0 = (ky * N + kx) * 25;
        size_t offset1 = (ky * N + (N - 1 - kx)) * 25;
        size_t offset2 = ((N - 1 - ky) * N + kx) * 25;
        size_t offset3 = ((N - 1 - ky) * N + (N - 1 - kx)) * 25;
        for (size_t py = 0; py < 5; ++py) {
          for (size_t px = 0; px < 5; ++px) {
            size_t j = 5 * ky + py;
            size_t i = 5 * kx + px;
            size_t my = std::min(i, j);
            size_t mx = std::max(i, j);
            float w = weights[5 * H * my - my * (my - 1) / 2 + mx - my];
            kernel_[offset0 + py * 5 + px] = w;
            kernel_[offset1 + py * 5 + (4 - px)] = w;
            kernel_[offset2 + (4 - py) * 5 + px] = w;
            kernel_[offset3 + (4 - py) * 5 + (4 - px)] = w;
          }
        }
      }
    }
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    constexpr HWY_FULL(float) df;
    size_t shift = settings_.shift_x;
    size_t N = 1 << shift;
    const size_t xsize_v = RoundUpTo(xsize, Lanes(df));
    // TODO(eustas): double-check (un)poison ranges.
    for (ptrdiff_t iy = -2; iy <= 2; iy++) {
      msan::UnpoisonMemory(GetInputRow(input_rows, c_, iy) + xsize + 2,
                           sizeof(float) * (xsize_v - xsize));
    }
    JXL_ENSURE(xextra_left == 0 && xextra_right == 0);
    for (size_t x = 0; x < xsize; x += kChunkSize) {
      size_t xend = std::min(x + kChunkSize, xsize);
      size_t len = xend - x;
      PreProcessRowImpl(input_rows, x, len, thread_id);
      if (N == 2) {
        ProcessRowImpl<2>(input_rows, output_rows, x, len, thread_id);
      }
      if (N == 4) {
        ProcessRowImpl<4>(input_rows, output_rows, x, len, thread_id);
      }
      if (N == 8) {
        ProcessRowImpl<8>(input_rows, output_rows, x, len, thread_id);
      }
    }
    for (size_t oy = 0; oy < N; oy++) {
      float* dst_row = GetOutputRow(output_rows, c_, oy);
      msan::PoisonMemory(dst_row + xsize * N,
                         sizeof(float) * (xsize_v - xsize) * N);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c == c_ ? RenderPipelineChannelMode::kInOut
                   : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "Upsample"; }

 private:
  JXL_INLINE float Kernel(size_t k, size_t i) const {
    return kernel_[k * 25 + i];
  }

  Status PrepareForThreads(size_t num_threads) override {
    constexpr HWY_FULL(float) df;
    size_t alloc_size =
        sizeof(float) * (kChunkSize + std::max(size_t{4}, Lanes(df)));
    for (size_t i = 0; i < 3; ++i) {
      temp_[i].resize(num_threads);
      for (size_t t = 0; t < num_threads; ++t) {
        JXL_ASSIGN_OR_RETURN(
            temp_[i][t], AlignedMemory::Create(memory_manager_, alloc_size));
      }
    }
    return true;
  }

  void PreProcessRowImpl(const RowInfo& input_rows, size_t x0, size_t len,
                         size_t thread_id) const {
    constexpr HWY_FULL(float) df;
    float* JXL_RESTRICT col_min = temp_[0][thread_id].address<float>();
    float* JXL_RESTRICT col_max = temp_[1][thread_id].address<float>();

    std::array<float* JXL_RESTRICT, 5> rows = {
        GetInputRow(input_rows, c_, -2) + x0 - 2,
        GetInputRow(input_rows, c_, -1) + x0 - 2,
        GetInputRow(input_rows, c_, 0) + x0 - 2,
        GetInputRow(input_rows, c_, 1) + x0 - 2,
        GetInputRow(input_rows, c_, 2) + x0 - 2};

    for (size_t x = 0; x < len + 4; x += Lanes(df)) {
      const auto v0 = LoadU(df, rows[0] + x);
      const auto v1 = LoadU(df, rows[1] + x);
      const auto min0 = Min(v0, v1);
      const auto max0 = Max(v0, v1);
      const auto v2 = LoadU(df, rows[2] + x);
      const auto v3 = LoadU(df, rows[3] + x);
      const auto min1 = Min(v2, v3);
      const auto max1 = Max(v2, v3);
      const auto v4 = LoadU(df, rows[4] + x);
      const auto min2 = Min(min0, min1);
      const auto max2 = Max(max0, max1);
      const auto min = Min(v4, min2);
      const auto max = Max(v4, max2);
      Store(min, df, col_min + x);
      Store(max, df, col_max + x);
    }

    float* JXL_RESTRICT mins = temp_[2][thread_id].address<float>();
    for (size_t x = 0; x < len; x += Lanes(df)) {
      const auto v0 = LoadU(df, col_min + x);
      const auto v1 = LoadU(df, col_min + x + 1);
      const auto min0 = Min(v0, v1);
      const auto v2 = LoadU(df, col_min + x + 2);
      const auto v3 = LoadU(df, col_min + x + 3);
      const auto min1 = Min(v2, v3);
      const auto v4 = LoadU(df, col_min + x + 4);
      const auto min2 = Min(min0, min1);
      const auto min = Min(v4, min2);
      Store(min, df, mins + x);
    }

    // col_mins will be overwritten
    float* JXL_RESTRICT maxs = temp_[0][thread_id].address<float>();
    for (size_t x = 0; x < len; x += Lanes(df)) {
      const auto v0 = LoadU(df, col_max + x);
      const auto v1 = LoadU(df, col_max + x + 1);
      const auto max0 = Max(v0, v1);
      const auto v2 = LoadU(df, col_max + x + 2);
      const auto v3 = LoadU(df, col_max + x + 3);
      const auto max1 = Max(v2, v3);
      const auto v4 = LoadU(df, col_max + x + 4);
      const auto max2 = Max(max0, max1);
      const auto max = Max(v4, max2);
      Store(max, df, maxs + x);
    }
  }

  template <ptrdiff_t N>
  void ProcessRowImpl(const RowInfo& input_rows, const RowInfo& output_rows,
                      size_t x0, size_t len, size_t thread_id) const {
    constexpr HWY_FULL(float) df;
    using V = hwy::HWY_NAMESPACE::Vec<HWY_FULL(float)>;
    V ups0, ups1, ups2, ups3, ups4, ups5, ups6, ups7;  // NOLINT
    (void)ups2, (void)ups3, (void)ups4, (void)ups5, (void)ups6, (void)ups7;
    // Once we have C++17 available, change this back to `V* ups[N]` and
    // initialize using `if constexpr` below.
    V* ups[8] = {};
    static_assert(N == 2 || N == 4 || N == 8, "N must be 2, 4, or 8");
    if (N >= 2) {
      ups[0] = &ups0;
      ups[1] = &ups1;
    }
    if (N >= 4) {
      ups[2] = &ups2;
      ups[3] = &ups3;
    }
    if (N == 8) {
      ups[4] = &ups4;
      ups[5] = &ups5;
      ups[6] = &ups6;
      ups[7] = &ups7;
    }

    float* JXL_RESTRICT mins = temp_[2][thread_id].address<float>();
    float* JXL_RESTRICT maxs = temp_[0][thread_id].address<float>();
    std::array<float*, 25> input;
    for (ptrdiff_t iy = -2; iy <= 2; ++iy) {
      for (ptrdiff_t ix = -2; ix <= 2; ++ix) {
        input[5 * (iy + 2) + (ix + 2)] =
            GetInputRow(input_rows, c_, iy) + x0 + ix;
      }
    }

    for (size_t x = 0; x < len; x += Lanes(df)) {
      for (size_t oy = 0; oy < N; oy++) {
        float* dst_row = GetOutputRow(output_rows, c_, oy) + x0 * N;
        for (size_t ox = 0; ox < N; ox++) {
          size_t k = N * oy + ox;
          auto acc0 = Mul(LoadU(df, input[0]), Set(df, Kernel(k, 0)));
          auto acc1 = Mul(LoadU(df, input[1]), Set(df, Kernel(k, 1)));
          auto acc2 = Mul(LoadU(df, input[2]), Set(df, Kernel(k, 2)));
          for (size_t i = 3; i < 24; i += 3) {
            acc0 = MulAdd(LoadU(df, input[i]), Set(df, Kernel(k, i)), acc0);
            acc1 = MulAdd(LoadU(df, input[i + 1]), Set(df, Kernel(k, i + 1)),
                          acc1);
            acc2 = MulAdd(LoadU(df, input[i + 2]), Set(df, Kernel(k, i + 2)),
                          acc2);
          }
          acc0 = MulAdd(LoadU(df, input[24]), Set(df, Kernel(k, 24)), acc0);
          auto result = Add(Add(acc1, acc2), acc0);
          *ups[ox] = Clamp(result, Load(df, mins + x), Load(df, maxs + x));
        }
        if (N == 2) {
          StoreInterleaved(df, ups0, ups1, dst_row + x * N);
        }
        if (N == 4) {
          StoreInterleaved(df, ups0, ups1, ups2, ups3, dst_row + x * N);
        }
        if (N == 8) {
          StoreInterleaved(df, ups0, ups1, ups2, ups3, ups4, ups5, ups6, ups7,
                           dst_row + x * N);
        }
      }
      for (size_t i = 0; i < 25; ++i) input[i] += Lanes(df);
    }
  }

  // Process row in chunks to keep per-thread buffers compact.
  std::array<std::vector<AlignedMemory>, 3> temp_;
  size_t c_;
  float kernel_[64 * 25];
  JxlMemoryManager* memory_manager_;
};

std::unique_ptr<RenderPipelineStage> GetUpsamplingStage(
    JxlMemoryManager* memory_manager, const CustomTransformData& ups_factors,
    size_t c, size_t shift) {
  return jxl::make_unique<UpsamplingStage>(memory_manager, ups_factors, c,
                                           shift);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetUpsamplingStage);

std::unique_ptr<RenderPipelineStage> GetUpsamplingStage(
    JxlMemoryManager* memory_manager, const CustomTransformData& ups_factors,
    size_t c, size_t shift) {
  if ((shift < 1) || (shift > 3)) {
    JXL_DEBUG_ABORT("internal: (shift != 0) && (shift <= 3)");
    return nullptr;
  }
  return HWY_DYNAMIC_DISPATCH(GetUpsamplingStage)(memory_manager, ups_factors,
                                                  c, shift);
}

}  // namespace jxl
#endif
