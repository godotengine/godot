// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_write.h"

#include <jxl/memory_manager.h>

#include <cstdint>
#include <type_traits>

#include "lib/jxl/alpha.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/memory_manager_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_write.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Clamp;
using hwy::HWY_NAMESPACE::Div;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::NearestInt;
using hwy::HWY_NAMESPACE::Or;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::ShiftLeftSame;
using hwy::HWY_NAMESPACE::ShiftRightSame;
using hwy::HWY_NAMESPACE::VFromD;

// 8x8 ordered dithering pattern from
// https://en.wikipedia.org/wiki/Ordered_dithering
// scaled to have an average of 0 and be fully contained in (-0.5, 0.5).
// Matrix is duplicated in width to avoid inconsistencies or out-of-bound-reads
// if doing unaligned operations.
const float kDither[(2 * 8) * 8] = {
    -0.4921875, 0.0078125, -0.3671875, 0.1328125,  //
    -0.4609375, 0.0390625, -0.3359375, 0.1640625,  //
    -0.4921875, 0.0078125, -0.3671875, 0.1328125,  //
    -0.4609375, 0.0390625, -0.3359375, 0.1640625,  //
                                                   //
    0.2578125, -0.2421875, 0.3828125, -0.1171875,  //
    0.2890625, -0.2109375, 0.4140625, -0.0859375,  //
    0.2578125, -0.2421875, 0.3828125, -0.1171875,  //
    0.2890625, -0.2109375, 0.4140625, -0.0859375,  //
                                                   //
    -0.3046875, 0.1953125, -0.4296875, 0.0703125,  //
    -0.2734375, 0.2265625, -0.3984375, 0.1015625,  //
    -0.3046875, 0.1953125, -0.4296875, 0.0703125,  //
    -0.2734375, 0.2265625, -0.3984375, 0.1015625,  //
                                                   //
    0.4453125, -0.0546875, 0.3203125, -0.1796875,  //
    0.4765625, -0.0234375, 0.3515625, -0.1484375,  //
    0.4453125, -0.0546875, 0.3203125, -0.1796875,  //
    0.4765625, -0.0234375, 0.3515625, -0.1484375,  //
                                                   //
    -0.4453125, 0.0546875, -0.3203125, 0.1796875,  //
    -0.4765625, 0.0234375, -0.3515625, 0.1484375,  //
    -0.4453125, 0.0546875, -0.3203125, 0.1796875,  //
    -0.4765625, 0.0234375, -0.3515625, 0.1484375,  //
                                                   //
    0.3046875, -0.1953125, 0.4296875, -0.0703125,  //
    0.2734375, -0.2265625, 0.3984375, -0.1015625,  //
    0.3046875, -0.1953125, 0.4296875, -0.0703125,  //
    0.2734375, -0.2265625, 0.3984375, -0.1015625,  //
                                                   //
    -0.2578125, 0.2421875, -0.3828125, 0.1171875,  //
    -0.2890625, 0.2109375, -0.4140625, 0.0859375,  //
    -0.2578125, 0.2421875, -0.3828125, 0.1171875,  //
    -0.2890625, 0.2109375, -0.4140625, 0.0859375,  //
                                                   //
    0.4921875, -0.0078125, 0.3671875, -0.1328125,  //
    0.4609375, -0.0390625, 0.3359375, -0.1640625,  //
    0.4921875, -0.0078125, 0.3671875, -0.1328125,  //
    0.4609375, -0.0390625, 0.3359375, -0.1640625,  //
};

using DF = HWY_FULL(float);

// Converts `v` to an appropriate value for the given unsigned type.
// If the unsigned type is an 8-bit type, performs ordered dithering.
template <typename T>
VFromD<Rebind<T, DF>> MakeUnsigned(VFromD<DF> v, size_t x0, size_t y0,
                                   VFromD<DF> mul) {
  static_assert(std::is_unsigned<T>::value, "T must be an unsigned type");
  using DU = Rebind<T, DF>;
  v = Mul(v, mul);
  // TODO(veluca): if constexpr with C++17
  if (sizeof(T) == 1) {
    size_t pos = (y0 % 8) * (2 * 8) + (x0 % 8);
#if HWY_TARGET != HWY_SCALAR
    auto dither = LoadDup128(DF(), kDither + pos);
#else
    auto dither = LoadU(DF(), kDither + pos);
#endif
    v = Add(v, dither);
  }
  v = Clamp(Zero(DF()), v, mul);
  return DemoteTo(DU(), NearestInt(v));
}

class WriteToOutputStage : public RenderPipelineStage {
 public:
  WriteToOutputStage(const ImageOutput& main_output, size_t width,
                     size_t height, bool has_alpha, bool unpremul_alpha,
                     size_t alpha_c, Orientation undo_orientation,
                     const std::vector<ImageOutput>& extra_output,
                     JxlMemoryManager* memory_manager)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        width_(width),
        height_(height),
        main_(main_output),
        num_color_(main_.num_channels_ < 3 ? 1 : 3),
        want_alpha_(main_.num_channels_ == 2 || main_.num_channels_ == 4),
        has_alpha_(has_alpha),
        unpremul_alpha_(unpremul_alpha),
        alpha_c_(alpha_c),
        flip_x_(ShouldFlipX(undo_orientation)),
        flip_y_(ShouldFlipY(undo_orientation)),
        transpose_(ShouldTranspose(undo_orientation)),
        opaque_alpha_(kMaxPixelsPerCall, 1.0f),
        memory_manager_(memory_manager) {
    for (size_t ec = 0; ec < extra_output.size(); ++ec) {
      if (extra_output[ec].callback.IsPresent() || extra_output[ec].buffer) {
        Output extra(extra_output[ec]);
        extra.channel_index_ = 3 + ec;
        extra_channels_.push_back(extra);
      }
    }
  }

  WriteToOutputStage(const WriteToOutputStage&) = delete;
  WriteToOutputStage& operator=(const WriteToOutputStage&) = delete;
  WriteToOutputStage(WriteToOutputStage&&) = delete;
  WriteToOutputStage& operator=(WriteToOutputStage&&) = delete;

  ~WriteToOutputStage() override {
    if (main_.run_opaque_) {
      main_.pixel_callback_.destroy(main_.run_opaque_);
    }
    for (auto& extra : extra_channels_) {
      if (extra.run_opaque_) {
        extra.pixel_callback_.destroy(extra.run_opaque_);
      }
    }
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    JXL_ENSURE(xextra == 0);
    JXL_ENSURE(main_.run_opaque_ || main_.buffer_);
    if (ypos >= height_) return true;
    if (xpos >= width_) return true;
    if (flip_y_) {
      ypos = height_ - 1u - ypos;
    }
    size_t limit = std::min(xsize, width_ - xpos);
    for (size_t x0 = 0; x0 < limit; x0 += kMaxPixelsPerCall) {
      size_t xstart = xpos + x0;
      size_t len = std::min<size_t>(kMaxPixelsPerCall, limit - x0);

      const float* line_buffers[4];
      for (size_t c = 0; c < num_color_; c++) {
        line_buffers[c] = GetInputRow(input_rows, c, 0) + x0;
      }
      if (has_alpha_) {
        line_buffers[num_color_] = GetInputRow(input_rows, alpha_c_, 0) + x0;
      } else {
        // opaque_alpha_ is a way to set all values to 1.0f.
        line_buffers[num_color_] = opaque_alpha_.data();
      }
      if (has_alpha_ && want_alpha_ && unpremul_alpha_) {
        UnpremulAlpha(thread_id, len, line_buffers);
      }
      OutputBuffers(main_, thread_id, ypos, xstart, len, line_buffers);
      for (const auto& extra : extra_channels_) {
        line_buffers[0] = GetInputRow(input_rows, extra.channel_index_, 0) + x0;
        OutputBuffers(extra, thread_id, ypos, xstart, len, line_buffers);
      }
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    if (c < num_color_ || (has_alpha_ && c == alpha_c_)) {
      return RenderPipelineChannelMode::kInput;
    }
    for (const auto& extra : extra_channels_) {
      if (c == extra.channel_index_) {
        return RenderPipelineChannelMode::kInput;
      }
    }
    return RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "WritePixelCB"; }

 private:
  struct Output {
    explicit Output(const ImageOutput& image_out)
        : pixel_callback_(image_out.callback),
          buffer_(image_out.buffer),
          buffer_size_(image_out.buffer_size),
          stride_(image_out.stride),
          num_channels_(image_out.format.num_channels),
          swap_endianness_(SwapEndianness(image_out.format.endianness)),
          data_type_(image_out.format.data_type),
          bits_per_sample_(image_out.bits_per_sample) {}

    Status PrepareForThreads(size_t num_threads) {
      if (pixel_callback_.IsPresent()) {
        run_opaque_ =
            pixel_callback_.Init(num_threads, /*num_pixels=*/kMaxPixelsPerCall);
        JXL_RETURN_IF_ERROR(run_opaque_ != nullptr);
      } else {
        JXL_RETURN_IF_ERROR(buffer_ != nullptr);
      }
      return true;
    }

    PixelCallback pixel_callback_;
    void* run_opaque_ = nullptr;
    void* buffer_ = nullptr;
    size_t buffer_size_;
    size_t stride_;
    size_t num_channels_;
    bool swap_endianness_;
    JxlDataType data_type_;
    size_t bits_per_sample_;
    size_t channel_index_;  // used for extra_channels
  };

  Status PrepareForThreads(size_t num_threads) override {
    JXL_RETURN_IF_ERROR(main_.PrepareForThreads(num_threads));
    for (auto& extra : extra_channels_) {
      JXL_RETURN_IF_ERROR(extra.PrepareForThreads(num_threads));
    }
    temp_out_.resize(num_threads);
    for (AlignedMemory& temp : temp_out_) {
      size_t alloc_size =
          sizeof(float) * kMaxPixelsPerCall * main_.num_channels_;
      JXL_ASSIGN_OR_RETURN(temp,
                           AlignedMemory::Create(memory_manager_, alloc_size));
    }
    if ((has_alpha_ && want_alpha_ && unpremul_alpha_) || flip_x_) {
      temp_in_.resize(num_threads * main_.num_channels_);
      for (AlignedMemory& temp : temp_in_) {
        size_t alloc_size = sizeof(float) * kMaxPixelsPerCall;
        JXL_ASSIGN_OR_RETURN(
            temp, AlignedMemory::Create(memory_manager_, alloc_size));
      }
    }
    return true;
  }
  static bool ShouldFlipX(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kFlipHorizontal ||
            undo_orientation == Orientation::kRotate180 ||
            undo_orientation == Orientation::kRotate270 ||
            undo_orientation == Orientation::kAntiTranspose);
  }
  static bool ShouldFlipY(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kFlipVertical ||
            undo_orientation == Orientation::kRotate180 ||
            undo_orientation == Orientation::kRotate90 ||
            undo_orientation == Orientation::kAntiTranspose);
  }
  static bool ShouldTranspose(Orientation undo_orientation) {
    return (undo_orientation == Orientation::kTranspose ||
            undo_orientation == Orientation::kRotate90 ||
            undo_orientation == Orientation::kRotate270 ||
            undo_orientation == Orientation::kAntiTranspose);
  }

  void UnpremulAlpha(size_t thread_id, size_t len,
                     const float** line_buffers) const {
    const HWY_FULL(float) d;
    auto one = Set(d, 1.0f);
    float* temp_in[4];
    for (size_t c = 0; c < main_.num_channels_; ++c) {
      size_t tix = thread_id * main_.num_channels_ + c;
      temp_in[c] = temp_in_[tix].address<float>();
      memcpy(temp_in[c], line_buffers[c], sizeof(float) * len);
    }
    auto small_alpha = Set(d, kSmallAlpha);
    for (size_t ix = 0; ix < len; ix += Lanes(d)) {
      auto alpha = LoadU(d, temp_in[num_color_] + ix);
      auto mul = Div(one, Max(small_alpha, alpha));
      for (size_t c = 0; c < num_color_; ++c) {
        auto val = LoadU(d, temp_in[c] + ix);
        StoreU(Mul(val, mul), d, temp_in[c] + ix);
      }
    }
    for (size_t c = 0; c < main_.num_channels_; ++c) {
      line_buffers[c] = temp_in[c];
    }
  }

  void OutputBuffers(const Output& out, size_t thread_id, size_t ypos,
                     size_t xstart, size_t len, const float* input[4]) const {
    if (flip_x_) {
      FlipX(out, thread_id, len, &xstart, input);
    }
    if (out.data_type_ == JXL_TYPE_UINT8) {
      uint8_t* JXL_RESTRICT temp = temp_out_[thread_id].address<uint8_t>();
      StoreUnsignedRow(out, input, len, temp, xstart, ypos);
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    } else if (out.data_type_ == JXL_TYPE_UINT16 ||
               out.data_type_ == JXL_TYPE_FLOAT16) {
      uint16_t* JXL_RESTRICT temp = temp_out_[thread_id].address<uint16_t>();
      if (out.data_type_ == JXL_TYPE_UINT16) {
        StoreUnsignedRow(out, input, len, temp, xstart, ypos);
      } else {
        StoreFloat16Row(out, input, len, temp);
      }
      if (out.swap_endianness_) {
        const HWY_FULL(uint16_t) du;
        size_t output_len = len * out.num_channels_;
        for (size_t j = 0; j < output_len; j += Lanes(du)) {
          auto v = LoadU(du, temp + j);
          auto vswap = Or(ShiftRightSame(v, 8), ShiftLeftSame(v, 8));
          StoreU(vswap, du, temp + j);
        }
      }
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    } else if (out.data_type_ == JXL_TYPE_FLOAT) {
      float* JXL_RESTRICT temp = temp_out_[thread_id].address<float>();
      StoreFloatRow(out, input, len, temp);
      if (out.swap_endianness_) {
        size_t output_len = len * out.num_channels_;
        for (size_t j = 0; j < output_len; ++j) {
          temp[j] = BSwapFloat(temp[j]);
        }
      }
      WriteToOutput(out, thread_id, ypos, xstart, len, temp);
    }
  }

  void FlipX(const Output& out, size_t thread_id, size_t len, size_t* xstart,
             const float** line_buffers) const {
    float* temp_in[4];
    for (size_t c = 0; c < out.num_channels_; ++c) {
      size_t tix = thread_id * main_.num_channels_ + c;
      temp_in[c] = temp_in_[tix].address<float>();
      if (temp_in[c] != line_buffers[c]) {
        memcpy(temp_in[c], line_buffers[c], sizeof(float) * len);
      }
    }
    size_t last = (len - 1u);
    size_t num = (len / 2);
    for (size_t i = 0; i < num; ++i) {
      for (size_t c = 0; c < out.num_channels_; ++c) {
        std::swap(temp_in[c][i], temp_in[c][last - i]);
      }
    }
    for (size_t c = 0; c < out.num_channels_; ++c) {
      line_buffers[c] = temp_in[c];
    }
    *xstart = width_ - *xstart - len;
  }

  template <typename T>
  void StoreUnsignedRow(const Output& out, const float* input[4], size_t len,
                        T* output, size_t xstart, size_t ypos) const {
    const HWY_FULL(float) d;
    auto mul = Set(d, (1u << (out.bits_per_sample_)) - 1);
    const Rebind<T, decltype(d)> du;
    const size_t padding = RoundUpTo(len, Lanes(d)) - len;
    for (size_t c = 0; c < out.num_channels_; ++c) {
      msan::UnpoisonMemory(input[c] + len, sizeof(input[c][0]) * padding);
    }
    if (out.num_channels_ == 1) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreU(MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul),
               du, &output[i]);
      }
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved2(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul), du,
            &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved3(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul),
            MakeUnsigned<T>(LoadU(d, &input[2][i]), xstart + i, ypos, mul), du,
            &output[3 * i]);
      }
    } else if (out.num_channels_ == 4) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved4(
            MakeUnsigned<T>(LoadU(d, &input[0][i]), xstart + i, ypos, mul),
            MakeUnsigned<T>(LoadU(d, &input[1][i]), xstart + i, ypos, mul),
            MakeUnsigned<T>(LoadU(d, &input[2][i]), xstart + i, ypos, mul),
            MakeUnsigned<T>(LoadU(d, &input[3][i]), xstart + i, ypos, mul), du,
            &output[4 * i]);
      }
    }
    msan::PoisonMemory(output + out.num_channels_ * len,
                       sizeof(output[0]) * out.num_channels_ * padding);
  }

  static void StoreFloat16Row(const Output& out, const float* input[4],
                              size_t len, uint16_t* output) {
    const HWY_FULL(float) d;
    const Rebind<uint16_t, decltype(d)> du;
    const Rebind<hwy::float16_t, decltype(d)> df16;
    const size_t padding = RoundUpTo(len, Lanes(d)) - len;
    for (size_t c = 0; c < out.num_channels_; ++c) {
      msan::UnpoisonMemory(input[c] + len, sizeof(input[c][0]) * padding);
    }
    if (out.num_channels_ == 1) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        StoreU(BitCast(du, DemoteTo(df16, v0)), du, &output[i]);
      }
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        StoreInterleaved2(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)), du, &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        auto v2 = LoadU(d, &input[2][i]);
        StoreInterleaved3(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)),
                          BitCast(du, DemoteTo(df16, v2)), du, &output[3 * i]);
      }
    } else if (out.num_channels_ == 4) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        auto v0 = LoadU(d, &input[0][i]);
        auto v1 = LoadU(d, &input[1][i]);
        auto v2 = LoadU(d, &input[2][i]);
        auto v3 = LoadU(d, &input[3][i]);
        StoreInterleaved4(BitCast(du, DemoteTo(df16, v0)),
                          BitCast(du, DemoteTo(df16, v1)),
                          BitCast(du, DemoteTo(df16, v2)),
                          BitCast(du, DemoteTo(df16, v3)), du, &output[4 * i]);
      }
    }
    msan::PoisonMemory(output + out.num_channels_ * len,
                       sizeof(output[0]) * out.num_channels_ * padding);
  }

  static void StoreFloatRow(const Output& out, const float* input[4],
                            size_t len, float* output) {
    const HWY_FULL(float) d;
    if (out.num_channels_ == 1) {
      memcpy(output, input[0], len * sizeof(output[0]));
    } else if (out.num_channels_ == 2) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved2(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]), d,
                          &output[2 * i]);
      }
    } else if (out.num_channels_ == 3) {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved3(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]),
                          LoadU(d, &input[2][i]), d, &output[3 * i]);
      }
    } else {
      for (size_t i = 0; i < len; i += Lanes(d)) {
        StoreInterleaved4(LoadU(d, &input[0][i]), LoadU(d, &input[1][i]),
                          LoadU(d, &input[2][i]), LoadU(d, &input[3][i]), d,
                          &output[4 * i]);
      }
    }
  }

  template <typename T>
  void WriteToOutput(const Output& out, size_t thread_id, size_t ypos,
                     size_t xstart, size_t len, T* output) const {
    if (transpose_) {
      // TODO(szabadka) Buffer 8x8 chunks and transpose with SIMD.
      if (out.run_opaque_) {
        for (size_t i = 0, j = 0; i < len; ++i, j += out.num_channels_) {
          out.pixel_callback_.run(out.run_opaque_, thread_id, ypos, xstart + i,
                                  1, output + j);
        }
      } else {
        const size_t pixel_stride = out.num_channels_ * sizeof(T);
        const size_t offset = xstart * out.stride_ + ypos * pixel_stride;
        for (size_t i = 0, j = 0; i < len; ++i, j += out.num_channels_) {
          const size_t ix = offset + i * out.stride_;
          JXL_DASSERT(ix + pixel_stride <= out.buffer_size_);
          memcpy(reinterpret_cast<uint8_t*>(out.buffer_) + ix, output + j,
                 pixel_stride);
        }
      }
    } else {
      if (out.run_opaque_) {
        out.pixel_callback_.run(out.run_opaque_, thread_id, xstart, ypos, len,
                                output);
      } else {
        const size_t pixel_stride = out.num_channels_ * sizeof(T);
        const size_t offset = ypos * out.stride_ + xstart * pixel_stride;
        JXL_DASSERT(offset + len * pixel_stride <= out.buffer_size_);
        memcpy(reinterpret_cast<uint8_t*>(out.buffer_) + offset, output,
               len * pixel_stride);
      }
    }
  }

  static constexpr size_t kMaxPixelsPerCall = 1024;
  size_t width_;
  size_t height_;
  Output main_;  // color + alpha
  size_t num_color_;
  bool want_alpha_;
  bool has_alpha_;
  bool unpremul_alpha_;
  size_t alpha_c_;
  bool flip_x_;
  bool flip_y_;
  bool transpose_;
  std::vector<Output> extra_channels_;
  std::vector<float> opaque_alpha_;
  JxlMemoryManager* memory_manager_;
  std::vector<AlignedMemory> temp_in_;
  std::vector<AlignedMemory> temp_out_;
};

#if JXL_CXX_LANG < JXL_CXX_17
constexpr size_t WriteToOutputStage::kMaxPixelsPerCall;
#endif

std::unique_ptr<RenderPipelineStage> GetWriteToOutputStage(
    const ImageOutput& main_output, size_t width, size_t height, bool has_alpha,
    bool unpremul_alpha, size_t alpha_c, Orientation undo_orientation,
    std::vector<ImageOutput>& extra_output, JxlMemoryManager* memory_manager) {
  return jxl::make_unique<WriteToOutputStage>(
      main_output, width, height, has_alpha, unpremul_alpha, alpha_c,
      undo_orientation, extra_output, memory_manager);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {

HWY_EXPORT(GetWriteToOutputStage);

namespace {
class WriteToImageBundleStage : public RenderPipelineStage {
 public:
  explicit WriteToImageBundleStage(
      ImageBundle* image_bundle, const OutputEncodingInfo& output_encoding_info)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        image_bundle_(image_bundle),
        color_encoding_(output_encoding_info.color_encoding) {}

  Status SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) override {
    JxlMemoryManager* memory_manager = image_bundle_->memory_manager();
    JXL_ENSURE(input_sizes.size() >= 3);
    for (size_t c = 1; c < input_sizes.size(); c++) {
      JXL_ENSURE(input_sizes[c].first == input_sizes[0].first);
      JXL_ENSURE(input_sizes[c].second == input_sizes[0].second);
    }
    // TODO(eustas): what should we do in the case of "want only ECs"?
    JXL_ASSIGN_OR_RETURN(Image3F tmp,
                         Image3F::Create(memory_manager, input_sizes[0].first,
                                         input_sizes[0].second));
    JXL_RETURN_IF_ERROR(
        image_bundle_->SetFromImage(std::move(tmp), color_encoding_));
    // TODO(veluca): consider not reallocating ECs if not needed.
    image_bundle_->extra_channels().clear();
    for (size_t c = 3; c < input_sizes.size(); c++) {
      JXL_ASSIGN_OR_RETURN(ImageF ch,
                           ImageF::Create(memory_manager, input_sizes[c].first,
                                          input_sizes[c].second));
      image_bundle_->extra_channels().emplace_back(std::move(ch));
    }
    return true;
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    for (size_t c = 0; c < 3; c++) {
      memcpy(image_bundle_->color()->PlaneRow(c, ypos) + xpos - xextra,
             GetInputRow(input_rows, c, 0) - xextra,
             sizeof(float) * (xsize + 2 * xextra));
    }
    for (size_t ec = 0; ec < image_bundle_->extra_channels().size(); ec++) {
      JXL_ENSURE(image_bundle_->extra_channels()[ec].xsize() >=
                 xpos + xsize + xextra);
      memcpy(image_bundle_->extra_channels()[ec].Row(ypos) + xpos - xextra,
             GetInputRow(input_rows, 3 + ec, 0) - xextra,
             sizeof(float) * (xsize + 2 * xextra));
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInput;
  }

  const char* GetName() const override { return "WriteIB"; }

 private:
  ImageBundle* image_bundle_;
  ColorEncoding color_encoding_;
};

class WriteToImage3FStage : public RenderPipelineStage {
 public:
  WriteToImage3FStage(JxlMemoryManager* memory_manager, Image3F* image)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        memory_manager_(memory_manager),
        image_(image) {}

  Status SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) override {
    JXL_ENSURE(input_sizes.size() >= 3);
    for (size_t c = 1; c < 3; ++c) {
      JXL_ENSURE(input_sizes[c].first == input_sizes[0].first);
      JXL_ENSURE(input_sizes[c].second == input_sizes[0].second);
    }
    JXL_ASSIGN_OR_RETURN(*image_,
                         Image3F::Create(memory_manager_, input_sizes[0].first,
                                         input_sizes[0].second));
    return true;
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    for (size_t c = 0; c < 3; c++) {
      memcpy(image_->PlaneRow(c, ypos) + xpos - xextra,
             GetInputRow(input_rows, c, 0) - xextra,
             sizeof(float) * (xsize + 2 * xextra));
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInput
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "WriteI3F"; }

 private:
  JxlMemoryManager* memory_manager_;
  Image3F* image_;
};

}  // namespace

std::unique_ptr<RenderPipelineStage> GetWriteToImageBundleStage(
    ImageBundle* image_bundle, const OutputEncodingInfo& output_encoding_info) {
  return jxl::make_unique<WriteToImageBundleStage>(image_bundle,
                                                   output_encoding_info);
}

std::unique_ptr<RenderPipelineStage> GetWriteToImage3FStage(
    JxlMemoryManager* memory_manager, Image3F* image) {
  return jxl::make_unique<WriteToImage3FStage>(memory_manager, image);
}

std::unique_ptr<RenderPipelineStage> GetWriteToOutputStage(
    const ImageOutput& main_output, size_t width, size_t height, bool has_alpha,
    bool unpremul_alpha, size_t alpha_c, Orientation undo_orientation,
    std::vector<ImageOutput>& extra_output, JxlMemoryManager* memory_manager) {
  return HWY_DYNAMIC_DISPATCH(GetWriteToOutputStage)(
      main_output, width, height, has_alpha, unpremul_alpha, alpha_c,
      undo_orientation, extra_output, memory_manager);
}

}  // namespace jxl

#endif
