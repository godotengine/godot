// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_spot.h"

#include <cstddef>
#include <memory>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {
class SpotColorStage : public RenderPipelineStage {
 public:
  explicit SpotColorStage(size_t spot_c_offset, const float* spot_color)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        spot_c_(3 + spot_c_offset),
        spot_color_(spot_color) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    // TODO(veluca): add SIMD.
    float scale = spot_color_[3];
    for (size_t c = 0; c < 3; c++) {
      float* JXL_RESTRICT p = GetInputRow(input_rows, c, 0);
      const float* JXL_RESTRICT s = GetInputRow(input_rows, spot_c_, 0);
      for (ssize_t x = -xextra; x < static_cast<ssize_t>(xsize + xextra); x++) {
        float mix = scale * s[x];
        p[x] = mix * spot_color_[c] + (1.0f - mix) * p[x];
      }
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3          ? RenderPipelineChannelMode::kInPlace
           : c == spot_c_ ? RenderPipelineChannelMode::kInput
                          : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "Spot"; }

 private:
  size_t spot_c_;
  const float* spot_color_;
};

std::unique_ptr<RenderPipelineStage> GetSpotColorStage(
    size_t spot_c_offset, const float* spot_color) {
  return jxl::make_unique<SpotColorStage>(spot_c_offset, spot_color);
}

}  // namespace jxl
