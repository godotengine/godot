// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_patches.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {
namespace {
class PatchDictionaryStage : public RenderPipelineStage {
 public:
  PatchDictionaryStage(const PatchDictionary* patches,
                       const std::vector<ExtraChannelInfo>* extra_channel_info)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        patches_(*patches),
        extra_channel_info_(extra_channel_info) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    JXL_ENSURE(xpos == 0 || xpos >= xextra);
    size_t x0 = xpos ? xpos - xextra : 0;
    size_t num_channels = 3 + extra_channel_info_->size();
    std::vector<float*> row_ptrs(num_channels);
    for (size_t i = 0; i < num_channels; i++) {
      row_ptrs[i] = GetInputRow(input_rows, i, 0) + x0 - xpos;
    }
    return patches_.AddOneRow(row_ptrs.data(), ypos, x0,
                              xsize + xextra + xpos - x0, *extra_channel_info_);
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    size_t num_channels = 3 + extra_channel_info_->size();
    return c < num_channels ? RenderPipelineChannelMode::kInPlace
                            : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "Patches"; }

 private:
  const PatchDictionary& patches_;
  const std::vector<ExtraChannelInfo>* extra_channel_info_;
};
}  // namespace

std::unique_ptr<RenderPipelineStage> GetPatchesStage(
    const PatchDictionary* patches,
    const std::vector<ExtraChannelInfo>* extra_channel_info) {
  return jxl::make_unique<PatchDictionaryStage>(patches, extra_channel_info);
}

}  // namespace jxl
