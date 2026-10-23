// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/blending.h"

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstring>
#include <vector>

#include "lib/jxl/alpha.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {

bool NeedsBlending(const FrameHeader& frame_header) {
  if (!(frame_header.frame_type == FrameType::kRegularFrame ||
        frame_header.frame_type == FrameType::kSkipProgressive)) {
    return false;
  }
  const auto& info = frame_header.blending_info;
  bool replace_all = (info.mode == BlendMode::kReplace);
  for (const auto& ec_i : frame_header.extra_channel_blending_info) {
    if (ec_i.mode != BlendMode::kReplace) {
      replace_all = false;
    }
  }
  // Replace the full frame: nothing to do.
  if (!frame_header.custom_size_or_origin && replace_all) {
    return false;
  }
  return true;
}

Status PerformBlending(
    JxlMemoryManager* memory_manager, const float* const* bg,
    const float* const* fg, float* const* out, size_t x0, size_t xsize,
    const PatchBlending& color_blending, const PatchBlending* ec_blending,
    const std::vector<ExtraChannelInfo>& extra_channel_info) {
  bool has_alpha = false;
  size_t num_ec = extra_channel_info.size();
  for (size_t i = 0; i < num_ec; i++) {
    if (extra_channel_info[i].type == jxl::ExtraChannel::kAlpha) {
      has_alpha = true;
      break;
    }
  }
  JXL_ASSIGN_OR_RETURN(ImageF tmp,
                       ImageF::Create(memory_manager, xsize, 3 + num_ec));
  // Blend extra channels first so that we use the pre-blending alpha.
  for (size_t i = 0; i < num_ec; i++) {
    switch (ec_blending[i].mode) {
      case PatchBlendMode::kAdd:
        for (size_t x = 0; x < xsize; x++) {
          tmp.Row(3 + i)[x] = bg[3 + i][x + x0] + fg[3 + i][x + x0];
        }
        continue;

      case PatchBlendMode::kBlendAbove: {
        size_t alpha = ec_blending[i].alpha_channel;
        bool is_premultiplied = extra_channel_info[alpha].alpha_associated;
        PerformAlphaBlending(bg[3 + i] + x0, bg[3 + alpha] + x0, fg[3 + i] + x0,
                             fg[3 + alpha] + x0, tmp.Row(3 + i), xsize,
                             is_premultiplied, ec_blending[i].clamp);
        continue;
      }

      case PatchBlendMode::kBlendBelow: {
        size_t alpha = ec_blending[i].alpha_channel;
        bool is_premultiplied = extra_channel_info[alpha].alpha_associated;
        PerformAlphaBlending(fg[3 + i] + x0, fg[3 + alpha] + x0, bg[3 + i] + x0,
                             bg[3 + alpha] + x0, tmp.Row(3 + i), xsize,
                             is_premultiplied, ec_blending[i].clamp);
        continue;
      }

      case PatchBlendMode::kAlphaWeightedAddAbove: {
        size_t alpha = ec_blending[i].alpha_channel;
        PerformAlphaWeightedAdd(bg[3 + i] + x0, fg[3 + i] + x0,
                                fg[3 + alpha] + x0, tmp.Row(3 + i), xsize,
                                ec_blending[i].clamp);
        continue;
      }

      case PatchBlendMode::kAlphaWeightedAddBelow: {
        size_t alpha = ec_blending[i].alpha_channel;
        PerformAlphaWeightedAdd(fg[3 + i] + x0, bg[3 + i] + x0,
                                bg[3 + alpha] + x0, tmp.Row(3 + i), xsize,
                                ec_blending[i].clamp);
        continue;
      }

      case PatchBlendMode::kMul:
        PerformMulBlending(bg[3 + i] + x0, fg[3 + i] + x0, tmp.Row(3 + i),
                           xsize, ec_blending[i].clamp);
        continue;

      case PatchBlendMode::kReplace:
        if (xsize) memcpy(tmp.Row(3 + i), fg[3 + i] + x0, xsize * sizeof(**fg));
        continue;

      case PatchBlendMode::kNone:
        if (xsize) memcpy(tmp.Row(3 + i), bg[3 + i] + x0, xsize * sizeof(**fg));
        continue;
    }
  }
  size_t alpha = color_blending.alpha_channel;

  const auto add = [&]() {
    for (int p = 0; p < 3; p++) {
      float* out = tmp.Row(p);
      for (size_t x = 0; x < xsize; x++) {
        out[x] = bg[p][x + x0] + fg[p][x + x0];
      }
    }
  };

  const auto blend_weighted = [&](const float* const* bottom,
                                  const float* const* top) {
    bool is_premultiplied = extra_channel_info[alpha].alpha_associated;
    PerformAlphaBlending(
        {bottom[0] + x0, bottom[1] + x0, bottom[2] + x0,
         bottom[3 + alpha] + x0},
        {top[0] + x0, top[1] + x0, top[2] + x0, top[3 + alpha] + x0},
        {tmp.Row(0), tmp.Row(1), tmp.Row(2), tmp.Row(3 + alpha)}, xsize,
        is_premultiplied, color_blending.clamp);
  };

  const auto add_weighted = [&](const float* const* bottom,
                                const float* const* top) {
    for (size_t c = 0; c < 3; c++) {
      PerformAlphaWeightedAdd(bottom[c] + x0, top[c] + x0, top[3 + alpha] + x0,
                              tmp.Row(c), xsize, color_blending.clamp);
    }
  };

  const auto copy = [&](const float* const* src) {
    for (size_t p = 0; p < 3; p++) {
      memcpy(tmp.Row(p), src[p] + x0, xsize * sizeof(**src));
    }
  };

  switch (color_blending.mode) {
    case PatchBlendMode::kAdd:
      add();
      break;

    case PatchBlendMode::kAlphaWeightedAddAbove:
      has_alpha ? add_weighted(bg, fg) : add();
      break;

    case PatchBlendMode::kAlphaWeightedAddBelow:
      has_alpha ? add_weighted(fg, bg) : add();
      break;

    case PatchBlendMode::kBlendAbove:
      has_alpha ? blend_weighted(bg, fg) : copy(fg);
      break;

    case PatchBlendMode::kBlendBelow:
      has_alpha ? blend_weighted(fg, bg) : copy(fg);
      break;

    case PatchBlendMode::kMul:
      for (int p = 0; p < 3; p++) {
        PerformMulBlending(bg[p] + x0, fg[p] + x0, tmp.Row(p), xsize,
                           color_blending.clamp);
      }
      break;

    case PatchBlendMode::kReplace:
      copy(fg);
      break;

    case PatchBlendMode::kNone:
      copy(bg);
  }

  for (size_t i = 0; i < 3 + num_ec; i++) {
    if (xsize != 0) memcpy(out[i] + x0, tmp.Row(i), xsize * sizeof(**out));
  }
  return true;
}

}  // namespace jxl
