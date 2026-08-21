// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_patch_dictionary.h"

#include <jxl/memory_manager.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/blending.h"
#include "lib/jxl/common.h"  // kMaxNumReferenceFrames
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/pack_signed.h"
#include "lib/jxl/patch_dictionary_internal.h"

namespace jxl {

Status PatchDictionary::Decode(JxlMemoryManager* memory_manager, BitReader* br,
                               size_t xsize, size_t ysize,
                               size_t num_extra_channels,
                               bool* uses_extra_channels) {
  positions_.clear();
  blendings_stride_ = num_extra_channels + 1;
  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(DecodeHistograms(
      memory_manager, br, kNumPatchDictionaryContexts, &code, &context_map));
  JXL_ASSIGN_OR_RETURN(ANSSymbolReader decoder,
                       ANSSymbolReader::Create(&code, br));

  auto read_num = [&](size_t context) {
    size_t r = decoder.ReadHybridUint(context, br, context_map);
    return r;
  };

  size_t num_ref_patch = read_num(kNumRefPatchContext);
  // Limit max memory usage of patches to about 66 bytes per pixel (assuming 8
  // bytes per size_t)
  const size_t num_pixels = xsize * ysize;
  const size_t max_ref_patches = 1024 + num_pixels / 4;
  const size_t max_patches = max_ref_patches * 4;
  const size_t max_blending_infos = max_patches * 4;
  if (num_ref_patch > max_ref_patches) {
    return JXL_FAILURE("Too many patches in dictionary");
  }

  size_t total_patches = 0;
  size_t next_size = 1;

  for (size_t id = 0; id < num_ref_patch; id++) {
    PatchReferencePosition ref_pos;
    ref_pos.ref = read_num(kReferenceFrameContext);
    if (ref_pos.ref >= kMaxNumReferenceFrames ||
        reference_frames_->at(ref_pos.ref).frame->xsize() == 0) {
      return JXL_FAILURE("Invalid reference frame ID");
    }
    if (!reference_frames_->at(ref_pos.ref).ib_is_in_xyb) {
      return JXL_FAILURE(
          "Patches cannot use frames saved post color transforms");
    }
    const ImageBundle& ib = *reference_frames_->at(ref_pos.ref).frame;
    ref_pos.x0 = read_num(kPatchReferencePositionContext);
    ref_pos.y0 = read_num(kPatchReferencePositionContext);
    ref_pos.xsize = read_num(kPatchSizeContext) + 1;
    ref_pos.ysize = read_num(kPatchSizeContext) + 1;
    if (ref_pos.x0 + ref_pos.xsize > ib.xsize()) {
      return JXL_FAILURE("Invalid position specified in reference frame");
    }
    if (ref_pos.y0 + ref_pos.ysize > ib.ysize()) {
      return JXL_FAILURE("Invalid position specified in reference frame");
    }
    size_t id_count = read_num(kPatchCountContext);
    if (id_count > max_patches) {
      return JXL_FAILURE("Too many patches in dictionary");
    }
    id_count++;
    total_patches += id_count;
    if (total_patches > max_patches) {
      return JXL_FAILURE("Too many patches in dictionary");
    }
    if (next_size < total_patches) {
      next_size *= 2;
      next_size = std::min<size_t>(next_size, max_patches);
    }
    if (next_size * blendings_stride_ > max_blending_infos) {
      return JXL_FAILURE("Too many patches in dictionary");
    }
    positions_.reserve(next_size);
    blendings_.reserve(next_size * blendings_stride_);
    bool choose_alpha = (num_extra_channels > 1);
    for (size_t i = 0; i < id_count; i++) {
      PatchPosition pos;
      pos.ref_pos_idx = ref_positions_.size();
      if (i == 0) {
        pos.x = read_num(kPatchPositionContext);
        pos.y = read_num(kPatchPositionContext);
      } else {
        ssize_t deltax = UnpackSigned(read_num(kPatchOffsetContext));
        if (deltax < 0 && static_cast<size_t>(-deltax) > positions_.back().x) {
          return JXL_FAILURE("Invalid patch: negative x coordinate (%" PRIuS
                             " base x %" PRIdS " delta x)",
                             positions_.back().x, deltax);
        }
        pos.x = positions_.back().x + deltax;
        ssize_t deltay = UnpackSigned(read_num(kPatchOffsetContext));
        if (deltay < 0 && static_cast<size_t>(-deltay) > positions_.back().y) {
          return JXL_FAILURE("Invalid patch: negative y coordinate (%" PRIuS
                             " base y %" PRIdS " delta y)",
                             positions_.back().y, deltay);
        }
        pos.y = positions_.back().y + deltay;
      }
      if (pos.x + ref_pos.xsize > xsize) {
        return JXL_FAILURE("Invalid patch x: at %" PRIuS " + %" PRIuS
                           " > %" PRIuS,
                           pos.x, ref_pos.xsize, xsize);
      }
      if (pos.y + ref_pos.ysize > ysize) {
        return JXL_FAILURE("Invalid patch y: at %" PRIuS " + %" PRIuS
                           " > %" PRIuS,
                           pos.y, ref_pos.ysize, ysize);
      }
      for (size_t j = 0; j < blendings_stride_; j++) {
        uint32_t blend_mode = read_num(kPatchBlendModeContext);
        if (blend_mode >= kNumPatchBlendModes) {
          return JXL_FAILURE("Invalid patch blend mode: %u", blend_mode);
        }
        PatchBlending info;
        info.mode = static_cast<PatchBlendMode>(blend_mode);
        if (UsesAlpha(info.mode)) {
          *uses_extra_channels = true;
        }
        if (info.mode != PatchBlendMode::kNone && j > 0) {
          *uses_extra_channels = true;
        }
        if (UsesAlpha(info.mode) && choose_alpha) {
          info.alpha_channel = read_num(kPatchAlphaChannelContext);
          if (info.alpha_channel >= num_extra_channels) {
            return JXL_FAILURE(
                "Invalid alpha channel for blending: %u out of %u\n",
                info.alpha_channel, static_cast<uint32_t>(num_extra_channels));
          }
        } else {
          info.alpha_channel = 0;
        }
        if (UsesClamp(info.mode)) {
          info.clamp = static_cast<bool>(read_num(kPatchClampContext));
        } else {
          info.clamp = false;
        }
        blendings_.push_back(info);
      }
      positions_.emplace_back(pos);
    }
    ref_positions_.emplace_back(ref_pos);
  }
  positions_.shrink_to_fit();

  if (!decoder.CheckANSFinalState()) {
    return JXL_FAILURE("ANS checksum failure.");
  }

  ComputePatchTree();
  return true;
}

int PatchDictionary::GetReferences() const {
  int result = 0;
  for (const auto& ref_pos : ref_positions_) {
    result |= (1 << static_cast<int>(ref_pos.ref));
  }
  return result;
}

namespace {
struct PatchInterval {
  size_t idx;
  size_t y0, y1;
};
}  // namespace

void PatchDictionary::ComputePatchTree() {
  patch_tree_.clear();
  num_patches_.clear();
  sorted_patches_y0_.clear();
  sorted_patches_y1_.clear();
  if (positions_.empty()) {
    return;
  }
  // Create a y-interval for each patch.
  std::vector<PatchInterval> intervals(positions_.size());
  for (size_t i = 0; i < positions_.size(); ++i) {
    const auto& pos = positions_[i];
    intervals[i].idx = i;
    intervals[i].y0 = pos.y;
    intervals[i].y1 = pos.y + ref_positions_[pos.ref_pos_idx].ysize;
  }
  auto sort_by_y0 = [&intervals](size_t start, size_t end) {
    std::sort(intervals.data() + start, intervals.data() + end,
              [](const PatchInterval& i0, const PatchInterval& i1) {
                return i0.y0 < i1.y0;
              });
  };
  auto sort_by_y1 = [&intervals](size_t start, size_t end) {
    std::sort(intervals.data() + start, intervals.data() + end,
              [](const PatchInterval& i0, const PatchInterval& i1) {
                return i0.y1 < i1.y1;
              });
  };
  // Count the number of patches for each row.
  sort_by_y1(0, intervals.size());
  num_patches_.resize(intervals.back().y1);
  for (auto iv : intervals) {
    for (size_t y = iv.y0; y < iv.y1; ++y) num_patches_[y]++;
  }
  PatchTreeNode root;
  root.start = 0;
  root.num = intervals.size();
  patch_tree_.push_back(root);
  size_t next = 0;
  while (next < patch_tree_.size()) {
    auto& node = patch_tree_[next];
    size_t start = node.start;
    size_t end = node.start + node.num;
    // Choose the y_center for this node to be the median of interval starts.
    sort_by_y0(start, end);
    size_t middle_idx = start + node.num / 2;
    node.y_center = intervals[middle_idx].y0;
    // Divide the intervals in [start, end) into three groups:
    //   * those completely to the right of y_center: [right_start, end)
    //   * those overlapping y_center: [left_end, right_start)
    //   * those completely to the left of y_center: [start, left_end)
    size_t right_start = middle_idx;
    while (right_start < end && intervals[right_start].y0 == node.y_center) {
      ++right_start;
    }
    sort_by_y1(start, right_start);
    size_t left_end = right_start;
    while (left_end > start && intervals[left_end - 1].y1 > node.y_center) {
      --left_end;
    }
    // Fill in sorted_patches_y0_ and sorted_patches_y1_ for the current node.
    node.num = right_start - left_end;
    node.start = sorted_patches_y0_.size();
    for (ssize_t i = static_cast<ssize_t>(right_start) - 1;
         i >= static_cast<ssize_t>(left_end); --i) {
      sorted_patches_y1_.emplace_back(intervals[i].y1, intervals[i].idx);
    }
    sort_by_y0(left_end, right_start);
    for (size_t i = left_end; i < right_start; ++i) {
      sorted_patches_y0_.emplace_back(intervals[i].y0, intervals[i].idx);
    }
    // Create the left and right nodes (if not empty).
    node.left_child = node.right_child = -1;
    if (left_end > start) {
      PatchTreeNode left;
      left.start = start;
      left.num = left_end - left.start;
      patch_tree_[next].left_child = patch_tree_.size();
      patch_tree_.push_back(left);
    }
    if (right_start < end) {
      PatchTreeNode right;
      right.start = right_start;
      right.num = end - right.start;
      patch_tree_[next].right_child = patch_tree_.size();
      patch_tree_.push_back(right);
    }
    ++next;
  }
}

std::vector<size_t> PatchDictionary::GetPatchesForRow(size_t y) const {
  std::vector<size_t> result;
  if (y < num_patches_.size() && num_patches_[y] > 0) {
    result.reserve(num_patches_[y]);
    for (ssize_t tree_idx = 0; tree_idx != -1;) {
      JXL_DASSERT(tree_idx < static_cast<ssize_t>(patch_tree_.size()));
      const auto& node = patch_tree_[tree_idx];
      if (y <= node.y_center) {
        for (size_t i = 0; i < node.num; ++i) {
          const auto& p = sorted_patches_y0_[node.start + i];
          if (y < p.first) break;
          result.push_back(p.second);
        }
        tree_idx = y < node.y_center ? node.left_child : -1;
      } else {
        for (size_t i = 0; i < node.num; ++i) {
          const auto& p = sorted_patches_y1_[node.start + i];
          if (y >= p.first) break;
          result.push_back(p.second);
        }
        tree_idx = node.right_child;
      }
    }
    // Ensure that he relative order of patches that affect the same pixels is
    // preserved. This is important for patches that have a blend mode
    // different from kAdd.
    std::sort(result.begin(), result.end());
  }
  return result;
}

// Adds patches to a segment of `xsize` pixels, starting at `inout`, assumed
// to be located at position (x0, y) in the frame.
Status PatchDictionary::AddOneRow(
    float* const* inout, size_t y, size_t x0, size_t xsize,
    const std::vector<ExtraChannelInfo>& extra_channel_info) const {
  size_t num_ec = extra_channel_info.size();
  JXL_ENSURE(num_ec + 1 <= blendings_stride_);
  std::vector<const float*> fg_ptrs(3 + num_ec);
  for (size_t pos_idx : GetPatchesForRow(y)) {
    const size_t blending_idx = pos_idx * blendings_stride_;
    const PatchPosition& pos = positions_[pos_idx];
    const PatchReferencePosition& ref_pos = ref_positions_[pos.ref_pos_idx];
    size_t by = pos.y;
    size_t bx = pos.x;
    size_t patch_xsize = ref_pos.xsize;
    JXL_ENSURE(y >= by);
    JXL_ENSURE(y < by + ref_pos.ysize);
    size_t iy = y - by;
    size_t ref = ref_pos.ref;
    if (bx >= x0 + xsize) continue;
    if (bx + patch_xsize < x0) continue;
    size_t patch_x0 = std::max(bx, x0);
    size_t patch_x1 = std::min(bx + patch_xsize, x0 + xsize);
    for (size_t c = 0; c < 3; c++) {
      fg_ptrs[c] = reference_frames_->at(ref).frame->color()->ConstPlaneRow(
                       c, ref_pos.y0 + iy) +
                   ref_pos.x0 + x0 - bx;
    }
    for (size_t i = 0; i < num_ec; i++) {
      fg_ptrs[3 + i] =
          reference_frames_->at(ref).frame->extra_channels()[i].ConstRow(
              ref_pos.y0 + iy) +
          ref_pos.x0 + x0 - bx;
    }
    JXL_RETURN_IF_ERROR(PerformBlending(
        memory_manager_, inout, fg_ptrs.data(), inout, patch_x0 - x0,
        patch_x1 - patch_x0, blendings_[blending_idx],
        blendings_.data() + blending_idx + 1, extra_channel_info));
  }
  return true;
}
}  // namespace jxl
