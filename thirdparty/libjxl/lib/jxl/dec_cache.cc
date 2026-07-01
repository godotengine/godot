// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_cache.h"

#include <jxl/memory_manager.h>

#include <algorithm>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/blending.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/common.h"  // JXL_HIGH_PRECISION
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/render_pipeline/stage_blending.h"
#include "lib/jxl/render_pipeline/stage_chroma_upsampling.h"
#include "lib/jxl/render_pipeline/stage_cms.h"
#include "lib/jxl/render_pipeline/stage_epf.h"
#include "lib/jxl/render_pipeline/stage_from_linear.h"
#include "lib/jxl/render_pipeline/stage_gaborish.h"
#include "lib/jxl/render_pipeline/stage_noise.h"
#include "lib/jxl/render_pipeline/stage_patches.h"
#include "lib/jxl/render_pipeline/stage_splines.h"
#include "lib/jxl/render_pipeline/stage_spot.h"
#include "lib/jxl/render_pipeline/stage_to_linear.h"
#include "lib/jxl/render_pipeline/stage_tone_mapping.h"
#include "lib/jxl/render_pipeline/stage_upsampling.h"
#include "lib/jxl/render_pipeline/stage_write.h"
#include "lib/jxl/render_pipeline/stage_xyb.h"
#include "lib/jxl/render_pipeline/stage_ycbcr.h"

namespace jxl {

Status GroupDecCache::InitOnce(JxlMemoryManager* memory_manager,
                               size_t num_passes, size_t used_acs) {
  for (size_t i = 0; i < num_passes; i++) {
    if (num_nzeroes[i].xsize() == 0) {
      // Allocate enough for a whole group - partial groups on the
      // right/bottom border just use a subset. The valid size is passed via
      // Rect.

      JXL_ASSIGN_OR_RETURN(num_nzeroes[i],
                           Image3I::Create(memory_manager, kGroupDimInBlocks,
                                           kGroupDimInBlocks));
    }
  }
  size_t max_block_area = 0;

  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    if ((used_acs & (1 << o)) == 0) continue;
    size_t area =
        acs.covered_blocks_x() * acs.covered_blocks_y() * kDCTBlockSize;
    max_block_area = std::max(area, max_block_area);
  }

  if (max_block_area > max_block_area_) {
    max_block_area_ = max_block_area;
    // We need 3x float blocks for dequantized coefficients and 1x for scratch
    // space for transforms.
    JXL_ASSIGN_OR_RETURN(
        float_memory_,
        AlignedMemory::Create(memory_manager,
                              max_block_area_ * 7 * sizeof(float)));
    // We need 3x int32 or int16 blocks for quantized coefficients.
    JXL_ASSIGN_OR_RETURN(
        int32_memory_,
        AlignedMemory::Create(memory_manager,
                              max_block_area_ * 3 * sizeof(int32_t)));
    JXL_ASSIGN_OR_RETURN(
        int16_memory_,
        AlignedMemory::Create(memory_manager,
                              max_block_area_ * 3 * sizeof(int16_t)));
  }

  dec_group_block = float_memory_.address<float>();
  scratch_space = dec_group_block + max_block_area_ * 3;
  dec_group_qblock = int32_memory_.address<int32_t>();
  dec_group_qblock16 = int16_memory_.address<int16_t>();
  return true;
}

// Initialize the decoder state after all of DC is decoded.
Status PassesDecoderState::InitForAC(size_t num_passes, ThreadPool* pool) {
  shared_storage.coeff_order_size = 0;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    if (((1 << o) & used_acs) == 0) continue;
    uint8_t ord = kStrategyOrder[o];
    shared_storage.coeff_order_size =
        std::max(kCoeffOrderOffset[3 * (ord + 1)] * kDCTBlockSize,
                 shared_storage.coeff_order_size);
  }
  size_t sz = num_passes * shared_storage.coeff_order_size;
  if (sz > shared_storage.coeff_orders.size()) {
    shared_storage.coeff_orders.resize(sz);
  }
  return true;
}

Status PassesDecoderState::PreparePipeline(const FrameHeader& frame_header,
                                           const ImageMetadata* metadata,
                                           ImageBundle* decoded,
                                           PipelineOptions options) {
  JxlMemoryManager* memory_manager = this->memory_manager();
  size_t num_c = 3 + frame_header.nonserialized_metadata->m.num_extra_channels;
  bool render_noise =
      (options.render_noise && (frame_header.flags & FrameHeader::kNoise) != 0);
  size_t num_tmp_c = render_noise ? 3 : 0;

  if (frame_header.CanBeReferenced()) {
    // Necessary so that SetInputSizes() can allocate output buffers as needed.
    frame_storage_for_referencing = ImageBundle(memory_manager, metadata);
  }

  RenderPipeline::Builder builder(memory_manager, num_c + num_tmp_c);

  if (options.use_slow_render_pipeline) {
    builder.UseSimpleImplementation();
  }

  if (!frame_header.chroma_subsampling.Is444()) {
    for (size_t c = 0; c < 3; c++) {
      if (frame_header.chroma_subsampling.HShift(c) != 0) {
        JXL_RETURN_IF_ERROR(
            builder.AddStage(GetChromaUpsamplingStage(c, /*horizontal=*/true)));
      }
      if (frame_header.chroma_subsampling.VShift(c) != 0) {
        JXL_RETURN_IF_ERROR(builder.AddStage(
            GetChromaUpsamplingStage(c, /*horizontal=*/false)));
      }
    }
  }

  if (frame_header.loop_filter.gab) {
    JXL_RETURN_IF_ERROR(
        builder.AddStage(GetGaborishStage(frame_header.loop_filter)));
  }

  {
    const LoopFilter& lf = frame_header.loop_filter;
    if (lf.epf_iters >= 3) {
      JXL_RETURN_IF_ERROR(
          builder.AddStage(GetEPFStage(lf, sigma, EpfStage::Zero)));
    }
    if (lf.epf_iters >= 1) {
      JXL_RETURN_IF_ERROR(
          builder.AddStage(GetEPFStage(lf, sigma, EpfStage::One)));
    }
    if (lf.epf_iters >= 2) {
      JXL_RETURN_IF_ERROR(
          builder.AddStage(GetEPFStage(lf, sigma, EpfStage::Two)));
    }
  }

  bool late_ec_upsample = frame_header.upsampling != 1;
  for (auto ecups : frame_header.extra_channel_upsampling) {
    if (ecups != frame_header.upsampling) {
      // If patches are applied, either frame_header.upsampling == 1 or
      // late_ec_upsample is true.
      late_ec_upsample = false;
    }
  }

  if (!late_ec_upsample) {
    for (size_t ec = 0; ec < frame_header.extra_channel_upsampling.size();
         ec++) {
      if (frame_header.extra_channel_upsampling[ec] != 1) {
        JXL_RETURN_IF_ERROR(builder.AddStage(GetUpsamplingStage(
            frame_header.nonserialized_metadata->transform_data, 3 + ec,
            CeilLog2Nonzero(frame_header.extra_channel_upsampling[ec]))));
      }
    }
  }

  if ((frame_header.flags & FrameHeader::kPatches) != 0) {
    JXL_RETURN_IF_ERROR(builder.AddStage(GetPatchesStage(
        &shared->image_features.patches,
        &frame_header.nonserialized_metadata->m.extra_channel_info)));
  }
  if ((frame_header.flags & FrameHeader::kSplines) != 0) {
    JXL_RETURN_IF_ERROR(
        builder.AddStage(GetSplineStage(&shared->image_features.splines)));
  }

  if (frame_header.upsampling != 1) {
    size_t nb_channels =
        3 +
        (late_ec_upsample ? frame_header.extra_channel_upsampling.size() : 0);
    for (size_t c = 0; c < nb_channels; c++) {
      JXL_RETURN_IF_ERROR(builder.AddStage(GetUpsamplingStage(
          frame_header.nonserialized_metadata->transform_data, c,
          CeilLog2Nonzero(frame_header.upsampling))));
    }
  }
  if (render_noise) {
    JXL_RETURN_IF_ERROR(builder.AddStage(GetConvolveNoiseStage(num_c)));
    JXL_RETURN_IF_ERROR(builder.AddStage(GetAddNoiseStage(
        shared->image_features.noise_params, shared->cmap.base(), num_c)));
  }
  if (frame_header.dc_level != 0) {
    JXL_RETURN_IF_ERROR(builder.AddStage(GetWriteToImage3FStage(
        memory_manager, &shared_storage.dc_frames[frame_header.dc_level - 1])));
  }

  if (frame_header.CanBeReferenced() &&
      frame_header.save_before_color_transform) {
    JXL_RETURN_IF_ERROR(builder.AddStage(GetWriteToImageBundleStage(
        &frame_storage_for_referencing, output_encoding_info)));
  }

  bool has_alpha = false;
  size_t alpha_c = 0;
  for (size_t i = 0; i < metadata->extra_channel_info.size(); i++) {
    if (metadata->extra_channel_info[i].type == ExtraChannel::kAlpha) {
      has_alpha = true;
      alpha_c = 3 + i;
      break;
    }
  }

  if (fast_xyb_srgb8_conversion) {
#if !JXL_HIGH_PRECISION
    JXL_ENSURE(!NeedsBlending(frame_header));
    JXL_ENSURE(!frame_header.CanBeReferenced() ||
               frame_header.save_before_color_transform);
    JXL_ENSURE(!options.render_spotcolors ||
               !metadata->Find(ExtraChannel::kSpotColor));
    bool is_rgba = (main_output.format.num_channels == 4);
    uint8_t* rgb_output = reinterpret_cast<uint8_t*>(main_output.buffer);
    JXL_RETURN_IF_ERROR(builder.AddStage(
        GetFastXYBTosRGB8Stage(rgb_output, main_output.stride, width, height,
                               is_rgba, has_alpha, alpha_c)));
#endif
  } else {
    bool linear = false;
    if (frame_header.color_transform == ColorTransform::kYCbCr) {
      JXL_RETURN_IF_ERROR(builder.AddStage(GetYCbCrStage()));
    } else if (frame_header.color_transform == ColorTransform::kXYB) {
      JXL_RETURN_IF_ERROR(builder.AddStage(GetXYBStage(output_encoding_info)));
      if (output_encoding_info.color_encoding.GetColorSpace() !=
          ColorSpace::kXYB) {
        linear = true;
      }
    }  // Nothing to do for kNone.

    if (options.coalescing && NeedsBlending(frame_header)) {
      if (linear) {
        JXL_RETURN_IF_ERROR(
            builder.AddStage(GetFromLinearStage(output_encoding_info)));
        linear = false;
      }
      JXL_RETURN_IF_ERROR(builder.AddStage(GetBlendingStage(
          frame_header, this, output_encoding_info.color_encoding)));
    }

    if (options.coalescing && frame_header.CanBeReferenced() &&
        !frame_header.save_before_color_transform) {
      if (linear) {
        JXL_RETURN_IF_ERROR(
            builder.AddStage(GetFromLinearStage(output_encoding_info)));
        linear = false;
      }
      JXL_RETURN_IF_ERROR(builder.AddStage(GetWriteToImageBundleStage(
          &frame_storage_for_referencing, output_encoding_info)));
    }

    if (options.render_spotcolors &&
        frame_header.nonserialized_metadata->m.Find(ExtraChannel::kSpotColor)) {
      for (size_t i = 0; i < metadata->extra_channel_info.size(); i++) {
        // Don't use Find() because there may be multiple spot color channels.
        const ExtraChannelInfo& eci = metadata->extra_channel_info[i];
        if (eci.type == ExtraChannel::kSpotColor) {
          JXL_RETURN_IF_ERROR(
              builder.AddStage(GetSpotColorStage(i, eci.spot_color)));
        }
      }
    }

    auto tone_mapping_stage = GetToneMappingStage(output_encoding_info);
    if (tone_mapping_stage) {
      if (!linear) {
        auto to_linear_stage = GetToLinearStage(output_encoding_info);
        if (!to_linear_stage) {
          if (!output_encoding_info.cms_set) {
            return JXL_FAILURE("Cannot tonemap this colorspace without a CMS");
          }
          auto cms_stage = GetCmsStage(output_encoding_info);
          if (cms_stage) {
            JXL_RETURN_IF_ERROR(builder.AddStage(std::move(cms_stage)));
          }
        } else {
          JXL_RETURN_IF_ERROR(builder.AddStage(std::move(to_linear_stage)));
        }
        linear = true;
      }
      JXL_RETURN_IF_ERROR(builder.AddStage(std::move(tone_mapping_stage)));
    }

    if (linear) {
      const size_t channels_src =
          (output_encoding_info.orig_color_encoding.IsCMYK()
               ? 4
               : output_encoding_info.orig_color_encoding.Channels());
      const size_t channels_dst =
          output_encoding_info.color_encoding.Channels();
      bool mixing_color_and_grey = (channels_dst != channels_src);
      if ((output_encoding_info.color_encoding_is_original) ||
          (!output_encoding_info.cms_set) || mixing_color_and_grey) {
        // in those cases we only need a linear stage in other cases we attempt
        // to obtain a cms stage: the cases are
        // - output_encoding_info.color_encoding_is_original: no cms stage
        // needed because it would be a no-op
        // - !output_encoding_info.cms_set: can't use the cms, so no point in
        // trying to add a cms stage
        // - mixing_color_and_grey: cms stage can't handle that
        // TODO(firsching): remove "mixing_color_and_grey" condition after
        // adding support for greyscale to cms stage.
        JXL_RETURN_IF_ERROR(
            builder.AddStage(GetFromLinearStage(output_encoding_info)));
      } else {
        if (!output_encoding_info.linear_color_encoding.CreateICC()) {
          return JXL_FAILURE("Failed to create ICC");
        }
        auto cms_stage = GetCmsStage(output_encoding_info);
        if (cms_stage) {
          JXL_RETURN_IF_ERROR(builder.AddStage(std::move(cms_stage)));
        }
      }
      linear = false;
    }
    (void)linear;

    if (main_output.callback.IsPresent() || main_output.buffer) {
      JXL_RETURN_IF_ERROR(builder.AddStage(GetWriteToOutputStage(
          main_output, width, height, has_alpha, unpremul_alpha, alpha_c,
          undo_orientation, extra_output, memory_manager)));
    } else {
      JXL_RETURN_IF_ERROR(builder.AddStage(
          GetWriteToImageBundleStage(decoded, output_encoding_info)));
    }
  }
  JXL_ASSIGN_OR_RETURN(render_pipeline,
                       std::move(builder).Finalize(shared->frame_dim));
  return render_pipeline->IsInitialized();
}

}  // namespace jxl
