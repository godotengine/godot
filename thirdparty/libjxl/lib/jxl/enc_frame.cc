// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_frame.h"

#include <jxl/cms_interface.h>
#include <jxl/encode.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_external_image.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/enc_ac_strategy.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_chroma_from_luma.h"
#include "lib/jxl/enc_coeff_order.h"
#include "lib/jxl/enc_context_map.h"
#include "lib/jxl/enc_entropy_coder.h"
#include "lib/jxl/enc_external_image.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_heuristics.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_noise.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/enc_photon_noise.h"
#include "lib/jxl/enc_progressive_split.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/enc_splines.h"
#include "lib/jxl/enc_toc.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/jpeg/enc_jpeg_data.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/noise.h"
#include "lib/jxl/padded_bytes.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/splines.h"
#include "lib/jxl/toc.h"

namespace jxl {

Status ParamsPostInit(CompressParams* p) {
  if (!p->manual_noise.empty() &&
      p->manual_noise.size() != NoiseParams::kNumNoisePoints) {
    return JXL_FAILURE("Invalid number of noise lut entries");
  }
  if (!p->manual_xyb_factors.empty() && p->manual_xyb_factors.size() != 3) {
    return JXL_FAILURE("Invalid number of XYB quantization factors");
  }
  if (!p->modular_mode && p->butteraugli_distance < kMinButteraugliDistance) {
    p->butteraugli_distance = kMinButteraugliDistance;
  }
  if (p->original_butteraugli_distance == -1.0) {
    p->original_butteraugli_distance = p->butteraugli_distance;
  }
  if (p->resampling <= 0) {
    p->resampling = 1;
    // For very low bit rates, using 2x2 resampling gives better results on
    // most photographic images, with an adjusted butteraugli score chosen to
    // give roughly the same amount of bits per pixel.
    if (!p->already_downsampled && p->butteraugli_distance >= 10) {
      // TODO(Jonnyawsom3): Explore 4x4 resampling at distance 25. Lower bpp
      // but results are inconsistent and images under 4K become far too blurry.
      p->resampling = 2;
      // Adding 0.25 balances photo with non-photo, shifting towards lower bpp
      // to avoid large overshoot while maintaining quality equal to before.
      p->butteraugli_distance = (p->butteraugli_distance * 0.25) + 0.25;
    }
  }
  if (p->ec_resampling <= 0) {
    p->ec_resampling = p->resampling;
  }
  // Modular has to be squeezed to show progressive HF passes.
  if (p->progressive_mode == Override::kOn ||
      p->qprogressive_mode == Override::kOn) {
    p->responsive = 1;
    if (p->IsLossless()) {
      p->qprogressive_mode = Override::kOn;
    }
  }
  return true;
}

namespace {

uint32_t GetGroupSizeShift(size_t xsize, size_t ysize,
                           const CompressParams& cparams) {
  if (!cparams.modular_mode) return 1;
  if (cparams.modular_group_size_shift >= 0) {
    return static_cast<uint32_t>(cparams.modular_group_size_shift);
  }
  // By default, use the smallest group size for faster decoding 2
  // and higher. Greatly speeds up decoding via multithreading at
  // the cost of density.
  if (cparams.decoding_speed_tier >= 2 ||
      (cparams.decoding_speed_tier >= 1 && cparams.responsive == 1 &&
       cparams.IsLossless())) {
    return 0;
  }
  // No point using groups when only one group is full and the others are
  // less than half full: multithreading will not really help much, while
  // compression does suffer; but no reason to have group larger than image.
  if (xsize <= 128 && ysize <= 128) return 0;
  if (xsize <= 256 && ysize <= 256) return 1;
  if (xsize <= 400 && ysize <= 400) return 2;
  return 1;
}

static size_t NumGroupsForFrame(size_t xsize, size_t ysize,
                                const CompressParams& cparams) {
  const size_t group_size_shift = GetGroupSizeShift(xsize, ysize, cparams);
  const size_t group_dim = (kGroupDim >> 1) << group_size_shift;
  return DivCeil(xsize, group_dim) * DivCeil(ysize, group_dim);
}

template <typename T>
uint32_t GetBitDepth(JxlBitDepth bit_depth, const T& metadata,
                     JxlPixelFormat format) {
  if (bit_depth.type == JXL_BIT_DEPTH_FROM_PIXEL_FORMAT) {
    return BitsPerChannel(format.data_type);
  } else if (bit_depth.type == JXL_BIT_DEPTH_FROM_CODESTREAM) {
    return metadata.bit_depth.bits_per_sample;
  } else if (bit_depth.type == JXL_BIT_DEPTH_CUSTOM) {
    return bit_depth.bits_per_sample;
  } else {
    return 0;
  }
}

Status CopyColorChannels(JxlChunkedFrameInputSource input, Rect rect,
                         const FrameInfo& frame_info,
                         const ImageMetadata& metadata, ThreadPool* pool,
                         Image3F* color, ImageF* alpha,
                         bool* has_interleaved_alpha) {
  JxlPixelFormat format = {4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  input.get_color_channels_pixel_format(input.opaque, &format);
  format.align = 0;  // align must be ignored.
  *has_interleaved_alpha = format.num_channels == 2 || format.num_channels == 4;
  size_t bits_per_sample =
      GetBitDepth(frame_info.image_bit_depth, metadata, format);
  size_t row_offset;
  auto buffer = GetColorBuffer(input, rect.x0(), rect.y0(), rect.xsize(),
                               rect.ysize(), &row_offset);
  if (!buffer) {
    return JXL_FAILURE("no buffer for color channels given");
  }
  size_t color_channels = frame_info.ib_needs_color_transform
                              ? metadata.color_encoding.Channels()
                              : 3;
  if (format.num_channels < color_channels) {
    return JXL_FAILURE("Expected %" PRIuS
                       " color channels, received only %u channels",
                       color_channels, format.num_channels);
  }
  const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer.get());
  for (size_t c = 0; c < color_channels; ++c) {
    JXL_RETURN_IF_ERROR(ConvertFromExternalPlaneNoSizeCheck(
        data, rect.xsize(), rect.ysize(), row_offset, bits_per_sample, format,
        c, pool, &color->Plane(c)));
  }
  if (color_channels == 1) {
    JXL_RETURN_IF_ERROR(CopyImageTo(color->Plane(0), &color->Plane(1)));
    JXL_RETURN_IF_ERROR(CopyImageTo(color->Plane(0), &color->Plane(2)));
  }
  if (alpha) {
    if (*has_interleaved_alpha) {
      JXL_RETURN_IF_ERROR(ConvertFromExternalPlaneNoSizeCheck(
          data, rect.xsize(), rect.ysize(), row_offset, bits_per_sample, format,
          format.num_channels - 1, pool, alpha));
    } else {
      // if alpha is not passed, but it is expected, then assume
      // it is all-opaque
      FillImage(1.0f, alpha);
    }
  }
  return true;
}

Status CopyExtraChannels(JxlChunkedFrameInputSource input, Rect rect,
                         const FrameInfo& frame_info,
                         const ImageMetadata& metadata,
                         bool has_interleaved_alpha, ThreadPool* pool,
                         std::vector<ImageF>* extra_channels) {
  for (size_t ec = 0; ec < metadata.num_extra_channels; ec++) {
    if (has_interleaved_alpha &&
        metadata.extra_channel_info[ec].type == ExtraChannel::kAlpha) {
      // Skip this alpha channel, but still request additional alpha channels
      // if they exist.
      has_interleaved_alpha = false;
      continue;
    }
    JxlPixelFormat ec_format = {1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    input.get_extra_channel_pixel_format(input.opaque, ec, &ec_format);
    ec_format.num_channels = 1;
    size_t row_offset;
    auto buffer =
        GetExtraChannelBuffer(input, ec, rect.x0(), rect.y0(), rect.xsize(),
                              rect.ysize(), &row_offset);
    if (!buffer) {
      return JXL_FAILURE("no buffer for extra channel given");
    }
    size_t bits_per_sample = GetBitDepth(
        frame_info.image_bit_depth, metadata.extra_channel_info[ec], ec_format);
    if (!ConvertFromExternalPlaneNoSizeCheck(
            reinterpret_cast<const uint8_t*>(buffer.get()), rect.xsize(),
            rect.ysize(), row_offset, bits_per_sample, ec_format, 0, pool,
            &(*extra_channels)[ec])) {
      return JXL_FAILURE("Failed to set buffer for extra channel");
    }
  }
  return true;
}

void SetProgressiveMode(const CompressParams& cparams,
                        ProgressiveSplitter* progressive_splitter) {
  constexpr PassDefinition progressive_passes_dc_vlf_lf_full_ac[] = {
      {/*num_coefficients=*/2, /*shift=*/0,
       /*suitable_for_downsampling_of_at_least=*/4},
      {/*num_coefficients=*/3, /*shift=*/0,
       /*suitable_for_downsampling_of_at_least=*/2},
      {/*num_coefficients=*/8, /*shift=*/0,
       /*suitable_for_downsampling_of_at_least=*/0},
  };
  constexpr PassDefinition progressive_passes_dc_quant_ac_full_ac[] = {
      {/*num_coefficients=*/8, /*shift=*/1,
       /*suitable_for_downsampling_of_at_least=*/2},
      {/*num_coefficients=*/8, /*shift=*/0,
       /*suitable_for_downsampling_of_at_least=*/0},
  };
  bool progressive_mode = ApplyOverride(cparams.progressive_mode, false);
  bool qprogressive_mode = ApplyOverride(cparams.qprogressive_mode, false);
  if (cparams.custom_progressive_mode) {
    progressive_splitter->SetProgressiveMode(*cparams.custom_progressive_mode);
  } else if (qprogressive_mode) {
    progressive_splitter->SetProgressiveMode(
        ProgressiveMode{progressive_passes_dc_quant_ac_full_ac});
  } else if (progressive_mode) {
    progressive_splitter->SetProgressiveMode(
        ProgressiveMode{progressive_passes_dc_vlf_lf_full_ac});
  }
}

uint64_t FrameFlagsFromParams(const CompressParams& cparams) {
  uint64_t flags = 0;

  const float dist = cparams.butteraugli_distance;

  // We don't add noise at low butteraugli distances because the original
  // noise is stored within the compressed image and adding noise makes things
  // worse.
  if (ApplyOverride(cparams.noise, dist >= kMinButteraugliForNoise) ||
      cparams.photon_noise_iso > 0 ||
      cparams.manual_noise.size() == NoiseParams::kNumNoisePoints) {
    flags |= FrameHeader::kNoise;
  }

  if (cparams.progressive_dc > 0 && cparams.modular_mode == false) {
    flags |= FrameHeader::kUseDcFrame;
  }

  return flags;
}

Status LoopFilterFromParams(const CompressParams& cparams, bool streaming_mode,
                            FrameHeader* JXL_RESTRICT frame_header) {
  LoopFilter* loop_filter = &frame_header->loop_filter;

  // Gaborish defaults to enabled in Hare or slower.
  loop_filter->gab = ApplyOverride(
      cparams.gaborish, cparams.speed_tier <= SpeedTier::kHare &&
                            frame_header->encoding == FrameEncoding::kVarDCT &&
                            cparams.decoding_speed_tier < 4 &&
                            cparams.butteraugli_distance > 0.5f &&
                            !cparams.disable_perceptual_optimizations);

  if (cparams.epf != -1) {
    loop_filter->epf_iters = cparams.epf;
  } else if (cparams.disable_perceptual_optimizations) {
    loop_filter->epf_iters = 0;
    return true;
  } else {
    if (frame_header->encoding == FrameEncoding::kModular) {
      loop_filter->epf_iters = 0;
    } else {
      constexpr float kThresholds[3] = {0.7, 1.5, 4.0};
      loop_filter->epf_iters = 0;
      if (cparams.decoding_speed_tier < 3) {
        for (size_t i = cparams.decoding_speed_tier == 2 ? 1 : 0; i < 3; i++) {
          if (cparams.butteraugli_distance >= kThresholds[i]) {
            loop_filter->epf_iters++;
          }
        }
      }
    }
  }
  // Strength of EPF in modular mode.
  if (frame_header->encoding == FrameEncoding::kModular &&
      !cparams.IsLossless()) {
    // TODO(veluca): this formula is nonsense.
    loop_filter->epf_sigma_for_modular =
        std::max(cparams.butteraugli_distance, 1.0f);
  }
  if (frame_header->encoding == FrameEncoding::kModular &&
      cparams.lossy_palette) {
    loop_filter->epf_sigma_for_modular = 1.0f;
  }

  return true;
}

Status MakeFrameHeader(size_t xsize, size_t ysize,
                       const CompressParams& cparams,
                       const ProgressiveSplitter& progressive_splitter,
                       const FrameInfo& frame_info,
                       const jpeg::JPEGData* jpeg_data, bool streaming_mode,
                       FrameHeader* JXL_RESTRICT frame_header) {
  frame_header->nonserialized_is_preview = frame_info.is_preview;
  frame_header->is_last = frame_info.is_last;
  frame_header->save_before_color_transform =
      frame_info.save_before_color_transform;
  frame_header->frame_type = frame_info.frame_type;
  frame_header->name = frame_info.name;

  JXL_RETURN_IF_ERROR(progressive_splitter.InitPasses(&frame_header->passes));

  if (cparams.modular_mode) {
    frame_header->encoding = FrameEncoding::kModular;
    frame_header->group_size_shift = GetGroupSizeShift(xsize, ysize, cparams);
  }

  if (jpeg_data) {
    // we are transcoding a JPEG, so we don't get to choose
    frame_header->encoding = FrameEncoding::kVarDCT;
    frame_header->x_qm_scale = 2;
    frame_header->b_qm_scale = 2;
    JXL_RETURN_IF_ERROR(SetChromaSubsamplingFromJpegData(
        *jpeg_data, &frame_header->chroma_subsampling));
    JXL_RETURN_IF_ERROR(SetColorTransformFromJpegData(
        *jpeg_data, &frame_header->color_transform));
  } else {
    frame_header->color_transform = cparams.color_transform;
    if (!cparams.modular_mode &&
        (frame_header->chroma_subsampling.MaxHShift() != 0 ||
         frame_header->chroma_subsampling.MaxVShift() != 0)) {
      return JXL_FAILURE(
          "Chroma subsampling is not supported in VarDCT mode when not "
          "recompressing JPEGs");
    }
  }

  if (frame_header->color_transform != ColorTransform::kYCbCr &&
      (frame_header->chroma_subsampling.MaxHShift() != 0 ||
       frame_header->chroma_subsampling.MaxVShift() != 0)) {
    return JXL_FAILURE(
        "Chroma subsampling is not supported when color transform is not "
        "YCbCr");
  }

  frame_header->flags = FrameFlagsFromParams(cparams);
  // Non-photon noise is not supported in the Modular encoder for now.
  if (frame_header->encoding != FrameEncoding::kVarDCT &&
      cparams.photon_noise_iso == 0 && cparams.manual_noise.empty()) {
    frame_header->UpdateFlag(false, FrameHeader::Flags::kNoise);
  }

  JXL_RETURN_IF_ERROR(
      LoopFilterFromParams(cparams, streaming_mode, frame_header));

  frame_header->dc_level = frame_info.dc_level;
  if (frame_header->dc_level > 2) {
    // With 3 or more progressive_dc frames, the implementation does not yet
    // work, see enc_cache.cc.
    return JXL_FAILURE("progressive_dc > 2 is not yet supported");
  }
  if (cparams.progressive_dc > 0 &&
      (cparams.ec_resampling != 1 || cparams.resampling != 1)) {
    return JXL_FAILURE("Resampling not supported with DC frames");
  }
  if (cparams.resampling != 1 && cparams.resampling != 2 &&
      cparams.resampling != 4 && cparams.resampling != 8) {
    return JXL_FAILURE("Invalid resampling factor");
  }
  if (cparams.ec_resampling != 1 && cparams.ec_resampling != 2 &&
      cparams.ec_resampling != 4 && cparams.ec_resampling != 8) {
    return JXL_FAILURE("Invalid ec_resampling factor");
  }
  // Resized frames.
  if (frame_info.frame_type != FrameType::kDCFrame) {
    frame_header->frame_origin = frame_info.origin;
    size_t ups = 1;
    if (cparams.already_downsampled) ups = cparams.resampling;

    // TODO(lode): this is not correct in case of odd original image sizes in
    // combination with cparams.already_downsampled. Likely these values should
    // be set to respectively frame_header->default_xsize() and
    // frame_header->default_ysize() instead, the original (non downsampled)
    // intended decoded image dimensions. But it may be more subtle than that
    // if combined with crop. This issue causes custom_size_or_origin to be
    // incorrectly set to true in case of already_downsampled with odd output
    // image size when no cropping is used.
    frame_header->frame_size.xsize = xsize * ups;
    frame_header->frame_size.ysize = ysize * ups;
    if (frame_info.origin.x0 != 0 || frame_info.origin.y0 != 0 ||
        frame_header->frame_size.xsize != frame_header->default_xsize() ||
        frame_header->frame_size.ysize != frame_header->default_ysize()) {
      frame_header->custom_size_or_origin = true;
    }
  }
  // Upsampling.
  frame_header->upsampling = cparams.resampling;
  const std::vector<ExtraChannelInfo>& extra_channels =
      frame_header->nonserialized_metadata->m.extra_channel_info;
  frame_header->extra_channel_upsampling.clear();
  frame_header->extra_channel_upsampling.resize(extra_channels.size(),
                                                cparams.ec_resampling);
  frame_header->save_as_reference = frame_info.save_as_reference;

  // Set blending-related information.
  if (frame_info.blend || frame_header->custom_size_or_origin) {
    // Set blend_channel to the first alpha channel. These values are only
    // encoded in case a blend mode involving alpha is used and there are more
    // than one extra channels.
    size_t index = 0;
    if (frame_info.alpha_channel == -1) {
      if (extra_channels.size() > 1) {
        for (size_t i = 0; i < extra_channels.size(); i++) {
          if (extra_channels[i].type == ExtraChannel::kAlpha) {
            index = i;
            break;
          }
        }
      }
    } else {
      index = static_cast<size_t>(frame_info.alpha_channel);
      JXL_ENSURE(index == 0 || index < extra_channels.size());
    }
    frame_header->blending_info.alpha_channel = index;
    frame_header->blending_info.mode =
        frame_info.blend ? frame_info.blendmode : BlendMode::kReplace;
    frame_header->blending_info.source = frame_info.source;
    frame_header->blending_info.clamp = frame_info.clamp;
    const auto& extra_channel_info = frame_info.extra_channel_blending_info;
    for (size_t i = 0; i < extra_channels.size(); i++) {
      if (i < extra_channel_info.size()) {
        frame_header->extra_channel_blending_info[i] = extra_channel_info[i];
      } else {
        frame_header->extra_channel_blending_info[i].alpha_channel = index;
        BlendMode default_blend = frame_info.blendmode;
        if (extra_channels[i].type != ExtraChannel::kBlack && i != index) {
          // K needs to be blended, spot colors and other stuff gets added
          default_blend = BlendMode::kAdd;
        }
        frame_header->extra_channel_blending_info[i].mode =
            frame_info.blend ? default_blend : BlendMode::kReplace;
        frame_header->extra_channel_blending_info[i].source = 1;
      }
    }
  }

  frame_header->animation_frame.duration = frame_info.duration;
  frame_header->animation_frame.timecode = frame_info.timecode;

  if (jpeg_data) {
    frame_header->UpdateFlag(false, FrameHeader::kUseDcFrame);
    frame_header->UpdateFlag(true, FrameHeader::kSkipAdaptiveDCSmoothing);
  }

  return true;
}

// Invisible (alpha = 0) pixels tend to be a mess in optimized PNGs.
// Since they have no visual impact whatsoever, we can replace them with
// something that compresses better and reduces artifacts near the edges. This
// does some kind of smooth stuff that seems to work.
// Replace invisible pixels with a weighted average of the pixel to the left,
// the pixel to the topright, and non-invisible neighbours.
// Produces downward-blurry smears, with in the upwards direction only a 1px
// edge duplication but not more. It would probably be better to smear in all
// directions. That requires an alpha-weighed convolution with a large enough
// kernel though, which might be overkill...
void SimplifyInvisible(Image3F* image, const ImageF& alpha, bool lossless) {
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      float* JXL_RESTRICT row = image->PlaneRow(c, y);
      const float* JXL_RESTRICT prow =
          (y > 0 ? image->PlaneRow(c, y - 1) : nullptr);
      const float* JXL_RESTRICT nrow =
          (y + 1 < image->ysize() ? image->PlaneRow(c, y + 1) : nullptr);
      const float* JXL_RESTRICT a = alpha.Row(y);
      const float* JXL_RESTRICT pa = (y > 0 ? alpha.Row(y - 1) : nullptr);
      const float* JXL_RESTRICT na =
          (y + 1 < image->ysize() ? alpha.Row(y + 1) : nullptr);
      for (size_t x = 0; x < image->xsize(); ++x) {
        if (a[x] == 0) {
          if (lossless) {
            row[x] = 0;
            continue;
          }
          float d = 0.f;
          row[x] = 0;
          if (x > 0) {
            row[x] += row[x - 1];
            d++;
            if (a[x - 1] > 0.f) {
              row[x] += row[x - 1];
              d++;
            }
          }
          if (x + 1 < image->xsize()) {
            if (y > 0) {
              row[x] += prow[x + 1];
              d++;
            }
            if (a[x + 1] > 0.f) {
              row[x] += 2.f * row[x + 1];
              d += 2.f;
            }
            if (y > 0 && pa[x + 1] > 0.f) {
              row[x] += 2.f * prow[x + 1];
              d += 2.f;
            }
            if (y + 1 < image->ysize() && na[x + 1] > 0.f) {
              row[x] += 2.f * nrow[x + 1];
              d += 2.f;
            }
          }
          if (y > 0 && pa[x] > 0.f) {
            row[x] += 2.f * prow[x];
            d += 2.f;
          }
          if (y + 1 < image->ysize() && na[x] > 0.f) {
            row[x] += 2.f * nrow[x];
            d += 2.f;
          }
          if (d > 1.f) row[x] /= d;
        }
      }
    }
  }
}

struct PixelStatsForChromacityAdjustment {
  float dx = 0;
  float db = 0;
  float exposed_blue = 0;
  static float CalcPlane(const ImageF* JXL_RESTRICT plane, const Rect& rect) {
    float xmax = 0;
    float ymax = 0;
    for (size_t ty = 1; ty < rect.ysize(); ++ty) {
      for (size_t tx = 1; tx < rect.xsize(); ++tx) {
        float cur = rect.Row(plane, ty)[tx];
        float prev_row = rect.Row(plane, ty - 1)[tx];
        float prev = rect.Row(plane, ty)[tx - 1];
        xmax = std::max(xmax, std::abs(cur - prev));
        ymax = std::max(ymax, std::abs(cur - prev_row));
      }
    }
    return std::max(xmax, ymax);
  }
  void CalcExposedBlue(const ImageF* JXL_RESTRICT plane_y,
                       const ImageF* JXL_RESTRICT plane_b, const Rect& rect) {
    float eb = 0;
    float xmax = 0;
    float ymax = 0;
    for (size_t ty = 1; ty < rect.ysize(); ++ty) {
      for (size_t tx = 1; tx < rect.xsize(); ++tx) {
        float cur_y = rect.Row(plane_y, ty)[tx];
        float cur_b = rect.Row(plane_b, ty)[tx];
        float exposed_b = cur_b - cur_y * 1.2;
        float diff_b = cur_b - cur_y;
        float prev_row = rect.Row(plane_b, ty - 1)[tx];
        float prev = rect.Row(plane_b, ty)[tx - 1];
        float diff_prev_row = prev_row - rect.Row(plane_y, ty - 1)[tx];
        float diff_prev = prev - rect.Row(plane_y, ty)[tx - 1];
        xmax = std::max(xmax, std::abs(diff_b - diff_prev));
        ymax = std::max(ymax, std::abs(diff_b - diff_prev_row));
        if (exposed_b >= 0) {
          exposed_b *= std::abs(cur_b - prev) + std::abs(cur_b - prev_row);
          eb = std::max(eb, exposed_b);
        }
      }
    }
    exposed_blue = eb;
    db = std::max(xmax, ymax);
  }
  void Calc(const Image3F* JXL_RESTRICT opsin, const Rect& rect) {
    dx = CalcPlane(&opsin->Plane(0), rect);
    CalcExposedBlue(&opsin->Plane(1), &opsin->Plane(2), rect);
  }
  int HowMuchIsXChannelPixelized() const {
    if (dx >= 0.026) {
      return 3;
    }
    if (dx >= 0.022) {
      return 2;
    }
    if (dx >= 0.015) {
      return 1;
    }
    return 0;
  }
  int HowMuchIsBChannelPixelized() const {
    int add = exposed_blue >= 0.13 ? 1 : 0;
    if (db > 0.38) {
      return 2 + add;
    }
    if (db > 0.33) {
      return 1 + add;
    }
    if (db > 0.28) {
      return add;
    }
    return 0;
  }
};

void ComputeChromacityAdjustments(const CompressParams& cparams,
                                  const Image3F& opsin, const Rect& rect,
                                  FrameHeader* frame_header) {
  if (frame_header->encoding != FrameEncoding::kVarDCT ||
      cparams.max_error_mode) {
    return;
  }
  // 1) Distance based approach for chromacity adjustment:
  float x_qm_scale_steps[3] = {2.5f, 5.5f, 9.5f};
  frame_header->x_qm_scale = 3;
  for (float x_qm_scale_step : x_qm_scale_steps) {
    if (cparams.original_butteraugli_distance > x_qm_scale_step) {
      frame_header->x_qm_scale++;
    }
  }
  // 2) Pixel-based approach for chromacity adjustment:
  // look at the individual pixels and make a guess how difficult
  // the image would be based on the worst case pixel.
  PixelStatsForChromacityAdjustment pixel_stats;
  if (cparams.speed_tier <= SpeedTier::kSquirrel) {
    pixel_stats.Calc(&opsin, rect);
  }
  // For X take the most severe adjustment.
  frame_header->x_qm_scale = std::max<int>(
      frame_header->x_qm_scale, 2 + pixel_stats.HowMuchIsXChannelPixelized());
  // B only adjusted by pixel-based approach.
  frame_header->b_qm_scale = 2 + pixel_stats.HowMuchIsBChannelPixelized();
}

void ComputeNoiseParams(const CompressParams& cparams, bool streaming_mode,
                        bool color_is_jpeg, const Image3F& opsin,
                        const FrameDimensions& frame_dim,
                        FrameHeader* frame_header, NoiseParams* noise_params) {
  if (frame_header->frame_type == FrameType::kDCFrame) {
    frame_header->flags &= ~FrameHeader::kNoise;
    return;
  }
  if (cparams.photon_noise_iso > 0) {
    FrameDimensions full_frame_dim = frame_header->ToFrameDimensions();
    *noise_params = SimulatePhotonNoise(full_frame_dim.xsize, full_frame_dim.ysize,
                                        cparams.photon_noise_iso);
  } else if (cparams.manual_noise.size() == NoiseParams::kNumNoisePoints) {
    for (size_t i = 0; i < NoiseParams::kNumNoisePoints; i++) {
      noise_params->lut[i] = cparams.manual_noise[i];
    }
  } else if (frame_header->encoding == FrameEncoding::kVarDCT &&
             frame_header->flags & FrameHeader::kNoise && !color_is_jpeg &&
             !streaming_mode) {
    // Don't start at zero amplitude since adding noise is expensive -- it
    // significantly slows down decoding, and this is unlikely to
    // completely go away even with advanced optimizations. After the
    // kNoiseModelingRampUpDistanceRange we have reached the full level,
    // i.e. noise is no longer represented by the compressed image, so we
    // can add full noise by the noise modeling itself.
    static const float kNoiseModelingRampUpDistanceRange = 0.6;
    static const float kNoiseLevelAtStartOfRampUp = 0.25;
    static const float kNoiseRampupStart = 1.0;
    // TODO(user) test and properly select quality_coef with smooth
    // filter
    float quality_coef = 1.0f;
    const float rampup = (cparams.butteraugli_distance - kNoiseRampupStart) /
                         kNoiseModelingRampUpDistanceRange;
    if (rampup < 1.0f) {
      quality_coef = kNoiseLevelAtStartOfRampUp +
                     (1.0f - kNoiseLevelAtStartOfRampUp) * rampup;
    }
    if (rampup < 0.0f) {
      quality_coef = kNoiseRampupStart;
    }
    if (!GetNoiseParameter(opsin, noise_params, quality_coef)) {
      frame_header->flags &= ~FrameHeader::kNoise;
    }
  }
}

Status DownsampleColorChannels(const CompressParams& cparams,
                               const FrameHeader& frame_header,
                               bool color_is_jpeg, Image3F* opsin) {
  if (color_is_jpeg || frame_header.upsampling == 1 ||
      cparams.already_downsampled) {
    return true;
  }
  if (frame_header.encoding == FrameEncoding::kVarDCT &&
      frame_header.upsampling == 2) {
    // TODO(lode): use the regular DownsampleImage, or adapt to the custom
    // coefficients, if there is are custom upscaling coefficients in
    // CustomTransformData
    if (cparams.speed_tier <= SpeedTier::kGlacier) {
      // TODO(Jonnyawsom3): Until optimized, enabled only for Glacier and
      // TectonicPlate. It's an 80% slowdown and downsampling is only active
      // at high distances by default anyway, making improvements negligible.
      JXL_RETURN_IF_ERROR(DownsampleImage2_Iterative(opsin));
    } else {
      JXL_RETURN_IF_ERROR(DownsampleImage2_Sharper(opsin));
    }
  } else {
    JXL_ASSIGN_OR_RETURN(*opsin,
                         DownsampleImage(*opsin, frame_header.upsampling));
  }
  if (frame_header.encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(PadImageToBlockMultipleInPlace(opsin));
  }
  return true;
}

template <size_t L, typename V, typename R>
void FindAvgIndexOfSumMaximum(const V* array, R* idx, V* sum) {
  static_assert(L > 0, "Empty arrays have undefined maximum");
  V maxval = 0;
  V val = 0;
  R maxidx_begin = 0;
  R maxidx_end = 0;
  for (size_t i = 0; i < L; ++i) {
    val += array[i];
    if (val > maxval) {
      maxval = val;
      maxidx_begin = i;
    }
    if (val == maxval) {
      maxidx_end = i;
    }
  }
  *idx = (maxidx_begin + maxidx_end + 1) >> 1;
  *sum = maxval;
}

Status ComputeJPEGTranscodingData(const jpeg::JPEGData& jpeg_data,
                                  const FrameHeader& frame_header,
                                  ThreadPool* pool,
                                  ModularFrameEncoder* enc_modular,
                                  PassesEncoderState* enc_state) {
  PassesSharedState& shared = enc_state->shared;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  const FrameDimensions& frame_dim = shared.frame_dim;

  const size_t xsize = frame_dim.xsize_padded;
  const size_t ysize = frame_dim.ysize_padded;
  const size_t xsize_blocks = frame_dim.xsize_blocks;
  const size_t ysize_blocks = frame_dim.ysize_blocks;

  // no-op chroma from luma
  JXL_ASSIGN_OR_RETURN(shared.cmap, ColorCorrelationMap::Create(
                                        memory_manager, xsize, ysize, false));
  shared.ac_strategy.FillDCT8();
  FillImage(static_cast<uint8_t>(0), &shared.epf_sharpness);

  enc_state->coeffs.clear();
  while (enc_state->coeffs.size() < enc_state->passes.size()) {
    JXL_ASSIGN_OR_RETURN(
        std::unique_ptr<ACImageT<int32_t>> coeffs,
        ACImageT<int32_t>::Make(memory_manager, kGroupDim * kGroupDim,
                                frame_dim.num_groups));
    enc_state->coeffs.emplace_back(std::move(coeffs));
  }

  // convert JPEG quantization table to a Quantizer object
  float dcquantization[3];
  std::vector<QuantEncoding> qe(kNumQuantTables, QuantEncoding::Library<0>());

  auto jpeg_c_map =
      JpegOrder(frame_header.color_transform, jpeg_data.components.size() == 1);

  std::vector<int> qt(kDCTBlockSize * 3);
  std::array<int32_t, 3> qt_dc;
  for (size_t c = 0; c < 3; c++) {
    size_t jpeg_c = jpeg_c_map[c];
    const int32_t* quant =
        jpeg_data.quant[jpeg_data.components[jpeg_c].quant_idx].values.data();

    dcquantization[c] = 255 * 8.0f / quant[0];
    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        // JPEG XL transposes the DCT, JPEG doesn't.
        qt[kDCTBlockSize * c + 8 * x + y] = quant[8 * y + x];
      }
    }
    qt_dc[c] = qt[kDCTBlockSize * c];
  }
  JXL_RETURN_IF_ERROR(DequantMatricesSetCustomDC(
      memory_manager, &shared.matrices, dcquantization));
  float dcquantization_r[3] = {1.0f / dcquantization[0],
                               1.0f / dcquantization[1],
                               1.0f / dcquantization[2]};

  // not transposed
  std::vector<int32_t> scaled_qtable(kDCTBlockSize * 3);
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < 8; y++) {
      for (size_t x = 0; x < 8; x++) {
        int coeffpos = y * 8 + x;
        int32_t ratio = (1 << kCFLFixedPointPrecision) *
                        qt[kDCTBlockSize + coeffpos] /
                        qt[kDCTBlockSize * c + coeffpos];
        if (ratio > kCFLFixedPointRatioMax) {
          return JXL_FAILURE(
              "Ratio of two entries in a JPEG quantization table is too large");
        }
        scaled_qtable[kDCTBlockSize * c + 8 * x + y] = ratio;
      }
    }
  }

  qe[static_cast<size_t>(AcStrategyType::DCT)] =
      QuantEncoding::RAW(std::move(qt));
  JXL_RETURN_IF_ERROR(
      DequantMatricesSetCustom(&shared.matrices, qe, enc_modular));

  // Ensure that InvGlobalScale() is 1.
  shared.quantizer = Quantizer(shared.matrices, 1, kGlobalScaleDenom);
  // Recompute MulDC() and InvMulDC().
  shared.quantizer.RecomputeFromGlobalScale();

  // Per-block dequant scaling should be 1.
  FillImage(static_cast<int32_t>(shared.quantizer.InvGlobalScale()),
            &shared.raw_quant_field);

  auto jpeg_row = [&](size_t c, size_t y) {
    return jpeg_data.components[jpeg_c_map[c]].coeffs.data() +
           jpeg_data.components[jpeg_c_map[c]].width_in_blocks * kDCTBlockSize *
               y;
  };

  bool DCzero = (frame_header.color_transform == ColorTransform::kYCbCr);
  // Compute chroma-from-luma for AC (doesn't seem to be useful for DC)
  if (frame_header.chroma_subsampling.Is444() &&
      enc_state->cparams.force_cfl_jpeg_recompression &&
      jpeg_data.components.size() == 3) {
    for (size_t c : {0, 2}) {
      ImageSB* map = (c == 0 ? &shared.cmap.ytox_map : &shared.cmap.ytob_map);
      const float kScale = kDefaultColorFactor;
      const int kOffset = 127;
      const float kBase = c == 0 ? shared.cmap.base().YtoXRatio(0)
                                 : shared.cmap.base().YtoBRatio(0);
      const float kZeroThresh =
          kScale * kZeroBiasDefault[c] *
          0.9999f;  // just epsilon less for better rounding

      auto process_row = [&](const uint32_t task,
                             const size_t thread) -> Status {
        size_t ty = task;
        int8_t* JXL_RESTRICT row_out = map->Row(ty);
        for (size_t tx = 0; tx < map->xsize(); ++tx) {
          const size_t y0 = ty * kColorTileDimInBlocks;
          const size_t x0 = tx * kColorTileDimInBlocks;
          const size_t y1 = std::min(frame_dim.ysize_blocks,
                                     (ty + 1) * kColorTileDimInBlocks);
          const size_t x1 = std::min(frame_dim.xsize_blocks,
                                     (tx + 1) * kColorTileDimInBlocks);
          int32_t d_num_zeros[257] = {0};
          // TODO(veluca): this needs SIMD + fixed point adaptation, and/or
          // conversion to the new CfL algorithm.
          for (size_t y = y0; y < y1; ++y) {
            const int16_t* JXL_RESTRICT row_m = jpeg_row(1, y);
            const int16_t* JXL_RESTRICT row_s = jpeg_row(c, y);
            for (size_t x = x0; x < x1; ++x) {
              for (size_t coeffpos = 1; coeffpos < kDCTBlockSize; coeffpos++) {
                const float scaled_m = row_m[x * kDCTBlockSize + coeffpos] *
                                       (1.0f / (1 << kCFLFixedPointPrecision)) *
                                       scaled_qtable[64 * c + coeffpos];
                const float scaled_s =
                    kScale * row_s[x * kDCTBlockSize + coeffpos] +
                    (kOffset - kBase * kScale) * scaled_m;
                if (std::abs(scaled_m) > 1e-8f) {
                  float from;
                  float to;
                  if (scaled_m > 0) {
                    from = (scaled_s - kZeroThresh) / scaled_m;
                    to = (scaled_s + kZeroThresh) / scaled_m;
                  } else {
                    from = (scaled_s + kZeroThresh) / scaled_m;
                    to = (scaled_s - kZeroThresh) / scaled_m;
                  }
                  if (from < 0.0f) {
                    from = 0.0f;
                  }
                  if (to > 255.0f) {
                    to = 255.0f;
                  }
                  // Instead of clamping the both values
                  // we just check that range is sane.
                  if (from <= to) {
                    d_num_zeros[static_cast<int>(std::ceil(from))]++;
                    d_num_zeros[static_cast<int>(std::floor(to + 1))]--;
                  }
                }
              }
            }
          }
          int best = 0;
          int32_t best_sum = 0;
          FindAvgIndexOfSumMaximum<256>(d_num_zeros, &best, &best_sum);
          int32_t offset_sum = 0;
          for (int i = 0; i <= kOffset; ++i) {
            offset_sum += d_num_zeros[i];
          }
          row_out[tx] = 0;
          if (best_sum > offset_sum + 1) {
            row_out[tx] = best - kOffset;
          }
        }
        return true;
      };

      JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, map->ysize(), ThreadPool::NoInit,
                                    process_row, "FindCorrelation"));
    }
  }

  JXL_ASSIGN_OR_RETURN(
      Image3F dc, Image3F::Create(memory_manager, xsize_blocks, ysize_blocks));
  if (!frame_header.chroma_subsampling.Is444()) {
    ZeroFillImage(&dc);
    for (auto& coeff : enc_state->coeffs) {
      coeff->ZeroFill();
    }
  }
  // JPEG DC is from -1024 to 1023.
  std::vector<size_t> dc_counts;
  dc_counts.resize(2048);
  size_t total_dc[3] = {};
  for (size_t c : {1, 0, 2}) {
    if (jpeg_data.components.size() == 1 && c != 1) {
      for (auto& coeff : enc_state->coeffs) {
        coeff->ZeroFillPlane(c);
      }
      ZeroFillImage(&dc.Plane(c));
      // Ensure no division by 0.
      total_dc[c] = 1;
      continue;
    }
    size_t hshift = frame_header.chroma_subsampling.HShift(c);
    size_t vshift = frame_header.chroma_subsampling.VShift(c);
    ImageSB& map = (c == 0 ? shared.cmap.ytox_map : shared.cmap.ytob_map);
    for (size_t group_index = 0; group_index < frame_dim.num_groups;
         group_index++) {
      const size_t gx = group_index % frame_dim.xsize_groups;
      const size_t gy = group_index / frame_dim.xsize_groups;
      int32_t* coeffs[kMaxNumPasses];
      for (size_t i = 0; i < enc_state->coeffs.size(); i++) {
        coeffs[i] = enc_state->coeffs[i]->PlaneRow(c, group_index, 0).ptr32;
      }
      int32_t block[64];
      for (size_t by = gy * kGroupDimInBlocks;
           by < ysize_blocks && by < (gy + 1) * kGroupDimInBlocks; ++by) {
        if ((by >> vshift) << vshift != by) continue;
        const int16_t* JXL_RESTRICT inputjpeg = jpeg_row(c, by >> vshift);
        const int16_t* JXL_RESTRICT inputjpegY = jpeg_row(1, by);
        float* JXL_RESTRICT fdc = dc.PlaneRow(c, by >> vshift);
        const int8_t* JXL_RESTRICT cm =
            map.ConstRow(by / kColorTileDimInBlocks);
        for (size_t bx = gx * kGroupDimInBlocks;
             bx < xsize_blocks && bx < (gx + 1) * kGroupDimInBlocks; ++bx) {
          if ((bx >> hshift) << hshift != bx) continue;
          size_t base = (bx >> hshift) * kDCTBlockSize;
          int idc;
          if (DCzero) {
            idc = inputjpeg[base];
          } else {
            idc = inputjpeg[base] + 1024 / qt_dc[c];
          }
          if (c == 1) {
            dc_counts[std::min<uint32_t>(idc + 1024, 2047)]++;
          }
          total_dc[c]++;
          fdc[bx >> hshift] = idc * dcquantization_r[c];
          if (c == 1 || !enc_state->cparams.force_cfl_jpeg_recompression ||
              !frame_header.chroma_subsampling.Is444()) {
            for (size_t y = 0; y < 8; y++) {
              for (size_t x = 0; x < 8; x++) {
                block[x * 8 + y] = inputjpeg[base + y * 8 + x];
              }
            }
          } else {
            const int32_t scale =
                ColorCorrelation::RatioJPEG(cm[bx / kColorTileDimInBlocks]);

            for (size_t y = 0; y < 8; y++) {
              for (size_t x = 0; x < 8; x++) {
                int coeffpos = y * 8 + x;
                int Y = inputjpegY[kDCTBlockSize * bx + coeffpos];
                int QChroma = inputjpeg[kDCTBlockSize * bx + coeffpos];
                // Fixed-point multiply of CfL scale with quant table ratio
                // first, and Y value second.
                int coeff_scale =
                    (scale * scaled_qtable[kDCTBlockSize * c + coeffpos] +
                     (1 << (kCFLFixedPointPrecision - 1))) >>
                    kCFLFixedPointPrecision;
                int cfl_factor =
                    (Y * coeff_scale + (1 << (kCFLFixedPointPrecision - 1))) >>
                    kCFLFixedPointPrecision;
                int QCR = QChroma - cfl_factor;
                block[x * 8 + y] = QCR;
              }
            }
          }
          enc_state->progressive_splitter.SplitACCoefficients(
              block, AcStrategy::FromRawStrategy(AcStrategyType::DCT), bx, by,
              coeffs);
          for (size_t i = 0; i < enc_state->coeffs.size(); i++) {
            coeffs[i] += kDCTBlockSize;
          }
        }
      }
    }
  }

  auto& dct = enc_state->shared.block_ctx_map.dc_thresholds;
  auto& num_dc_ctxs = enc_state->shared.block_ctx_map.num_dc_ctxs;

  for (size_t i = 0; i < 3; i++) {
    dct[i].clear();
  }
  // use more contexts for larger and higher quality images
  int num_thresholds = CeilLog2Nonzero(total_dc[1]) -
                       CeilLog2Nonzero(static_cast<unsigned>(
                           qt[1] + qt[2] + qt[3] + qt[4] + qt[5])) -
                       7;
  // up to 8 buckets, based on luma only
  num_thresholds = jxl::Clamp1(num_thresholds, 1, 7);
  size_t cumsum = 0;
  size_t cut = total_dc[1] / (num_thresholds + 1);
  for (int j = 0; j < 2048; j++) {
    cumsum += dc_counts[j];
    if (cumsum > cut) {
      dct[1].push_back(j - 1025);
      cut = total_dc[1] * (dct[1].size() + 1) / (num_thresholds + 1);
    }
  }
  num_dc_ctxs = dct[1].size() + 1;

  auto& ctx_map = enc_state->shared.block_ctx_map.ctx_map;
  ctx_map.clear();
  ctx_map.resize(3 * kNumOrders * num_dc_ctxs, 0);

  for (size_t i = 0; i < num_dc_ctxs; i++) {
    // luma: one context per luma DC bucket
    ctx_map[i] = i;
    if (jpeg_data.components.size() == 1) {
      // grayscale -> one context for all chroma
      ctx_map[kNumOrders * num_dc_ctxs + i] =
          ctx_map[2 * kNumOrders * num_dc_ctxs + i] = num_dc_ctxs;
    } else {
      // color -> multiple contexts per chroma component
      ctx_map[kNumOrders * num_dc_ctxs + i] = num_dc_ctxs + i / 2;
      ctx_map[2 * kNumOrders * num_dc_ctxs + i] =
          num_dc_ctxs + (num_dc_ctxs - 1) / 2 + 1 + i / 2;
    }
  }
  enc_state->shared.block_ctx_map.num_ctxs =
      *std::max_element(ctx_map.begin(), ctx_map.end()) + 1;

  JXL_ENSURE(enc_state->shared.block_ctx_map.num_ctxs <= 16);

  // disable DC frame for now
  auto compute_dc_coeffs = [&](const uint32_t group_index,
                               size_t /* thread */) -> Status {
    const Rect r = enc_state->shared.frame_dim.DCGroupRect(group_index);
    JXL_RETURN_IF_ERROR(enc_modular->AddVarDCTDC(frame_header, dc, r,
                                                 group_index,
                                                 /*nl_dc=*/false, enc_state,
                                                 /*jpeg_transcode=*/true));
    JXL_RETURN_IF_ERROR(enc_modular->AddACMetadata(
        r, group_index, /*jpeg_transcode=*/true, enc_state));
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                ThreadPool::NoInit, compute_dc_coeffs,
                                "Compute DC coeffs"));

  return true;
}

Status ComputeVarDCTEncodingData(const FrameHeader& frame_header,
                                 const Image3F* linear,
                                 Image3F* JXL_RESTRICT opsin, const Rect& rect,
                                 const JxlCmsInterface& cms, ThreadPool* pool,
                                 ModularFrameEncoder* enc_modular,
                                 PassesEncoderState* enc_state,
                                 AuxOut* aux_out) {
  JXL_ENSURE((rect.xsize() % kBlockDim) == 0 &&
             (rect.ysize() % kBlockDim) == 0);
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  // Save pre-Gaborish opsin for AR control field heuristics computation.
  Image3F orig_opsin;
  JXL_ASSIGN_OR_RETURN(
      orig_opsin, Image3F::Create(memory_manager, rect.xsize(), rect.ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(rect, *opsin, Rect(orig_opsin), &orig_opsin));
  JXL_RETURN_IF_ERROR(orig_opsin.ShrinkTo(enc_state->shared.frame_dim.xsize,
                                          enc_state->shared.frame_dim.ysize));

  JXL_RETURN_IF_ERROR(LossyFrameHeuristics(frame_header, enc_state, enc_modular,
                                           linear, opsin, rect, cms, pool,
                                           aux_out));

  JXL_RETURN_IF_ERROR(InitializePassesEncoder(
      frame_header, *opsin, rect, cms, pool, enc_state, enc_modular, aux_out));

  JXL_RETURN_IF_ERROR(
      ComputeARHeuristics(frame_header, enc_state, orig_opsin, rect, pool));

  JXL_RETURN_IF_ERROR(ComputeACMetadata(pool, enc_state, enc_modular));

  return true;
}

Status ComputeAllCoeffOrders(PassesEncoderState& enc_state,
                             const FrameDimensions& frame_dim) {
  auto used_orders_info = ComputeUsedOrders(
      enc_state.cparams.speed_tier, enc_state.shared.ac_strategy,
      Rect(enc_state.shared.raw_quant_field));
  enc_state.used_orders.resize(enc_state.progressive_splitter.GetNumPasses());
  for (size_t i = 0; i < enc_state.progressive_splitter.GetNumPasses(); i++) {
    JXL_RETURN_IF_ERROR(ComputeCoeffOrder(
        enc_state.cparams.speed_tier, *enc_state.coeffs[i],
        enc_state.shared.ac_strategy, frame_dim, enc_state.used_orders[i],
        enc_state.used_acs, used_orders_info.first, used_orders_info.second,
        &enc_state.shared.coeff_orders[i * enc_state.shared.coeff_order_size]));
  }
  enc_state.used_acs |= used_orders_info.first;
  return true;
}

// Working area for TokenizeCoefficients (per-group!)
struct EncCache {
  // Allocates memory when first called.
  Status InitOnce(JxlMemoryManager* memory_manager) {
    if (num_nzeroes.xsize() == 0) {
      JXL_ASSIGN_OR_RETURN(num_nzeroes,
                           Image3I::Create(memory_manager, kGroupDimInBlocks,
                                           kGroupDimInBlocks));
    }
    return true;
  }
  // TokenizeCoefficients
  Image3I num_nzeroes;
};

Status TokenizeAllCoefficients(const FrameHeader& frame_header,
                               ThreadPool* pool,
                               PassesEncoderState* enc_state) {
  PassesSharedState& shared = enc_state->shared;
  std::vector<EncCache> group_caches;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  const auto tokenize_group_init = [&](const size_t num_threads) -> Status {
    group_caches.resize(num_threads);
    return true;
  };
  const auto tokenize_group = [&](const uint32_t group_index,
                                  const size_t thread) -> Status {
    // Tokenize coefficients.
    const Rect rect = shared.frame_dim.BlockGroupRect(group_index);
    for (size_t idx_pass = 0; idx_pass < enc_state->passes.size(); idx_pass++) {
      JXL_ENSURE(enc_state->coeffs[idx_pass]->Type() == ACType::k32);
      const int32_t* JXL_RESTRICT ac_rows[3] = {
          enc_state->coeffs[idx_pass]->PlaneRow(0, group_index, 0).ptr32,
          enc_state->coeffs[idx_pass]->PlaneRow(1, group_index, 0).ptr32,
          enc_state->coeffs[idx_pass]->PlaneRow(2, group_index, 0).ptr32,
      };
      // Ensure group cache is initialized.
      JXL_RETURN_IF_ERROR(group_caches[thread].InitOnce(memory_manager));
      JXL_RETURN_IF_ERROR(TokenizeCoefficients(
          &shared.coeff_orders[idx_pass * shared.coeff_order_size], rect,
          ac_rows, shared.ac_strategy, frame_header.chroma_subsampling,
          &group_caches[thread].num_nzeroes,
          &enc_state->passes[idx_pass].ac_tokens[group_index], shared.quant_dc,
          shared.raw_quant_field, shared.block_ctx_map));
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_groups,
                                tokenize_group_init, tokenize_group,
                                "TokenizeGroup"));
  return true;
}

Status EncodeGlobalDCInfo(const PassesSharedState& shared, BitWriter* writer,
                          AuxOut* aux_out) {
  // Encode quantizer DC and global scale.
  QuantizerParams params = shared.quantizer.GetParams();
  JXL_RETURN_IF_ERROR(
      WriteQuantizerParams(params, writer, LayerType::Quant, aux_out));
  JXL_RETURN_IF_ERROR(EncodeBlockCtxMap(shared.block_ctx_map, writer, aux_out));
  JXL_RETURN_IF_ERROR(ColorCorrelationEncodeDC(shared.cmap.base(), writer,
                                               LayerType::Dc, aux_out));
  return true;
}

// In streaming mode, this function only performs the histogram clustering and
// saves the histogram bitstreams in enc_state, the actual AC global bitstream
// is written in OutputAcGlobal() function after all the groups are processed.
Status EncodeGlobalACInfo(PassesEncoderState* enc_state, BitWriter* writer,
                          ModularFrameEncoder* enc_modular, AuxOut* aux_out) {
  PassesSharedState& shared = enc_state->shared;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  JXL_RETURN_IF_ERROR(DequantMatricesEncode(memory_manager, shared.matrices,
                                            writer, LayerType::Quant, aux_out,
                                            enc_modular));
  size_t num_histo_bits = CeilLog2Nonzero(shared.frame_dim.num_groups);
  if (!enc_state->streaming_mode && num_histo_bits != 0) {
    JXL_RETURN_IF_ERROR(
        writer->WithMaxBits(num_histo_bits, LayerType::Ac, aux_out, [&] {
          writer->Write(num_histo_bits, shared.num_histograms - 1);
          return true;
        }));
  }

  for (size_t i = 0; i < enc_state->progressive_splitter.GetNumPasses(); i++) {
    // Encode coefficient orders.
    if (!enc_state->streaming_mode) {
      size_t order_bits = 0;
      JXL_RETURN_IF_ERROR(U32Coder::CanEncode(
          kOrderEnc, enc_state->used_orders[i], &order_bits));
      JXL_RETURN_IF_ERROR(
          writer->WithMaxBits(order_bits, LayerType::Order, aux_out, [&] {
            return U32Coder::Write(kOrderEnc, enc_state->used_orders[i],
                                   writer);
          }));
      JXL_RETURN_IF_ERROR(
          EncodeCoeffOrders(enc_state->used_orders[i],
                            &shared.coeff_orders[i * shared.coeff_order_size],
                            writer, LayerType::Order, aux_out));
    }

    // Encode histograms.
    HistogramParams hist_params(enc_state->cparams.speed_tier,
                                shared.block_ctx_map.NumACContexts());
    if (enc_state->cparams.speed_tier > SpeedTier::kTortoise) {
      hist_params.lz77_method = HistogramParams::LZ77Method::kNone;
    }
    if (enc_state->cparams.decoding_speed_tier >= 1) {
      hist_params.max_histograms = 6;
    }
    size_t num_histogram_groups = shared.num_histograms;
    if (enc_state->streaming_mode) {
      size_t prev_num_histograms =
          enc_state->passes[i].codes.encoding_info.size();
      if (enc_state->initialize_global_state) {
        prev_num_histograms += kNumFixedHistograms;
        hist_params.add_fixed_histograms = true;
      }
      size_t remaining_histograms = kClustersLimit - prev_num_histograms;
      // Heuristic to assign budget of new histograms to DC groups.
      // TODO(szabadka) Tune this together with the DC group ordering.
      size_t max_histograms = remaining_histograms < 20
                                  ? std::min<size_t>(remaining_histograms, 4)
                                  : remaining_histograms / 4;
      hist_params.max_histograms =
          std::min(max_histograms, hist_params.max_histograms);
      num_histogram_groups = 1;
    }
    hist_params.streaming_mode = enc_state->streaming_mode;
    hist_params.initialize_global_state = enc_state->initialize_global_state;
    JXL_ASSIGN_OR_RETURN(
        size_t cost,
        BuildAndEncodeHistograms(
            memory_manager, hist_params,
            num_histogram_groups * shared.block_ctx_map.NumACContexts(),
            enc_state->passes[i].ac_tokens, &enc_state->passes[i].codes, writer,
            LayerType::Ac, aux_out));
    (void)cost;
  }

  return true;
}
Status EncodeGroups(const FrameHeader& frame_header,
                    PassesEncoderState* enc_state,
                    ModularFrameEncoder* enc_modular, ThreadPool* pool,
                    std::vector<std::unique_ptr<BitWriter>>* group_codes,
                    AuxOut* aux_out) {
  const PassesSharedState& shared = enc_state->shared;
  JxlMemoryManager* memory_manager = shared.memory_manager;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const size_t num_groups = frame_dim.num_groups;
  const size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
  const size_t global_ac_index = frame_dim.num_dc_groups + 1;
  const bool is_small_image =
      !enc_state->streaming_mode && num_groups == 1 && num_passes == 1;
  const size_t num_toc_entries =
      is_small_image ? 1
                     : AcGroupIndex(0, 0, num_groups, frame_dim.num_dc_groups) +
                           num_groups * num_passes;
  JXL_ENSURE(group_codes->empty());
  group_codes->reserve(num_toc_entries);
  for (size_t i = 0; i < num_toc_entries; ++i) {
    group_codes->emplace_back(jxl::make_unique<BitWriter>(memory_manager));
  }

  const auto get_output = [&](const size_t index) -> BitWriter* {
    return (*group_codes)[is_small_image ? 0 : index].get();
  };
  auto ac_group_code = [&](size_t pass, size_t group) {
    return get_output(AcGroupIndex(pass, group, frame_dim.num_groups,
                                   frame_dim.num_dc_groups));
  };

  if (enc_state->initialize_global_state) {
    if (frame_header.flags & FrameHeader::kPatches) {
      JXL_RETURN_IF_ERROR(PatchDictionaryEncoder::Encode(
          shared.image_features.patches, get_output(0), LayerType::Dictionary,
          aux_out));
    }
    if (frame_header.flags & FrameHeader::kSplines) {
      JXL_RETURN_IF_ERROR(EncodeSplines(shared.image_features.splines,
                                        get_output(0), LayerType::Splines,
                                        HistogramParams(), aux_out));
    }
    if (frame_header.flags & FrameHeader::kNoise) {
      JXL_RETURN_IF_ERROR(EncodeNoise(shared.image_features.noise_params,
                                      get_output(0), LayerType::Noise,
                                      aux_out));
    }

    JXL_RETURN_IF_ERROR(DequantMatricesEncodeDC(shared.matrices, get_output(0),
                                                LayerType::Quant, aux_out));
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      JXL_RETURN_IF_ERROR(EncodeGlobalDCInfo(shared, get_output(0), aux_out));
    }
    JXL_RETURN_IF_ERROR(enc_modular->EncodeGlobalInfo(enc_state->streaming_mode,
                                                      get_output(0), aux_out));
    JXL_RETURN_IF_ERROR(enc_modular->EncodeStream(get_output(0), aux_out,
                                                  LayerType::ModularGlobal,
                                                  ModularStreamId::Global()));
  }

  std::vector<std::unique_ptr<AuxOut>> aux_outs;
  auto resize_aux_outs = [&aux_outs,
                          aux_out](const size_t num_threads) -> Status {
    if (aux_out == nullptr) {
      aux_outs.resize(num_threads);
    } else {
      while (aux_outs.size() > num_threads) {
        aux_out->Assimilate(*aux_outs.back());
        aux_outs.pop_back();
      }
      while (num_threads > aux_outs.size()) {
        aux_outs.emplace_back(jxl::make_unique<AuxOut>());
      }
    }
    return true;
  };

  const auto process_dc_group = [&](const uint32_t group_index,
                                    const size_t thread) -> Status {
    AuxOut* my_aux_out = aux_outs[thread].get();
    uint32_t input_index = enc_state->streaming_mode ? 0 : group_index;
    BitWriter* output = get_output(input_index + 1);
    if (frame_header.encoding == FrameEncoding::kVarDCT &&
        !(frame_header.flags & FrameHeader::kUseDcFrame)) {
      JXL_RETURN_IF_ERROR(
          output->WithMaxBits(2, LayerType::Dc, my_aux_out, [&] {
            output->Write(2, enc_modular->extra_dc_precision[group_index]);
            return true;
          }));
      JXL_RETURN_IF_ERROR(
          enc_modular->EncodeStream(output, my_aux_out, LayerType::Dc,
                                    ModularStreamId::VarDCTDC(group_index)));
    }
    JXL_RETURN_IF_ERROR(
        enc_modular->EncodeStream(output, my_aux_out, LayerType::ModularDcGroup,
                                  ModularStreamId::ModularDC(group_index)));
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      const Rect& rect = enc_state->shared.frame_dim.DCGroupRect(input_index);
      size_t nb_bits = CeilLog2Nonzero(rect.xsize() * rect.ysize());
      if (nb_bits != 0) {
        JXL_RETURN_IF_ERROR(output->WithMaxBits(
            nb_bits, LayerType::ControlFields, my_aux_out, [&] {
              output->Write(nb_bits,
                            enc_modular->ac_metadata_size[group_index] - 1);
              return true;
            }));
      }
      JXL_RETURN_IF_ERROR(enc_modular->EncodeStream(
          output, my_aux_out, LayerType::ControlFields,
          ModularStreamId::ACMetadata(group_index)));
    }
    return true;
  };
  if (enc_state->streaming_mode) {
    JXL_ENSURE(frame_dim.num_dc_groups == 1);
    JXL_RETURN_IF_ERROR(resize_aux_outs(1));
    JXL_RETURN_IF_ERROR(process_dc_group(enc_state->dc_group_index, 0));
  } else {
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_dc_groups,
                                  resize_aux_outs, process_dc_group,
                                  "EncodeDCGroup"));
  }
  if (frame_header.encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(EncodeGlobalACInfo(
        enc_state, get_output(global_ac_index), enc_modular, aux_out));
  }

  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) -> Status {
    AuxOut* my_aux_out = aux_outs[thread].get();

    size_t ac_group_id =
        enc_state->streaming_mode
            ? enc_modular->ComputeStreamingAbsoluteAcGroupId(
                  enc_state->dc_group_index, group_index, shared.frame_dim)
            : group_index;

    for (size_t i = 0; i < num_passes; i++) {
      JXL_DEBUG_V(2, "Encoding AC group %u [abs %" PRIuS "] pass %" PRIuS,
                  group_index, ac_group_id, i);
      if (frame_header.encoding == FrameEncoding::kVarDCT) {
        JXL_RETURN_IF_ERROR(EncodeGroupTokenizedCoefficients(
            group_index, i, enc_state->histogram_idx[group_index], *enc_state,
            ac_group_code(i, group_index), my_aux_out));
      }
      // Write all modular encoded data (color?, alpha, depth, extra channels)
      JXL_RETURN_IF_ERROR(enc_modular->EncodeStream(
          ac_group_code(i, group_index), my_aux_out, LayerType::ModularAcGroup,
          ModularStreamId::ModularAC(ac_group_id, i)));
      JXL_DEBUG_V(2,
                  "AC group %u [abs %" PRIuS "] pass %" PRIuS
                  " encoded size is %" PRIuS " bits",
                  group_index, ac_group_id, i,
                  ac_group_code(i, group_index)->BitsWritten());
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_groups, resize_aux_outs,
                                process_group, "EncodeGroupCoefficients"));
  // Resizing aux_outs to 0 also Assimilates the array.
  static_cast<void>(resize_aux_outs(0));

  for (std::unique_ptr<BitWriter>& bw : *group_codes) {
    JXL_RETURN_IF_ERROR(bw->WithMaxBits(8, LayerType::Ac, aux_out, [&] {
      bw->ZeroPadToByte();  // end of group.
      return true;
    }));
  }
  return true;
}

Status ComputeEncodingData(
    const CompressParams& cparams, const FrameInfo& frame_info,
    const CodecMetadata* metadata, JxlEncoderChunkedFrameAdapter& frame_data,
    const jpeg::JPEGData* jpeg_data, size_t x0, size_t y0, size_t xsize,
    size_t ysize, const JxlCmsInterface& cms, ThreadPool* pool,
    FrameHeader& mutable_frame_header, ModularFrameEncoder& enc_modular,
    PassesEncoderState& enc_state,
    std::vector<std::unique_ptr<BitWriter>>* group_codes, AuxOut* aux_out) {
  JXL_ENSURE(x0 + xsize <= frame_data.xsize);
  JXL_ENSURE(y0 + ysize <= frame_data.ysize);
  JxlMemoryManager* memory_manager = enc_state.memory_manager();
  const FrameHeader& frame_header = mutable_frame_header;
  PassesSharedState& shared = enc_state.shared;
  shared.metadata = metadata;
  if (enc_state.streaming_mode) {
    shared.frame_dim.Set(
        xsize, ysize, frame_header.group_size_shift,
        /*max_hshift=*/0, /*max_vshift=*/0,
        mutable_frame_header.encoding == FrameEncoding::kModular,
        /*upsampling=*/1);
  } else {
    shared.frame_dim = frame_header.ToFrameDimensions();
  }

  shared.image_features.patches.SetShared(&shared.reference_frames);
  const FrameDimensions& frame_dim = shared.frame_dim;
  JXL_ASSIGN_OR_RETURN(
      shared.ac_strategy,
      AcStrategyImage::Create(memory_manager, frame_dim.xsize_blocks,
                              frame_dim.ysize_blocks));
  JXL_ASSIGN_OR_RETURN(shared.raw_quant_field,
                       ImageI::Create(memory_manager, frame_dim.xsize_blocks,
                                      frame_dim.ysize_blocks));
  JXL_ASSIGN_OR_RETURN(shared.epf_sharpness,
                       ImageB::Create(memory_manager, frame_dim.xsize_blocks,
                                      frame_dim.ysize_blocks));
  JXL_ASSIGN_OR_RETURN(
      shared.cmap, ColorCorrelationMap::Create(memory_manager, frame_dim.xsize,
                                               frame_dim.ysize));
  shared.coeff_order_size = kCoeffOrderMaxSize;
  if (frame_header.encoding == FrameEncoding::kVarDCT) {
    shared.coeff_orders.resize(frame_header.passes.num_passes *
                               kCoeffOrderMaxSize);
  }

  JXL_ASSIGN_OR_RETURN(shared.quant_dc,
                       ImageB::Create(memory_manager, frame_dim.xsize_blocks,
                                      frame_dim.ysize_blocks));
  JXL_ASSIGN_OR_RETURN(shared.dc_storage,
                       Image3F::Create(memory_manager, frame_dim.xsize_blocks,
                                       frame_dim.ysize_blocks));
  shared.dc = &shared.dc_storage;

  const size_t num_extra_channels = metadata->m.num_extra_channels;
  const ExtraChannelInfo* alpha_eci = metadata->m.Find(ExtraChannel::kAlpha);
  const ExtraChannelInfo* black_eci = metadata->m.Find(ExtraChannel::kBlack);
  const size_t alpha_idx = alpha_eci - metadata->m.extra_channel_info.data();
  const size_t black_idx = black_eci - metadata->m.extra_channel_info.data();
  const ColorEncoding c_enc = metadata->m.color_encoding;

  // Make the image patch bigger than the currently processed group in
  // streaming mode so that we can take into account border pixels around the
  // group when computing inverse Gaborish and adaptive quantization map.
  int max_border = enc_state.streaming_mode ? kBlockDim : 0;
  Rect frame_rect(0, 0, frame_data.xsize, frame_data.ysize);
  Rect frame_area_rect = Rect(x0, y0, xsize, ysize);
  Rect patch_rect = frame_area_rect.Extend(max_border, frame_rect);
  JXL_ENSURE(patch_rect.IsInside(frame_rect));

  // Allocating a large enough image avoids a copy when padding.
  JXL_ASSIGN_OR_RETURN(
      Image3F color,
      Image3F::Create(memory_manager, RoundUpToBlockDim(patch_rect.xsize()),
                      RoundUpToBlockDim(patch_rect.ysize())));
  JXL_RETURN_IF_ERROR(color.ShrinkTo(patch_rect.xsize(), patch_rect.ysize()));
  std::vector<ImageF> extra_channels(num_extra_channels);
  for (auto& extra_channel : extra_channels) {
    JXL_ASSIGN_OR_RETURN(
        extra_channel,
        ImageF::Create(memory_manager, patch_rect.xsize(), patch_rect.ysize()));
  }
  ImageF* alpha = alpha_eci ? &extra_channels[alpha_idx] : nullptr;
  ImageF* black = black_eci ? &extra_channels[black_idx] : nullptr;
  bool has_interleaved_alpha = false;
  JxlChunkedFrameInputSource input = frame_data.GetInputSource();
  if (!jpeg_data) {
    JXL_RETURN_IF_ERROR(CopyColorChannels(input, patch_rect, frame_info,
                                          metadata->m, pool, &color, alpha,
                                          &has_interleaved_alpha));
  }
  JXL_RETURN_IF_ERROR(CopyExtraChannels(input, patch_rect, frame_info,
                                        metadata->m, has_interleaved_alpha,
                                        pool, &extra_channels));

  enc_state.cparams = cparams;

  Image3F linear_storage;
  Image3F* linear = nullptr;

  if (!jpeg_data) {
    if (frame_header.color_transform == ColorTransform::kXYB &&
        frame_info.ib_needs_color_transform) {
      if (frame_header.encoding == FrameEncoding::kVarDCT &&
          cparams.speed_tier <= SpeedTier::kKitten) {
        JXL_ASSIGN_OR_RETURN(linear_storage,
                             Image3F::Create(memory_manager, patch_rect.xsize(),
                                             patch_rect.ysize()));
        linear = &linear_storage;
      }
      JXL_RETURN_IF_ERROR(ToXYB(c_enc, metadata->m.IntensityTarget(), black,
                                pool, &color, cms, linear));
    } else {
      // Nothing to do.
      // RGB or YCbCr: forward YCbCr is not implemented, this is only used
      // when the input is already in YCbCr If encoding a special DC or
      // reference frame: input is already in XYB.
    }
    bool lossless = cparams.IsLossless();
    if (alpha && !alpha_eci->alpha_associated &&
        frame_header.frame_type == FrameType::kRegularFrame &&
        !ApplyOverride(cparams.keep_invisible, cparams.IsLossless()) &&
        cparams.ec_resampling == cparams.resampling &&
        !cparams.disable_perceptual_optimizations) {
      // simplify invisible pixels
      SimplifyInvisible(&color, *alpha, lossless);
      if (linear) {
        SimplifyInvisible(linear, *alpha, lossless);
      }
    }
    JXL_RETURN_IF_ERROR(PadImageToBlockMultipleInPlace(&color));
  }

  // Rectangle within color that corresponds to the currently processed group
  // in streaming mode.
  Rect group_rect(x0 - patch_rect.x0(), y0 - patch_rect.y0(),
                  RoundUpToBlockDim(xsize), RoundUpToBlockDim(ysize));

  if (enc_state.initialize_global_state && !jpeg_data) {
    ComputeChromacityAdjustments(cparams, color, group_rect,
                                 &mutable_frame_header);
  }

  bool has_jpeg_data = (jpeg_data != nullptr);
  ComputeNoiseParams(cparams, enc_state.streaming_mode, has_jpeg_data, color,
                     frame_dim, &mutable_frame_header,
                     &shared.image_features.noise_params);

  JXL_RETURN_IF_ERROR(
      DownsampleColorChannels(cparams, frame_header, has_jpeg_data, &color));

  if (cparams.ec_resampling != 1 && !cparams.already_downsampled) {
    for (ImageF& ec : extra_channels) {
      JXL_ASSIGN_OR_RETURN(ec, DownsampleImage(ec, cparams.ec_resampling));
    }
  }

  if (!enc_state.streaming_mode) {
    group_rect = Rect(color);
  }

  if (frame_header.encoding == FrameEncoding::kVarDCT) {
    enc_state.passes.resize(enc_state.progressive_splitter.GetNumPasses());
    for (PassesEncoderState::PassData& pass : enc_state.passes) {
      pass.ac_tokens.resize(shared.frame_dim.num_groups);
    }
    if (jpeg_data) {
      JXL_RETURN_IF_ERROR(ComputeJPEGTranscodingData(
          *jpeg_data, frame_header, pool, &enc_modular, &enc_state));
    } else {
      JXL_RETURN_IF_ERROR(ComputeVarDCTEncodingData(
          frame_header, linear, &color, group_rect, cms, pool, &enc_modular,
          &enc_state, aux_out));
    }
    JXL_RETURN_IF_ERROR(ComputeAllCoeffOrders(enc_state, frame_dim));
    if (!enc_state.streaming_mode) {
      shared.num_histograms = 1;
      enc_state.histogram_idx.resize(frame_dim.num_groups);
    }
    JXL_RETURN_IF_ERROR(
        TokenizeAllCoefficients(frame_header, pool, &enc_state));
  }

  if (cparams.modular_mode || !extra_channels.empty()) {
    JXL_RETURN_IF_ERROR(enc_modular.ComputeEncodingData(
        frame_header, metadata->m, &color, extra_channels, group_rect,
        frame_dim, frame_area_rect, &enc_state, cms, pool, aux_out,
        /*do_color=*/cparams.modular_mode));
  }

  if (!enc_state.streaming_mode) {
    // If checks pass here and CanDoStreamingEncoding returns false
    // a Global MA tree is used.
    if (cparams.speed_tier < SpeedTier::kTortoise ||
        !cparams.ModularPartIsLossless() || cparams.lossy_palette ||
        // Allow Local trees for progressive lossless.
        // TODO(Jonnyawsom3): Figure out how to allow local trees for
        // extra channels on VarDCT images without failing tests.
        (!(cparams.responsive == 1 && cparams.IsLossless()) &&
         cparams.buffering < 3) ||
        !cparams.custom_fixed_tree.empty()) {
      JXL_RETURN_IF_ERROR(enc_modular.ComputeTree(pool));
      JXL_RETURN_IF_ERROR(enc_modular.ComputeTokens(pool));
    }
    mutable_frame_header.UpdateFlag(shared.image_features.patches.HasAny(),
                                    FrameHeader::kPatches);
    mutable_frame_header.UpdateFlag(shared.image_features.splines.HasAny(),
                                    FrameHeader::kSplines);
  }

  JXL_RETURN_IF_ERROR(EncodeGroups(frame_header, &enc_state, &enc_modular, pool,
                                   group_codes, aux_out));
  if (enc_state.streaming_mode) {
    const size_t group_index = enc_state.dc_group_index;
    enc_modular.ClearStreamData(ModularStreamId::VarDCTDC(group_index));
    enc_modular.ClearStreamData(ModularStreamId::ACMetadata(group_index));
    enc_modular.ClearModularStreamData();
  }
  return true;
}

Status PermuteGroups(const CompressParams& cparams,
                     const FrameDimensions& frame_dim, size_t num_passes,
                     std::vector<coeff_order_t>* permutation,
                     std::vector<std::unique_ptr<BitWriter>>* group_codes) {
  const size_t num_groups = frame_dim.num_groups;
  if (!cparams.centerfirst || (num_passes == 1 && num_groups == 1)) {
    return true;
  }
  // Don't permute global DC/AC or DC.
  permutation->resize(frame_dim.num_dc_groups + 2);
  std::iota(permutation->begin(), permutation->end(), 0);
  std::vector<coeff_order_t> ac_group_order(num_groups);
  std::iota(ac_group_order.begin(), ac_group_order.end(), 0);
  size_t group_dim = frame_dim.group_dim;

  // The center of the image is either given by parameters or chosen
  // to be the middle of the image by default if center_x, center_y resp.
  // are not provided.

  int64_t imag_cx;
  if (cparams.center_x != static_cast<size_t>(-1)) {
    JXL_RETURN_IF_ERROR(cparams.center_x < frame_dim.xsize);
    imag_cx = cparams.center_x;
  } else {
    imag_cx = frame_dim.xsize / 2;
  }

  int64_t imag_cy;
  if (cparams.center_y != static_cast<size_t>(-1)) {
    JXL_RETURN_IF_ERROR(cparams.center_y < frame_dim.ysize);
    imag_cy = cparams.center_y;
  } else {
    imag_cy = frame_dim.ysize / 2;
  }

  // The center of the group containing the center of the image.
  int64_t cx = (imag_cx / group_dim) * group_dim + group_dim / 2;
  int64_t cy = (imag_cy / group_dim) * group_dim + group_dim / 2;
  // This identifies in what area of the central group the center of the image
  // lies in.
  double direction = -std::atan2(imag_cy - cy, imag_cx - cx);
  // This identifies the side of the central group the center of the image
  // lies closest to. This can take values 0, 1, 2, 3 corresponding to left,
  // bottom, right, top.
  int64_t side = std::fmod((direction + 5 * kPi / 4), 2 * kPi) * 2 / kPi;
  auto get_distance_from_center = [&](size_t gid) {
    Rect r = frame_dim.GroupRect(gid);
    int64_t gcx = r.x0() + group_dim / 2;
    int64_t gcy = r.y0() + group_dim / 2;
    int64_t dx = gcx - cx;
    int64_t dy = gcy - cy;
    // The angle is determined by taking atan2 and adding an appropriate
    // starting point depending on the side we want to start on.
    double angle = std::remainder(
        std::atan2(dy, dx) + kPi / 4 + side * (kPi / 2), 2 * kPi);
    // Concentric squares in clockwise order.
    return std::make_pair(std::max(std::abs(dx), std::abs(dy)), angle);
  };
  std::sort(ac_group_order.begin(), ac_group_order.end(),
            [&](coeff_order_t a, coeff_order_t b) {
              return get_distance_from_center(a) < get_distance_from_center(b);
            });
  std::vector<coeff_order_t> inv_ac_group_order(ac_group_order.size(), 0);
  for (size_t i = 0; i < ac_group_order.size(); i++) {
    inv_ac_group_order[ac_group_order[i]] = i;
  }
  for (size_t i = 0; i < num_passes; i++) {
    size_t pass_start = permutation->size();
    for (coeff_order_t v : inv_ac_group_order) {
      permutation->push_back(pass_start + v);
    }
  }
  if (group_codes == nullptr) return true;
  std::vector<std::unique_ptr<BitWriter>> new_group_codes(group_codes->size());
  for (size_t i = 0; i < permutation->size(); i++) {
    new_group_codes[(*permutation)[i]] = std::move((*group_codes)[i]);
  }
  group_codes->swap(new_group_codes);
  return true;
}

bool CanDoStreamingEncoding(const CompressParams& cparams,
                            const FrameInfo& frame_info,
                            const CodecMetadata& metadata,
                            const JxlEncoderChunkedFrameAdapter& frame_data) {
  if (cparams.buffering == -1) {
    if (cparams.speed_tier < SpeedTier::kTortoise) return false;
    if (cparams.speed_tier < SpeedTier::kSquirrel &&
        cparams.butteraugli_distance > 0.5f) {
      return false;
    }
    if (cparams.speed_tier == SpeedTier::kSquirrel &&
        cparams.butteraugli_distance >= 3.f) {
      return false;
    }
  }
  if (cparams.buffering == 0) {
    return false;
  }
  if (cparams.buffering == 1 && frame_data.xsize <= 2048 &&
      frame_data.ysize <= 2048) {
    return false;
  }

  // Random heuristic: disable streaming for frames with less than 8 groups.
  // (too little speed benefit, too much compression penalty)
  // Can change this but have to make sure numgroups > 1, because the streaming
  // path assumes that.
  if (NumGroupsForFrame(frame_data.xsize, frame_data.ysize, cparams) <= 8) {
    return false;
  }
  if (frame_data.IsJPEG()) {
    return false;
  }
  if (cparams.noise == Override::kOn || cparams.patches == Override::kOn) {
    return false;
  }
  if (cparams.progressive_dc != 0 || frame_info.dc_level != 0) {
    return false;
  }
  if (cparams.custom_progressive_mode ||
      cparams.qprogressive_mode == Override::kOn ||
      cparams.progressive_mode == Override::kOn) {
    return false;
  }
  if (cparams.resampling != 1 || cparams.ec_resampling != 1) {
    return false;
  }
  if (cparams.lossy_palette) {
    return false;
  }
  if (cparams.max_error_mode) {
    return false;
  }
  // Progressive lossless uses Local MA trees, but requires a full
  // buffer to compress well, so no special check.
  if (!cparams.ModularPartIsLossless() || cparams.responsive > 0) {
    if (metadata.m.num_extra_channels > 0 || cparams.modular_mode) {
      return false;
    }
  }
  ColorTransform ok_color_transform =
      cparams.modular_mode ? ColorTransform::kNone : ColorTransform::kXYB;
  if (cparams.color_transform != ok_color_transform) {
    return false;
  }
  return true;
}

Status ComputePermutationForStreaming(size_t xsize, size_t ysize,
                                      size_t group_size, size_t num_passes,
                                      std::vector<coeff_order_t>& permutation,
                                      std::vector<size_t>& dc_group_order) {
  // This is only valid in VarDCT mode, otherwise there can be group shift.
  const size_t dc_group_size = group_size * kBlockDim;
  const size_t group_xsize = DivCeil(xsize, group_size);
  const size_t group_ysize = DivCeil(ysize, group_size);
  const size_t dc_group_xsize = DivCeil(xsize, dc_group_size);
  const size_t dc_group_ysize = DivCeil(ysize, dc_group_size);
  const size_t num_groups = group_xsize * group_ysize;
  const size_t num_dc_groups = dc_group_xsize * dc_group_ysize;
  const size_t num_sections = 2 + num_dc_groups + num_passes * num_groups;
  permutation.resize(num_sections);
  size_t new_ix = 0;
  // DC Global is first
  permutation[0] = new_ix++;
  // TODO(szabadka) Change the dc group order to center-first.
  for (size_t dc_y = 0; dc_y < dc_group_ysize; ++dc_y) {
    for (size_t dc_x = 0; dc_x < dc_group_xsize; ++dc_x) {
      size_t dc_ix = dc_y * dc_group_xsize + dc_x;
      dc_group_order.push_back(dc_ix);
      permutation[1 + dc_ix] = new_ix++;
      size_t ac_y0 = dc_y * kBlockDim;
      size_t ac_x0 = dc_x * kBlockDim;
      size_t ac_y1 = std::min<size_t>(group_ysize, ac_y0 + kBlockDim);
      size_t ac_x1 = std::min<size_t>(group_xsize, ac_x0 + kBlockDim);
      for (size_t pass = 0; pass < num_passes; ++pass) {
        for (size_t ac_y = ac_y0; ac_y < ac_y1; ++ac_y) {
          for (size_t ac_x = ac_x0; ac_x < ac_x1; ++ac_x) {
            size_t group_ix = ac_y * group_xsize + ac_x;
            size_t old_ix =
                AcGroupIndex(pass, group_ix, num_groups, num_dc_groups);
            permutation[old_ix] = new_ix++;
          }
        }
      }
    }
  }
  // AC Global is last
  permutation[1 + num_dc_groups] = new_ix++;
  JXL_ENSURE(new_ix == num_sections);
  return true;
}

constexpr size_t kGroupSizeOffset[4] = {
    static_cast<size_t>(0),
    static_cast<size_t>(1024),
    static_cast<size_t>(17408),
    static_cast<size_t>(4211712),
};
constexpr size_t kTOCBits[4] = {12, 16, 24, 32};

size_t TOCBucket(size_t group_size) {
  size_t bucket = 0;
  while (bucket < 3 && group_size >= kGroupSizeOffset[bucket + 1]) ++bucket;
  return bucket;
}

size_t TOCSize(const std::vector<size_t>& group_sizes) {
  size_t toc_bits = 0;
  for (size_t group_size : group_sizes) {
    toc_bits += kTOCBits[TOCBucket(group_size)];
  }
  return (toc_bits + 7) / 8;
}

Status ComputeGroupDataOffset(size_t frame_header_size, size_t dc_global_size,
                              size_t num_sections, size_t& min_dc_global_size,
                              size_t& group_offset) {
  size_t max_toc_bits = (num_sections - 1) * 32;
  size_t min_toc_bits = (num_sections - 1) * 12;
  size_t max_padding = (max_toc_bits - min_toc_bits + 7) / 8;
  min_dc_global_size = dc_global_size;
  size_t dc_global_bucket = TOCBucket(min_dc_global_size);
  while (TOCBucket(min_dc_global_size + max_padding) > dc_global_bucket) {
    dc_global_bucket = TOCBucket(min_dc_global_size + max_padding);
    min_dc_global_size = kGroupSizeOffset[dc_global_bucket];
  }
  JXL_ENSURE(TOCBucket(min_dc_global_size) == dc_global_bucket);
  JXL_ENSURE(TOCBucket(min_dc_global_size + max_padding) == dc_global_bucket);
  max_toc_bits += kTOCBits[dc_global_bucket];
  size_t max_toc_size = (max_toc_bits + 7) / 8;
  group_offset = frame_header_size + max_toc_size + min_dc_global_size;
  return true;
}

size_t ComputeDcGlobalPadding(const std::vector<size_t>& group_sizes,
                              size_t frame_header_size,
                              size_t group_data_offset,
                              size_t min_dc_global_size) {
  std::vector<size_t> new_group_sizes = group_sizes;
  new_group_sizes[0] = min_dc_global_size;
  size_t toc_size = TOCSize(new_group_sizes);
  size_t actual_offset = frame_header_size + toc_size + group_sizes[0];
  return group_data_offset - actual_offset;
}

// write_fn is called once per section (dc_group, then each ac_group).
Status OutputGroups(std::vector<std::unique_ptr<BitWriter>>&& group_codes,
                    std::function<Status(PaddedBytes&&)> write_fn) {
  JXL_ENSURE(group_codes.size() >= 4);
  JXL_RETURN_IF_ERROR(write_fn(std::move(*group_codes[1]).TakeBytes()));
  for (size_t i = 3; i < group_codes.size(); ++i) {
    JXL_RETURN_IF_ERROR(write_fn(std::move(*group_codes[i]).TakeBytes()));
  }
  return true;
}

void RemoveUnusedHistograms(EntropyEncodingData& codes) {
  std::vector<int> remap(256, -1);
  std::vector<uint8_t> inv_remap;
  for (uint8_t& context : codes.context_map) {
    const uint8_t histo_ix = context;
    if (remap[histo_ix] == -1) {
      remap[histo_ix] = inv_remap.size();
      inv_remap.push_back(histo_ix);
    }
    context = remap[histo_ix];
  }
  EntropyEncodingData new_codes;
  new_codes.use_prefix_code = codes.use_prefix_code;
  new_codes.lz77 = codes.lz77;
  new_codes.context_map = std::move(codes.context_map);
  for (uint8_t histo_idx : inv_remap) {
    new_codes.encoding_info.emplace_back(
        std::move(codes.encoding_info[histo_idx]));
    new_codes.uint_config.emplace_back(codes.uint_config[histo_idx]);
    new_codes.encoded_histograms.emplace_back(
        std::move(codes.encoded_histograms[histo_idx]));
  }
  codes = std::move(new_codes);
}

StatusOr<std::unique_ptr<BitWriter>> OutputAcGlobal(
    PassesEncoderState& enc_state, const FrameDimensions& frame_dim,
    AuxOut* aux_out) {
  JXL_ENSURE(frame_dim.num_groups > 1);
  JxlMemoryManager* memory_manager = enc_state.memory_manager();
  std::unique_ptr<BitWriter> writer =
      jxl::make_unique<BitWriter>(memory_manager);
  {
    size_t num_histo_bits = CeilLog2Nonzero(frame_dim.num_groups);
    JXL_RETURN_IF_ERROR(
        writer->WithMaxBits(num_histo_bits + 1, LayerType::Ac, aux_out, [&] {
          writer->Write(1, 1);  // default dequant matrices
          writer->Write(num_histo_bits, frame_dim.num_dc_groups - 1);
          return true;
        }));
  }
  const PassesSharedState& shared = enc_state.shared;
  for (size_t i = 0; i < enc_state.progressive_splitter.GetNumPasses(); i++) {
    // Encode coefficient orders.
    size_t order_bits = 0;
    JXL_RETURN_IF_ERROR(
        U32Coder::CanEncode(kOrderEnc, enc_state.used_orders[i], &order_bits));
    JXL_RETURN_IF_ERROR(
        writer->WithMaxBits(order_bits, LayerType::Order, aux_out, [&] {
          return U32Coder::Write(kOrderEnc, enc_state.used_orders[i],
                                 writer.get());
        }));
    JXL_RETURN_IF_ERROR(
        EncodeCoeffOrders(enc_state.used_orders[i],
                          &shared.coeff_orders[i * shared.coeff_order_size],
                          writer.get(), LayerType::Order, aux_out));
    // Fix up context map and entropy codes to remove any fix histograms that
    // were not selected by clustering.
    RemoveUnusedHistograms(enc_state.passes[i].codes);
    JXL_RETURN_IF_ERROR(EncodeHistograms(enc_state.passes[i].codes,
                                         writer.get(), LayerType::Ac, aux_out));
  }
  JXL_RETURN_IF_ERROR(writer->WithMaxBits(8, LayerType::Ac, aux_out, [&] {
    writer->ZeroPadToByte();  // end of group.
    return true;
  }));
  return writer;
}

JXL_NOINLINE Status EncodeFrameStreaming(
    JxlMemoryManager* memory_manager, const CompressParams& cparams,
    const FrameInfo& frame_info, const CodecMetadata* metadata,
    JxlEncoderChunkedFrameAdapter& frame_data, const JxlCmsInterface& cms,
    ThreadPool* pool, JxlEncoderOutputProcessorWrapper* output_processor,
    AuxOut* aux_out, uint32_t* jxlp_counter) {
  const int output_mode = cparams.output_mode;
  auto enc_state = jxl::make_unique<PassesEncoderState>(memory_manager);
  SetProgressiveMode(cparams, &enc_state->progressive_splitter);
  FrameHeader frame_header(metadata);
  std::unique_ptr<jpeg::JPEGData> jpeg_data;
  if (frame_data.IsJPEG()) {
    jpeg_data = frame_data.TakeJPEGData();
    JXL_ENSURE(jpeg_data);
  }
  JXL_RETURN_IF_ERROR(MakeFrameHeader(frame_data.xsize, frame_data.ysize,
                                      cparams, enc_state->progressive_splitter,
                                      frame_info, jpeg_data.get(), true,
                                      &frame_header));
  const size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
  JXL_ASSIGN_OR_RETURN(
      auto enc_modular,
      ModularFrameEncoder::Create(memory_manager, frame_header, cparams, true));
  std::vector<coeff_order_t> permutation;
  std::vector<size_t> dc_group_order;
  size_t group_size = frame_header.ToFrameDimensions().group_dim;
  JXL_RETURN_IF_ERROR(ComputePermutationForStreaming(
      frame_data.xsize, frame_data.ysize, group_size, num_passes, permutation,
      dc_group_order));
  enc_state->shared.num_histograms = dc_group_order.size();
  size_t dc_group_size = group_size * kBlockDim;
  size_t dc_group_xsize = DivCeil(frame_data.xsize, dc_group_size);
  size_t min_dc_global_size = 0;
  size_t group_data_offset = 0;
  PaddedBytes frame_header_bytes{memory_manager};
  PaddedBytes dc_global_bytes{memory_manager};
  std::vector<size_t> group_sizes;
  const bool streaming_output = (output_mode == 1);
  const bool streaming = (output_mode >= 1);
  const bool ooo = (output_mode == 2 && jxlp_counter != nullptr);
  size_t start_pos = (streaming_output && output_processor)
                         ? output_processor->CurrentPosition()
                         : 0;
  const size_t total_sections =
      NumTocEntries(frame_header.ToFrameDimensions().num_groups,
                    dc_group_order.size(), num_passes);
  std::vector<std::unique_ptr<BitWriter>> global_group_codes(
      streaming ? 0 : total_sections);

  // Compute group_order_perm for all modes (centerfirst, if requested).
  // For ooo mode, also compute inv_perm (encoding_pos → canonical_idx).
  std::vector<coeff_order_t> group_order_perm;
  JXL_RETURN_IF_ERROR(PermuteGroups(cparams, frame_header.ToFrameDimensions(),
                                    num_passes, &group_order_perm, nullptr));
  size_t encoding_pos = 0;
  std::vector<size_t> inv_perm;
  if (ooo) {
    inv_perm.resize(permutation.size());
    for (size_t j = 0; j < permutation.size(); j++) {
      inv_perm[permutation[j]] = j;
    }
  }

  auto write_jxlp = [&](uint32_t counter, bool is_last,
                        Span<const uint8_t> data) -> Status {
    uint8_t hdr[kLargeBoxHeaderSize + 4];
    const size_t hdr_size =
        WriteBoxHeader(MakeBoxType("jxlp"), 4 + data.size(),
                       /*unbounded=*/false, /*force_large_box=*/false, hdr);
    StoreBE32(counter | (is_last ? 0x80000000u : 0u), hdr + hdr_size);
    JXL_RETURN_IF_ERROR(
        AppendData(*output_processor, Span<const uint8_t>(hdr, hdr_size + 4)));
    return AppendData(*output_processor, data);
  };

  const uint32_t jxlp_base = jxlp_counter ? *jxlp_counter : 0;

  auto write_group_section = [&](PaddedBytes&& bytes) -> Status {
    group_sizes.push_back(bytes.size());
    if (ooo) {
      const size_t canonical_idx = inv_perm[encoding_pos++];
      const size_t cs_pos = group_order_perm.empty()
                                ? canonical_idx
                                : group_order_perm[canonical_idx];
      const bool is_last = (cs_pos + 1 == total_sections) && frame_info.is_last;
      return write_jxlp(jxlp_base + static_cast<uint32_t>(cs_pos) + 2, is_last,
                        Span<const uint8_t>(bytes.data(), bytes.size()));
    }
    return AppendData(*output_processor, bytes);
  };
  for (size_t i = 0; i < dc_group_order.size(); ++i) {
    size_t dc_ix = dc_group_order[i];
    size_t dc_y = dc_ix / dc_group_xsize;
    size_t dc_x = dc_ix % dc_group_xsize;
    size_t y0 = dc_y * dc_group_size;
    size_t x0 = dc_x * dc_group_size;
    size_t ysize = std::min<size_t>(dc_group_size, frame_data.ysize - y0);
    size_t xsize = std::min<size_t>(dc_group_size, frame_data.xsize - x0);
    size_t group_xsize = DivCeil(xsize, group_size);
    size_t group_ysize = DivCeil(ysize, group_size);
    JXL_DEBUG_V(2,
                "Encoding DC group #%" PRIuS " dc_y = %" PRIuS " dc_x = %" PRIuS
                " (x0, y0) = (%" PRIuS ", %" PRIuS ") (xsize, ysize) = (%" PRIuS
                ", %" PRIuS ")",
                dc_ix, dc_y, dc_x, x0, y0, xsize, ysize);
    enc_state->streaming_mode = true;
    enc_state->initialize_global_state = (i == 0);
    enc_state->dc_group_index = dc_ix;
    enc_state->histogram_idx =
        std::vector<size_t>(group_xsize * group_ysize, i);
    std::vector<std::unique_ptr<BitWriter>> group_codes;
    JXL_RETURN_IF_ERROR(ComputeEncodingData(
        cparams, frame_info, metadata, frame_data, jpeg_data.get(), x0, y0,
        xsize, ysize, cms, pool, frame_header, *enc_modular, *enc_state,
        &group_codes, aux_out));
    JXL_ENSURE(enc_state->special_frames.empty());
    if (i == 0) {
      BitWriter writer{memory_manager};
      JXL_RETURN_IF_ERROR(WriteFrameHeader(frame_header, &writer, aux_out));
      JXL_RETURN_IF_ERROR(WriteTocPermutation(
          streaming_output ? permutation : group_order_perm, &writer, aux_out));
      frame_header_bytes = std::move(writer).TakeBytes();
      if (streaming_output) {
        dc_global_bytes = std::move(*group_codes[0]).TakeBytes();
        JXL_RETURN_IF_ERROR(ComputeGroupDataOffset(
            frame_header_bytes.size(), dc_global_bytes.size(),
            permutation.size(), min_dc_global_size, group_data_offset));
        JXL_DEBUG_V(2, "Frame header size: %" PRIuS, frame_header_bytes.size());
        JXL_DEBUG_V(2, "DC global size: %" PRIuS ", min size for TOC: %" PRIuS,
                    dc_global_bytes.size(), min_dc_global_size);
        JXL_DEBUG_V(2, "Num groups: %" PRIuS " group data offset: %" PRIuS,
                    permutation.size(), group_data_offset);
        group_sizes.push_back(dc_global_bytes.size());
        JXL_RETURN_IF_ERROR(
            output_processor->Seek(start_pos + group_data_offset));
      } else if (ooo) {
        JXL_RETURN_IF_ERROR(
            write_jxlp(jxlp_base, /*is_last=*/false,
                       Span<const uint8_t>(frame_header_bytes.data(),
                                           frame_header_bytes.size())));
        JXL_RETURN_IF_ERROR(
            write_group_section(std::move(*group_codes[0]).TakeBytes()));
      }
    }
    if (streaming) {
      JXL_RETURN_IF_ERROR(
          OutputGroups(std::move(group_codes), write_group_section));
    } else {
      JXL_ENSURE(group_codes.size() >= 4);
      if (i == 0) {
        JXL_RETURN_IF_ERROR(group_codes[0]->Shrink());
        global_group_codes[0] = std::move(group_codes[0]);
      }
      JXL_RETURN_IF_ERROR(group_codes[1]->Shrink());
      global_group_codes[1 + i] = std::move(group_codes[1]);
      for (size_t j = 3; j < group_codes.size(); j++) {
        FrameDimensions patch_dim;
        patch_dim.Set(xsize, ysize, frame_header.group_size_shift,
                      /*max_hshift=*/0, /*max_vshift=*/0,
                      frame_header.encoding == FrameEncoding::kModular,
                      /*upsampling=*/1);
        size_t global_ac_idx =
            enc_modular->ComputeStreamingAbsoluteAcGroupId(i, j - 3, patch_dim);
        JXL_RETURN_IF_ERROR(group_codes[j]->Shrink());
        global_group_codes[2 + dc_group_order.size() + global_ac_idx] =
            std::move(group_codes[j]);
      }
    }
  }
  if (frame_header.encoding == FrameEncoding::kVarDCT) {
    JXL_ASSIGN_OR_RETURN(
        std::unique_ptr<BitWriter> writer,
        OutputAcGlobal(*enc_state, frame_header.ToFrameDimensions(), aux_out));
    JXL_RETURN_IF_ERROR(writer->Shrink());
    if (streaming) {
      JXL_RETURN_IF_ERROR(write_group_section(std::move(*writer).TakeBytes()));
    } else {
      global_group_codes[1 + dc_group_order.size()] = std::move(writer);
    }
  } else {
    if (streaming) {
      JXL_RETURN_IF_ERROR(write_group_section(PaddedBytes{memory_manager}));
    } else {
      global_group_codes[1 + dc_group_order.size()] =
          jxl::make_unique<BitWriter>(memory_manager);
    }
  }
  if (streaming_output) {
    JXL_ENSURE(group_sizes.size() == permutation.size());
    size_t end_pos = output_processor->CurrentPosition();
    JXL_RETURN_IF_ERROR(output_processor->Seek(start_pos));
    size_t padding_size =
        ComputeDcGlobalPadding(group_sizes, frame_header_bytes.size(),
                               group_data_offset, min_dc_global_size);
    group_sizes[0] += padding_size;
    BitWriter toc_writer{memory_manager};
    JXL_RETURN_IF_ERROR(WriteTocSizes(group_sizes, &toc_writer, aux_out));
    PaddedBytes toc_bytes = std::move(toc_writer).TakeBytes();
    std::vector<uint8_t> padding_bytes(padding_size);
    JXL_RETURN_IF_ERROR(AppendData(*output_processor, frame_header_bytes));
    JXL_RETURN_IF_ERROR(AppendData(*output_processor, toc_bytes));
    JXL_RETURN_IF_ERROR(AppendData(*output_processor, dc_global_bytes));
    JXL_RETURN_IF_ERROR(AppendData(*output_processor, padding_bytes));
    JXL_DEBUG_V(2,
                "TOC size: %" PRIuS " padding bytes after DC global: %" PRIuS,
                toc_bytes.size(), padding_size);
    JXL_ENSURE(output_processor->CurrentPosition() ==
               start_pos + group_data_offset);
    JXL_RETURN_IF_ERROR(output_processor->Seek(end_pos));
  } else if (ooo) {
    // Write TOC (in codestream order) as jxlp[jxlp_base+1]; counter +1 means
    // the decoder processes it before the group sections (counters >= +2).
    JXL_ENSURE(group_sizes.size() == permutation.size());
    std::vector<size_t> codestream_sizes(permutation.size());
    for (size_t canonical_idx = 0; canonical_idx < permutation.size();
         canonical_idx++) {
      const size_t cs_pos = group_order_perm.empty()
                                ? canonical_idx
                                : group_order_perm[canonical_idx];
      codestream_sizes[cs_pos] = group_sizes[permutation[canonical_idx]];
    }
    BitWriter toc_writer{memory_manager};
    JXL_RETURN_IF_ERROR(WriteTocSizes(codestream_sizes, &toc_writer, aux_out));
    PaddedBytes toc_bytes = std::move(toc_writer).TakeBytes();
    JXL_RETURN_IF_ERROR(
        write_jxlp(jxlp_base + 1, /*is_last=*/false,
                   Span<const uint8_t>(toc_bytes.data(), toc_bytes.size())));
    if (jxlp_counter != nullptr) {
      *jxlp_counter = jxlp_base + static_cast<uint32_t>(total_sections) + 2;
    }
  } else {
    for (auto& g : global_group_codes) {
      group_sizes.push_back(g->BitsWritten() / 8);
    }
    JXL_ENSURE(group_sizes.size() == permutation.size());
    if (!group_order_perm.empty()) {
      std::vector<std::unique_ptr<BitWriter>> reordered(
          group_order_perm.size());
      std::vector<size_t> reordered_sizes(group_sizes.size());
      for (size_t i = 0; i < group_order_perm.size(); i++) {
        reordered[group_order_perm[i]] = std::move(global_group_codes[i]);
        reordered_sizes[group_order_perm[i]] = group_sizes[i];
      }
      global_group_codes.swap(reordered);
      group_sizes.swap(reordered_sizes);
    }
    BitWriter combined{memory_manager};
    JXL_RETURN_IF_ERROR(WriteFrameHeader(frame_header, &combined, aux_out));
    JXL_RETURN_IF_ERROR(
        WriteTocPermutation(group_order_perm, &combined, aux_out));
    JXL_RETURN_IF_ERROR(WriteTocSizes(group_sizes, &combined, aux_out));
    JXL_RETURN_IF_ERROR(
        AppendData(*output_processor, std::move(combined).TakeBytes()));
    for (auto& g : global_group_codes) {
      JXL_RETURN_IF_ERROR(
          AppendData(*output_processor, std::move(*g).TakeBytes()));
    }
  }
  return true;
}

Status EncodeFrameOneShot(JxlMemoryManager* memory_manager,
                          const CompressParams& cparams,
                          const FrameInfo& frame_info,
                          const CodecMetadata* metadata,
                          JxlEncoderChunkedFrameAdapter& frame_data,
                          const JxlCmsInterface& cms, ThreadPool* pool,
                          JxlEncoderOutputProcessorWrapper* output_processor,
                          AuxOut* aux_out) {
  auto enc_state = jxl::make_unique<PassesEncoderState>(memory_manager);
  SetProgressiveMode(cparams, &enc_state->progressive_splitter);
  FrameHeader frame_header(metadata);
  std::unique_ptr<jpeg::JPEGData> jpeg_data;
  if (frame_data.IsJPEG()) {
    jpeg_data = frame_data.TakeJPEGData();
    JXL_ENSURE(jpeg_data);
  }
  JXL_RETURN_IF_ERROR(MakeFrameHeader(frame_data.xsize, frame_data.ysize,
                                      cparams, enc_state->progressive_splitter,
                                      frame_info, jpeg_data.get(), false,
                                      &frame_header));
  const size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
  JXL_ASSIGN_OR_RETURN(auto enc_modular,
                       ModularFrameEncoder::Create(memory_manager, frame_header,
                                                   cparams, false));
  std::vector<std::unique_ptr<BitWriter>> group_codes;
  JXL_RETURN_IF_ERROR(ComputeEncodingData(
      cparams, frame_info, metadata, frame_data, jpeg_data.get(), 0, 0,
      frame_data.xsize, frame_data.ysize, cms, pool, frame_header, *enc_modular,
      *enc_state, &group_codes, aux_out));

  BitWriter writer{memory_manager};
  JXL_RETURN_IF_ERROR(writer.AppendByteAligned(enc_state->special_frames));
  JXL_RETURN_IF_ERROR(WriteFrameHeader(frame_header, &writer, aux_out));

  std::vector<coeff_order_t> permutation;
  JXL_RETURN_IF_ERROR(PermuteGroups(cparams, enc_state->shared.frame_dim,
                                    num_passes, &permutation, &group_codes));

  std::vector<size_t> group_sizes;
  group_sizes.reserve(group_codes.size());
  for (const auto& bw : group_codes) {
    JXL_ENSURE(bw->BitsWritten() % kBitsPerByte == 0);
    group_sizes.push_back(bw->BitsWritten() / kBitsPerByte);
  }
  JXL_RETURN_IF_ERROR(WriteTocPermutation(permutation, &writer, aux_out));
  JXL_RETURN_IF_ERROR(WriteTocSizes(group_sizes, &writer, aux_out));

  JXL_RETURN_IF_ERROR(writer.AppendByteAligned(group_codes));
  PaddedBytes frame_bytes = std::move(writer).TakeBytes();
  JXL_RETURN_IF_ERROR(AppendData(*output_processor, frame_bytes));

  return true;
}

}  // namespace

std::vector<CompressParams> TectonicPlateSettingsLessPalette(
    const CompressParams& cparams_orig) {
  std::vector<CompressParams> all_params;
  CompressParams cparams_attempt = cparams_orig;
  cparams_attempt.speed_tier = SpeedTier::kGlacier;

  cparams_attempt.options.max_properties = 4;
  cparams_attempt.options.nb_repeats = 1.0f;
  cparams_attempt.modular_group_size_shift = 0;
  cparams_attempt.channel_colors_percent = 0;
  cparams_attempt.options.predictor = Predictor::Variable;
  cparams_attempt.channel_colors_pre_transform_percent = 95.f;
  cparams_attempt.palette_colors = 1024;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.patches = Override::kDefault;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_percent = 80.f;
  cparams_attempt.modular_group_size_shift = 1;
  cparams_attempt.palette_colors = 0;
  cparams_attempt.channel_colors_pre_transform_percent = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_pre_transform_percent = 95.f;
  cparams_attempt.modular_group_size_shift = 2;
  all_params.push_back(cparams_attempt);
  cparams_attempt.modular_group_size_shift = 3;
  cparams_attempt.patches = Override::kOff;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.palette_colors = 1024;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  all_params.push_back(cparams_attempt);
  cparams_attempt.patches = Override::kDefault;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.channel_colors_pre_transform_percent = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_pre_transform_percent = 95.f;
  cparams_attempt.options.nb_repeats = 0.9f;
  cparams_attempt.modular_group_size_shift = 2;
  all_params.push_back(cparams_attempt);
  cparams_attempt.modular_group_size_shift = 3;
  cparams_attempt.palette_colors = 0;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.channel_colors_pre_transform_percent = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.palette_colors = 1024;
  cparams_attempt.options.nb_repeats = 0.95f;
  cparams_attempt.modular_group_size_shift = 1;
  cparams_attempt.channel_colors_percent = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.modular_group_size_shift = 2;
  cparams_attempt.palette_colors = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_percent = 80.f;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.palette_colors = 1024;
  cparams_attempt.channel_colors_pre_transform_percent = 95.f;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.modular_group_size_shift = 3;
  all_params.push_back(cparams_attempt);
  cparams_attempt.palette_colors = 0;
  cparams_attempt.patches = Override::kOff;
  all_params.push_back(cparams_attempt);
  cparams_attempt.patches = Override::kDefault;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.palette_colors = 1024;
  cparams_attempt.patches = Override::kOff;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.nb_repeats = 0.5f;
  cparams_attempt.patches = Override::kDefault;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.predictor = Predictor::Zero;
  cparams_attempt.options.nb_repeats = 0;
  cparams_attempt.channel_colors_percent = 0;
  cparams_attempt.channel_colors_pre_transform_percent = 0;
  cparams_attempt.patches = Override::kOff;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_percent = 80.f;
  cparams_attempt.channel_colors_pre_transform_percent = 95.f;
  cparams_attempt.options.nb_repeats = 1.0f;
  cparams_attempt.palette_colors = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.patches = Override::kDefault;
  cparams_attempt.options.predictor = Predictor::Best;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.nb_repeats = 0.9f;
  cparams_attempt.patches = Override::kOff;
  all_params.push_back(cparams_attempt);
  cparams_attempt.palette_colors = 1024;
  cparams_attempt.patches = Override::kDefault;
  cparams_attempt.options.predictor = Predictor::Weighted;
  cparams_attempt.options.nb_repeats = 1.0f;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.nb_repeats = 0.95f;
  cparams_attempt.modular_group_size_shift = 2;
  cparams_attempt.palette_colors = 0;
  cparams_attempt.channel_colors_pre_transform_percent = 0;
  all_params.push_back(cparams_attempt);
  return all_params;
}

std::vector<CompressParams> TectonicPlateSettingsMorePalette(
    const CompressParams& cparams_orig) {
  std::vector<CompressParams> all_params;
  CompressParams cparams_attempt = cparams_orig;
  cparams_attempt.speed_tier = SpeedTier::kGlacier;

  cparams_attempt.options.max_properties = 4;
  cparams_attempt.options.nb_repeats = 1.0f;
  cparams_attempt.modular_group_size_shift = 0;
  cparams_attempt.palette_colors = 70000;
  cparams_attempt.options.predictor = Predictor::Variable;
  cparams_attempt.channel_colors_percent = 80.f;
  cparams_attempt.channel_colors_pre_transform_percent = 95.f;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.patches = Override::kDefault;
  all_params.push_back(cparams_attempt);
  cparams_attempt.modular_group_size_shift = 2;
  cparams_attempt.channel_colors_percent = 0;
  cparams_attempt.patches = Override::kOff;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_percent = 80.f;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.modular_group_size_shift = 3;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.nb_repeats = 0.9f;
  all_params.push_back(cparams_attempt);
  cparams_attempt.patches = Override::kDefault;
  cparams_attempt.options.nb_repeats = 0.95f;
  cparams_attempt.modular_group_size_shift = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.modular_group_size_shift = 3;
  all_params.push_back(cparams_attempt);
  cparams_attempt.patches = Override::kOff;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.nb_repeats = 0.5f;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.options.predictor = Predictor::Zero;
  cparams_attempt.options.nb_repeats = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.patches = Override::kDefault;
  cparams_attempt.channel_colors_pre_transform_percent = 0;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.nb_repeats = 0.01f;
  cparams_attempt.palette_colors = 0;
  cparams_attempt.patches = Override::kOff;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_pre_transform_percent = 95.f;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.palette_colors = 70000;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.nb_repeats = 1.0f;
  cparams_attempt.modular_group_size_shift = 0;
  cparams_attempt.channel_colors_percent = 0;
  cparams_attempt.channel_colors_pre_transform_percent = 0;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_pre_transform_percent = 95.f;
  cparams_attempt.modular_group_size_shift = 1;
  all_params.push_back(cparams_attempt);
  cparams_attempt.modular_group_size_shift = 2;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_percent = 80.f;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.modular_group_size_shift = 3;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.nb_repeats = 0.5f;
  cparams_attempt.modular_group_size_shift = 1;
  cparams_attempt.channel_colors_percent = 0;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.modular_group_size_shift = 2;
  all_params.push_back(cparams_attempt);
  cparams_attempt.channel_colors_percent = 80.f;
  cparams_attempt.modular_group_size_shift = 3;
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
  all_params.push_back(cparams_attempt);
  cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
  cparams_attempt.options.predictor = Predictor::Select;
  cparams_attempt.options.nb_repeats = 1.0f;
  all_params.push_back(cparams_attempt);
  return all_params;
}

Status EncodeFrame(JxlMemoryManager* memory_manager,
                   const CompressParams& cparams_orig,
                   const FrameInfo& frame_info, const CodecMetadata* metadata,
                   JxlEncoderChunkedFrameAdapter& frame_data,
                   const JxlCmsInterface& cms, ThreadPool* pool,
                   JxlEncoderOutputProcessorWrapper* output_processor,
                   AuxOut* aux_out, uint32_t* jxlp_counter) {
  CompressParams cparams = cparams_orig;
  if (cparams.speed_tier == SpeedTier::kTectonicPlate &&
      !cparams.IsLossless()) {
    cparams.speed_tier = SpeedTier::kGlacier;
  }
  // Lightning mode is handled externally, so switch to Thunder mode to handle
  // potentially weird cases.
  if (cparams.speed_tier == SpeedTier::kLightning) {
    cparams.speed_tier = SpeedTier::kThunder;
  }
  if (cparams.speed_tier == SpeedTier::kTectonicPlate) {
    // Test palette performance to inform later trials.
    std::vector<CompressParams> all_params;
    CompressParams cparams_attempt = cparams;
    cparams_attempt.speed_tier = SpeedTier::kGlacier;

    cparams_attempt.options.max_properties = 4;
    cparams_attempt.options.nb_repeats = 1.0f;
    cparams_attempt.modular_group_size_shift = 3;
    cparams_attempt.palette_colors = 0;
    cparams_attempt.options.predictor = Predictor::Variable;
    cparams_attempt.channel_colors_percent = 80.f;
    cparams_attempt.channel_colors_pre_transform_percent = 95.f;
    cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kDefault;
    cparams_attempt.patches = Override::kDefault;
    all_params.push_back(cparams_attempt);
    cparams_attempt.options.predictor = Predictor::Zero;
    cparams_attempt.options.nb_repeats = 0.01f;
    cparams_attempt.palette_colors = 70000;
    cparams_attempt.patches = Override::kOff;
    cparams_attempt.options.wp_tree_mode = ModularOptions::TreeMode::kNoWP;
    all_params.push_back(cparams_attempt);

    std::vector<size_t> size;
    size.resize(all_params.size());

    const auto process_variant = [&](size_t task, size_t) -> Status {
      JxlEncoderOutputProcessorWrapper local_output(memory_manager);
      JXL_RETURN_IF_ERROR(EncodeFrame(
          memory_manager, all_params[task], frame_info, metadata, frame_data,
          cms, nullptr, &local_output, aux_out, nullptr));
      size[task] = local_output.CurrentPosition();
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, all_params.size(),
                                  ThreadPool::NoInit, process_variant,
                                  "Compress kTectonicPlate"));

    std::vector<CompressParams> all_params_test = all_params;
    std::vector<size_t> size_test = size;
    size_t best_idx_test = 0;

    if (size_test[0] <= size_test[1]) {
      all_params = TectonicPlateSettingsLessPalette(cparams);
    } else {
      best_idx_test = 1;
      all_params = TectonicPlateSettingsMorePalette(cparams);
    }

    size.clear();
    size.resize(all_params.size());

    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, all_params.size(),
                                  ThreadPool::NoInit, process_variant,
                                  "Compress kTectonicPlate"));

    size_t best_idx = 0;
    for (size_t i = 1; i < all_params.size(); i++) {
      if (size[best_idx] > size[i]) {
        best_idx = i;
      }
    }
    if (size[best_idx] < size_test[best_idx_test]) {
      cparams = all_params[best_idx];
    } else {
      cparams = all_params_test[best_idx_test];
    }
  }

  JXL_RETURN_IF_ERROR(ParamsPostInit(&cparams));

  if (cparams.butteraugli_distance < 0) {
    return JXL_FAILURE("Expected non-negative distance");
  }

  if (cparams.progressive_dc < 0) {
    if (cparams.progressive_dc != -1) {
      return JXL_FAILURE("Invalid progressive DC setting value (%d)",
                         cparams.progressive_dc);
    }
    cparams.progressive_dc = 0;
  }
  if (cparams.ec_resampling < cparams.resampling) {
    cparams.ec_resampling = cparams.resampling;
  }
  if (cparams.resampling > 1 ||
      frame_info.is_preview
      // LF frame extra channels not implemented yet, for images with alpha
      // level 1 doesn't render and level 2 corrupts the image.
      || metadata->m.num_extra_channels > 0) {
    cparams.progressive_dc = 0;
  }

  if (frame_info.dc_level + cparams.progressive_dc > 4) {
    return JXL_FAILURE("Too many levels of progressive DC");
  }

  if (cparams.modular_mode == false &&
      cparams.butteraugli_distance < kMinButteraugliDistance) {
    return JXL_FAILURE("Butteraugli distance is too low (%f)",
                       cparams.butteraugli_distance);
  }

  if (frame_data.IsJPEG()) {
    cparams.gaborish = Override::kOff;
    cparams.epf = 0;
    cparams.modular_mode = false;
  }

  if (frame_data.xsize == 0 || frame_data.ysize == 0) {
    return JXL_FAILURE("Empty image");
  }

  // Assert that this metadata is correctly set up for the compression params,
  // this should have been done by enc_file.cc
  JXL_ENSURE(metadata->m.xyb_encoded ==
             (cparams.color_transform == ColorTransform::kXYB));

  if (frame_data.IsJPEG() && cparams.color_transform == ColorTransform::kXYB) {
    return JXL_FAILURE("Can't add JPEG frame to XYB codestream");
  }

  if (CanDoStreamingEncoding(cparams, frame_info, *metadata, frame_data)) {
    return EncodeFrameStreaming(memory_manager, cparams, frame_info, metadata,
                                frame_data, cms, pool, output_processor,
                                aux_out, jxlp_counter);
  } else {
    if (cparams.output_mode == 2 && jxlp_counter != nullptr) {
      // Leave a 12-byte gap for the jxlp box header; encode.cc fills it in.
      JXL_RETURN_IF_ERROR(
          output_processor->Seek(output_processor->CurrentPosition() + 12));
    }
    return EncodeFrameOneShot(memory_manager, cparams, frame_info, metadata,
                              frame_data, cms, pool, output_processor, aux_out);
  }
}

Status EncodeFrame(JxlMemoryManager* memory_manager,
                   const CompressParams& cparams_orig,
                   const FrameInfo& frame_info, const CodecMetadata* metadata,
                   ImageBundle& ib, const JxlCmsInterface& cms,
                   ThreadPool* pool, BitWriter* writer, AuxOut* aux_out) {
  JxlEncoderChunkedFrameAdapter frame_data(ib.xsize(), ib.ysize(),
                                           ib.extra_channels().size());
  std::vector<uint8_t> color;
  if (ib.IsJPEG()) {
    frame_data.SetJPEGData(std::move(ib.jpeg_data));
  } else {
    uint32_t num_channels =
        ib.IsGray() && frame_info.ib_needs_color_transform ? 1 : 3;
    size_t stride = ib.xsize() * num_channels * 4;
    color.resize(ib.ysize() * stride);
    JXL_RETURN_IF_ERROR(ConvertToExternal(
        ib, /*bits_per_sample=*/32, /*float_out=*/true, num_channels,
        JXL_NATIVE_ENDIAN, stride, pool, color.data(), color.size(),
        /*out_callback=*/{}, Orientation::kIdentity));
    JxlPixelFormat format{num_channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    frame_data.SetFromBuffer(0, color.data(), color.size(), format);
  }
  for (size_t ec = 0; ec < ib.extra_channels().size(); ++ec) {
    JxlPixelFormat ec_format{1, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    size_t ec_stride = ib.xsize() * 4;
    std::vector<uint8_t> ec_data(ib.ysize() * ec_stride);
    const ImageF* channel = &ib.extra_channels()[ec];
    JXL_RETURN_IF_ERROR(ConvertChannelsToExternal(
        &channel, 1,
        /*bits_per_sample=*/32,
        /*float_out=*/true, JXL_NATIVE_ENDIAN, ec_stride, pool, ec_data.data(),
        ec_data.size(), /*out_callback=*/{}, Orientation::kIdentity));
    frame_data.SetFromBuffer(1 + ec, ec_data.data(), ec_data.size(), ec_format);
  }
  FrameInfo fi = frame_info;
  fi.origin = ib.origin;
  fi.blend = ib.blend;
  fi.blendmode = ib.blendmode;
  fi.duration = ib.duration;
  fi.timecode = ib.timecode;
  fi.name = ib.name;
  JxlEncoderOutputProcessorWrapper output_processor(memory_manager);
  JXL_RETURN_IF_ERROR(EncodeFrame(memory_manager, cparams_orig, fi, metadata,
                                  frame_data, cms, pool, &output_processor,
                                  aux_out, nullptr));
  JXL_RETURN_IF_ERROR(output_processor.SetFinalizedPosition());
  std::vector<uint8_t> output;
  JXL_RETURN_IF_ERROR(output_processor.CopyOutput(output));
  JXL_RETURN_IF_ERROR(writer->AppendByteAligned(Bytes(output)));
  return true;
}

}  // namespace jxl
