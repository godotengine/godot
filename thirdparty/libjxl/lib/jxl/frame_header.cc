// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/frame_header.h"

#if JXL_DEBUG_V_LEVEL >= 1
#include <sstream>
#include <string>
#endif

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/fields.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

constexpr uint8_t YCbCrChromaSubsampling::kHShift[] = {0, 1, 1, 0};
constexpr uint8_t YCbCrChromaSubsampling::kVShift[] = {0, 1, 0, 1};

static Status VisitBlendMode(Visitor* JXL_RESTRICT visitor,
                             BlendMode default_value, BlendMode* blend_mode) {
  uint32_t encoded = static_cast<uint32_t>(*blend_mode);

  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
      Val(static_cast<uint32_t>(BlendMode::kReplace)),
      Val(static_cast<uint32_t>(BlendMode::kAdd)),
      Val(static_cast<uint32_t>(BlendMode::kBlend)), BitsOffset(2, 3),
      static_cast<uint32_t>(default_value), &encoded));
  if (encoded > static_cast<uint32_t>(BlendMode::kMul)) {
    return JXL_FAILURE("Invalid blend_mode");
  }
  *blend_mode = static_cast<BlendMode>(encoded);
  return true;
}

static Status VisitFrameType(Visitor* JXL_RESTRICT visitor,
                             FrameType default_value, FrameType* frame_type) {
  uint32_t encoded = static_cast<uint32_t>(*frame_type);

  JXL_QUIET_RETURN_IF_ERROR(
      visitor->U32(Val(static_cast<uint32_t>(FrameType::kRegularFrame)),
                   Val(static_cast<uint32_t>(FrameType::kDCFrame)),
                   Val(static_cast<uint32_t>(FrameType::kReferenceOnly)),
                   Val(static_cast<uint32_t>(FrameType::kSkipProgressive)),
                   static_cast<uint32_t>(default_value), &encoded));
  *frame_type = static_cast<FrameType>(encoded);
  return true;
}

BlendingInfo::BlendingInfo() { Bundle::Init(this); }

Status BlendingInfo::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(
      VisitBlendMode(visitor, BlendMode::kReplace, &mode));
  if (visitor->Conditional(nonserialized_num_extra_channels > 0 &&
                           (mode == BlendMode::kBlend ||
                            mode == BlendMode::kAlphaWeightedAdd))) {
    // Up to 11 alpha channels for blending.
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
        Val(0), Val(1), Val(2), BitsOffset(3, 3), 0, &alpha_channel));
    if (visitor->IsReading() &&
        alpha_channel >= nonserialized_num_extra_channels) {
      return JXL_FAILURE("Invalid alpha channel for blending");
    }
  }
  if (visitor->Conditional((nonserialized_num_extra_channels > 0 &&
                            (mode == BlendMode::kBlend ||
                             mode == BlendMode::kAlphaWeightedAdd)) ||
                           mode == BlendMode::kMul)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &clamp));
  }
  // 'old' frame for blending. Only necessary if this is not a full frame, or
  // blending is not kReplace.
  if (visitor->Conditional(mode != BlendMode::kReplace ||
                           nonserialized_is_partial_frame)) {
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->U32(Val(0), Val(1), Val(2), Val(3), 0, &source));
  }
  return true;
}

#if JXL_DEBUG_V_LEVEL >= 1
std::string BlendingInfo::DebugString() const {
  std::ostringstream os;
  os << (mode == BlendMode::kReplace            ? "Replace"
         : mode == BlendMode::kAdd              ? "Add"
         : mode == BlendMode::kBlend            ? "Blend"
         : mode == BlendMode::kAlphaWeightedAdd ? "AlphaWeightedAdd"
                                                : "Mul");
  if (nonserialized_num_extra_channels > 0 &&
      (mode == BlendMode::kBlend || mode == BlendMode::kAlphaWeightedAdd)) {
    os << ",alpha=" << alpha_channel << ",clamp=" << clamp;
  } else if (mode == BlendMode::kMul) {
    os << ",clamp=" << clamp;
  }
  if (mode != BlendMode::kReplace || nonserialized_is_partial_frame) {
    os << ",source=" << source;
  }
  return os.str();
}
#endif

AnimationFrame::AnimationFrame(const CodecMetadata* metadata)
    : nonserialized_metadata(metadata) {
  Bundle::Init(this);
}
Status AnimationFrame::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->Conditional(nonserialized_metadata != nullptr &&
                           nonserialized_metadata->m.have_animation)) {
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->U32(Val(0), Val(1), Bits(8), Bits(32), 0, &duration));
  }

  if (visitor->Conditional(
          nonserialized_metadata != nullptr &&
          nonserialized_metadata->m.animation.have_timecodes)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(32, 0, &timecode));
  }
  return true;
}

YCbCrChromaSubsampling::YCbCrChromaSubsampling() { Bundle::Init(this); }
Passes::Passes() { Bundle::Init(this); }
Status Passes::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(
      visitor->U32(Val(1), Val(2), Val(3), BitsOffset(3, 4), 1, &num_passes));
  JXL_ENSURE(num_passes <= kMaxNumPasses);  // Cannot happen when reading

  if (visitor->Conditional(num_passes != 1)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
        Val(0), Val(1), Val(2), BitsOffset(1, 3), 0, &num_downsample));
    JXL_ENSURE(num_downsample <= 4);  // 1,2,4,8
    if (num_downsample > num_passes) {
      return JXL_FAILURE("num_downsample %u > num_passes %u", num_downsample,
                         num_passes);
    }

    for (uint32_t i = 0; i < num_passes - 1; i++) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(2, 0, &shift[i]));
    }
    shift[num_passes - 1] = 0;

    for (uint32_t i = 0; i < num_downsample; ++i) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->U32(Val(1), Val(2), Val(4), Val(8), 1, &downsample[i]));
      if (i > 0 && downsample[i] >= downsample[i - 1]) {
        return JXL_FAILURE("downsample sequence should be decreasing");
      }
    }
    for (uint32_t i = 0; i < num_downsample; ++i) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->U32(Val(0), Val(1), Val(2), Bits(3), 0, &last_pass[i]));
      if (i > 0 && last_pass[i] <= last_pass[i - 1]) {
        return JXL_FAILURE("last_pass sequence should be increasing");
      }
      if (last_pass[i] >= num_passes) {
        return JXL_FAILURE("last_pass %u >= num_passes %u", last_pass[i],
                           num_passes);
      }
    }
  }

  return true;
}

#if JXL_DEBUG_V_LEVEL >= 1
std::string Passes::DebugString() const {
  std::ostringstream os;
  os << "p=" << num_passes;
  if (num_downsample) {
    os << ",ds=";
    for (uint32_t i = 0; i < num_downsample; ++i) {
      os << last_pass[i] << ":" << downsample[i];
      if (i + 1 < num_downsample) os << ";";
    }
  }
  bool have_shifts = false;
  for (uint32_t i = 0; i < num_passes; ++i) {
    if (shift[i]) have_shifts = true;
  }
  if (have_shifts) {
    os << ",shifts=";
    for (uint32_t i = 0; i < num_passes; ++i) {
      os << shift[i];
      if (i + 1 < num_passes) os << ";";
    }
  }
  return os.str();
}
#endif

FrameHeader::FrameHeader(const CodecMetadata* metadata)
    : animation_frame(metadata), nonserialized_metadata(metadata) {
  Bundle::Init(this);
}

Status ReadFrameHeader(BitReader* JXL_RESTRICT reader,
                       FrameHeader* JXL_RESTRICT frame) {
  return Bundle::Read(reader, frame);
}

Status FrameHeader::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }

  JXL_QUIET_RETURN_IF_ERROR(
      VisitFrameType(visitor, FrameType::kRegularFrame, &frame_type));
  if (visitor->IsReading() && nonserialized_is_preview &&
      frame_type != kRegularFrame) {
    return JXL_FAILURE("Only regular frame could be a preview");
  }

  // FrameEncoding.
  bool is_modular = (encoding == FrameEncoding::kModular);
  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &is_modular));
  encoding = (is_modular ? FrameEncoding::kModular : FrameEncoding::kVarDCT);

  // Flags
  JXL_QUIET_RETURN_IF_ERROR(visitor->U64(0, &flags));

  // Color transform
  bool xyb_encoded = nonserialized_metadata == nullptr ||
                     nonserialized_metadata->m.xyb_encoded;

  if (xyb_encoded) {
    color_transform = ColorTransform::kXYB;
  } else {
    // Alternate if kYCbCr.
    bool alternate = color_transform == ColorTransform::kYCbCr;
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &alternate));
    color_transform =
        (alternate ? ColorTransform::kYCbCr : ColorTransform::kNone);
  }

  // Chroma subsampling for YCbCr, if no DC frame is used.
  if (visitor->Conditional(color_transform == ColorTransform::kYCbCr &&
                           ((flags & kUseDcFrame) == 0))) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&chroma_subsampling));
  }

  size_t num_extra_channels =
      nonserialized_metadata != nullptr
          ? nonserialized_metadata->m.extra_channel_info.size()
          : 0;

  // Upsampling
  if (visitor->Conditional((flags & kUseDcFrame) == 0)) {
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->U32(Val(1), Val(2), Val(4), Val(8), 1, &upsampling));
    if (nonserialized_metadata != nullptr &&
        visitor->Conditional(num_extra_channels != 0)) {
      const std::vector<ExtraChannelInfo>& extra_channels =
          nonserialized_metadata->m.extra_channel_info;
      extra_channel_upsampling.resize(extra_channels.size(), 1);
      for (size_t i = 0; i < extra_channels.size(); ++i) {
        uint32_t dim_shift =
            nonserialized_metadata->m.extra_channel_info[i].dim_shift;
        uint32_t& ec_upsampling = extra_channel_upsampling[i];
        ec_upsampling >>= dim_shift;
        JXL_QUIET_RETURN_IF_ERROR(
            visitor->U32(Val(1), Val(2), Val(4), Val(8), 1, &ec_upsampling));
        ec_upsampling <<= dim_shift;
        if (ec_upsampling < upsampling) {
          return JXL_FAILURE(
              "EC upsampling (%u) < color upsampling (%u), which is invalid.",
              ec_upsampling, upsampling);
        }
        if (ec_upsampling > 8) {
          return JXL_FAILURE("EC upsampling too large (%u)", ec_upsampling);
        }
      }
    } else {
      extra_channel_upsampling.clear();
    }
  }

  // Modular- or VarDCT-specific data.
  if (visitor->Conditional(encoding == FrameEncoding::kModular)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(2, 1, &group_size_shift));
  }
  if (visitor->Conditional(encoding == FrameEncoding::kVarDCT &&
                           color_transform == ColorTransform::kXYB)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(3, 3, &x_qm_scale));
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(3, 2, &b_qm_scale));
  } else {
    x_qm_scale = b_qm_scale = 2;  // noop
  }

  // Not useful for kPatchSource
  if (visitor->Conditional(frame_type != FrameType::kReferenceOnly)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&passes));
  }

  if (visitor->Conditional(frame_type == FrameType::kDCFrame)) {
    // Up to 4 pyramid levels - for up to 16384x downsampling.
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->U32(Val(1), Val(2), Val(3), Val(4), 1, &dc_level));
  }
  if (frame_type != FrameType::kDCFrame) {
    dc_level = 0;
  }

  bool is_partial_frame = false;
  if (visitor->Conditional(frame_type != FrameType::kDCFrame)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &custom_size_or_origin));
    if (visitor->Conditional(custom_size_or_origin)) {
      const U32Enc enc(Bits(8), BitsOffset(11, 256), BitsOffset(14, 2304),
                       BitsOffset(30, 18688));
      // Frame offset, only if kRegularFrame or kSkipProgressive.
      if (visitor->Conditional(frame_type == FrameType::kRegularFrame ||
                               frame_type == FrameType::kSkipProgressive)) {
        uint32_t ux0 = PackSigned(frame_origin.x0);
        uint32_t uy0 = PackSigned(frame_origin.y0);
        JXL_QUIET_RETURN_IF_ERROR(visitor->U32(enc, 0, &ux0));
        JXL_QUIET_RETURN_IF_ERROR(visitor->U32(enc, 0, &uy0));
        frame_origin.x0 = UnpackSigned(ux0);
        frame_origin.y0 = UnpackSigned(uy0);
      }
      // Frame size
      JXL_QUIET_RETURN_IF_ERROR(visitor->U32(enc, 0, &frame_size.xsize));
      JXL_QUIET_RETURN_IF_ERROR(visitor->U32(enc, 0, &frame_size.ysize));
      if (custom_size_or_origin &&
          (frame_size.xsize == 0 || frame_size.ysize == 0)) {
        return JXL_FAILURE(
            "Invalid crop dimensions for frame: zero width or height");
      }
      int32_t image_xsize = default_xsize();
      int32_t image_ysize = default_ysize();
      if (frame_type == FrameType::kRegularFrame ||
          frame_type == FrameType::kSkipProgressive) {
        is_partial_frame |= frame_origin.x0 > 0;
        is_partial_frame |= frame_origin.y0 > 0;
        is_partial_frame |= (static_cast<int32_t>(frame_size.xsize) +
                             frame_origin.x0) < image_xsize;
        is_partial_frame |= (static_cast<int32_t>(frame_size.ysize) +
                             frame_origin.y0) < image_ysize;
      }
    }
  }

  // Blending info, animation info and whether this is the last frame or not.
  if (visitor->Conditional(frame_type == FrameType::kRegularFrame ||
                           frame_type == FrameType::kSkipProgressive)) {
    blending_info.nonserialized_num_extra_channels = num_extra_channels;
    blending_info.nonserialized_is_partial_frame = is_partial_frame;
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&blending_info));
    bool replace_all = (blending_info.mode == BlendMode::kReplace);
    extra_channel_blending_info.resize(num_extra_channels);
    for (size_t i = 0; i < num_extra_channels; i++) {
      auto& ec_blending_info = extra_channel_blending_info[i];
      ec_blending_info.nonserialized_is_partial_frame = is_partial_frame;
      ec_blending_info.nonserialized_num_extra_channels = num_extra_channels;
      JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&ec_blending_info));
      replace_all &= (ec_blending_info.mode == BlendMode::kReplace);
    }
    if (visitor->IsReading() && nonserialized_is_preview) {
      if (!replace_all || custom_size_or_origin) {
        return JXL_FAILURE("Preview is not compatible with blending");
      }
    }
    if (visitor->Conditional(nonserialized_metadata != nullptr &&
                             nonserialized_metadata->m.have_animation)) {
      animation_frame.nonserialized_metadata = nonserialized_metadata;
      JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&animation_frame));
    }
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(true, &is_last));
  } else {
    is_last = false;
  }

  // ID of that can be used to refer to this frame. 0 for a non-zero-duration
  // frame means that it will not be referenced. Not necessary for the last
  // frame.
  if (visitor->Conditional(frame_type != kDCFrame && !is_last)) {
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->U32(Val(0), Val(1), Val(2), Val(3), 0, &save_as_reference));
  }

  // If this frame is not blended on another frame post-color-transform, it may
  // be stored for being referenced either before or after the color transform.
  // If it is blended post-color-transform, it must be blended after. It must
  // also be blended after if this is a kRegular frame that does not cover the
  // full frame, as samples outside the partial region are from a
  // post-color-transform frame.
  if (frame_type != FrameType::kDCFrame) {
    if (visitor->Conditional(CanBeReferenced() &&
                             blending_info.mode == BlendMode::kReplace &&
                             !is_partial_frame &&
                             (frame_type == FrameType::kRegularFrame ||
                              frame_type == FrameType::kSkipProgressive))) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Bool(false, &save_before_color_transform));
    } else if (visitor->Conditional(frame_type == FrameType::kReferenceOnly)) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Bool(true, &save_before_color_transform));
      size_t xsize = custom_size_or_origin ? frame_size.xsize
                                           : nonserialized_metadata->xsize();
      size_t ysize = custom_size_or_origin ? frame_size.ysize
                                           : nonserialized_metadata->ysize();
      if (!save_before_color_transform &&
          (xsize < nonserialized_metadata->xsize() ||
           ysize < nonserialized_metadata->ysize() || frame_origin.x0 != 0 ||
           frame_origin.y0 != 0)) {
        return JXL_FAILURE(
            "non-patch reference frame with invalid crop: %" PRIuS "x%" PRIuS
            "%+d%+d",
            xsize, ysize, static_cast<int>(frame_origin.x0),
            static_cast<int>(frame_origin.y0));
      }
    }
  } else {
    save_before_color_transform = true;
  }

  JXL_QUIET_RETURN_IF_ERROR(VisitNameString(visitor, &name));

  loop_filter.nonserialized_is_modular = is_modular;
  JXL_RETURN_IF_ERROR(visitor->VisitNested(&loop_filter));

  JXL_QUIET_RETURN_IF_ERROR(visitor->BeginExtensions(&extensions));
  // Extensions: in chronological order of being added to the format.
  return visitor->EndExtensions();
}

#if JXL_DEBUG_V_LEVEL >= 1
std::string FrameHeader::DebugString() const {
  std::ostringstream os;
  os << (encoding == FrameEncoding::kVarDCT ? "VarDCT" : "Modular");
  os << ",";
  os << (frame_type == FrameType::kRegularFrame    ? "Regular"
         : frame_type == FrameType::kDCFrame       ? "DC"
         : frame_type == FrameType::kReferenceOnly ? "Reference"
                                                   : "SkipProgressive");
  if (frame_type == FrameType::kDCFrame) {
    os << "(lv" << dc_level << ")";
  }

  if (flags) {
    os << ",";
    uint32_t remaining = flags;

#define TEST_FLAG(name)           \
  if (flags & Flags::k##name) {   \
    remaining &= ~Flags::k##name; \
    os << #name;                  \
    if (remaining) os << "|";     \
  }
    TEST_FLAG(Noise);
    TEST_FLAG(Patches);
    TEST_FLAG(Splines);
    TEST_FLAG(UseDcFrame);
    TEST_FLAG(SkipAdaptiveDCSmoothing);
#undef TEST_FLAG
  }

  os << ",";
  os << (color_transform == ColorTransform::kXYB     ? "XYB"
         : color_transform == ColorTransform::kYCbCr ? "YCbCr"
                                                     : "None");

  if (encoding == FrameEncoding::kModular) {
    os << ",shift=" << group_size_shift;
  } else if (color_transform == ColorTransform::kXYB) {
    os << ",qm=" << x_qm_scale << ";" << b_qm_scale;
  }
  if (frame_type != FrameType::kReferenceOnly) {
    os << "," << passes.DebugString();
  }
  if (custom_size_or_origin) {
    os << ",xs=" << frame_size.xsize;
    os << ",ys=" << frame_size.ysize;
    if (frame_type == FrameType::kRegularFrame ||
        frame_type == FrameType::kSkipProgressive) {
      os << ",x0=" << frame_origin.x0;
      os << ",y0=" << frame_origin.y0;
    }
  }
  if (upsampling > 1) os << ",up=" << upsampling;
  if (loop_filter.gab) os << ",Gaborish";
  if (loop_filter.epf_iters > 0) os << ",epf=" << loop_filter.epf_iters;
  if (animation_frame.duration > 0) os << ",dur=" << animation_frame.duration;
  if (frame_type == FrameType::kRegularFrame ||
      frame_type == FrameType::kSkipProgressive) {
    os << ",";
    os << blending_info.DebugString();
    for (size_t i = 0; i < extra_channel_blending_info.size(); ++i) {
      os << (i == 0 ? "[" : ";");
      os << extra_channel_blending_info[i].DebugString();
      if (i + 1 == extra_channel_blending_info.size()) os << "]";
    }
  }
  if (save_as_reference > 0) os << ",ref=" << save_as_reference;
  os << "," << (save_before_color_transform ? "before" : "after") << "_ct";
  if (is_last) os << ",last";
  return os.str();
}
#endif

}  // namespace jxl
