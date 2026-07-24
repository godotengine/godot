// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_FRAME_HEADER_H_
#define LIB_JXL_FRAME_HEADER_H_

// Frame header with backward and forward-compatible extension capability and
// compressed integer fields.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/loop_filter.h"

namespace jxl {

// TODO(eustas): move to proper place?
// Also used by extra channel names.
static inline Status VisitNameString(Visitor* JXL_RESTRICT visitor,
                                     std::string* name) {
  uint32_t name_length = static_cast<uint32_t>(name->length());
  // Allows layer name lengths up to 1071 bytes
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(0), Bits(4), BitsOffset(5, 16),
                                         BitsOffset(10, 48), 0, &name_length));
  if (visitor->IsReading()) {
    name->resize(name_length);
  }
  for (size_t i = 0; i < name_length; i++) {
    uint32_t c = static_cast<uint8_t>((*name)[i]);
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(8, 0, &c));
    (*name)[i] = static_cast<char>(c);
  }
  return true;
}

enum class FrameEncoding : uint32_t {
  kVarDCT,
  kModular,
};

enum class ColorTransform : uint32_t {
  kXYB,    // Values are encoded with XYB. May only be used if
           // ImageBundle::xyb_encoded.
  kNone,   // Values are encoded according to the attached color profile. May
           // only be used if !ImageBundle::xyb_encoded.
  kYCbCr,  // Values are encoded according to the attached color profile, but
           // transformed to YCbCr. May only be used if
           // !ImageBundle::xyb_encoded.
};

inline std::array<int, 3> JpegOrder(ColorTransform ct, bool is_gray) {
  if (is_gray) {
    return {{0, 0, 0}};
  }
  if (ct == ColorTransform::kYCbCr) {
    return {{1, 0, 2}};
  } else if (ct == ColorTransform::kNone) {
    return {{0, 1, 2}};
  } else {
    JXL_DEBUG_ABORT("Internal logic error");
    return {{0, 1, 2}};
  }
}

struct YCbCrChromaSubsampling : public Fields {
  YCbCrChromaSubsampling();
  JXL_FIELDS_NAME(YCbCrChromaSubsampling)
  size_t HShift(size_t c) const { return maxhs_ - kHShift[channel_mode_[c]]; }
  size_t VShift(size_t c) const { return maxvs_ - kVShift[channel_mode_[c]]; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override {
    // TODO(veluca): consider allowing 4x downsamples
    for (uint32_t& ch : channel_mode_) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(2, 0, &ch));
    }
    Recompute();
    return true;
  }

  uint8_t MaxHShift() const { return maxhs_; }
  uint8_t MaxVShift() const { return maxvs_; }

  uint8_t RawHShift(size_t c) const { return kHShift[channel_mode_[c]]; }
  uint8_t RawVShift(size_t c) const { return kVShift[channel_mode_[c]]; }

  // Uses JPEG channel order (Y, Cb, Cr).
  Status Set(const uint8_t* hsample, const uint8_t* vsample) {
    for (size_t c = 0; c < 3; c++) {
      size_t cjpeg = c < 2 ? c ^ 1 : c;
      size_t i = 0;
      for (; i < 4; i++) {
        if (1 << kHShift[i] == hsample[cjpeg] &&
            1 << kVShift[i] == vsample[cjpeg]) {
          channel_mode_[c] = i;
          break;
        }
      }
      if (i == 4) {
        return JXL_FAILURE("Invalid subsample mode");
      }
    }
    Recompute();
    return true;
  }

  bool Is444() const {
    return HShift(0) == 0 && VShift(0) == 0 &&  // Cb
           HShift(2) == 0 && VShift(2) == 0 &&  // Cr
           HShift(1) == 0 && VShift(1) == 0;    // Y
  }

  bool Is420() const {
    return HShift(0) == 1 && VShift(0) == 1 &&  // Cb
           HShift(2) == 1 && VShift(2) == 1 &&  // Cr
           HShift(1) == 0 && VShift(1) == 0;    // Y
  }

  bool Is422() const {
    return HShift(0) == 1 && VShift(0) == 0 &&  // Cb
           HShift(2) == 1 && VShift(2) == 0 &&  // Cr
           HShift(1) == 0 && VShift(1) == 0;    // Y
  }

  bool Is440() const {
    return HShift(0) == 0 && VShift(0) == 1 &&  // Cb
           HShift(2) == 0 && VShift(2) == 1 &&  // Cr
           HShift(1) == 0 && VShift(1) == 0;    // Y
  }

  std::string DebugString() const {
    if (Is444()) return "444";
    if (Is420()) return "420";
    if (Is422()) return "422";
    if (Is440()) return "440";
    return "cs" + std::to_string(channel_mode_[0]) +
           std::to_string(channel_mode_[1]) + std::to_string(channel_mode_[2]);
  }

 private:
  void Recompute() {
    maxhs_ = 0;
    maxvs_ = 0;
    for (uint32_t ch : channel_mode_) {
      maxhs_ = std::max(maxhs_, kHShift[ch]);
      maxvs_ = std::max(maxvs_, kVShift[ch]);
    }
  }
  static const uint8_t kHShift[4];
  static const uint8_t kVShift[4];
  uint32_t channel_mode_[3];
  uint8_t maxhs_;
  uint8_t maxvs_;
};

// Indicates how to combine the current frame with a previously-saved one. Can
// be independently controlled for color and extra channels. Formulas are
// indicative and treat alpha as if it is in range 0.0-1.0. In descriptions
// below, alpha channel is the extra channel of type alpha used for blending
// according to the blend_channel, or fully opaque if there is no alpha channel.
// The blending specified here is used for performing blending *after* color
// transforms - in linear sRGB if blending a XYB-encoded frame on another
// XYB-encoded frame, in sRGB if blending a frame with kColorSpace == kSRGB, or
// in the original colorspace otherwise. Blending in XYB or YCbCr is done by
// using patches.
enum class BlendMode {
  // The new values (in the crop) replace the old ones: sample = new
  kReplace = 0,
  // The new values (in the crop) get added to the old ones: sample = old + new
  kAdd = 1,
  // The new values (in the crop) replace the old ones if alpha>0:
  // For the alpha channel that is used as source:
  // alpha = old + new * (1 - old)
  // For other channels if !alpha_associated:
  // sample = ((1 - new_alpha) * old * old_alpha + new_alpha * new) / alpha
  // For other channels if alpha_associated:
  // sample = (1 - new_alpha) * old + new
  // The alpha formula applies to the alpha used for the division in the other
  // channels formula, and applies to the alpha channel itself if its
  // blend_channel value matches itself.
  kBlend = 2,
  // The new values (in the crop) are added to the old ones if alpha>0:
  // For the alpha channel that is used as source:
  // sample = sample = old + new * (1 - old)
  // For other channels: sample = old + alpha * new
  kAlphaWeightedAdd = 3,
  // The new values (in the crop) get multiplied by the old ones:
  // sample = old * new
  // The range of the new value matters for multiplication purposes, and its
  // nominal range of 0..1 is computed the same way as this is done for the
  // alpha values in kBlend and kAlphaWeightedAdd.
  // If using kMul as a blend mode for color channels, no color transform is
  // performed on the current frame.
  kMul = 4,
};

struct BlendingInfo : public Fields {
  BlendingInfo();
  JXL_FIELDS_NAME(BlendingInfo)
  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;
  BlendMode mode;
  // Which extra channel to use as alpha channel for blending, only encoded
  // for blend modes that involve alpha and if there are more than 1 extra
  // channels.
  uint32_t alpha_channel;
  // Clamp alpha or channel values to 0-1 range.
  bool clamp;
  // Frame ID to copy from (0-3). Only encoded if blend_mode is not kReplace.
  uint32_t source;

  std::string DebugString() const;

  size_t nonserialized_num_extra_channels = 0;
  bool nonserialized_is_partial_frame = false;
};

// Origin of the current frame. Not present for frames of type
// kOnlyPatches.
struct FrameOrigin {
  int32_t x0, y0;  // can be negative.
};

// Size of the current frame.
struct FrameSize {
  uint32_t xsize, ysize;
};

// AnimationFrame defines duration of animation frames.
struct AnimationFrame : public Fields {
  explicit AnimationFrame(const CodecMetadata* metadata);
  JXL_FIELDS_NAME(AnimationFrame)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  // How long to wait [in ticks, see Animation{}] after rendering.
  // May be 0 if the current frame serves as a foundation for another frame.
  uint32_t duration;

  uint32_t timecode;  // 0xHHMMSSFF

  // Must be set to the one ImageMetadata acting as the full codestream header,
  // with correct xyb_encoded, list of extra channels, etc...
  const CodecMetadata* nonserialized_metadata = nullptr;
};

// For decoding to lower resolutions. Only used for kRegular frames.
struct Passes : public Fields {
  Passes();
  JXL_FIELDS_NAME(Passes)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  void GetDownsamplingBracket(size_t pass, int& minShift, int& maxShift) const {
    maxShift = 2;
    minShift = 3;
    for (size_t i = 0;; i++) {
      for (uint32_t j = 0; j < num_downsample; ++j) {
        if (i == last_pass[j]) {
          if (downsample[j] == 8) minShift = 3;
          if (downsample[j] == 4) minShift = 2;
          if (downsample[j] == 2) minShift = 1;
          if (downsample[j] == 1) minShift = 0;
        }
      }
      if (i == num_passes - 1) minShift = 0;
      if (i == pass) return;
      maxShift = minShift - 1;
    }
  }

  uint32_t GetDownsamplingTargetForCompletedPasses(uint32_t num_p) const {
    if (num_p >= num_passes) return 1;
    uint32_t retval = 8;
    for (uint32_t i = 0; i < num_downsample; ++i) {
      if (num_p > last_pass[i]) {
        retval = std::min(retval, downsample[i]);
      }
    }
    return retval;
  }

  std::string DebugString() const;

  uint32_t num_passes;      // <= kMaxNumPasses
  uint32_t num_downsample;  // <= num_passes

  // Array of num_downsample pairs. downsample=1/last_pass=num_passes-1 and
  // downsample=8/last_pass=0 need not be specified; they are implicit.
  uint32_t downsample[kMaxNumPasses];
  uint32_t last_pass[kMaxNumPasses];
  // Array of shift values for each pass. It is implicitly assumed to be 0 for
  // the last pass.
  uint32_t shift[kMaxNumPasses];
};

enum FrameType {
  // A "regular" frame: might be a crop, and will be blended on a previous
  // frame, if any, and displayed or blended in future frames.
  kRegularFrame = 0,
  // A DC frame: this frame is downsampled and will be *only* used as the DC of
  // a future frame and, possibly, for previews. Cannot be cropped, blended, or
  // referenced by patches or blending modes. Frames that *use* a DC frame
  // cannot have non-default sizes either.
  kDCFrame = 1,
  // A PatchesSource frame: this frame will be only used as a source frame for
  // taking patches. Can be cropped, but cannot have non-(0, 0) x0 and y0.
  kReferenceOnly = 2,
  // Same as kRegularFrame, but not used for progressive rendering. This also
  // implies no early display of DC.
  kSkipProgressive = 3,
};

// Image/frame := one of more of these, where the last has is_last = true.
// Starts at a byte-aligned address "a"; the next pass starts at "a + size".
struct FrameHeader : public Fields {
  // Optional postprocessing steps. These flags are the source of truth;
  // Override must set/clear them rather than change their meaning. Values
  // chosen such that typical flags == 0 (encoded in only two bits).
  enum Flags {
    // Often but not always off => low bit value:

    // Inject noise into decoded output.
    kNoise = 1,

    // Overlay patches.
    kPatches = 2,

    // 4, 8 = reserved for future sometimes-off

    // Overlay splines.
    kSplines = 16,

    kUseDcFrame = 32,  // Implies kSkipAdaptiveDCSmoothing.

    // 64 = reserved for future often-off

    // Almost always on => negated:

    kSkipAdaptiveDCSmoothing = 128,
  };

  explicit FrameHeader(const CodecMetadata* metadata);
  JXL_FIELDS_NAME(FrameHeader)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  // Sets/clears `flag` based upon `condition`.
  void UpdateFlag(const bool condition, const uint64_t flag) {
    if (condition) {
      flags |= flag;
    } else {
      flags &= ~flag;
    }
  }

  // Returns true if this frame is supposed to be saved for future usage by
  // other frames.
  bool CanBeReferenced() const {
    // DC frames cannot be referenced. The last frame cannot be referenced. A
    // duration 0 frame makes little sense if it is not referenced. A
    // non-duration 0 frame may or may not be referenced.
    return !is_last && frame_type != FrameType::kDCFrame &&
           (animation_frame.duration == 0 || save_as_reference != 0);
  }

  mutable bool all_default;

  // Always present
  FrameEncoding encoding;
  // Some versions of UBSAN complain in VisitFrameType if not initialized.
  FrameType frame_type = FrameType::kRegularFrame;

  uint64_t flags;

  ColorTransform color_transform;
  YCbCrChromaSubsampling chroma_subsampling;

  uint32_t group_size_shift;  // only if encoding == kModular;

  uint32_t x_qm_scale;  // only if VarDCT and color_transform == kXYB
  uint32_t b_qm_scale;  // only if VarDCT and color_transform == kXYB

  std::string name;

  // Skipped for kReferenceOnly.
  Passes passes;

  // Skipped for kDCFrame
  bool custom_size_or_origin;
  FrameSize frame_size;

  // upsampling factors for color and extra channels.
  // Upsampling is always performed before applying any inverse color transform.
  // Skipped (1) if kUseDCFrame
  uint32_t upsampling;
  std::vector<uint32_t> extra_channel_upsampling;

  // Only for kRegular frames.
  FrameOrigin frame_origin;

  BlendingInfo blending_info;
  std::vector<BlendingInfo> extra_channel_blending_info;

  // Animation info for this frame.
  AnimationFrame animation_frame;

  // This is the last frame.
  bool is_last;

  // ID to refer to this frame with. 0-3, not present if kDCFrame.
  // 0 has a special meaning for kRegular frames of nonzero duration: it defines
  // a frame that will not be referenced in the future.
  uint32_t save_as_reference;

  // Whether to save this frame before or after the color transform. A frame
  // that is saved before the color transform can only be used for blending
  // through patches. On the contrary, a frame that is saved after the color
  // transform can only be used for blending through blending modes.
  // Irrelevant for extra channel blending. Can only be true if
  // blending_info.mode == kReplace and this is not a partial kRegularFrame; if
  // this is a DC frame, it is always true.
  bool save_before_color_transform;

  uint32_t dc_level;  // 1-4 if kDCFrame (0 otherwise).

  // Must be set to the one ImageMetadata acting as the full codestream header,
  // with correct xyb_encoded, list of extra channels, etc...
  const CodecMetadata* nonserialized_metadata = nullptr;

  // NOTE: This is ignored by AllDefault.
  LoopFilter loop_filter;

  bool nonserialized_is_preview = false;

  size_t default_xsize() const {
    if (!nonserialized_metadata) return 0;
    if (nonserialized_is_preview) {
      return nonserialized_metadata->m.preview_size.xsize();
    }
    return nonserialized_metadata->xsize();
  }

  size_t default_ysize() const {
    if (!nonserialized_metadata) return 0;
    if (nonserialized_is_preview) {
      return nonserialized_metadata->m.preview_size.ysize();
    }
    return nonserialized_metadata->ysize();
  }

  FrameDimensions ToFrameDimensions() const {
    size_t xsize = default_xsize();
    size_t ysize = default_ysize();

    xsize = frame_size.xsize ? frame_size.xsize : xsize;
    ysize = frame_size.ysize ? frame_size.ysize : ysize;

    if (dc_level != 0) {
      xsize = DivCeil(xsize, 1 << (3 * dc_level));
      ysize = DivCeil(ysize, 1 << (3 * dc_level));
    }

    FrameDimensions frame_dim;
    frame_dim.Set(xsize, ysize, group_size_shift,
                  chroma_subsampling.MaxHShift(),
                  chroma_subsampling.MaxVShift(),
                  encoding == FrameEncoding::kModular, upsampling);
    return frame_dim;
  }

  // True if a color transform should be applied to this frame.
  bool needs_color_transform() const {
    return !save_before_color_transform ||
           frame_type == FrameType::kRegularFrame ||
           frame_type == FrameType::kSkipProgressive;
  }

  std::string DebugString() const;

  uint64_t extensions;
};

Status ReadFrameHeader(BitReader* JXL_RESTRICT reader,
                       FrameHeader* JXL_RESTRICT frame);

// Shared by enc/dec. 5F and 13 are by far the most common for d1/2/4/8, 0
// ensures low overhead for small images.
static constexpr U32Enc kOrderEnc =
    U32Enc(Val(0x5F), Val(0x13), Val(0), Bits(kNumOrders));

}  // namespace jxl

#endif  // LIB_JXL_FRAME_HEADER_H_
