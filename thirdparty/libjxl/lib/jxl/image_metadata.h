// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Main codestream header bundles, the metadata that applies to all frames.
// Enums must align with the C API definitions in codestream_header.h.

#ifndef LIB_JXL_IMAGE_METADATA_H_
#define LIB_JXL_IMAGE_METADATA_H_

#include <jxl/codestream_header.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/matrix_ops.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/headers.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

// EXIF orientation of the image. This field overrides any field present in
// actual EXIF metadata. The value tells which transformation the decoder must
// apply after decoding to display the image with the correct orientation.
enum class Orientation : uint32_t {
  // Values 1..8 match the EXIF definitions.
  kIdentity = JXL_ORIENT_IDENTITY,
  kFlipHorizontal = JXL_ORIENT_FLIP_HORIZONTAL,
  kRotate180 = JXL_ORIENT_ROTATE_180,
  kFlipVertical = JXL_ORIENT_FLIP_VERTICAL,
  kTranspose = JXL_ORIENT_TRANSPOSE,
  kRotate90 = JXL_ORIENT_ROTATE_90_CW,
  kAntiTranspose = JXL_ORIENT_ANTI_TRANSPOSE,
  kRotate270 = JXL_ORIENT_ROTATE_90_CCW,
};
// Don't need an EnumBits because Orientation is not read via Enum().

enum class ExtraChannel : uint32_t {
  // First two enumerators (most common) are cheaper to encode
  kAlpha = JXL_CHANNEL_ALPHA,
  kDepth = JXL_CHANNEL_DEPTH,

  kSpotColor = JXL_CHANNEL_SPOT_COLOR,
  kSelectionMask = JXL_CHANNEL_SELECTION_MASK,
  kBlack = JXL_CHANNEL_BLACK,  // for CMYK
  kCFA = JXL_CHANNEL_CFA,      // Bayer channel
  kThermal = JXL_CHANNEL_THERMAL,
  kReserved0 = JXL_CHANNEL_RESERVED0,
  kReserved1 = JXL_CHANNEL_RESERVED1,
  kReserved2 = JXL_CHANNEL_RESERVED2,
  kReserved3 = JXL_CHANNEL_RESERVED3,
  kReserved4 = JXL_CHANNEL_RESERVED4,
  kReserved5 = JXL_CHANNEL_RESERVED5,
  kReserved6 = JXL_CHANNEL_RESERVED6,
  kReserved7 = JXL_CHANNEL_RESERVED7,
  // disambiguated via name string, raise warning if unsupported
  kUnknown = JXL_CHANNEL_UNKNOWN,
  // like kUnknown but can silently be ignored
  kOptional = JXL_CHANNEL_OPTIONAL
};
static inline const char* EnumName(ExtraChannel /*unused*/) {
  return "ExtraChannel";
}
static inline constexpr uint64_t EnumBits(ExtraChannel /*unused*/) {
  using EC = ExtraChannel;
  return MakeBit(EC::kAlpha) | MakeBit(EC::kDepth) | MakeBit(EC::kSpotColor) |
         MakeBit(EC::kSelectionMask) | MakeBit(EC::kBlack) | MakeBit(EC::kCFA) |
         MakeBit(EC::kThermal) | MakeBit(EC::kUnknown) | MakeBit(EC::kOptional);
}

// Used in ImageMetadata and ExtraChannelInfo.
struct BitDepth : public Fields {
  BitDepth();
  JXL_FIELDS_NAME(BitDepth)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  std::string DebugString() const;

  // Whether the original (uncompressed) samples are floating point or
  // unsigned integer.
  bool floating_point_sample;

  // Bit depth of the original (uncompressed) image samples. Must be in the
  // range [1, 32].
  uint32_t bits_per_sample;

  // Floating point exponent bits of the original (uncompressed) image samples,
  // only used if floating_point_sample is true.
  // If used, the samples are floating point with:
  // - 1 sign bit
  // - exponent_bits_per_sample exponent bits
  // - (bits_per_sample - exponent_bits_per_sample - 1) mantissa bits
  // If used, exponent_bits_per_sample must be in the range
  // [2, 8] and amount of mantissa bits must be in the range [2, 23].
  // NOTE: exponent_bits_per_sample is 8 for single precision binary32
  // point, 5 for half precision binary16, 7 for fp24.
  uint32_t exponent_bits_per_sample;
};

// Describes one extra channel.
struct ExtraChannelInfo : public Fields {
  ExtraChannelInfo();
  JXL_FIELDS_NAME(ExtraChannelInfo)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  std::string DebugString() const;

  mutable bool all_default;

  ExtraChannel type;
  BitDepth bit_depth;
  uint32_t dim_shift;  // downsampled by 2^dim_shift on each axis

  std::string name;  // UTF-8

  // Conditional:
  bool alpha_associated;  // i.e. premultiplied
  float spot_color[4];    // spot color in linear RGBA
  uint32_t cfa_channel;
};

struct OpsinInverseMatrix : public Fields {
  OpsinInverseMatrix();
  JXL_FIELDS_NAME(OpsinInverseMatrix)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  mutable bool all_default;

  Matrix3x3 inverse_matrix;
  float opsin_biases[3];
  float quant_biases[4];
};

// Information useful for mapping HDR images to lower dynamic range displays.
struct ToneMapping : public Fields {
  ToneMapping();
  JXL_FIELDS_NAME(ToneMapping)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  mutable bool all_default;

  // Upper bound on the intensity level present in the image. For unsigned
  // integer pixel encodings, this is the brightness of the largest
  // representable value. The image does not necessarily contain a pixel
  // actually this bright. An encoder is allowed to set 255 for SDR images
  // without computing a histogram.
  float intensity_target;  // [nits]

  // Lower bound on the intensity level present in the image. This may be
  // loose, i.e. lower than the actual darkest pixel. When tone mapping, a
  // decoder will map [min_nits, intensity_target] to the display range.
  float min_nits;

  bool relative_to_max_display;  // see below
  // The tone mapping will leave unchanged (linear mapping) any pixels whose
  // brightness is strictly below this. The interpretation depends on
  // relative_to_max_display. If true, this is a ratio [0, 1] of the maximum
  // display brightness [nits], otherwise an absolute brightness [nits].
  float linear_below;
};

// Contains weights to customize some transforms - in particular, XYB and
// upsampling.
struct CustomTransformData : public Fields {
  CustomTransformData();
  JXL_FIELDS_NAME(CustomTransformData)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  // Must be set before calling VisitFields. Must equal xyb_encoded of
  // ImageMetadata, should be set by ImageMetadata during VisitFields.
  bool nonserialized_xyb_encoded = false;

  mutable bool all_default;

  OpsinInverseMatrix opsin_inverse_matrix;

  uint32_t custom_weights_mask;
  float upsampling2_weights[15];
  float upsampling4_weights[55];
  float upsampling8_weights[210];
};

// Properties of the original image bundle. This enables Encode(Decode()) to
// re-create an equivalent image without user input.
struct ImageMetadata : public Fields {
  ImageMetadata();
  JXL_FIELDS_NAME(ImageMetadata)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  // Returns bit depth of the JPEG XL compressed alpha channel, or 0 if no alpha
  // channel present. In the theoretical case that there are multiple alpha
  // channels, returns the bit depth of the first.
  uint32_t GetAlphaBits() const {
    const ExtraChannelInfo* alpha = Find(ExtraChannel::kAlpha);
    if (alpha == nullptr) return 0;
    JXL_DASSERT(alpha->bit_depth.bits_per_sample != 0);
    return alpha->bit_depth.bits_per_sample;
  }

  // Sets bit depth of alpha channel, adding extra channel if needed, or
  // removing all alpha channels if bits is 0.
  // Assumes integer alpha channel and not designed to support multiple
  // alpha channels (it's possible to use those features by manipulating
  // extra_channel_info directly).
  //
  // Callers must insert the actual channel image at the same index before any
  // further modifications to extra_channel_info.
  void SetAlphaBits(uint32_t bits, bool alpha_is_premultiplied = false);

  bool HasAlpha() const { return GetAlphaBits() != 0; }

  // Sets the original bit depth fields to indicate unsigned integer of the
  // given bit depth.
  // TODO(lode): move function to BitDepth
  void SetUintSamples(uint32_t bits) {
    bit_depth.bits_per_sample = bits;
    bit_depth.exponent_bits_per_sample = 0;
    bit_depth.floating_point_sample = false;
    // RCT / Squeeze may add one bit each, and this is about int16_t,
    // so uint13 should still be OK but limiting it to 12 seems safer.
    // TODO(jon): figure out a better way to set this header field.
    // (in particular, if modular mode is not used it doesn't matter,
    // and if transforms are restricted, up to 15-bit could be done)
    if (bits > 12) modular_16_bit_buffer_sufficient = false;
  }
  // Sets the original bit depth fields to indicate single precision floating
  // point.
  // TODO(lode): move function to BitDepth
  void SetFloat32Samples() {
    bit_depth.bits_per_sample = 32;
    bit_depth.exponent_bits_per_sample = 8;
    bit_depth.floating_point_sample = true;
    modular_16_bit_buffer_sufficient = false;
  }

  void SetFloat16Samples() {
    bit_depth.bits_per_sample = 16;
    bit_depth.exponent_bits_per_sample = 5;
    bit_depth.floating_point_sample = true;
    modular_16_bit_buffer_sufficient = false;
  }

  void SetIntensityTarget(float intensity_target) {
    tone_mapping.intensity_target = intensity_target;
  }
  float IntensityTarget() const {
    JXL_DASSERT(tone_mapping.intensity_target != 0.0f);
    return tone_mapping.intensity_target;
  }

  // Returns first ExtraChannelInfo of the given type, or nullptr if none.
  const ExtraChannelInfo* Find(ExtraChannel type) const {
    for (const ExtraChannelInfo& eci : extra_channel_info) {
      if (eci.type == type) return &eci;
    }
    return nullptr;
  }

  // Returns first ExtraChannelInfo of the given type, or nullptr if none.
  ExtraChannelInfo* Find(ExtraChannel type) {
    for (ExtraChannelInfo& eci : extra_channel_info) {
      if (eci.type == type) return &eci;
    }
    return nullptr;
  }

  Orientation GetOrientation() const {
    return static_cast<Orientation>(orientation);
  }

  bool ExtraFieldsDefault() const;

  std::string DebugString() const;

  mutable bool all_default;

  BitDepth bit_depth;
  bool modular_16_bit_buffer_sufficient;  // otherwise 32 is.

  // Whether the colors values of the pixels of frames are encoded in the
  // codestream using the absolute XYB color space, or the using values that
  // follow the color space defined by the ColorEncoding or ICC profile. This
  // determines when or whether a CMS (Color Management System) is needed to get
  // the pixels in a desired color space. In one case, the pixels have one known
  // color space and a CMS is needed to convert them to the original image's
  // color space, in the other case the pixels have the color space of the
  // original image and a CMS is required if a different display space, or a
  // single known consistent color space for multiple decoded images, is
  // desired. In all cases, the color space of all frames from a single image is
  // the same, both VarDCT and modular frames.
  //
  // If true: then frames can be decoded to XYB (which can also be converted to
  // linear and non-linear sRGB with the built in conversion without CMS). The
  // attached ColorEncoding or ICC profile has no effect on the meaning of the
  // pixel's color values, but instead indicates what the color profile of the
  // original image was, and what color profile one should convert to when
  // decoding to integers to prevent clipping and precision loss. To do that
  // conversion requires a CMS.
  //
  // If false: then the color values of decoded frames are in the space defined
  // by the attached ColorEncoding or ICC profile. To instead get the pixels in
  // a chosen known color space, such as sRGB, requires a CMS, since the
  // attached ColorEncoding or ICC profile could be any arbitrary color space.
  // This mode is typically used for lossless images encoded as integers.
  // Frames can also use YCbCr encoding, some frames may and some may not, but
  // this is not a different color space but a certain encoding of the RGB
  // values.
  //
  // Note: if !xyb_encoded, but the attached color profile indicates XYB (which
  // can happen either if it's a ColorEncoding with color_space_ ==
  // ColorSpace::kXYB, or if it's an ICC Profile that has been crafted to
  // represent XYB), then the frames still may not use ColorEncoding kXYB, they
  // must still use kNone (or kYCbCr, which would mean applying the YCbCr
  // transform to the 3-channel XYB data), since with !xyb_encoded, the 3
  // channels are stored as-is, no matter what meaning the color profile assigns
  // to them. To use ColorSpace::kXYB, xyb_encoded must be true.
  //
  // This value is defined in image metadata because this is the global
  // codestream header. This value does not affect the image itself, so is not
  // image metadata per se, it only affects the encoding, and what color space
  // the decoder can receive the pixels in without needing a CMS.
  bool xyb_encoded;

  ColorEncoding color_encoding;

  // These values are initialized to defaults such that the 'extra_fields'
  // condition in VisitFields uses correctly initialized values.
  uint32_t orientation = 1;
  bool have_preview = false;
  bool have_animation = false;
  bool have_intrinsic_size = false;

  // If present, the stored image has the dimensions of the first SizeHeader,
  // but decoders are advised to resample or display per `intrinsic_size`.
  SizeHeader intrinsic_size;  // only if have_intrinsic_size

  ToneMapping tone_mapping;

  // When reading: deserialized. When writing: automatically set from vector.
  uint32_t num_extra_channels;
  std::vector<ExtraChannelInfo> extra_channel_info;

  // Only present if m.have_preview.
  PreviewHeader preview_size;
  // Only present if m.have_animation.
  AnimationHeader animation;

  uint64_t extensions;

  // Option to stop parsing after basic info, and treat as if the later
  // fields do not participate. Use to parse only basic image information
  // excluding the final larger or variable sized data.
  bool nonserialized_only_parse_basic_info = false;
};

Status ReadImageMetadata(BitReader* JXL_RESTRICT reader,
                         ImageMetadata* JXL_RESTRICT metadata);

Status WriteImageMetadata(const ImageMetadata& metadata,
                          BitWriter* JXL_RESTRICT writer, LayerType layer,
                          AuxOut* aux_out);

// All metadata applicable to the entire codestream (dimensions, extra channels,
// ...)
struct CodecMetadata {
  // TODO(lode): use the preview and animation fields too, in place of the
  // nonserialized_ ones in ImageMetadata.
  ImageMetadata m;
  // The size of the codestream: this is the nominal size applicable to all
  // frames, although some frames can have a different effective size through
  // crop, dc_level or representing a the preview.
  SizeHeader size;
  // Often default.
  CustomTransformData transform_data;

  size_t xsize() const { return size.xsize(); }
  size_t ysize() const { return size.ysize(); }
  size_t oriented_xsize(bool keep_orientation) const {
    if (static_cast<uint32_t>(m.GetOrientation()) > 4 && !keep_orientation) {
      return ysize();
    } else {
      return xsize();
    }
  }
  size_t oriented_preview_xsize(bool keep_orientation) const {
    if (static_cast<uint32_t>(m.GetOrientation()) > 4 && !keep_orientation) {
      return m.preview_size.ysize();
    } else {
      return m.preview_size.xsize();
    }
  }
  size_t oriented_ysize(bool keep_orientation) const {
    if (static_cast<uint32_t>(m.GetOrientation()) > 4 && !keep_orientation) {
      return xsize();
    } else {
      return ysize();
    }
  }
  size_t oriented_preview_ysize(bool keep_orientation) const {
    if (static_cast<uint32_t>(m.GetOrientation()) > 4 && !keep_orientation) {
      return m.preview_size.xsize();
    } else {
      return m.preview_size.ysize();
    }
  }

  std::string DebugString() const;
};

}  // namespace jxl

#endif  // LIB_JXL_IMAGE_METADATA_H_
