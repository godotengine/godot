// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_COLOR_ENCODING_INTERNAL_H_
#define LIB_JXL_COLOR_ENCODING_INTERNAL_H_

// Metadata for color space conversions.

#include <jxl/cms_interface.h>
#include <jxl/color_encoding.h>
#include <jxl/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>  // free
#include <ostream>
#include <string>
#include <utility>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/color_encoding_cms.h"
#include "lib/jxl/cms/jxl_cms_internal.h"
#include "lib/jxl/field_encodings.h"

namespace jxl {

using IccBytes = ::jxl::cms::IccBytes;
using ColorSpace = ::jxl::cms::ColorSpace;
using WhitePoint = ::jxl::cms::WhitePoint;
using Primaries = ::jxl::cms::Primaries;
using TransferFunction = ::jxl::cms::TransferFunction;
using RenderingIntent = ::jxl::cms::RenderingIntent;
using CIExy = ::jxl::cms::CIExy;
using PrimariesCIExy = ::jxl::cms::PrimariesCIExy;

namespace cms {

static inline const char* EnumName(ColorSpace /*unused*/) {
  return "ColorSpace";
}
static inline constexpr uint64_t EnumBits(ColorSpace /*unused*/) {
  using CS = ColorSpace;
  return MakeBit(CS::kRGB) | MakeBit(CS::kGray) | MakeBit(CS::kXYB) |
         MakeBit(CS::kUnknown);
}

static inline const char* EnumName(WhitePoint /*unused*/) {
  return "WhitePoint";
}
static inline constexpr uint64_t EnumBits(WhitePoint /*unused*/) {
  return MakeBit(WhitePoint::kD65) | MakeBit(WhitePoint::kCustom) |
         MakeBit(WhitePoint::kE) | MakeBit(WhitePoint::kDCI);
}

static inline const char* EnumName(Primaries /*unused*/) { return "Primaries"; }
static inline constexpr uint64_t EnumBits(Primaries /*unused*/) {
  using Pr = Primaries;
  return MakeBit(Pr::kSRGB) | MakeBit(Pr::kCustom) | MakeBit(Pr::k2100) |
         MakeBit(Pr::kP3);
}

static inline const char* EnumName(TransferFunction /*unused*/) {
  return "TransferFunction";
}

static inline constexpr uint64_t EnumBits(TransferFunction /*unused*/) {
  using TF = TransferFunction;
  return MakeBit(TF::k709) | MakeBit(TF::kLinear) | MakeBit(TF::kSRGB) |
         MakeBit(TF::kPQ) | MakeBit(TF::kDCI) | MakeBit(TF::kHLG) |
         MakeBit(TF::kUnknown);
}

static inline const char* EnumName(RenderingIntent /*unused*/) {
  return "RenderingIntent";
}
static inline constexpr uint64_t EnumBits(RenderingIntent /*unused*/) {
  using RI = RenderingIntent;
  return MakeBit(RI::kPerceptual) | MakeBit(RI::kRelative) |
         MakeBit(RI::kSaturation) | MakeBit(RI::kAbsolute);
}

}  // namespace cms

struct ColorEncoding;

// Serializable form of CIExy.
struct Customxy : public Fields {
  Customxy();
  JXL_FIELDS_NAME(Customxy)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

 private:
  friend struct ColorEncoding;
  ::jxl::cms::Customxy storage_;
};

struct CustomTransferFunction : public Fields {
  CustomTransferFunction();
  JXL_FIELDS_NAME(CustomTransferFunction)

  // Sets fields and returns true if nonserialized_color_space has an implicit
  // transfer function, otherwise leaves fields unchanged and returns false.
  bool SetImplicit();

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  // Must be set before calling VisitFields!
  ColorSpace nonserialized_color_space = ColorSpace::kRGB;

 private:
  friend struct ColorEncoding;
  ::jxl::cms::CustomTransferFunction storage_;
};

// Compact encoding of data required to interpret and translate pixels to a
// known color space. Stored in Metadata. Thread-compatible.
struct ColorEncoding : public Fields {
  ColorEncoding();
  JXL_FIELDS_NAME(ColorEncoding)

  // Returns ready-to-use color encodings (initialized on-demand).
  static const ColorEncoding& SRGB(bool is_gray = false);
  static const ColorEncoding& LinearSRGB(bool is_gray = false);

  // Returns true if an ICC profile was successfully created from fields.
  // Must be called after modifying fields. Defined in color_management.cc.
  Status CreateICC() {
    storage_.icc.clear();
    const JxlColorEncoding external = ToExternal();
    if (!MaybeCreateProfile(external, &storage_.icc)) {
      storage_.icc.clear();
      return JXL_FAILURE("Failed to create ICC profile");
    }
    return true;
  }

  // Returns non-empty and valid ICC profile, unless:
  // - WantICC() == true and SetICC() was not yet called;
  // - after a failed call to SetSRGB(), SetICC(), or CreateICC().
  const IccBytes& ICC() const { return storage_.icc; }

  // Returns true if `icc` is assigned and decoded successfully. If so,
  // subsequent WantICC() will return true until DecideIfWantICC() changes it.
  // Returning false indicates data has been lost.
  Status SetICC(IccBytes&& icc, const JxlCmsInterface* cms) {
    JXL_ENSURE(cms != nullptr);
    JXL_ENSURE(!icc.empty());
    want_icc_ = storage_.SetFieldsFromICC(std::move(icc), *cms);
    return want_icc_;
  }

  // Sets the raw ICC profile bytes, without parsing the ICC, and without
  // updating the direct fields such as white point, primaries and color
  // space. Functions to get and set fields, such as SetWhitePoint, cannot be
  // used anymore after this and functions such as IsSRGB return false no matter
  // what the contents of the icc profile.
  void SetICCRaw(IccBytes&& icc) {
    JXL_DASSERT(!icc.empty());
    storage_.icc = std::move(icc);
    storage_.have_fields = false;
    want_icc_ = true;
  }

  // Returns whether to send the ICC profile in the codestream.
  bool WantICC() const { return want_icc_; }

  // Return whether the direct fields are set, if false but ICC is set, only
  // raw ICC bytes are known.
  bool HaveFields() const { return storage_.have_fields; }

  // Causes WantICC() to return false if ICC() can be reconstructed from fields.
  void DecideIfWantICC(const JxlCmsInterface& cms);

  bool IsGray() const { return storage_.color_space == ColorSpace::kGray; }
  bool IsCMYK() const { return storage_.cmyk; }
  size_t Channels() const { return storage_.Channels(); }

  // Returns false if the field is invalid and unusable.
  bool HasPrimaries() const { return storage_.HasPrimaries(); }

  // Returns true after setting the field to a value defined by color_space,
  // otherwise false and leaves the field unchanged.
  bool ImplicitWhitePoint() {
    // TODO(eustas): inline
    if (storage_.color_space == ColorSpace::kXYB) {
      storage_.white_point = WhitePoint::kD65;
      return true;
    }
    return false;
  }

  // Returns whether the color space is known to be sRGB. If a raw unparsed ICC
  // profile is set without the fields being set, this returns false, even if
  // the content of the ICC profile would match sRGB.
  bool IsSRGB() const {
    if (!storage_.have_fields) return false;
    if (!IsGray() && storage_.color_space != ColorSpace::kRGB) return false;
    if (storage_.white_point != WhitePoint::kD65) return false;
    if (storage_.primaries != Primaries::kSRGB) return false;
    if (!storage_.tf.IsSRGB()) return false;
    return true;
  }

  // Returns whether the color space is known to be linear sRGB. If a raw
  // unparsed ICC profile is set without the fields being set, this returns
  // false, even if the content of the ICC profile would match linear sRGB.
  bool IsLinearSRGB() const {
    if (!storage_.have_fields) return false;
    if (!IsGray() && storage_.color_space != ColorSpace::kRGB) return false;
    if (storage_.white_point != WhitePoint::kD65) return false;
    if (storage_.primaries != Primaries::kSRGB) return false;
    if (!storage_.tf.IsLinear()) return false;
    return true;
  }

  Status SetSRGB(const ColorSpace cs,
                 const RenderingIntent ri = RenderingIntent::kRelative) {
    storage_.icc.clear();
    JXL_ENSURE(cs == ColorSpace::kGray || cs == ColorSpace::kRGB);
    storage_.color_space = cs;
    storage_.white_point = WhitePoint::kD65;
    storage_.primaries = Primaries::kSRGB;
    storage_.tf.transfer_function = TransferFunction::kSRGB;
    storage_.rendering_intent = ri;
    return CreateICC();
  }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  // Accessors ensure tf.nonserialized_color_space is updated at the same time.
  ColorSpace GetColorSpace() const { return storage_.color_space; }
  void SetColorSpace(const ColorSpace cs) { storage_.color_space = cs; }
  CIExy GetWhitePoint() const { return storage_.GetWhitePoint(); }

  WhitePoint GetWhitePointType() const { return storage_.white_point; }
  Status SetWhitePointType(const WhitePoint& wp);
  Status GetPrimaries(PrimariesCIExy& p) const {
    return storage_.GetPrimaries(p);
  }

  Primaries GetPrimariesType() const { return storage_.primaries; }
  Status SetPrimariesType(const Primaries& p);

  jxl::cms::CustomTransferFunction& Tf() { return storage_.tf; }
  const jxl::cms::CustomTransferFunction& Tf() const { return storage_.tf; }

  RenderingIntent GetRenderingIntent() const {
    return storage_.rendering_intent;
  }
  void SetRenderingIntent(const RenderingIntent& ri) {
    storage_.rendering_intent = ri;
  }

  bool SameColorEncoding(const ColorEncoding& other) const {
    return storage_.SameColorEncoding(other.storage_);
  }

  mutable bool all_default;

  JxlColorEncoding ToExternal() const { return storage_.ToExternal(); }
  Status FromExternal(const JxlColorEncoding& external) {
    JXL_RETURN_IF_ERROR(storage_.FromExternal(external));
    (void)CreateICC();
    return true;
  }
  const jxl::cms::ColorEncoding& View() const { return storage_; }
  std::string Description() const;

 private:
  static std::array<ColorEncoding, 2> CreateC2(Primaries pr,
                                               TransferFunction tf);

  // If true, the codestream contains an ICC profile and we do not serialize
  // fields. Otherwise, fields are serialized and we create an ICC profile.
  bool want_icc_;

  ::jxl::cms::ColorEncoding storage_;
  // Only used if white_point == kCustom.
  Customxy white_;

  // Only valid if HaveFields()
  CustomTransferFunction tf_;

  // Only used if primaries == kCustom.
  Customxy red_;
  Customxy green_;
  Customxy blue_;
};

static inline std::string Description(const ColorEncoding& c) {
  const JxlColorEncoding external = c.View().ToExternal();
  return ColorEncodingDescription(external);
}

static inline std::ostream& operator<<(std::ostream& os,
                                       const ColorEncoding& c) {
  return os << Description(c);
}

class ColorSpaceTransform {
 public:
  explicit ColorSpaceTransform(const JxlCmsInterface& cms) : cms_(cms) {}
  ~ColorSpaceTransform() {
    if (cms_data_ != nullptr) {
      cms_.destroy(cms_data_);
    }
  }

  // Cannot copy.
  ColorSpaceTransform(const ColorSpaceTransform&) = delete;
  ColorSpaceTransform& operator=(const ColorSpaceTransform&) = delete;

  Status Init(const ColorEncoding& c_src, const ColorEncoding& c_dst,
              float intensity_target, size_t xsize, size_t num_threads) {
    JxlColorProfile input_profile;
    icc_src_ = c_src.ICC();
    input_profile.icc.data = icc_src_.data();
    input_profile.icc.size = icc_src_.size();
    input_profile.color_encoding = c_src.ToExternal();
    input_profile.num_channels = c_src.IsCMYK() ? 4 : c_src.Channels();
    JxlColorProfile output_profile;
    icc_dst_ = c_dst.ICC();
    output_profile.icc.data = icc_dst_.data();
    output_profile.icc.size = icc_dst_.size();
    output_profile.color_encoding = c_dst.ToExternal();
    if (c_dst.IsCMYK())
      return JXL_FAILURE("Conversion to CMYK is not supported");
    output_profile.num_channels = c_dst.Channels();
    cms_data_ = cms_.init(cms_.init_data, num_threads, xsize, &input_profile,
                          &output_profile, intensity_target);
    JXL_RETURN_IF_ERROR(cms_data_ != nullptr);
    return true;
  }

  float* BufSrc(const size_t thread) const {
    return cms_.get_src_buf(cms_data_, thread);
  }

  float* BufDst(const size_t thread) const {
    return cms_.get_dst_buf(cms_data_, thread);
  }

  Status Run(const size_t thread, const float* buf_src, float* buf_dst,
             size_t xsize) {
    // TODO(eustas): convert false to Status?
    return FROM_JXL_BOOL(cms_.run(cms_data_, thread, buf_src, buf_dst, xsize));
  }

 private:
  JxlCmsInterface cms_;
  void* cms_data_ = nullptr;
  // The interface may retain pointers into these.
  IccBytes icc_src_;
  IccBytes icc_dst_;
};

}  // namespace jxl

#endif  // LIB_JXL_COLOR_ENCODING_INTERNAL_H_
