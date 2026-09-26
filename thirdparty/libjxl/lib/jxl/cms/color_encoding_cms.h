// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_CMS_COLOR_ENCODING_CMS_H_
#define LIB_JXL_CMS_COLOR_ENCODING_CMS_H_

#include <jxl/cms_interface.h>
#include <jxl/color_encoding.h>
#include <jxl/types.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "lib/jxl/base/status.h"

namespace jxl {
namespace cms {

using IccBytes = std::vector<uint8_t>;

// Returns whether the two inputs are approximately equal.
static inline bool ApproxEq(const double a, const double b,
                            double max_l1 = 1E-3) {
  // Threshold should be sufficient for ICC's 15-bit fixed-point numbers.
  // We have seen differences of 7.1E-5 with lcms2 and 1E-3 with skcms.
  return std::abs(a - b) <= max_l1;
}

// (All CIE units are for the standard 1931 2 degree observer)

// Color space the color pixel data is encoded in. The color pixel data is
// 3-channel in all cases except in case of kGray, where it uses only 1 channel.
// This also determines the amount of channels used in modular encoding.
enum class ColorSpace : uint32_t {
  // Trichromatic color data. This also includes CMYK if a kBlack
  // ExtraChannelInfo is present. This implies, if there is an ICC profile, that
  // the ICC profile uses a 3-channel color space if no kBlack extra channel is
  // present, or uses color space 'CMYK' if a kBlack extra channel is present.
  kRGB,
  // Single-channel data. This implies, if there is an ICC profile, that the ICC
  // profile also represents single-channel data and has the appropriate color
  // space ('GRAY').
  kGray,
  // Like kRGB, but implies fixed values for primaries etc.
  kXYB,
  // For non-RGB/gray data, e.g. from non-electro-optical sensors. Otherwise
  // the same conditions as kRGB apply.
  kUnknown
  // NB: don't forget to update EnumBits!
};

// Values from CICP ColourPrimaries.
enum class WhitePoint : uint32_t {
  kD65 = 1,     // sRGB/BT.709/Display P3/BT.2020
  kCustom = 2,  // Actual values encoded in separate fields
  kE = 10,      // XYZ
  kDCI = 11,    // DCI-P3
  // NB: don't forget to update EnumBits!
};

// Values from CICP ColourPrimaries
enum class Primaries : uint32_t {
  kSRGB = 1,    // Same as BT.709
  kCustom = 2,  // Actual values encoded in separate fields
  k2100 = 9,    // Same as BT.2020
  kP3 = 11,
  // NB: don't forget to update EnumBits!
};

// Values from CICP TransferCharacteristics
enum class TransferFunction : uint32_t {
  k709 = 1,
  kUnknown = 2,
  kLinear = 8,
  kSRGB = 13,
  kPQ = 16,   // from BT.2100
  kDCI = 17,  // from SMPTE RP 431-2 reference projector
  kHLG = 18,  // from BT.2100
  // NB: don't forget to update EnumBits!
};

enum class RenderingIntent : uint32_t {
  // Values match ICC sRGB encodings.
  kPerceptual = 0,  // good for photos, requires a profile with LUT.
  kRelative,        // good for logos.
  kSaturation,      // perhaps useful for CG with fully saturated colors.
  kAbsolute,        // leaves white point unchanged; good for proofing.
  // NB: don't forget to update EnumBits!
};

// Chromaticity (Y is omitted because it is 1 for white points and implicit for
// primaries)
struct CIExy {
  CIExy() = default;
  CIExy(double x, double y) : x(x), y(y) {}
  double x = 0.0;
  double y = 0.0;
};

struct PrimariesCIExy {
  CIExy r;
  CIExy g;
  CIExy b;
};

// Serializable form of CIExy.
struct Customxy {
  static constexpr uint32_t kMul = 1000000;
  static constexpr double kRoughLimit = 4.0;
  static constexpr int32_t kMin = -0x200000;
  static constexpr int32_t kMax = 0x1FFFFF;

  int32_t x = 0;
  int32_t y = 0;

  CIExy GetValue() const {
    CIExy xy;
    xy.x = x * (1.0 / kMul);
    xy.y = y * (1.0 / kMul);
    return xy;
  }

  Status SetValue(const CIExy& xy) {
    bool ok = (std::abs(xy.x) < kRoughLimit) && (std::abs(xy.y) < kRoughLimit);
    if (!ok) return JXL_FAILURE("X or Y is out of bounds");
    x = static_cast<int32_t>(roundf(xy.x * kMul));
    if (x < kMin || x > kMax) return JXL_FAILURE("X is out of bounds");
    y = static_cast<int32_t>(roundf(xy.y * kMul));
    if (y < kMin || y > kMax) return JXL_FAILURE("Y is out of bounds");
    return true;
  }

  bool IsSame(const Customxy& other) const {
    return (x == other.x) && (y == other.y);
  }
};

static inline Status WhitePointFromExternal(const JxlWhitePoint external,
                                            WhitePoint* out) {
  switch (external) {
    case JXL_WHITE_POINT_D65:
      *out = WhitePoint::kD65;
      return true;
    case JXL_WHITE_POINT_CUSTOM:
      *out = WhitePoint::kCustom;
      return true;
    case JXL_WHITE_POINT_E:
      *out = WhitePoint::kE;
      return true;
    case JXL_WHITE_POINT_DCI:
      *out = WhitePoint::kDCI;
      return true;
  }
  return JXL_FAILURE("Invalid WhitePoint enum value %d",
                     static_cast<int>(external));
}

static inline Status PrimariesFromExternal(const JxlPrimaries external,
                                           Primaries* out) {
  switch (external) {
    case JXL_PRIMARIES_SRGB:
      *out = Primaries::kSRGB;
      return true;
    case JXL_PRIMARIES_CUSTOM:
      *out = Primaries::kCustom;
      return true;
    case JXL_PRIMARIES_2100:
      *out = Primaries::k2100;
      return true;
    case JXL_PRIMARIES_P3:
      *out = Primaries::kP3;
      return true;
  }
  return JXL_FAILURE("Invalid Primaries enum value");
}

static inline Status RenderingIntentFromExternal(
    const JxlRenderingIntent external, RenderingIntent* out) {
  switch (external) {
    case JXL_RENDERING_INTENT_PERCEPTUAL:
      *out = RenderingIntent::kPerceptual;
      return true;
    case JXL_RENDERING_INTENT_RELATIVE:
      *out = RenderingIntent::kRelative;
      return true;
    case JXL_RENDERING_INTENT_SATURATION:
      *out = RenderingIntent::kSaturation;
      return true;
    case JXL_RENDERING_INTENT_ABSOLUTE:
      *out = RenderingIntent::kAbsolute;
      return true;
  }
  return JXL_FAILURE("Invalid RenderingIntent enum value");
}

struct CustomTransferFunction {
  // Highest reasonable value for the gamma of a transfer curve.
  static constexpr uint32_t kMaxGamma = 8192;
  static constexpr uint32_t kGammaMul = 10000000;

  bool have_gamma = false;

  // OETF exponent to go from linear to gamma-compressed.
  uint32_t gamma = 0;  // Only used if have_gamma_.

  // Can be kUnknown.
  TransferFunction transfer_function =
      TransferFunction::kSRGB;  // Only used if !have_gamma_.

  TransferFunction GetTransferFunction() const {
    JXL_DASSERT(!have_gamma);
    return have_gamma ? TransferFunction::kUnknown : transfer_function;
  }
  void SetTransferFunction(const TransferFunction tf) {
    have_gamma = false;
    transfer_function = tf;
  }

  bool IsUnknown() const {
    return !have_gamma && (transfer_function == TransferFunction::kUnknown);
  }
  bool IsSRGB() const {
    return !have_gamma && (transfer_function == TransferFunction::kSRGB);
  }
  bool IsLinear() const {
    return !have_gamma && (transfer_function == TransferFunction::kLinear);
  }
  bool IsPQ() const {
    return !have_gamma && (transfer_function == TransferFunction::kPQ);
  }
  bool IsHLG() const {
    return !have_gamma && (transfer_function == TransferFunction::kHLG);
  }
  bool Is709() const {
    return !have_gamma && (transfer_function == TransferFunction::k709);
  }
  bool IsDCI() const {
    return !have_gamma && (transfer_function == TransferFunction::kDCI);
  }

  double GetGamma() const {
    JXL_DASSERT(have_gamma);
    if (!have_gamma) return 0.0;
    return gamma * (1.0 / kGammaMul);  // (0, 1)
  }
  Status SetGamma(double new_gamma) {
    if (new_gamma < (1.0 / kMaxGamma) || new_gamma > 1.0) {
      return JXL_FAILURE("Invalid gamma %f", new_gamma);
    }

    have_gamma = false;
    if (ApproxEq(new_gamma, 1.0)) {
      transfer_function = TransferFunction::kLinear;
      return true;
    }
    if (ApproxEq(new_gamma, 1.0 / 2.6)) {
      transfer_function = TransferFunction::kDCI;
      return true;
    }
    // Don't translate 0.45.. to kSRGB nor k709 - that might change pixel
    // values because those curves also have a linear part.

    have_gamma = true;
    gamma = roundf(new_gamma * kGammaMul);
    transfer_function = TransferFunction::kUnknown;
    return true;
  }

  bool IsSame(const CustomTransferFunction& other) const {
    if (have_gamma != other.have_gamma) {
      return false;
    }
    if (have_gamma) {
      if (gamma != other.gamma) {
        return false;
      }
    } else {
      if (transfer_function != other.transfer_function) {
        return false;
      }
    }
    return true;
  }
};

static inline Status ConvertExternalToInternalTransferFunction(
    const JxlTransferFunction external, TransferFunction* internal) {
  switch (external) {
    case JXL_TRANSFER_FUNCTION_709:
      *internal = TransferFunction::k709;
      return true;
    case JXL_TRANSFER_FUNCTION_UNKNOWN:
      *internal = TransferFunction::kUnknown;
      return true;
    case JXL_TRANSFER_FUNCTION_LINEAR:
      *internal = TransferFunction::kLinear;
      return true;
    case JXL_TRANSFER_FUNCTION_SRGB:
      *internal = TransferFunction::kSRGB;
      return true;
    case JXL_TRANSFER_FUNCTION_PQ:
      *internal = TransferFunction::kPQ;
      return true;
    case JXL_TRANSFER_FUNCTION_DCI:
      *internal = TransferFunction::kDCI;
      return true;
    case JXL_TRANSFER_FUNCTION_HLG:
      *internal = TransferFunction::kHLG;
      return true;
    case JXL_TRANSFER_FUNCTION_GAMMA:
      return JXL_FAILURE("Gamma should be handled separately");
  }
  return JXL_FAILURE("Invalid TransferFunction enum value");
}

// Compact encoding of data required to interpret and translate pixels to a
// known color space. Stored in Metadata. Thread-compatible.
struct ColorEncoding {
  // Only valid if HaveFields()
  WhitePoint white_point = WhitePoint::kD65;
  Primaries primaries = Primaries::kSRGB;  // Only valid if HasPrimaries()
  RenderingIntent rendering_intent = RenderingIntent::kRelative;

  // When false, fields such as white_point and tf are invalid and must not be
  // used. This occurs after setting a raw bytes-only ICC profile, only the
  // ICC bytes may be used. The color_space_ field is still valid.
  bool have_fields = true;

  IccBytes icc;  // Valid ICC profile

  ColorSpace color_space = ColorSpace::kRGB;  // Can be kUnknown
  bool cmyk = false;

  // "late sync" fields
  CustomTransferFunction tf;
  Customxy white;  // Only used if white_point == kCustom
  Customxy red;    // Only used if primaries == kCustom
  Customxy green;  // Only used if primaries == kCustom
  Customxy blue;   // Only used if primaries == kCustom

  // Returns false if the field is invalid and unusable.
  bool HasPrimaries() const {
    return (color_space != ColorSpace::kGray) &&
           (color_space != ColorSpace::kXYB);
  }

  size_t Channels() const { return (color_space == ColorSpace::kGray) ? 1 : 3; }

  Status GetPrimaries(PrimariesCIExy& xy) const {
    JXL_ENSURE(have_fields);
    JXL_ENSURE(HasPrimaries());
    xy = {};
    switch (primaries) {
      case Primaries::kCustom:
        xy.r = red.GetValue();
        xy.g = green.GetValue();
        xy.b = blue.GetValue();
        break;

      case Primaries::kSRGB:
        xy.r.x = 0.639998686;
        xy.r.y = 0.330010138;
        xy.g.x = 0.300003784;
        xy.g.y = 0.600003357;
        xy.b.x = 0.150002046;
        xy.b.y = 0.059997204;
        break;

      case Primaries::k2100:
        xy.r.x = 0.708;
        xy.r.y = 0.292;
        xy.g.x = 0.170;
        xy.g.y = 0.797;
        xy.b.x = 0.131;
        xy.b.y = 0.046;
        break;

      case Primaries::kP3:
        xy.r.x = 0.680;
        xy.r.y = 0.320;
        xy.g.x = 0.265;
        xy.g.y = 0.690;
        xy.b.x = 0.150;
        xy.b.y = 0.060;
        break;

      default:
        JXL_DEBUG_ABORT("internal: unexpected Primaries: %d",
                        static_cast<int>(primaries));
    }
    return true;
  }

  Status SetPrimaries(const PrimariesCIExy& xy) {
    JXL_ENSURE(have_fields);
    JXL_ENSURE(HasPrimaries());
    if (xy.r.x == 0.0 || xy.r.y == 0.0 || xy.g.x == 0.0 || xy.g.y == 0.0 ||
        xy.b.x == 0.0 || xy.b.y == 0.0) {
      return JXL_FAILURE("Invalid primaries %f %f %f %f %f %f", xy.r.x, xy.r.y,
                         xy.g.x, xy.g.y, xy.b.x, xy.b.y);
    }

    if (ApproxEq(xy.r.x, 0.64) && ApproxEq(xy.r.y, 0.33) &&
        ApproxEq(xy.g.x, 0.30) && ApproxEq(xy.g.y, 0.60) &&
        ApproxEq(xy.b.x, 0.15) && ApproxEq(xy.b.y, 0.06)) {
      primaries = Primaries::kSRGB;
      return true;
    }

    if (ApproxEq(xy.r.x, 0.708) && ApproxEq(xy.r.y, 0.292) &&
        ApproxEq(xy.g.x, 0.170) && ApproxEq(xy.g.y, 0.797) &&
        ApproxEq(xy.b.x, 0.131) && ApproxEq(xy.b.y, 0.046)) {
      primaries = Primaries::k2100;
      return true;
    }
    if (ApproxEq(xy.r.x, 0.680) && ApproxEq(xy.r.y, 0.320) &&
        ApproxEq(xy.g.x, 0.265) && ApproxEq(xy.g.y, 0.690) &&
        ApproxEq(xy.b.x, 0.150) && ApproxEq(xy.b.y, 0.060)) {
      primaries = Primaries::kP3;
      return true;
    }

    primaries = Primaries::kCustom;
    JXL_RETURN_IF_ERROR(red.SetValue(xy.r));
    JXL_RETURN_IF_ERROR(green.SetValue(xy.g));
    JXL_RETURN_IF_ERROR(blue.SetValue(xy.b));
    return true;
  }

  CIExy GetWhitePoint() const {
    CIExy xy{};
    JXL_DASSERT(have_fields);
    if (!have_fields) return xy;
    switch (white_point) {
      case WhitePoint::kCustom:
        xy = white.GetValue();
        break;

      case WhitePoint::kD65:
        xy.x = 0.3127;
        xy.y = 0.3290;
        break;

      case WhitePoint::kDCI:
        // From https://ieeexplore.ieee.org/document/7290729 C.2 page 11
        xy.x = 0.314;
        xy.y = 0.351;
        break;

      case WhitePoint::kE:
        xy.x = xy.y = 1.0 / 3;
        break;

      default:
        JXL_DEBUG_ABORT("internal: unexpected WhitePoint: %d",
                        static_cast<int>(white_point));
    }
    return xy;
  }

  Status SetWhitePoint(const CIExy& xy) {
    JXL_ENSURE(have_fields);
    if (xy.x == 0.0 || xy.y == 0.0) {
      return JXL_FAILURE("Invalid white point %f %f", xy.x, xy.y);
    }
    if (ApproxEq(xy.x, 0.3127) && ApproxEq(xy.y, 0.3290)) {
      white_point = WhitePoint::kD65;
      return true;
    }
    if (ApproxEq(xy.x, 1.0 / 3) && ApproxEq(xy.y, 1.0 / 3)) {
      white_point = WhitePoint::kE;
      return true;
    }
    if (ApproxEq(xy.x, 0.314) && ApproxEq(xy.y, 0.351)) {
      white_point = WhitePoint::kDCI;
      return true;
    }
    white_point = WhitePoint::kCustom;
    return white.SetValue(xy);
  }

  // Checks if the color spaces (including white point / primaries) are the
  // same, but ignores the transfer function, rendering intent and ICC bytes.
  bool SameColorSpace(const ColorEncoding& other) const {
    if (color_space != other.color_space) return false;

    if (white_point != other.white_point) return false;
    if (white_point == WhitePoint::kCustom) {
      if (!white.IsSame(other.white)) {
        return false;
      }
    }

    if (HasPrimaries() != other.HasPrimaries()) return false;
    if (HasPrimaries()) {
      if (primaries != other.primaries) return false;
      if (primaries == Primaries::kCustom) {
        if (!red.IsSame(other.red)) return false;
        if (!green.IsSame(other.green)) return false;
        if (!blue.IsSame(other.blue)) return false;
      }
    }
    return true;
  }

  // Checks if the color space and transfer function are the same, ignoring
  // rendering intent and ICC bytes
  bool SameColorEncoding(const ColorEncoding& other) const {
    return SameColorSpace(other) && tf.IsSame(other.tf);
  }

  // Returns true if all fields have been initialized (possibly to kUnknown).
  // Returns false if the ICC profile is invalid or decoding it fails.
  Status SetFieldsFromICC(IccBytes&& new_icc, const JxlCmsInterface& cms) {
    // In case parsing fails, mark the ColorEncoding as invalid.
    JXL_ENSURE(!new_icc.empty());
    color_space = ColorSpace::kUnknown;
    tf.transfer_function = TransferFunction::kUnknown;
    icc.clear();

    JxlColorEncoding external;
    JXL_BOOL new_cmyk;
    JXL_RETURN_IF_ERROR(cms.set_fields_from_icc(cms.set_fields_data,
                                                new_icc.data(), new_icc.size(),
                                                &external, &new_cmyk));
    cmyk = static_cast<bool>(new_cmyk);
    JXL_RETURN_IF_ERROR(FromExternal(external));
    icc = std::move(new_icc);
    return true;
  }

  JxlColorEncoding ToExternal() const {
    JxlColorEncoding external = {};
    auto set_error = [&]() {
      external.color_space = JXL_COLOR_SPACE_UNKNOWN;
      external.primaries = JXL_PRIMARIES_CUSTOM;
      external.rendering_intent = JXL_RENDERING_INTENT_PERCEPTUAL;  //?
      external.transfer_function = JXL_TRANSFER_FUNCTION_UNKNOWN;
      external.white_point = JXL_WHITE_POINT_CUSTOM;
    };
    if (!have_fields) {
      set_error();
      return external;
    }
    external.color_space = static_cast<JxlColorSpace>(color_space);

    external.white_point = static_cast<JxlWhitePoint>(white_point);

    CIExy wp = GetWhitePoint();
    external.white_point_xy[0] = wp.x;
    external.white_point_xy[1] = wp.y;

    if (external.color_space == JXL_COLOR_SPACE_RGB ||
        external.color_space == JXL_COLOR_SPACE_UNKNOWN) {
      external.primaries = static_cast<JxlPrimaries>(primaries);
      PrimariesCIExy p;
      if (!GetPrimaries(p)) {
        set_error();
        return external;
      }
      external.primaries_red_xy[0] = p.r.x;
      external.primaries_red_xy[1] = p.r.y;
      external.primaries_green_xy[0] = p.g.x;
      external.primaries_green_xy[1] = p.g.y;
      external.primaries_blue_xy[0] = p.b.x;
      external.primaries_blue_xy[1] = p.b.y;
    }

    if (tf.have_gamma) {
      external.transfer_function = JXL_TRANSFER_FUNCTION_GAMMA;
      external.gamma = tf.GetGamma();
    } else {
      external.transfer_function =
          static_cast<JxlTransferFunction>(tf.GetTransferFunction());
      external.gamma = 0;
    }

    external.rendering_intent =
        static_cast<JxlRenderingIntent>(rendering_intent);
    return external;
  }

  // NB: does not create ICC.
  Status FromExternal(const JxlColorEncoding& external) {
    // TODO(eustas): update non-serializable on call-site
    color_space = static_cast<ColorSpace>(external.color_space);

    JXL_RETURN_IF_ERROR(
        WhitePointFromExternal(external.white_point, &white_point));
    if (external.white_point == JXL_WHITE_POINT_CUSTOM) {
      CIExy wp;
      wp.x = external.white_point_xy[0];
      wp.y = external.white_point_xy[1];
      JXL_RETURN_IF_ERROR(SetWhitePoint(wp));
    }

    if (external.color_space == JXL_COLOR_SPACE_RGB ||
        external.color_space == JXL_COLOR_SPACE_UNKNOWN) {
      JXL_RETURN_IF_ERROR(
          PrimariesFromExternal(external.primaries, &primaries));
      if (external.primaries == JXL_PRIMARIES_CUSTOM) {
        PrimariesCIExy primaries;
        primaries.r.x = external.primaries_red_xy[0];
        primaries.r.y = external.primaries_red_xy[1];
        primaries.g.x = external.primaries_green_xy[0];
        primaries.g.y = external.primaries_green_xy[1];
        primaries.b.x = external.primaries_blue_xy[0];
        primaries.b.y = external.primaries_blue_xy[1];
        JXL_RETURN_IF_ERROR(SetPrimaries(primaries));
      }
    }
    CustomTransferFunction tf;
    if (external.transfer_function == JXL_TRANSFER_FUNCTION_GAMMA) {
      JXL_RETURN_IF_ERROR(tf.SetGamma(external.gamma));
    } else {
      TransferFunction tf_enum;
      // JXL_TRANSFER_FUNCTION_GAMMA is not handled by this function since
      // there's no internal enum value for it.
      JXL_RETURN_IF_ERROR(ConvertExternalToInternalTransferFunction(
          external.transfer_function, &tf_enum));
      tf.SetTransferFunction(tf_enum);
    }
    this->tf = tf;

    JXL_RETURN_IF_ERROR(RenderingIntentFromExternal(external.rendering_intent,
                                                    &rendering_intent));

    icc.clear();

    return true;
  }
};

}  // namespace cms
}  // namespace jxl

#endif  // LIB_JXL_CMS_COLOR_ENCODING_CMS_H_
