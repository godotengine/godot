// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/color_encoding_internal.h"

#include <array>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/color_encoding_cms.h"
#include "lib/jxl/cms/jxl_cms_internal.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

bool CustomTransferFunction::SetImplicit() {
  if (nonserialized_color_space == ColorSpace::kXYB) {
    JXL_RETURN_IF_ERROR(storage_.SetGamma(1.0 / 3));
    return true;
  }
  return false;
}

std::array<ColorEncoding, 2> ColorEncoding::CreateC2(Primaries pr,
                                                     TransferFunction tf) {
  std::array<ColorEncoding, 2> c2;

  ColorEncoding* c_rgb = c2.data() + 0;
  c_rgb->SetColorSpace(ColorSpace::kRGB);
  c_rgb->storage_.white_point = WhitePoint::kD65;
  c_rgb->storage_.primaries = pr;
  c_rgb->storage_.tf.SetTransferFunction(tf);
  Status status = c_rgb->CreateICC();
  (void)status;
  JXL_DASSERT(status);

  ColorEncoding* c_gray = c2.data() + 1;
  c_gray->SetColorSpace(ColorSpace::kGray);
  c_gray->storage_.white_point = WhitePoint::kD65;
  c_gray->storage_.primaries = pr;
  c_gray->storage_.tf.SetTransferFunction(tf);
  status = c_gray->CreateICC();
  (void)status;
  JXL_DASSERT(status);

  return c2;
}

const ColorEncoding& ColorEncoding::SRGB(bool is_gray) {
  static std::array<ColorEncoding, 2> c2 =
      CreateC2(Primaries::kSRGB, TransferFunction::kSRGB);
  return c2[is_gray ? 1 : 0];
}
const ColorEncoding& ColorEncoding::LinearSRGB(bool is_gray) {
  static std::array<ColorEncoding, 2> c2 =
      CreateC2(Primaries::kSRGB, TransferFunction::kLinear);
  return c2[is_gray ? 1 : 0];
}

Status ColorEncoding::SetWhitePointType(const WhitePoint& wp) {
  JXL_ENSURE(storage_.have_fields);
  storage_.white_point = wp;
  return true;
}

Status ColorEncoding::SetPrimariesType(const Primaries& p) {
  JXL_ENSURE(storage_.have_fields);
  JXL_ENSURE(HasPrimaries());
  storage_.primaries = p;
  return true;
}

void ColorEncoding::DecideIfWantICC(const JxlCmsInterface& cms) {
  if (storage_.icc.empty()) return;

  JxlColorEncoding c;
  JXL_BOOL cmyk;
  if (!cms.set_fields_from_icc(cms.set_fields_data, storage_.icc.data(),
                               storage_.icc.size(), &c, &cmyk)) {
    return;
  }
  if (cmyk) return;

  std::vector<uint8_t> icc;
  if (!MaybeCreateProfile(c, &icc)) return;

  want_icc_ = false;
}

Customxy::Customxy() { Bundle::Init(this); }
Status Customxy::VisitFields(Visitor* JXL_RESTRICT visitor) {
  uint32_t ux = PackSigned(storage_.x);
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Bits(19), BitsOffset(19, 524288),
                                         BitsOffset(20, 1048576),
                                         BitsOffset(21, 2097152), 0, &ux));
  storage_.x = UnpackSigned(ux);
  uint32_t uy = PackSigned(storage_.y);
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Bits(19), BitsOffset(19, 524288),
                                         BitsOffset(20, 1048576),
                                         BitsOffset(21, 2097152), 0, &uy));
  storage_.y = UnpackSigned(uy);
  return true;
}

CustomTransferFunction::CustomTransferFunction() { Bundle::Init(this); }
Status CustomTransferFunction::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->Conditional(!SetImplicit())) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &storage_.have_gamma));

    if (visitor->Conditional(storage_.have_gamma)) {
      // Gamma is represented as a 24-bit int, the exponent used is
      // gamma_ / 1e7. Valid values are (0, 1]. On the low end side, we also
      // limit it to kMaxGamma/1e7.
      JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(
          24, ::jxl::cms::CustomTransferFunction::kGammaMul, &storage_.gamma));
      if (storage_.gamma > ::jxl::cms::CustomTransferFunction::kGammaMul ||
          static_cast<uint64_t>(storage_.gamma) *
                  ::jxl::cms::CustomTransferFunction::kMaxGamma <
              ::jxl::cms::CustomTransferFunction::kGammaMul) {
        return JXL_FAILURE("Invalid gamma %u", storage_.gamma);
      }
    }

    if (visitor->Conditional(!storage_.have_gamma)) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Enum(TransferFunction::kSRGB, &storage_.transfer_function));
    }
  }

  return true;
}

ColorEncoding::ColorEncoding() { Bundle::Init(this); }
Status ColorEncoding::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &want_icc_));

  // Always send even if want_icc_ because this affects decoding.
  // We can skip the white point/primaries because they do not.
  JXL_QUIET_RETURN_IF_ERROR(
      visitor->Enum(ColorSpace::kRGB, &storage_.color_space));

  if (visitor->Conditional(!WantICC())) {
    // Serialize enums. NOTE: we set the defaults to the most common values so
    // ImageMetadata.all_default is true in the common case.

    if (visitor->Conditional(!ImplicitWhitePoint())) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Enum(WhitePoint::kD65, &storage_.white_point));
      if (visitor->Conditional(storage_.white_point == WhitePoint::kCustom)) {
        white_.storage_ = storage_.white;
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&white_));
        storage_.white = white_.storage_;
      }
    }

    if (visitor->Conditional(HasPrimaries())) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Enum(Primaries::kSRGB, &storage_.primaries));
      if (visitor->Conditional(storage_.primaries == Primaries::kCustom)) {
        red_.storage_ = storage_.red;
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&red_));
        storage_.red = red_.storage_;
        green_.storage_ = storage_.green;
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&green_));
        storage_.green = green_.storage_;
        blue_.storage_ = storage_.blue;
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&blue_));
        storage_.blue = blue_.storage_;
      }
    }

    tf_.nonserialized_color_space = storage_.color_space;
    tf_.storage_ = storage_.tf;
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&tf_));
    storage_.tf = tf_.storage_;

    JXL_QUIET_RETURN_IF_ERROR(
        visitor->Enum(RenderingIntent::kRelative, &storage_.rendering_intent));

    // We didn't have ICC, so all fields should be known.
    if (storage_.color_space == ColorSpace::kUnknown ||
        storage_.tf.IsUnknown()) {
      return JXL_FAILURE(
          "No ICC but cs %u and tf %u%s",
          static_cast<unsigned int>(storage_.color_space),
          storage_.tf.have_gamma
              ? 0
              : static_cast<unsigned int>(storage_.tf.transfer_function),
          storage_.tf.have_gamma ? "(gamma)" : "");
    }

    JXL_RETURN_IF_ERROR(CreateICC());
  }

  if (WantICC() && visitor->IsReading()) {
    // Haven't called SetICC() yet, do nothing.
  } else {
    if (ICC().empty()) return JXL_FAILURE("Empty ICC");
  }

  return true;
}

}  // namespace jxl
