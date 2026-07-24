// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_CMS_JXL_CMS_INTERNAL_H_
#define LIB_JXL_CMS_JXL_CMS_INTERNAL_H_

// ICC profiles and color space conversions.

#include <jxl/color_encoding.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/matrix_ops.h"
#include "lib/jxl/base/span.h"  // Bytes
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/cms/tone_mapping.h"
#include "lib/jxl/cms/transfer_functions.h"

#ifndef JXL_ENABLE_3D_ICC_TONEMAPPING
#define JXL_ENABLE_3D_ICC_TONEMAPPING 1
#endif

namespace jxl {

enum class ExtraTF {
  kNone,
  kPQ,
  kHLG,
  kSRGB,
};

static Status PrimariesToXYZ(float rx, float ry, float gx, float gy, float bx,
                             float by, float wx, float wy, Matrix3x3& matrix) {
  bool ok = (wx >= 0) && (wx <= 1) && (wy > 0) && (wy <= 1);
  if (!ok) {
    return JXL_FAILURE("Invalid white point");
  }
  // TODO(lode): also require rx, ry, gx, gy, bx, to be in range 0-1? ICC
  // profiles in theory forbid negative XYZ values, but in practice the ACES P0
  // color space uses a negative y for the blue primary.
  Matrix3x3 primaries{{{rx, gx, bx},
                       {ry, gy, by},
                       {1.0f - rx - ry, 1.0f - gx - gy, 1.0f - bx - by}}};
  Matrix3x3 primaries_inv;
  primaries_inv = primaries;
  JXL_RETURN_IF_ERROR(Inv3x3Matrix(primaries_inv));

  Vector3 w{wx / wy, 1.0f, (1.0f - wx - wy) / wy};
  // 1 / tiny float can still overflow
  JXL_RETURN_IF_ERROR(std::isfinite(w[0]) && std::isfinite(w[2]));
  Vector3 xyz;
  Mul3x3Vector(primaries_inv, w, xyz);

  Matrix3x3 a{{{xyz[0], 0, 0}, {0, xyz[1], 0}, {0, 0, xyz[2]}}};

  Mul3x3Matrix(primaries, a, matrix);
  return true;
}

/* Chromatic adaptation matrices*/
constexpr Matrix3x3 kBradford{{{0.8951f, 0.2664f, -0.1614f},
                               {-0.7502f, 1.7135f, 0.0367f},
                               {0.0389f, -0.0685f, 1.0296f}}};
constexpr Matrix3x3 kBradfordInv{{{0.9869929f, -0.1470543f, 0.1599627f},
                                  {0.4323053f, 0.5183603f, 0.0492912f},
                                  {-0.0085287f, 0.0400428f, 0.9684867f}}};

// Adapts white point x, y to D50
static Status AdaptToXYZD50(float wx, float wy, Matrix3x3& matrix) {
  bool ok = (wx >= 0) && (wx <= 1) && (wy > 0) && (wy <= 1);
  if (!ok) {
    // Out of range values can cause division through zero
    // further down with the bradford adaptation too.
    return JXL_FAILURE("Invalid white point");
  }
  Vector3 w{wx / wy, 1.0f, (1.0f - wx - wy) / wy};
  // 1 / tiny float can still overflow
  JXL_RETURN_IF_ERROR(std::isfinite(w[0]) && std::isfinite(w[2]));
  Vector3 w50{0.96422f, 1.0f, 0.82521f};

  Vector3 lms;
  Vector3 lms50;

  Mul3x3Vector(kBradford, w, lms);
  Mul3x3Vector(kBradford, w50, lms50);

  if (lms[0] == 0 || lms[1] == 0 || lms[2] == 0) {
    return JXL_FAILURE("Invalid white point");
  }
  Matrix3x3 a{{{lms50[0] / lms[0], 0, 0},
               {0, lms50[1] / lms[1], 0},
               {0, 0, lms50[2] / lms[2]}}};
  if (!std::isfinite(a[0][0]) || !std::isfinite(a[1][1]) ||
      !std::isfinite(a[2][2])) {
    return JXL_FAILURE("Invalid white point");
  }

  Matrix3x3 b;
  Mul3x3Matrix(a, kBradford, b);
  Mul3x3Matrix(kBradfordInv, b, matrix);

  return true;
}

static Status PrimariesToXYZD50(float rx, float ry, float gx, float gy,
                                float bx, float by, float wx, float wy,
                                Matrix3x3& matrix) {
  Matrix3x3 toXYZ;
  JXL_RETURN_IF_ERROR(PrimariesToXYZ(rx, ry, gx, gy, bx, by, wx, wy, toXYZ));
  Matrix3x3 d50;
  JXL_RETURN_IF_ERROR(AdaptToXYZD50(wx, wy, d50));

  Mul3x3Matrix(d50, toXYZ, matrix);
  return true;
}

static Status ToneMapPixel(const JxlColorEncoding& c, const float in[3],
                           uint8_t pcslab_out[3]) {
  Matrix3x3 primaries_XYZ;
  JXL_RETURN_IF_ERROR(PrimariesToXYZ(
      c.primaries_red_xy[0], c.primaries_red_xy[1], c.primaries_green_xy[0],
      c.primaries_green_xy[1], c.primaries_blue_xy[0], c.primaries_blue_xy[1],
      c.white_point_xy[0], c.white_point_xy[1], primaries_XYZ));
  const Vector3 luminances = primaries_XYZ[1];
  Color linear;
  JxlTransferFunction tf = c.transfer_function;
  if (tf == JXL_TRANSFER_FUNCTION_PQ) {
    for (size_t i = 0; i < 3; ++i) {
      linear[i] = TF_PQ_Base::DisplayFromEncoded(
          /*display_intensity_target=*/10000.0, in[i]);
    }
  } else {
    for (size_t i = 0; i < 3; ++i) {
      linear[i] = TF_HLG_Base::DisplayFromEncoded(in[i]);
    }
  }
  if (tf == JXL_TRANSFER_FUNCTION_PQ) {
    Rec2408ToneMapperBase tone_mapper({0.0f, 10000.0f}, {0.0f, 250.0f},
                                      luminances);
    tone_mapper.ToneMap(linear);
  } else {
    HlgOOTF_Base ootf(/*source_luminance=*/300, /*target_luminance=*/80,
                      luminances);
    ootf.Apply(linear);
  }
  GamutMapScalar(linear, luminances,
                 /*preserve_saturation=*/0.3f);

  Matrix3x3 chad;
  JXL_RETURN_IF_ERROR(
      AdaptToXYZD50(c.white_point_xy[0], c.white_point_xy[1], chad));
  Matrix3x3 to_xyzd50;
  Mul3x3Matrix(chad, primaries_XYZ, to_xyzd50);

  Vector3 xyz{0, 0, 0};
  for (size_t xyz_c = 0; xyz_c < 3; ++xyz_c) {
    for (size_t rgb_c = 0; rgb_c < 3; ++rgb_c) {
      xyz[xyz_c] += linear[rgb_c] * to_xyzd50[xyz_c][rgb_c];
    }
  }

  const auto lab_f = [](const float x) {
    static constexpr float kDelta = 6. / 29;
    return x <= kDelta * kDelta * kDelta
               ? x * (1 / (3 * kDelta * kDelta)) + 4.f / 29
               : std::cbrt(x);
  };
  static constexpr float kXn = 0.964212;
  static constexpr float kYn = 1;
  static constexpr float kZn = 0.825188;

  const float f_x = lab_f(xyz[0] / kXn);
  const float f_y = lab_f(xyz[1] / kYn);
  const float f_z = lab_f(xyz[2] / kZn);

  pcslab_out[0] = static_cast<uint8_t>(
      std::lroundf(255.f * Clamp1(1.16f * f_y - .16f, 0.f, 1.f)));
  pcslab_out[1] = static_cast<uint8_t>(
      std::lroundf(128.f + Clamp1(500 * (f_x - f_y), -128.f, 127.f)));
  pcslab_out[2] = static_cast<uint8_t>(
      std::lroundf(128.f + Clamp1(200 * (f_y - f_z), -128.f, 127.f)));

  return true;
}

template <size_t N, ExtraTF tf>
static std::vector<uint16_t> CreateTableCurve(bool tone_map) {
  // The generated PQ curve will make room for highlights up to this luminance.
  // TODO(sboukortt): make this variable?
  static constexpr float kPQIntensityTarget = 10000;

  static_assert(N <= 4096);  // ICC MFT2 only allows 4K entries
  static_assert(tf == ExtraTF::kPQ || tf == ExtraTF::kHLG);

  static constexpr Vector3 kLuminances{1.f / 3, 1.f / 3, 1.f / 3};
  Rec2408ToneMapperBase tone_mapper(
      {0.0f, kPQIntensityTarget}, {0.0f, kDefaultIntensityTarget}, kLuminances);
  // No point using float - LCMS converts to 16-bit for A2B/MFT.
  std::vector<uint16_t> table(N);
  for (uint32_t i = 0; i < N; ++i) {
    const float x = static_cast<float>(i) / (N - 1);  // 1.0 at index N - 1.
    const double dx = static_cast<double>(x);
    // LCMS requires EOTF (e.g. 2.4 exponent).
    double y = (tf == ExtraTF::kHLG)
                   ? TF_HLG_Base::DisplayFromEncoded(dx)
                   : TF_PQ_Base::DisplayFromEncoded(kPQIntensityTarget, dx);
    if (tone_map && tf == ExtraTF::kPQ &&
        kPQIntensityTarget > kDefaultIntensityTarget) {
      float l = y * 10000 / kPQIntensityTarget;
      Color gray{l, l, l};
      tone_mapper.ToneMap(gray);
      y = gray[0];
    }
    JXL_DASSERT(y >= 0.0);
    // Clamp to table range - necessary for HLG.
    y = Clamp1(y, 0.0, 1.0);
    // 1.0 corresponds to table value 0xFFFF.
    table[i] = static_cast<uint16_t>(roundf(y * 65535.0));
  }
  return table;
}

static Status CIEXYZFromWhiteCIExy(double wx, double wy, Color& XYZ) {
  // Target Y = 1.
  if (std::abs(wy) < 1e-12) return JXL_FAILURE("Y value is too small");
  const float factor = 1 / wy;
  XYZ[0] = wx * factor;
  XYZ[1] = 1;
  XYZ[2] = (1 - wx - wy) * factor;
  return true;
}

namespace detail {

constexpr bool kEnable3DToneMapping = JXL_ENABLE_3D_ICC_TONEMAPPING;

static bool CanToneMap(const JxlColorEncoding& encoding) {
  // If the color space cannot be represented by a CICP tag in the ICC profile
  // then the rest of the profile must unambiguously identify it; we have less
  // freedom to do use it for tone mapping.
  JxlTransferFunction tf = encoding.transfer_function;
  JxlPrimaries p = encoding.primaries;
  JxlWhitePoint wp = encoding.white_point;
  return encoding.color_space == JXL_COLOR_SPACE_RGB &&
         (tf == JXL_TRANSFER_FUNCTION_PQ || tf == JXL_TRANSFER_FUNCTION_HLG) &&
         ((p == JXL_PRIMARIES_P3 &&
           (wp == JXL_WHITE_POINT_D65 || wp == JXL_WHITE_POINT_DCI)) ||
          (p != JXL_PRIMARIES_CUSTOM && wp == JXL_WHITE_POINT_D65));
}

static void ICCComputeMD5(const std::vector<uint8_t>& data, uint8_t sum[16])
    JXL_NO_SANITIZE("unsigned-integer-overflow") {
  std::vector<uint8_t> data64 = data;
  data64.push_back(128);
  // Add bytes such that ((size + 8) & 63) == 0.
  size_t extra = ((64 - ((data64.size() + 8) & 63)) & 63);
  data64.resize(data64.size() + extra, 0);
  for (uint64_t i = 0; i < 64; i += 8) {
    data64.push_back(static_cast<uint64_t>(data.size() << 3u) >> i);
  }

  static const uint32_t sineparts[64] = {
      0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
      0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
      0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
      0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
      0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
      0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
      0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
      0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
      0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
      0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
      0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
  };
  static const uint32_t shift[64] = {
      7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
      5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
      4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
      6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  };

  uint32_t a0 = 0x67452301;
  uint32_t b0 = 0xefcdab89;
  uint32_t c0 = 0x98badcfe;
  uint32_t d0 = 0x10325476;

  for (size_t i = 0; i < data64.size(); i += 64) {
    uint32_t a = a0;
    uint32_t b = b0;
    uint32_t c = c0;
    uint32_t d = d0;
    uint32_t f;
    uint32_t g;
    for (size_t j = 0; j < 64; j++) {
      if (j < 16) {
        f = (b & c) | ((~b) & d);
        g = j;
      } else if (j < 32) {
        f = (d & b) | ((~d) & c);
        g = (5 * j + 1) & 0xf;
      } else if (j < 48) {
        f = b ^ c ^ d;
        g = (3 * j + 5) & 0xf;
      } else {
        f = c ^ (b | (~d));
        g = (7 * j) & 0xf;
      }
      uint32_t dg0 = data64[i + g * 4 + 0];
      uint32_t dg1 = data64[i + g * 4 + 1];
      uint32_t dg2 = data64[i + g * 4 + 2];
      uint32_t dg3 = data64[i + g * 4 + 3];
      uint32_t u = dg0 | (dg1 << 8u) | (dg2 << 16u) | (dg3 << 24u);
      f += a + sineparts[j] + u;
      a = d;
      d = c;
      c = b;
      b += (f << shift[j]) | (f >> (32u - shift[j]));
    }
    a0 += a;
    b0 += b;
    c0 += c;
    d0 += d;
  }
  sum[0] = a0;
  sum[1] = a0 >> 8u;
  sum[2] = a0 >> 16u;
  sum[3] = a0 >> 24u;
  sum[4] = b0;
  sum[5] = b0 >> 8u;
  sum[6] = b0 >> 16u;
  sum[7] = b0 >> 24u;
  sum[8] = c0;
  sum[9] = c0 >> 8u;
  sum[10] = c0 >> 16u;
  sum[11] = c0 >> 24u;
  sum[12] = d0;
  sum[13] = d0 >> 8u;
  sum[14] = d0 >> 16u;
  sum[15] = d0 >> 24u;
}

static Status CreateICCChadMatrix(double wx, double wy, Matrix3x3& result) {
  Matrix3x3 m;
  if (wy == 0) {  // WhitePoint can not be pitch-black.
    return JXL_FAILURE("Invalid WhitePoint");
  }
  JXL_RETURN_IF_ERROR(AdaptToXYZD50(wx, wy, m));
  result = m;
  return true;
}

// Creates RGB to XYZ matrix given RGB primaries and white point in xy.
static Status CreateICCRGBMatrix(double rx, double ry, double gx, double gy,
                                 double bx, double by, double wx, double wy,
                                 Matrix3x3& result) {
  Matrix3x3 m;
  JXL_RETURN_IF_ERROR(PrimariesToXYZD50(rx, ry, gx, gy, bx, by, wx, wy, m));
  result = m;
  return true;
}

static void WriteICCUint32(uint32_t value, size_t pos,
                           std::vector<uint8_t>* icc) {
  if (icc->size() < pos + 4) icc->resize(pos + 4);
  (*icc)[pos + 0] = (value >> 24u) & 255;
  (*icc)[pos + 1] = (value >> 16u) & 255;
  (*icc)[pos + 2] = (value >> 8u) & 255;
  (*icc)[pos + 3] = value & 255;
}

static void WriteICCUint16(uint16_t value, size_t pos,
                           std::vector<uint8_t>* icc) {
  if (icc->size() < pos + 2) icc->resize(pos + 2);
  (*icc)[pos + 0] = (value >> 8u) & 255;
  (*icc)[pos + 1] = value & 255;
}

static void WriteICCUint8(uint8_t value, size_t pos,
                          std::vector<uint8_t>* icc) {
  if (icc->size() < pos + 1) icc->resize(pos + 1);
  (*icc)[pos] = value;
}

// Writes a 4-character tag
static void WriteICCTag(const char* value, size_t pos,
                        std::vector<uint8_t>* icc) {
  if (icc->size() < pos + 4) icc->resize(pos + 4);
  memcpy(icc->data() + pos, value, 4);
}

static Status WriteICCS15Fixed16(float value, size_t pos,
                                 std::vector<uint8_t>* icc) {
  // "nextafterf" for 32768.0f towards zero are:
  // 32767.998046875, 32767.99609375, 32767.994140625
  // Even the first value works well,...
  bool ok = (-32767.995f <= value) && (value <= 32767.995f);
  if (!ok) return JXL_FAILURE("ICC value is out of range / NaN");
  int32_t i = static_cast<int32_t>(std::lround(value * 65536.0f));
  // Use two's complement
  uint32_t u = static_cast<uint32_t>(i);
  WriteICCUint32(u, pos, icc);
  return true;
}

static Status CreateICCHeader(const JxlColorEncoding& c,
                              std::vector<uint8_t>* header) {
  // TODO(lode): choose color management engine name, e.g. "skia" if
  // integrated in skia.
  static const char* kCmm = "jxl ";

  header->resize(128, 0);

  WriteICCUint32(0, 0, header);  // size, correct value filled in at end
  WriteICCTag(kCmm, 4, header);
  WriteICCUint32(0x04400000u, 8, header);
  const char* profile_type =
      c.color_space == JXL_COLOR_SPACE_XYB ? "scnr" : "mntr";
  WriteICCTag(profile_type, 12, header);
  WriteICCTag(c.color_space == JXL_COLOR_SPACE_GRAY ? "GRAY" : "RGB ", 16,
              header);
  if (kEnable3DToneMapping && CanToneMap(c)) {
    // We are going to use a 3D LUT for tone mapping, which will be more compact
    // with an 8-bit LUT to CIELAB than with a 16-bit LUT to XYZ. 8-bit XYZ
    // would not be viable due to XYZ being linear, whereas it is fine with
    // CIELAB's ~cube root.
    WriteICCTag("Lab ", 20, header);
  } else {
    WriteICCTag("XYZ ", 20, header);
  }

  // Three uint32_t's date/time encoding.
  // TODO(lode): encode actual date and time, this is a placeholder
  uint32_t year = 2019;
  uint32_t month = 12;
  uint32_t day = 1;
  uint32_t hour = 0;
  uint32_t minute = 0;
  uint32_t second = 0;
  WriteICCUint16(year, 24, header);
  WriteICCUint16(month, 26, header);
  WriteICCUint16(day, 28, header);
  WriteICCUint16(hour, 30, header);
  WriteICCUint16(minute, 32, header);
  WriteICCUint16(second, 34, header);

  WriteICCTag("acsp", 36, header);
  WriteICCTag("APPL", 40, header);
  WriteICCUint32(0, 44, header);  // flags
  WriteICCUint32(0, 48, header);  // device manufacturer
  WriteICCUint32(0, 52, header);  // device model
  WriteICCUint32(0, 56, header);  // device attributes
  WriteICCUint32(0, 60, header);  // device attributes
  WriteICCUint32(static_cast<uint32_t>(c.rendering_intent), 64, header);

  // Mandatory D50 white point of profile connection space
  WriteICCUint32(0x0000f6d6, 68, header);
  WriteICCUint32(0x00010000, 72, header);
  WriteICCUint32(0x0000d32d, 76, header);

  WriteICCTag(kCmm, 80, header);

  return true;
}

static void AddToICCTagTable(const char* tag, size_t offset, size_t size,
                             std::vector<uint8_t>* tagtable,
                             std::vector<size_t>* offsets) {
  WriteICCTag(tag, tagtable->size(), tagtable);
  // writing true offset deferred to later
  WriteICCUint32(0, tagtable->size(), tagtable);
  offsets->push_back(offset);
  WriteICCUint32(size, tagtable->size(), tagtable);
}

static void FinalizeICCTag(std::vector<uint8_t>* tags, size_t* offset,
                           size_t* size) {
  while ((tags->size() & 3) != 0) {
    tags->push_back(0);
  }
  *offset += *size;
  *size = tags->size() - *offset;
}

// The input text must be ASCII, writing other characters to UTF-16 is not
// implemented.
static void CreateICCMlucTag(const std::string& text,
                             std::vector<uint8_t>* tags) {
  WriteICCTag("mluc", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  WriteICCUint32(1, tags->size(), tags);
  WriteICCUint32(12, tags->size(), tags);
  WriteICCTag("enUS", tags->size(), tags);
  WriteICCUint32(text.size() * 2, tags->size(), tags);
  WriteICCUint32(28, tags->size(), tags);
  for (char c : text) {
    tags->push_back(0);  // prepend 0 for UTF-16
    tags->push_back(c);
  }
}

static Status CreateICCXYZTag(const Color& xyz, std::vector<uint8_t>* tags) {
  WriteICCTag("XYZ ", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  for (size_t i = 0; i < 3; ++i) {
    JXL_RETURN_IF_ERROR(WriteICCS15Fixed16(xyz[i], tags->size(), tags));
  }
  return true;
}

static Status CreateICCChadTag(const Matrix3x3& chad,
                               std::vector<uint8_t>* tags) {
  WriteICCTag("sf32", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      JXL_RETURN_IF_ERROR(WriteICCS15Fixed16(chad[j][i], tags->size(), tags));
    }
  }
  return true;
}

static void MaybeCreateICCCICPTag(const JxlColorEncoding& c,
                                  std::vector<uint8_t>* tags, size_t* offset,
                                  size_t* size, std::vector<uint8_t>* tagtable,
                                  std::vector<size_t>* offsets) {
  if (c.color_space != JXL_COLOR_SPACE_RGB) {
    return;
  }
  uint8_t primaries = 0;
  if (c.primaries == JXL_PRIMARIES_P3) {
    if (c.white_point == JXL_WHITE_POINT_D65) {
      primaries = 12;
    } else if (c.white_point == JXL_WHITE_POINT_DCI) {
      primaries = 11;
    } else {
      return;
    }
  } else if (c.primaries != JXL_PRIMARIES_CUSTOM &&
             c.white_point == JXL_WHITE_POINT_D65) {
    primaries = static_cast<uint8_t>(c.primaries);
  } else {
    return;
  }
  JxlTransferFunction tf = c.transfer_function;
  if (tf == JXL_TRANSFER_FUNCTION_UNKNOWN ||
      tf == JXL_TRANSFER_FUNCTION_GAMMA) {
    return;
  }
  WriteICCTag("cicp", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  WriteICCUint8(primaries, tags->size(), tags);
  WriteICCUint8(static_cast<uint8_t>(tf), tags->size(), tags);
  // Matrix
  WriteICCUint8(0, tags->size(), tags);
  // Full range
  WriteICCUint8(1, tags->size(), tags);
  FinalizeICCTag(tags, offset, size);
  AddToICCTagTable("cicp", *offset, *size, tagtable, offsets);
}

static void CreateICCCurvCurvTag(const std::vector<uint16_t>& curve,
                                 std::vector<uint8_t>* tags) {
  size_t pos = tags->size();
  tags->resize(tags->size() + 12 + curve.size() * 2, 0);
  WriteICCTag("curv", pos, tags);
  WriteICCUint32(0, pos + 4, tags);
  WriteICCUint32(curve.size(), pos + 8, tags);
  for (size_t i = 0; i < curve.size(); i++) {
    WriteICCUint16(curve[i], pos + 12 + i * 2, tags);
  }
}

// Writes 12 + 4*params.size() bytes
static Status CreateICCCurvParaTag(const std::vector<float>& params,
                                   size_t curve_type,
                                   std::vector<uint8_t>* tags) {
  WriteICCTag("para", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  WriteICCUint16(curve_type, tags->size(), tags);
  WriteICCUint16(0, tags->size(), tags);
  for (float param : params) {
    JXL_RETURN_IF_ERROR(WriteICCS15Fixed16(param, tags->size(), tags));
  }
  return true;
}

static Status CreateICCLutAtoBTagForXYB(std::vector<uint8_t>* tags) {
  WriteICCTag("mAB ", tags->size(), tags);
  // 4 reserved bytes set to 0
  WriteICCUint32(0, tags->size(), tags);
  // number of input channels
  WriteICCUint8(3, tags->size(), tags);
  // number of output channels
  WriteICCUint8(3, tags->size(), tags);
  // 2 reserved bytes for padding
  WriteICCUint16(0, tags->size(), tags);
  // offset to first B curve
  WriteICCUint32(32, tags->size(), tags);
  // offset to matrix
  WriteICCUint32(244, tags->size(), tags);
  // offset to first M curve
  WriteICCUint32(148, tags->size(), tags);
  // offset to CLUT
  WriteICCUint32(80, tags->size(), tags);
  // offset to first A curve
  // (reuse linear B curves)
  WriteICCUint32(32, tags->size(), tags);

  // offset = 32
  // no-op curves
  JXL_RETURN_IF_ERROR(CreateICCCurvParaTag({1.0f}, 0, tags));
  JXL_RETURN_IF_ERROR(CreateICCCurvParaTag({1.0f}, 0, tags));
  JXL_RETURN_IF_ERROR(CreateICCCurvParaTag({1.0f}, 0, tags));
  // offset = 80
  // number of grid points for each input channel
  for (int i = 0; i < 16; ++i) {
    WriteICCUint8(i < 3 ? 2 : 0, tags->size(), tags);
  }
  // precision = 2
  WriteICCUint8(2, tags->size(), tags);
  // 3 bytes of padding
  WriteICCUint8(0, tags->size(), tags);
  WriteICCUint16(0, tags->size(), tags);
  // 2*2*2*3 entries of 2 bytes each = 48 bytes
  const jxl::cms::ColorCube3D& cube = jxl::cms::UnscaledA2BCube();
  for (size_t ix = 0; ix < 2; ++ix) {
    for (size_t iy = 0; iy < 2; ++iy) {
      for (size_t ib = 0; ib < 2; ++ib) {
        const jxl::cms::ColorCube0D& out_f = cube[ix][iy][ib];
        for (int i = 0; i < 3; ++i) {
          int32_t val = static_cast<int32_t>(std::lroundf(65535 * out_f[i]));
          JXL_DASSERT(val >= 0 && val <= 65535);
          WriteICCUint16(val, tags->size(), tags);
        }
      }
    }
  }
  // offset = 148
  // 3 curves with 5 parameters = 3 * (12 + 5 * 4) = 96 bytes
  for (size_t i = 0; i < 3; ++i) {
    const float b = -jxl::cms::kXYBOffset[i] -
                    std::cbrt(jxl::cms::kNegOpsinAbsorbanceBiasRGB[i]);
    std::vector<float> params = {
        3,
        1.0f / jxl::cms::kXYBScale[i],
        b,
        0,                                           // unused
        std::max(0.f, -b * jxl::cms::kXYBScale[i]),  // make skcms happy
    };
    JXL_RETURN_IF_ERROR(CreateICCCurvParaTag(params, 3, tags));
  }
  // offset = 244
  const double matrix[] = {1.5170095, -1.1065225, 0.071623,
                           -0.050022, 0.5683655,  -0.018344,
                           -1.387676, 1.1145555,  0.6857255};
  // 12 * 4 = 48 bytes
  for (double v : matrix) {
    JXL_RETURN_IF_ERROR(WriteICCS15Fixed16(v, tags->size(), tags));
  }
  for (size_t i = 0; i < 3; ++i) {
    float intercept = 0;
    for (size_t j = 0; j < 3; ++j) {
      intercept += matrix[i * 3 + j] * jxl::cms::kNegOpsinAbsorbanceBiasRGB[j];
    }
    JXL_RETURN_IF_ERROR(WriteICCS15Fixed16(intercept, tags->size(), tags));
  }
  return true;
}

static Status CreateICCLutAtoBTagForHDR(JxlColorEncoding c,
                                        std::vector<uint8_t>* tags) {
  static constexpr size_t k3DLutDim = 9;
  WriteICCTag("mft1", tags->size(), tags);
  // 4 reserved bytes set to 0
  WriteICCUint32(0, tags->size(), tags);
  // number of input channels
  WriteICCUint8(3, tags->size(), tags);
  // number of output channels
  WriteICCUint8(3, tags->size(), tags);
  // number of CLUT grid points
  WriteICCUint8(k3DLutDim, tags->size(), tags);
  // 1 reserved bytes for padding
  WriteICCUint8(0, tags->size(), tags);

  // Matrix (per specification, must be identity if input is not XYZ)
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      JXL_RETURN_IF_ERROR(
          WriteICCS15Fixed16(i == j ? 1.f : 0.f, tags->size(), tags));
    }
  }

  // Input tables
  for (size_t c = 0; c < 3; ++c) {
    for (size_t i = 0; i < 256; ++i) {
      WriteICCUint8(i, tags->size(), tags);
    }
  }

  for (size_t ix = 0; ix < k3DLutDim; ++ix) {
    for (size_t iy = 0; iy < k3DLutDim; ++iy) {
      for (size_t ib = 0; ib < k3DLutDim; ++ib) {
        float f[3] = {ix * (1.0f / (k3DLutDim - 1)),
                      iy * (1.0f / (k3DLutDim - 1)),
                      ib * (1.0f / (k3DLutDim - 1))};
        uint8_t pcslab_out[3];
        JXL_RETURN_IF_ERROR(ToneMapPixel(c, f, pcslab_out));
        for (uint8_t val : pcslab_out) {
          WriteICCUint8(val, tags->size(), tags);
        }
      }
    }
  }

  // Output tables
  for (size_t c = 0; c < 3; ++c) {
    for (size_t i = 0; i < 256; ++i) {
      WriteICCUint8(i, tags->size(), tags);
    }
  }

  return true;
}

// Some software (Apple Safari, Preview) requires this.
static Status CreateICCNoOpBToATag(std::vector<uint8_t>* tags) {
  WriteICCTag("mBA ", tags->size(), tags);  // notypo
  // 4 reserved bytes set to 0
  WriteICCUint32(0, tags->size(), tags);
  // number of input channels
  WriteICCUint8(3, tags->size(), tags);
  // number of output channels
  WriteICCUint8(3, tags->size(), tags);
  // 2 reserved bytes for padding
  WriteICCUint16(0, tags->size(), tags);
  // offset to first B curve
  WriteICCUint32(32, tags->size(), tags);
  // offset to matrix
  WriteICCUint32(0, tags->size(), tags);
  // offset to first M curve
  WriteICCUint32(0, tags->size(), tags);
  // offset to CLUT
  WriteICCUint32(0, tags->size(), tags);
  // offset to first A curve
  WriteICCUint32(0, tags->size(), tags);

  JXL_RETURN_IF_ERROR(CreateICCCurvParaTag({1.0f}, 0, tags));
  JXL_RETURN_IF_ERROR(CreateICCCurvParaTag({1.0f}, 0, tags));
  JXL_RETURN_IF_ERROR(CreateICCCurvParaTag({1.0f}, 0, tags));

  return true;
}

// These strings are baked into Description - do not change.

static std::string ToString(JxlColorSpace color_space) {
  switch (color_space) {
    case JXL_COLOR_SPACE_RGB:
      return "RGB";
    case JXL_COLOR_SPACE_GRAY:
      return "Gra";
    case JXL_COLOR_SPACE_XYB:
      return "XYB";
    case JXL_COLOR_SPACE_UNKNOWN:
      return "CS?";
    default:
      // Should not happen - visitor fails if enum is invalid.
      JXL_DEBUG_ABORT("Invalid ColorSpace %u",
                      static_cast<uint32_t>(color_space));
      return "Invalid";
  }
}

static std::string ToString(JxlWhitePoint white_point) {
  switch (white_point) {
    case JXL_WHITE_POINT_D65:
      return "D65";
    case JXL_WHITE_POINT_CUSTOM:
      return "Cst";
    case JXL_WHITE_POINT_E:
      return "EER";
    case JXL_WHITE_POINT_DCI:
      return "DCI";
    default:
      // Should not happen - visitor fails if enum is invalid.
      JXL_DEBUG_ABORT("Invalid WhitePoint %u",
                      static_cast<uint32_t>(white_point));
      return "Invalid";
  }
}

static std::string ToString(JxlPrimaries primaries) {
  switch (primaries) {
    case JXL_PRIMARIES_SRGB:
      return "SRG";
    case JXL_PRIMARIES_2100:
      return "202";
    case JXL_PRIMARIES_P3:
      return "DCI";
    case JXL_PRIMARIES_CUSTOM:
      return "Cst";
    default:
      // Should not happen - visitor fails if enum is invalid.
      JXL_DEBUG_ABORT("Invalid Primaries %u", static_cast<uint32_t>(primaries));
      return "Invalid";
  }
}

static std::string ToString(JxlTransferFunction transfer_function) {
  switch (transfer_function) {
    case JXL_TRANSFER_FUNCTION_SRGB:
      return "SRG";
    case JXL_TRANSFER_FUNCTION_LINEAR:
      return "Lin";
    case JXL_TRANSFER_FUNCTION_709:
      return "709";
    case JXL_TRANSFER_FUNCTION_PQ:
      return "PeQ";
    case JXL_TRANSFER_FUNCTION_HLG:
      return "HLG";
    case JXL_TRANSFER_FUNCTION_DCI:
      return "DCI";
    case JXL_TRANSFER_FUNCTION_UNKNOWN:
      return "TF?";
    case JXL_TRANSFER_FUNCTION_GAMMA:
      JXL_DEBUG_ABORT("Invalid TransferFunction: gamma");
      return "Invalid";
    default:
      // Should not happen - visitor fails if enum is invalid.
      JXL_DEBUG_ABORT("Invalid TransferFunction %u",
                      static_cast<uint32_t>(transfer_function));
      return "Invalid";
  }
}

static std::string ToString(JxlRenderingIntent rendering_intent) {
  switch (rendering_intent) {
    case JXL_RENDERING_INTENT_PERCEPTUAL:
      return "Per";
    case JXL_RENDERING_INTENT_RELATIVE:
      return "Rel";
    case JXL_RENDERING_INTENT_SATURATION:
      return "Sat";
    case JXL_RENDERING_INTENT_ABSOLUTE:
      return "Abs";
  }
  // Should not happen - visitor fails if enum is invalid.
  JXL_DEBUG_ABORT("Invalid RenderingIntent %u",
                  static_cast<uint32_t>(rendering_intent));
  return "Invalid";
}

static std::string ColorEncodingDescriptionImpl(const JxlColorEncoding& c) {
  if (c.color_space == JXL_COLOR_SPACE_RGB &&
      c.white_point == JXL_WHITE_POINT_D65) {
    if (c.rendering_intent == JXL_RENDERING_INTENT_PERCEPTUAL &&
        c.transfer_function == JXL_TRANSFER_FUNCTION_SRGB) {
      if (c.primaries == JXL_PRIMARIES_SRGB) return "sRGB";
      if (c.primaries == JXL_PRIMARIES_P3) return "DisplayP3";
    }
    if (c.rendering_intent == JXL_RENDERING_INTENT_RELATIVE &&
        c.primaries == JXL_PRIMARIES_2100) {
      if (c.transfer_function == JXL_TRANSFER_FUNCTION_PQ) return "Rec2100PQ";
      if (c.transfer_function == JXL_TRANSFER_FUNCTION_HLG) return "Rec2100HLG";
    }
  }

  std::string d = ToString(c.color_space);

  bool explicit_wp_tf = (c.color_space != JXL_COLOR_SPACE_XYB);
  if (explicit_wp_tf) {
    d += '_';
    if (c.white_point == JXL_WHITE_POINT_CUSTOM) {
      d += jxl::ToString(c.white_point_xy[0]) + ';';
      d += jxl::ToString(c.white_point_xy[1]);
    } else {
      d += ToString(c.white_point);
    }
  }

  if ((c.color_space != JXL_COLOR_SPACE_GRAY) &&
      (c.color_space != JXL_COLOR_SPACE_XYB)) {
    d += '_';
    if (c.primaries == JXL_PRIMARIES_CUSTOM) {
      d += jxl::ToString(c.primaries_red_xy[0]) + ';';
      d += jxl::ToString(c.primaries_red_xy[1]) + ';';
      d += jxl::ToString(c.primaries_green_xy[0]) + ';';
      d += jxl::ToString(c.primaries_green_xy[1]) + ';';
      d += jxl::ToString(c.primaries_blue_xy[0]) + ';';
      d += jxl::ToString(c.primaries_blue_xy[1]);
    } else {
      d += ToString(c.primaries);
    }
  }

  d += '_';
  d += ToString(c.rendering_intent);

  if (explicit_wp_tf) {
    JxlTransferFunction tf = c.transfer_function;
    d += '_';
    if (tf == JXL_TRANSFER_FUNCTION_GAMMA) {
      d += 'g';
      d += jxl::ToString(c.gamma);
    } else {
      d += ToString(tf);
    }
  }
  return d;
}

static Status MaybeCreateProfileImpl(const JxlColorEncoding& c,
                                     std::vector<uint8_t>* icc) {
  std::vector<uint8_t> header;
  std::vector<uint8_t> tagtable;
  std::vector<uint8_t> tags;
  JxlTransferFunction tf = c.transfer_function;
  if (c.color_space == JXL_COLOR_SPACE_UNKNOWN ||
      tf == JXL_TRANSFER_FUNCTION_UNKNOWN) {
    return false;  // Not an error
  }

  switch (c.color_space) {
    case JXL_COLOR_SPACE_RGB:
    case JXL_COLOR_SPACE_GRAY:
    case JXL_COLOR_SPACE_XYB:
      break;  // OK
    default:
      return JXL_FAILURE("Invalid CS %u",
                         static_cast<unsigned int>(c.color_space));
  }

  if (c.color_space == JXL_COLOR_SPACE_XYB &&
      c.rendering_intent != JXL_RENDERING_INTENT_PERCEPTUAL) {
    return JXL_FAILURE(
        "Only perceptual rendering intent implemented for XYB "
        "ICC profile.");
  }

  JXL_RETURN_IF_ERROR(CreateICCHeader(c, &header));

  std::vector<size_t> offsets;
  // tag count, deferred to later
  WriteICCUint32(0, tagtable.size(), &tagtable);

  size_t tag_offset = 0;
  size_t tag_size = 0;

  CreateICCMlucTag(ColorEncodingDescriptionImpl(c), &tags);
  FinalizeICCTag(&tags, &tag_offset, &tag_size);
  AddToICCTagTable("desc", tag_offset, tag_size, &tagtable, &offsets);

  const std::string copyright = "CC0";
  CreateICCMlucTag(copyright, &tags);
  FinalizeICCTag(&tags, &tag_offset, &tag_size);
  AddToICCTagTable("cprt", tag_offset, tag_size, &tagtable, &offsets);

  // TODO(eustas): isn't it the other way round: gray image has d50 WhitePoint?
  if (c.color_space == JXL_COLOR_SPACE_GRAY) {
    Color wtpt;
    JXL_RETURN_IF_ERROR(
        CIEXYZFromWhiteCIExy(c.white_point_xy[0], c.white_point_xy[1], wtpt));
    JXL_RETURN_IF_ERROR(CreateICCXYZTag(wtpt, &tags));
  } else {
    Color d50{0.964203, 1.0, 0.824905};
    JXL_RETURN_IF_ERROR(CreateICCXYZTag(d50, &tags));
  }
  FinalizeICCTag(&tags, &tag_offset, &tag_size);
  AddToICCTagTable("wtpt", tag_offset, tag_size, &tagtable, &offsets);

  if (c.color_space != JXL_COLOR_SPACE_GRAY) {
    // Chromatic adaptation matrix
    Matrix3x3 chad;
    JXL_RETURN_IF_ERROR(
        CreateICCChadMatrix(c.white_point_xy[0], c.white_point_xy[1], chad));

    JXL_RETURN_IF_ERROR(CreateICCChadTag(chad, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("chad", tag_offset, tag_size, &tagtable, &offsets);
  }

  if (c.color_space == JXL_COLOR_SPACE_RGB) {
    MaybeCreateICCCICPTag(c, &tags, &tag_offset, &tag_size, &tagtable,
                          &offsets);

    Matrix3x3 m;
    JXL_RETURN_IF_ERROR(CreateICCRGBMatrix(
        c.primaries_red_xy[0], c.primaries_red_xy[1], c.primaries_green_xy[0],
        c.primaries_green_xy[1], c.primaries_blue_xy[0], c.primaries_blue_xy[1],
        c.white_point_xy[0], c.white_point_xy[1], m));
    Color r{m[0][0], m[1][0], m[2][0]};
    Color g{m[0][1], m[1][1], m[2][1]};
    Color b{m[0][2], m[1][2], m[2][2]};

    JXL_RETURN_IF_ERROR(CreateICCXYZTag(r, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("rXYZ", tag_offset, tag_size, &tagtable, &offsets);

    JXL_RETURN_IF_ERROR(CreateICCXYZTag(g, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("gXYZ", tag_offset, tag_size, &tagtable, &offsets);

    JXL_RETURN_IF_ERROR(CreateICCXYZTag(b, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("bXYZ", tag_offset, tag_size, &tagtable, &offsets);
  }

  if (c.color_space == JXL_COLOR_SPACE_XYB) {
    JXL_RETURN_IF_ERROR(CreateICCLutAtoBTagForXYB(&tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("A2B0", tag_offset, tag_size, &tagtable, &offsets);
    JXL_RETURN_IF_ERROR(CreateICCNoOpBToATag(&tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("B2A0", tag_offset, tag_size, &tagtable, &offsets);
  } else if (kEnable3DToneMapping && CanToneMap(c)) {
    JXL_RETURN_IF_ERROR(CreateICCLutAtoBTagForHDR(c, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("A2B0", tag_offset, tag_size, &tagtable, &offsets);
    JXL_RETURN_IF_ERROR(CreateICCNoOpBToATag(&tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("B2A0", tag_offset, tag_size, &tagtable, &offsets);
  } else {
    if (tf == JXL_TRANSFER_FUNCTION_GAMMA) {
      float gamma = 1.0 / c.gamma;
      JXL_RETURN_IF_ERROR(CreateICCCurvParaTag({gamma}, 0, &tags));
    } else if (c.color_space != JXL_COLOR_SPACE_XYB) {
      switch (tf) {
        case JXL_TRANSFER_FUNCTION_HLG:
          CreateICCCurvCurvTag(
              CreateTableCurve<64, ExtraTF::kHLG>(CanToneMap(c)), &tags);
          break;
        case JXL_TRANSFER_FUNCTION_PQ:
          CreateICCCurvCurvTag(
              CreateTableCurve<64, ExtraTF::kPQ>(CanToneMap(c)), &tags);
          break;
        case JXL_TRANSFER_FUNCTION_SRGB:
          JXL_RETURN_IF_ERROR(CreateICCCurvParaTag(
              {2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045}, 3,
              &tags));
          break;
        case JXL_TRANSFER_FUNCTION_709:
          JXL_RETURN_IF_ERROR(CreateICCCurvParaTag(
              {1.0 / 0.45, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081}, 3,
              &tags));
          break;
        case JXL_TRANSFER_FUNCTION_LINEAR:
          JXL_RETURN_IF_ERROR(
              CreateICCCurvParaTag({1.0, 1.0, 0.0, 1.0, 0.0}, 3, &tags));
          break;
        case JXL_TRANSFER_FUNCTION_DCI:
          JXL_RETURN_IF_ERROR(
              CreateICCCurvParaTag({2.6, 1.0, 0.0, 1.0, 0.0}, 3, &tags));
          break;
        default:
          return JXL_UNREACHABLE("unknown TF %u",
                                 static_cast<unsigned int>(tf));
      }
    }
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    if (c.color_space == JXL_COLOR_SPACE_GRAY) {
      AddToICCTagTable("kTRC", tag_offset, tag_size, &tagtable, &offsets);
    } else {
      AddToICCTagTable("rTRC", tag_offset, tag_size, &tagtable, &offsets);
      AddToICCTagTable("gTRC", tag_offset, tag_size, &tagtable, &offsets);
      AddToICCTagTable("bTRC", tag_offset, tag_size, &tagtable, &offsets);
    }
  }

  // Tag count
  WriteICCUint32(offsets.size(), 0, &tagtable);
  for (size_t i = 0; i < offsets.size(); i++) {
    WriteICCUint32(offsets[i] + header.size() + tagtable.size(), 4 + 12 * i + 4,
                   &tagtable);
  }

  // ICC profile size
  WriteICCUint32(header.size() + tagtable.size() + tags.size(), 0, &header);

  *icc = header;
  Bytes(tagtable).AppendTo(*icc);
  Bytes(tags).AppendTo(*icc);

  // The MD5 checksum must be computed on the profile with profile flags,
  // rendering intent, and region of the checksum itself, set to 0.
  // TODO(lode): manually verify with a reliable tool that this creates correct
  // signature (profile id) for ICC profiles.
  std::vector<uint8_t> icc_sum = *icc;
  if (icc_sum.size() >= 64 + 4) {
    memset(icc_sum.data() + 44, 0, 4);
    memset(icc_sum.data() + 64, 0, 4);
  }
  uint8_t checksum[16];
  detail::ICCComputeMD5(icc_sum, checksum);

  memcpy(icc->data() + 84, checksum, sizeof(checksum));

  return true;
}

}  // namespace detail

// Returns a representation of the ColorEncoding fields (not icc).
// Example description: "RGB_D65_SRG_Rel_Lin"
static JXL_MAYBE_UNUSED std::string ColorEncodingDescription(
    const JxlColorEncoding& c) {
  return detail::ColorEncodingDescriptionImpl(c);
}

// NOTE: for XYB colorspace, the created profile can be used to transform a
// *scaled* XYB image (created by ScaleXYB()) to another colorspace.
static JXL_MAYBE_UNUSED Status MaybeCreateProfile(const JxlColorEncoding& c,
                                                  std::vector<uint8_t>* icc) {
  return detail::MaybeCreateProfileImpl(c, icc);
}

}  // namespace jxl

#endif  // LIB_JXL_CMS_JXL_CMS_INTERNAL_H_
