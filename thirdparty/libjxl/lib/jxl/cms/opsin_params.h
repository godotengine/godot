// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_CMS_OPSIN_PARAMS_H_
#define LIB_JXL_CMS_OPSIN_PARAMS_H_

#include <array>
#include <cstddef>

#include "lib/jxl/base/matrix_ops.h"

// Constants that define the XYB color space.

namespace jxl {
namespace cms {

// Parameters for opsin absorbance.
constexpr float kM02 = 0.078f;
constexpr float kM00 = 0.30f;
constexpr float kM01 = 1.0f - kM02 - kM00;

constexpr float kM12 = 0.078f;
constexpr float kM10 = 0.23f;
constexpr float kM11 = 1.0f - kM12 - kM10;

constexpr float kM20 = 0.24342268924547819f;
constexpr float kM21 = 0.20476744424496821f;
constexpr float kM22 = 1.0f - kM20 - kM21;

constexpr float kBScale = 1.0f;
constexpr float kYToBRatio = 1.0f;  // works better with 0.50017729543783418
constexpr float kBToYRatio = 1.0f / kYToBRatio;

constexpr float kOpsinAbsorbanceBias0 = 0.0037930732552754493f;
constexpr float kOpsinAbsorbanceBias1 = kOpsinAbsorbanceBias0;
constexpr float kOpsinAbsorbanceBias2 = kOpsinAbsorbanceBias0;

// Opsin absorbance matrix is now frozen.
constexpr Matrix3x3 kOpsinAbsorbanceMatrix{
    {{kM00, kM01, kM02}, {kM10, kM11, kM12}, {kM20, kM21, kM22}}};

constexpr Matrix3x3 kDefaultInverseOpsinAbsorbanceMatrix{
    {{11.031566901960783f, -9.866943921568629f, -0.16462299647058826f},
     {-3.254147380392157f, 4.418770392156863f, -0.16462299647058826f},
     {-3.6588512862745097f, 2.7129230470588235f, 1.9459282392156863f}}};

// Must be the inverse matrix of kOpsinAbsorbanceMatrix and match the spec.
static inline const Matrix3x3& DefaultInverseOpsinAbsorbanceMatrix() {
  return kDefaultInverseOpsinAbsorbanceMatrix;
}

constexpr Vector3 kOpsinAbsorbanceBias = {
    kOpsinAbsorbanceBias0,
    kOpsinAbsorbanceBias1,
    kOpsinAbsorbanceBias2,
};

constexpr std::array<float, 4> kNegOpsinAbsorbanceBiasRGB = {
    -kOpsinAbsorbanceBias0, -kOpsinAbsorbanceBias1, -kOpsinAbsorbanceBias2,
    1.0f};

constexpr float kScaledXYBOffset0 = 0.015386134f;
constexpr float kScaledXYBOffset1 = 0.0f;
constexpr float kScaledXYBOffset2 = 0.27770459f;

constexpr Vector3 kScaledXYBOffset = {kScaledXYBOffset0, kScaledXYBOffset1,
                                      kScaledXYBOffset2};

constexpr float kScaledXYBScale0 = 22.995788804f;
constexpr float kScaledXYBScale1 = 1.183000077f;
constexpr float kScaledXYBScale2 = 1.502141333f;

constexpr Vector3 kScaledXYBScale = {
    kScaledXYBScale0,
    kScaledXYBScale1,
    kScaledXYBScale2,
};

// NB(eustas): following function/variable names are just "namos".

// More precise calculation of 1 / ((1 / r1) + (1 / r2))
constexpr float ReciprocialSum(float r1, float r2) {
  return (r1 * r2) / (r1 + r2);
}

constexpr float kXYBOffset0 = kScaledXYBOffset0 + kScaledXYBOffset1;
constexpr float kXYBOffset1 =
    kScaledXYBOffset1 - kScaledXYBOffset0 + (1.0f / kScaledXYBScale0);
constexpr float kXYBOffset2 = kScaledXYBOffset1 + kScaledXYBOffset2;

constexpr std::array<float, 3> kXYBOffset = {kXYBOffset0, kXYBOffset1,
                                             kXYBOffset2};

constexpr float kXYBScale0 = ReciprocialSum(kScaledXYBScale0, kScaledXYBScale1);
constexpr float kXYBScale1 = ReciprocialSum(kScaledXYBScale0, kScaledXYBScale1);
constexpr float kXYBScale2 = ReciprocialSum(kScaledXYBScale1, kScaledXYBScale2);

constexpr std::array<float, 3> kXYBScale = {kXYBScale0, kXYBScale1, kXYBScale2};

template <size_t idx>
constexpr float ScaledXYBScale() {
  return (idx == 0)   ? kScaledXYBScale0
         : (idx == 1) ? kScaledXYBScale1
                      : kScaledXYBScale2;
}

template <size_t idx>
constexpr float ScaledXYBOffset() {
  return (idx == 0)   ? kScaledXYBOffset0
         : (idx == 1) ? kScaledXYBOffset1
                      : kScaledXYBOffset2;
}

template <size_t x, size_t y, size_t b, size_t idx>
constexpr float XYBCorner() {
  return (((idx == 0)   ? x
           : (idx == 1) ? y
                        : b) /
          ScaledXYBScale<idx>()) -
         ScaledXYBOffset<idx>();
}

template <size_t x, size_t y, size_t b, size_t idx>
constexpr float ScaledA2BCorner() {
  return (idx == 0)   ? (XYBCorner<x, y, b, 1>() + XYBCorner<x, y, b, 0>())
         : (idx == 1) ? (XYBCorner<x, y, b, 1>() - XYBCorner<x, y, b, 0>())
                      : (XYBCorner<x, y, b, 2>() + XYBCorner<x, y, b, 1>());
}

typedef std::array<float, 3> ColorCube0D;
template <size_t x, size_t y, size_t b>
constexpr ColorCube0D UnscaledA2BCorner() {
  return {(ScaledA2BCorner<x, y, b, 0>() + kXYBOffset0) * kXYBScale0,
          (ScaledA2BCorner<x, y, b, 1>() + kXYBOffset1) * kXYBScale1,
          (ScaledA2BCorner<x, y, b, 2>() + kXYBOffset2) * kXYBScale2};
}

typedef std::array<ColorCube0D, 2> ColorCube1D;
template <size_t x, size_t y>
constexpr ColorCube1D UnscaledA2BCubeXY() {
  return {UnscaledA2BCorner<x, y, 0>(), UnscaledA2BCorner<x, y, 1>()};
}

typedef std::array<ColorCube1D, 2> ColorCube2D;
template <size_t x>
constexpr ColorCube2D UnscaledA2BCubeX() {
  return {UnscaledA2BCubeXY<x, 0>(), UnscaledA2BCubeXY<x, 1>()};
}

typedef std::array<ColorCube2D, 2> ColorCube3D;
constexpr ColorCube3D UnscaledA2BCube() {
  return {UnscaledA2BCubeX<0>(), UnscaledA2BCubeX<1>()};
}

constexpr ColorCube3D kUnscaledA2BCube = UnscaledA2BCube();

}  // namespace cms
}  // namespace jxl

#endif  // LIB_JXL_CMS_OPSIN_PARAMS_H_
