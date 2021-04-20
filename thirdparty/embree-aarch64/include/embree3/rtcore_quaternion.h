// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore_common.h"

RTC_NAMESPACE_BEGIN

/*
 * Structure for transformation respresentation as a matrix decomposition using
 * a quaternion
 */
struct RTC_ALIGN(16) RTCQuaternionDecomposition
{
  float scale_x;
  float scale_y;
  float scale_z;
  float skew_xy;
  float skew_xz;
  float skew_yz;
  float shift_x;
  float shift_y;
  float shift_z;
  float quaternion_r;
  float quaternion_i;
  float quaternion_j;
  float quaternion_k;
  float translation_x;
  float translation_y;
  float translation_z;
};

RTC_FORCEINLINE void rtcInitQuaternionDecomposition(struct RTCQuaternionDecomposition* qdecomp)
{
  qdecomp->scale_x = 1.f;
  qdecomp->scale_y = 1.f;
  qdecomp->scale_z = 1.f;
  qdecomp->skew_xy = 0.f;
  qdecomp->skew_xz = 0.f;
  qdecomp->skew_yz = 0.f;
  qdecomp->shift_x = 0.f;
  qdecomp->shift_y = 0.f;
  qdecomp->shift_z = 0.f;
  qdecomp->quaternion_r = 1.f;
  qdecomp->quaternion_i = 0.f;
  qdecomp->quaternion_j = 0.f;
  qdecomp->quaternion_k = 0.f;
  qdecomp->translation_x = 0.f;
  qdecomp->translation_y = 0.f;
  qdecomp->translation_z = 0.f;
}

RTC_FORCEINLINE void rtcQuaternionDecompositionSetQuaternion(
  struct RTCQuaternionDecomposition* qdecomp,
  float r, float i, float j, float k)
{
  qdecomp->quaternion_r = r;
  qdecomp->quaternion_i = i;
  qdecomp->quaternion_j = j;
  qdecomp->quaternion_k = k;
}

RTC_FORCEINLINE void rtcQuaternionDecompositionSetScale(
  struct RTCQuaternionDecomposition* qdecomp,
  float scale_x, float scale_y, float scale_z)
{
  qdecomp->scale_x = scale_x;
  qdecomp->scale_y = scale_y;
  qdecomp->scale_z = scale_z;
}

RTC_FORCEINLINE void rtcQuaternionDecompositionSetSkew(
  struct RTCQuaternionDecomposition* qdecomp,
  float skew_xy, float skew_xz, float skew_yz)
{
  qdecomp->skew_xy = skew_xy;
  qdecomp->skew_xz = skew_xz;
  qdecomp->skew_yz = skew_yz;
}

RTC_FORCEINLINE void rtcQuaternionDecompositionSetShift(
  struct RTCQuaternionDecomposition* qdecomp,
  float shift_x, float shift_y, float shift_z)
{
  qdecomp->shift_x = shift_x;
  qdecomp->shift_y = shift_y;
  qdecomp->shift_z = shift_z;
}

RTC_FORCEINLINE void rtcQuaternionDecompositionSetTranslation(
  struct RTCQuaternionDecomposition* qdecomp,
  float translation_x, float translation_y, float translation_z)
{
  qdecomp->translation_x = translation_x;
  qdecomp->translation_y = translation_y;
  qdecomp->translation_z = translation_z;
}

RTC_NAMESPACE_END

