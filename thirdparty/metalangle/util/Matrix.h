//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Matrix:
//   Helper class for doing matrix math.
//

#ifndef UTIL_MATRIX_H
#define UTIL_MATRIX_H

#include "common/vector_utils.h"
#include "util/util_export.h"

struct ANGLE_UTIL_EXPORT Matrix4
{
    float data[16];

    Matrix4();
    Matrix4(float m00,
            float m01,
            float m02,
            float m03,
            float m10,
            float m11,
            float m12,
            float m13,
            float m20,
            float m21,
            float m22,
            float m23,
            float m30,
            float m31,
            float m32,
            float m33);

    static Matrix4 identity();
    static Matrix4 rotate(float angle, const angle::Vector3 &p);
    static Matrix4 translate(const angle::Vector3 &t);
    static Matrix4 scale(const angle::Vector3 &s);
    static Matrix4 frustum(float l, float r, float b, float t, float n, float f);
    static Matrix4 perspective(float fov, float aspectRatio, float n, float f);
    static Matrix4 ortho(float l, float r, float b, float t, float n, float f);
    static Matrix4 rollPitchYaw(float roll, float pitch, float yaw);

    static Matrix4 invert(const Matrix4 &mat);
    static Matrix4 transpose(const Matrix4 &mat);
    static angle::Vector3 transform(const Matrix4 &mat, const angle::Vector3 &pt);
    static angle::Vector3 transform(const Matrix4 &mat, const angle::Vector4 &pt);
};

ANGLE_UTIL_EXPORT Matrix4 operator*(const Matrix4 &a, const Matrix4 &b);
ANGLE_UTIL_EXPORT Matrix4 &operator*=(Matrix4 &a, const Matrix4 &b);
ANGLE_UTIL_EXPORT Matrix4 operator*(const Matrix4 &a, float b);
ANGLE_UTIL_EXPORT Matrix4 &operator*=(Matrix4 &a, float b);
ANGLE_UTIL_EXPORT angle::Vector4 operator*(const Matrix4 &a, const angle::Vector4 &b);

ANGLE_UTIL_EXPORT bool operator==(const Matrix4 &a, const Matrix4 &b);
ANGLE_UTIL_EXPORT bool operator!=(const Matrix4 &a, const Matrix4 &b);

#endif  // UTIL_MATRIX_H
