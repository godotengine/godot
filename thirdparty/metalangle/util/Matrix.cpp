//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Matrix:
//   Helper class for doing matrix math.
//

#include "Matrix.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <cstddef>

using namespace angle;

Matrix4::Matrix4()
{
    data[0]  = 1.0f;
    data[4]  = 0.0f;
    data[8]  = 0.0f;
    data[12] = 0.0f;
    data[1]  = 0.0f;
    data[5]  = 1.0f;
    data[9]  = 0.0f;
    data[13] = 0.0f;
    data[2]  = 0.0f;
    data[6]  = 0.0f;
    data[10] = 1.0f;
    data[14] = 0.0f;
    data[3]  = 0.0f;
    data[7]  = 0.0f;
    data[11] = 0.0f;
    data[15] = 1.0f;
}

Matrix4::Matrix4(float m00,
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
                 float m33)
{
    data[0]  = m00;
    data[4]  = m01;
    data[8]  = m02;
    data[12] = m03;
    data[1]  = m10;
    data[5]  = m11;
    data[9]  = m12;
    data[13] = m13;
    data[2]  = m20;
    data[6]  = m21;
    data[10] = m22;
    data[14] = m23;
    data[3]  = m30;
    data[7]  = m31;
    data[11] = m32;
    data[15] = m33;
}

Matrix4 Matrix4::identity()
{
    return Matrix4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 1.0f);
}

Matrix4 Matrix4::rotate(float angle, const Vector3 &p)
{
    Vector3 u   = p.normalized();
    float theta = static_cast<float>(angle * (M_PI / 180.0f));
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    return Matrix4(cos_t + (u.x() * u.x() * (1.0f - cos_t)),
                   (u.x() * u.y() * (1.0f - cos_t)) - (u.z() * sin_t),
                   (u.x() * u.z() * (1.0f - cos_t)) + (u.y() * sin_t), 0.0f,
                   (u.y() * u.x() * (1.0f - cos_t)) + (u.z() * sin_t),
                   cos_t + (u.y() * u.y() * (1.0f - cos_t)),
                   (u.y() * u.z() * (1.0f - cos_t)) - (u.x() * sin_t), 0.0f,
                   (u.z() * u.x() * (1.0f - cos_t)) - (u.y() * sin_t),
                   (u.z() * u.y() * (1.0f - cos_t)) + (u.x() * sin_t),
                   cos_t + (u.z() * u.z() * (1.0f - cos_t)), 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix4 Matrix4::translate(const Vector3 &t)
{
    return Matrix4(1.0f, 0.0f, 0.0f, t.x(), 0.0f, 1.0f, 0.0f, t.y(), 0.0f, 0.0f, 1.0f, t.z(), 0.0f,
                   0.0f, 0.0f, 1.0f);
}

Matrix4 Matrix4::scale(const Vector3 &s)
{
    return Matrix4(s.x(), 0.0f, 0.0f, 0.0f, 0.0f, s.y(), 0.0f, 0.0f, 0.0f, 0.0f, s.z(), 0.0f, 0.0f,
                   0.0f, 0.0f, 1.0f);
}

Matrix4 Matrix4::frustum(float l, float r, float b, float t, float n, float f)
{
    return Matrix4((2.0f * n) / (r - l), 0.0f, (r + l) / (r - l), 0.0f, 0.0f, (2.0f * n) / (t - b),
                   (t + b) / (t - b), 0.0f, 0.0f, 0.0f, -(f + n) / (f - n),
                   -(2.0f * f * n) / (f - n), 0.0f, 0.0f, -1.0f, 0.0f);
}

Matrix4 Matrix4::perspective(float fovY, float aspectRatio, float nearZ, float farZ)
{
    const float frustumHeight = tanf(static_cast<float>(fovY / 360.0f * M_PI)) * nearZ;
    const float frustumWidth  = frustumHeight * aspectRatio;
    return frustum(-frustumWidth, frustumWidth, -frustumHeight, frustumHeight, nearZ, farZ);
}

Matrix4 Matrix4::ortho(float l, float r, float b, float t, float n, float f)
{
    return Matrix4(2.0f / (r - l), 0.0f, 0.0f, -(r + l) / (r - l), 0.0f, 2.0f / (t - b), 0.0f,
                   -(t + b) / (t - b), 0.0f, 0.0f, -2.0f / (f - n), -(f + n) / (f - n), 0.0f, 0.0f,
                   0.0f, 1.0f);
}

Matrix4 Matrix4::rollPitchYaw(float roll, float pitch, float yaw)
{
    return rotate(yaw, Vector3(0, 0, 1)) * rotate(pitch, Vector3(0, 1, 0)) *
           rotate(roll, Vector3(1, 0, 0));
}

Matrix4 Matrix4::invert(const Matrix4 &mat)
{
    Matrix4 inverted(
        mat.data[5] * mat.data[10] * mat.data[15] - mat.data[5] * mat.data[11] * mat.data[14] -
            mat.data[9] * mat.data[6] * mat.data[15] + mat.data[9] * mat.data[7] * mat.data[14] +
            mat.data[13] * mat.data[6] * mat.data[11] - mat.data[13] * mat.data[7] * mat.data[10],
        -mat.data[4] * mat.data[10] * mat.data[15] + mat.data[4] * mat.data[11] * mat.data[14] +
            mat.data[8] * mat.data[6] * mat.data[15] - mat.data[8] * mat.data[7] * mat.data[14] -
            mat.data[12] * mat.data[6] * mat.data[11] + mat.data[12] * mat.data[7] * mat.data[10],
        mat.data[4] * mat.data[9] * mat.data[15] - mat.data[4] * mat.data[11] * mat.data[13] -
            mat.data[8] * mat.data[5] * mat.data[15] + mat.data[8] * mat.data[7] * mat.data[13] +
            mat.data[12] * mat.data[5] * mat.data[11] - mat.data[12] * mat.data[7] * mat.data[9],
        -mat.data[4] * mat.data[9] * mat.data[14] + mat.data[4] * mat.data[10] * mat.data[13] +
            mat.data[8] * mat.data[5] * mat.data[14] - mat.data[8] * mat.data[6] * mat.data[13] -
            mat.data[12] * mat.data[5] * mat.data[10] + mat.data[12] * mat.data[6] * mat.data[9],
        -mat.data[1] * mat.data[10] * mat.data[15] + mat.data[1] * mat.data[11] * mat.data[14] +
            mat.data[9] * mat.data[2] * mat.data[15] - mat.data[9] * mat.data[3] * mat.data[14] -
            mat.data[13] * mat.data[2] * mat.data[11] + mat.data[13] * mat.data[3] * mat.data[10],
        mat.data[0] * mat.data[10] * mat.data[15] - mat.data[0] * mat.data[11] * mat.data[14] -
            mat.data[8] * mat.data[2] * mat.data[15] + mat.data[8] * mat.data[3] * mat.data[14] +
            mat.data[12] * mat.data[2] * mat.data[11] - mat.data[12] * mat.data[3] * mat.data[10],
        -mat.data[0] * mat.data[9] * mat.data[15] + mat.data[0] * mat.data[11] * mat.data[13] +
            mat.data[8] * mat.data[1] * mat.data[15] - mat.data[8] * mat.data[3] * mat.data[13] -
            mat.data[12] * mat.data[1] * mat.data[11] + mat.data[12] * mat.data[3] * mat.data[9],
        mat.data[0] * mat.data[9] * mat.data[14] - mat.data[0] * mat.data[10] * mat.data[13] -
            mat.data[8] * mat.data[1] * mat.data[14] + mat.data[8] * mat.data[2] * mat.data[13] +
            mat.data[12] * mat.data[1] * mat.data[10] - mat.data[12] * mat.data[2] * mat.data[9],
        mat.data[1] * mat.data[6] * mat.data[15] - mat.data[1] * mat.data[7] * mat.data[14] -
            mat.data[5] * mat.data[2] * mat.data[15] + mat.data[5] * mat.data[3] * mat.data[14] +
            mat.data[13] * mat.data[2] * mat.data[7] - mat.data[13] * mat.data[3] * mat.data[6],
        -mat.data[0] * mat.data[6] * mat.data[15] + mat.data[0] * mat.data[7] * mat.data[14] +
            mat.data[4] * mat.data[2] * mat.data[15] - mat.data[4] * mat.data[3] * mat.data[14] -
            mat.data[12] * mat.data[2] * mat.data[7] + mat.data[12] * mat.data[3] * mat.data[6],
        mat.data[0] * mat.data[5] * mat.data[15] - mat.data[0] * mat.data[7] * mat.data[13] -
            mat.data[4] * mat.data[1] * mat.data[15] + mat.data[4] * mat.data[3] * mat.data[13] +
            mat.data[12] * mat.data[1] * mat.data[7] - mat.data[12] * mat.data[3] * mat.data[5],
        -mat.data[0] * mat.data[5] * mat.data[14] + mat.data[0] * mat.data[6] * mat.data[13] +
            mat.data[4] * mat.data[1] * mat.data[14] - mat.data[4] * mat.data[2] * mat.data[13] -
            mat.data[12] * mat.data[1] * mat.data[6] + mat.data[12] * mat.data[2] * mat.data[5],
        -mat.data[1] * mat.data[6] * mat.data[11] + mat.data[1] * mat.data[7] * mat.data[10] +
            mat.data[5] * mat.data[2] * mat.data[11] - mat.data[5] * mat.data[3] * mat.data[10] -
            mat.data[9] * mat.data[2] * mat.data[7] + mat.data[9] * mat.data[3] * mat.data[6],
        mat.data[0] * mat.data[6] * mat.data[11] - mat.data[0] * mat.data[7] * mat.data[10] -
            mat.data[4] * mat.data[2] * mat.data[11] + mat.data[4] * mat.data[3] * mat.data[10] +
            mat.data[8] * mat.data[2] * mat.data[7] - mat.data[8] * mat.data[3] * mat.data[6],
        -mat.data[0] * mat.data[5] * mat.data[11] + mat.data[0] * mat.data[7] * mat.data[9] +
            mat.data[4] * mat.data[1] * mat.data[11] - mat.data[4] * mat.data[3] * mat.data[9] -
            mat.data[8] * mat.data[1] * mat.data[7] + mat.data[8] * mat.data[3] * mat.data[5],
        mat.data[0] * mat.data[5] * mat.data[10] - mat.data[0] * mat.data[6] * mat.data[9] -
            mat.data[4] * mat.data[1] * mat.data[10] + mat.data[4] * mat.data[2] * mat.data[9] +
            mat.data[8] * mat.data[1] * mat.data[6] - mat.data[8] * mat.data[2] * mat.data[5]);

    float determinant = mat.data[0] * inverted.data[0] + mat.data[1] * inverted.data[4] +
                        mat.data[2] * inverted.data[8] + mat.data[3] * inverted.data[12];

    if (determinant != 0.0f)
    {
        inverted *= 1.0f / determinant;
    }
    else
    {
        inverted = identity();
    }

    return inverted;
}

Matrix4 Matrix4::transpose(const Matrix4 &mat)
{
    return Matrix4(mat.data[0], mat.data[1], mat.data[2], mat.data[3], mat.data[4], mat.data[5],
                   mat.data[6], mat.data[7], mat.data[8], mat.data[9], mat.data[10], mat.data[11],
                   mat.data[12], mat.data[13], mat.data[14], mat.data[15]);
}

Vector3 Matrix4::transform(const Matrix4 &mat, const Vector3 &pt)
{
    Vector4 transformed = (mat * Vector4(pt, 1.0f)).normalized();
    return Vector3(transformed.x(), transformed.y(), transformed.z());
}

Vector3 Matrix4::transform(const Matrix4 &mat, const Vector4 &pt)
{
    Vector4 transformed = (mat * pt).normalized();
    return Vector3(transformed.x(), transformed.y(), transformed.z());
}

Matrix4 operator*(const Matrix4 &a, const Matrix4 &b)
{
    return Matrix4(a.data[0] * b.data[0] + a.data[4] * b.data[1] + a.data[8] * b.data[2] +
                       a.data[12] * b.data[3],
                   a.data[0] * b.data[4] + a.data[4] * b.data[5] + a.data[8] * b.data[6] +
                       a.data[12] * b.data[7],
                   a.data[0] * b.data[8] + a.data[4] * b.data[9] + a.data[8] * b.data[10] +
                       a.data[12] * b.data[11],
                   a.data[0] * b.data[12] + a.data[4] * b.data[13] + a.data[8] * b.data[14] +
                       a.data[12] * b.data[15],
                   a.data[1] * b.data[0] + a.data[5] * b.data[1] + a.data[9] * b.data[2] +
                       a.data[13] * b.data[3],
                   a.data[1] * b.data[4] + a.data[5] * b.data[5] + a.data[9] * b.data[6] +
                       a.data[13] * b.data[7],
                   a.data[1] * b.data[8] + a.data[5] * b.data[9] + a.data[9] * b.data[10] +
                       a.data[13] * b.data[11],
                   a.data[1] * b.data[12] + a.data[5] * b.data[13] + a.data[9] * b.data[14] +
                       a.data[13] * b.data[15],
                   a.data[2] * b.data[0] + a.data[6] * b.data[1] + a.data[10] * b.data[2] +
                       a.data[14] * b.data[3],
                   a.data[2] * b.data[4] + a.data[6] * b.data[5] + a.data[10] * b.data[6] +
                       a.data[14] * b.data[7],
                   a.data[2] * b.data[8] + a.data[6] * b.data[9] + a.data[10] * b.data[10] +
                       a.data[14] * b.data[11],
                   a.data[2] * b.data[12] + a.data[6] * b.data[13] + a.data[10] * b.data[14] +
                       a.data[14] * b.data[15],
                   a.data[3] * b.data[0] + a.data[7] * b.data[1] + a.data[11] * b.data[2] +
                       a.data[15] * b.data[3],
                   a.data[3] * b.data[4] + a.data[7] * b.data[5] + a.data[11] * b.data[6] +
                       a.data[15] * b.data[7],
                   a.data[3] * b.data[8] + a.data[7] * b.data[9] + a.data[11] * b.data[10] +
                       a.data[15] * b.data[11],
                   a.data[3] * b.data[12] + a.data[7] * b.data[13] + a.data[11] * b.data[14] +
                       a.data[15] * b.data[15]);
}

Matrix4 &operator*=(Matrix4 &a, const Matrix4 &b)
{
    a = a * b;
    return a;
}

Matrix4 operator*(const Matrix4 &a, float b)
{
    Matrix4 ret(a);
    for (size_t i = 0; i < 16; i++)
    {
        ret.data[i] *= b;
    }
    return ret;
}

Matrix4 &operator*=(Matrix4 &a, float b)
{
    for (size_t i = 0; i < 16; i++)
    {
        a.data[i] *= b;
    }
    return a;
}

Vector4 operator*(const Matrix4 &a, const Vector4 &b)
{
    return Vector4(a.data[0] * b.x() + a.data[4] * b.y() + a.data[8] * b.z() + a.data[12] * b.w(),
                   a.data[1] * b.x() + a.data[5] * b.y() + a.data[9] * b.z() + a.data[13] * b.w(),
                   a.data[2] * b.x() + a.data[6] * b.y() + a.data[10] * b.z() + a.data[14] * b.w(),
                   a.data[3] * b.x() + a.data[7] * b.y() + a.data[11] * b.z() + a.data[15] * b.w());
}

bool operator==(const Matrix4 &a, const Matrix4 &b)
{
    for (size_t i = 0; i < 16; i++)
    {
        if (a.data[i] != b.data[i])
        {
            return false;
        }
    }
    return true;
}

bool operator!=(const Matrix4 &a, const Matrix4 &b)
{
    return !(a == b);
}
