//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// matrix_utils.cpp: Contains implementations for Mat4 methods.

#include "common/matrix_utils.h"

namespace angle
{

Mat4::Mat4() : Mat4(1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f)
{}

Mat4::Mat4(const Matrix<float> generalMatrix) : Matrix(std::vector<float>(16, 0), 4, 4)
{
    unsigned int minCols = std::min((unsigned int)4, generalMatrix.columns());
    unsigned int minRows = std::min((unsigned int)4, generalMatrix.rows());
    for (unsigned int i = 0; i < minCols; i++)
    {
        for (unsigned int j = 0; j < minRows; j++)
        {
            mElements[j * minCols + i] = generalMatrix.at(j, i);
        }
    }
}

Mat4::Mat4(const std::vector<float> &elements) : Matrix(elements, 4) {}

Mat4::Mat4(const float *elements) : Matrix(elements, 4) {}

Mat4::Mat4(float m00,
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
    : Matrix(std::vector<float>(16, 0), 4, 4)
{
    mElements[0]  = m00;
    mElements[1]  = m01;
    mElements[2]  = m02;
    mElements[3]  = m03;
    mElements[4]  = m10;
    mElements[5]  = m11;
    mElements[6]  = m12;
    mElements[7]  = m13;
    mElements[8]  = m20;
    mElements[9]  = m21;
    mElements[10] = m22;
    mElements[11] = m23;
    mElements[12] = m30;
    mElements[13] = m31;
    mElements[14] = m32;
    mElements[15] = m33;
}

// static
Mat4 Mat4::Rotate(float angle, const Vector3 &axis)
{
    auto axis_normalized = axis.normalized();
    float angle_radians  = angle * (3.14159265358979323f / 180.0f);
    float c              = cos(angle_radians);
    float ci             = 1.f - c;
    float s              = sin(angle_radians);

    float x = axis_normalized.x();
    float y = axis_normalized.y();
    float z = axis_normalized.z();

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    float xy = x * y;
    float yz = y * z;
    float zx = z * x;

    float r00 = c + ci * x2;
    float r01 = ci * xy + s * z;
    float r02 = ci * zx - s * y;
    float r03 = 0.f;

    float r10 = ci * xy - s * z;
    float r11 = c + ci * y2;
    float r12 = ci * yz + s * x;
    float r13 = 0.f;

    float r20 = ci * zx + s * y;
    float r21 = ci * yz - s * x;
    float r22 = c + ci * z2;
    float r23 = 0.f;

    float r30 = 0.f;
    float r31 = 0.f;
    float r32 = 0.f;
    float r33 = 1.f;

    return Mat4(r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23, r30, r31, r32, r33);
}

// static
Mat4 Mat4::Translate(const Vector3 &t)
{
    float r00 = 1.f;
    float r01 = 0.f;
    float r02 = 0.f;
    float r03 = 0.f;

    float r10 = 0.f;
    float r11 = 1.f;
    float r12 = 0.f;
    float r13 = 0.f;

    float r20 = 0.f;
    float r21 = 0.f;
    float r22 = 1.f;
    float r23 = 0.f;

    float r30 = t.x();
    float r31 = t.y();
    float r32 = t.z();
    float r33 = 1.f;

    return Mat4(r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23, r30, r31, r32, r33);
}

// static
Mat4 Mat4::Scale(const Vector3 &s)
{
    float r00 = s.x();
    float r01 = 0.f;
    float r02 = 0.f;
    float r03 = 0.f;

    float r10 = 0.f;
    float r11 = s.y();
    float r12 = 0.f;
    float r13 = 0.f;

    float r20 = 0.f;
    float r21 = 0.f;
    float r22 = s.z();
    float r23 = 0.f;

    float r30 = 0.f;
    float r31 = 0.f;
    float r32 = 0.f;
    float r33 = 1.f;

    return Mat4(r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23, r30, r31, r32, r33);
}

// static
Mat4 Mat4::Frustum(float l, float r, float b, float t, float n, float f)
{
    float nn  = 2.f * n;
    float fpn = f + n;
    float fmn = f - n;
    float tpb = t + b;
    float tmb = t - b;
    float rpl = r + l;
    float rml = r - l;

    float r00 = nn / rml;
    float r01 = 0.f;
    float r02 = 0.f;
    float r03 = 0.f;

    float r10 = 0.f;
    float r11 = nn / tmb;
    float r12 = 0.f;
    float r13 = 0.f;

    float r20 = rpl / rml;
    float r21 = tpb / tmb;
    float r22 = -fpn / fmn;
    float r23 = -1.f;

    float r30 = 0.f;
    float r31 = 0.f;
    float r32 = -nn * f / fmn;
    float r33 = 0.f;

    return Mat4(r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23, r30, r31, r32, r33);
}

// static
Mat4 Mat4::Perspective(float fov, float aspectRatio, float n, float f)
{
    const float frustumHeight = tanf(static_cast<float>(fov / 360.0f * 3.14159265358979323)) * n;
    const float frustumWidth  = frustumHeight * aspectRatio;
    return Frustum(-frustumWidth, frustumWidth, -frustumHeight, frustumHeight, n, f);
}

// static
Mat4 Mat4::Ortho(float l, float r, float b, float t, float n, float f)
{
    float fpn = f + n;
    float fmn = f - n;
    float tpb = t + b;
    float tmb = t - b;
    float rpl = r + l;
    float rml = r - l;

    float r00 = 2.f / rml;
    float r01 = 0.f;
    float r02 = 0.f;
    float r03 = 0.f;

    float r10 = 0.f;
    float r11 = 2.f / tmb;
    float r12 = 0.f;
    float r13 = 0.f;

    float r20 = 0.f;
    float r21 = 0.f;
    float r22 = -2.f / fmn;
    float r23 = 0.f;

    float r30 = -rpl / rml;
    float r31 = -tpb / tmb;
    float r32 = -fpn / fmn;
    float r33 = 1.f;

    return Mat4(r00, r01, r02, r03, r10, r11, r12, r13, r20, r21, r22, r23, r30, r31, r32, r33);
}

Mat4 Mat4::product(const Mat4 &m)
{
    const float *a = mElements.data();
    const float *b = m.mElements.data();

    return Mat4(a[0] * b[0] + a[4] * b[1] + a[8] * b[2] + a[12] * b[3],
                a[1] * b[0] + a[5] * b[1] + a[9] * b[2] + a[13] * b[3],
                a[2] * b[0] + a[6] * b[1] + a[10] * b[2] + a[14] * b[3],
                a[3] * b[0] + a[7] * b[1] + a[11] * b[2] + a[15] * b[3],

                a[0] * b[4] + a[4] * b[5] + a[8] * b[6] + a[12] * b[7],
                a[1] * b[4] + a[5] * b[5] + a[9] * b[6] + a[13] * b[7],
                a[2] * b[4] + a[6] * b[5] + a[10] * b[6] + a[14] * b[7],
                a[3] * b[4] + a[7] * b[5] + a[11] * b[6] + a[15] * b[7],

                a[0] * b[8] + a[4] * b[9] + a[8] * b[10] + a[12] * b[11],
                a[1] * b[8] + a[5] * b[9] + a[9] * b[10] + a[13] * b[11],
                a[2] * b[8] + a[6] * b[9] + a[10] * b[10] + a[14] * b[11],
                a[3] * b[8] + a[7] * b[9] + a[11] * b[10] + a[15] * b[11],

                a[0] * b[12] + a[4] * b[13] + a[8] * b[14] + a[12] * b[15],
                a[1] * b[12] + a[5] * b[13] + a[9] * b[14] + a[13] * b[15],
                a[2] * b[12] + a[6] * b[13] + a[10] * b[14] + a[14] * b[15],
                a[3] * b[12] + a[7] * b[13] + a[11] * b[14] + a[15] * b[15]);
}

Vector4 Mat4::product(const Vector4 &b)
{
    return Vector4(
        mElements[0] * b.x() + mElements[4] * b.y() + mElements[8] * b.z() + mElements[12] * b.w(),
        mElements[1] * b.x() + mElements[5] * b.y() + mElements[9] * b.z() + mElements[13] * b.w(),
        mElements[2] * b.x() + mElements[6] * b.y() + mElements[10] * b.z() + mElements[14] * b.w(),
        mElements[3] * b.x() + mElements[7] * b.y() + mElements[11] * b.z() +
            mElements[15] * b.w());
}

void Mat4::dump()
{
    printf("[ %f %f %f %f ]\n", mElements[0], mElements[4], mElements[8], mElements[12]);
    printf("[ %f %f %f %f ]\n", mElements[1], mElements[5], mElements[9], mElements[13]);
    printf("[ %f %f %f %f ]\n", mElements[2], mElements[6], mElements[10], mElements[14]);
    printf("[ %f %f %f %f ]\n", mElements[3], mElements[7], mElements[11], mElements[15]);
}

}  // namespace angle
