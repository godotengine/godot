// Copyright (c) 2017-2024, The Khronos Group Inc.
// Copyright (c) 2016, Oculus VR, LLC.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: J.M.P. van Waveren
//

#ifndef XR_LINEAR_H_
#define XR_LINEAR_H_

#include <openxr/openxr.h>

/* REUSE-IgnoreStart */
/* The following has copyright notices that duplicate the header above */

/*
================================================================================================

Description  : Vector, matrix and quaternion math.
Orig. Author : J.M.P. van Waveren
Orig. Date   : 12/10/2016
Language     : C99
Copyright    : Copyright (c) 2016 Oculus VR, LLC. All Rights reserved.


DESCRIPTION
===========

All matrices are column-major.

INTERFACE
=========

XrVector2f
XrVector3f
XrVector4f
XrQuaternionf
XrPosef
XrMatrix4x4f

inline static void XrVector3f_Set(XrVector3f* v, const float value);
inline static void XrVector3f_Add(XrVector3f* result, const XrVector3f* a, const XrVector3f* b);
inline static void XrVector3f_Sub(XrVector3f* result, const XrVector3f* a, const XrVector3f* b);
inline static void XrVector3f_Min(XrVector3f* result, const XrVector3f* a, const XrVector3f* b);
inline static void XrVector3f_Max(XrVector3f* result, const XrVector3f* a, const XrVector3f* b);
inline static void XrVector3f_Decay(XrVector3f* result, const XrVector3f* a, const float value);
inline static void XrVector3f_Lerp(XrVector3f* result, const XrVector3f* a, const XrVector3f* b, const float fraction);
inline static void XrVector3f_Scale(XrVector3f* result, const XrVector3f* a, const float scaleFactor);
inline static void XrVector3f_Normalize(XrVector3f* v);
inline static float XrVector3f_Length(const XrVector3f* v);

inline static void XrQuaternionf_CreateIdentity(XrQuaternionf* q);
inline static void XrQuaternionf_CreateFromAxisAngle(XrQuaternionf* result, const XrVector3f* axis, const float angleInRadians);
inline static void XrQuaternionf_Lerp(XrQuaternionf* result, const XrQuaternionf* a, const XrQuaternionf* b, const float fraction);
inline static void XrQuaternionf_Multiply(XrQuaternionf* result, const XrQuaternionf* a, const XrQuaternionf* b);
inline static void XrQuaternionf_Invert(XrQuaternionf* result, const XrQuaternionf* q);
inline static void XrQuaternionf_Normalize(XrQuaternionf* q);
inline static void XrQuaternionf_RotateVector3f(XrVector3f* result, const XrQuaternionf* a, const XrVector3f* v);

inline static void XrPosef_CreateIdentity(XrPosef* result);
inline static void XrPosef_TransformVector3f(XrVector3f* result, const XrPosef* a, const XrVector3f* v);
inline static void XrPosef_Multiply(XrPosef* result, const XrPosef* a, const XrPosef* b);
inline static void XrPosef_Invert(XrPosef* result, const XrPosef* a);

inline static void XrMatrix4x4f_CreateIdentity(XrMatrix4x4f* result);
inline static void XrMatrix4x4f_CreateTranslation(XrMatrix4x4f* result, const float x, const float y, const float z);
inline static void XrMatrix4x4f_CreateRotation(XrMatrix4x4f* result, const float degreesX, const float degreesY,
                                               const float degreesZ);
inline static void XrMatrix4x4f_CreateScale(XrMatrix4x4f* result, const float x, const float y, const float z);
inline static void XrMatrix4x4f_CreateTranslationRotationScale(XrMatrix4x4f* result, const XrVector3f* translation,
                                                               const XrQuaternionf* rotation, const XrVector3f* scale);
inline static void XrMatrix4x4f_CreateFromRigidTransform(XrMatrix4x4f* result, const XrPosef* s);
inline static void XrMatrix4x4f_CreateProjection(XrMatrix4x4f* result, GraphicsAPI graphicsApi, const float tanAngleLeft,
                                                 const float tanAngleRight, const float tanAngleUp, float const tanAngleDown,
                                                 const float nearZ, const float farZ);
inline static void XrMatrix4x4f_CreateProjectionFov(XrMatrix4x4f* result, GraphicsAPI graphicsApi, const XrFovf fov,
                                                    const float nearZ, const float farZ);
inline static void XrMatrix4x4f_CreateFromQuaternion(XrMatrix4x4f* result, const XrQuaternionf* quat);
inline static void XrMatrix4x4f_CreateOffsetScaleForBounds(XrMatrix4x4f* result, const XrMatrix4x4f* matrix, const XrVector3f* mins,
                                                           const XrVector3f* maxs);

inline static bool XrMatrix4x4f_IsAffine(const XrMatrix4x4f* matrix, const float epsilon);
inline static bool XrMatrix4x4f_IsOrthogonal(const XrMatrix4x4f* matrix, const float epsilon);
inline static bool XrMatrix4x4f_IsOrthonormal(const XrMatrix4x4f* matrix, const float epsilon);
inline static bool XrMatrix4x4f_IsRigidBody(const XrMatrix4x4f* matrix, const float epsilon);

inline static void XrMatrix4x4f_GetTranslation(XrVector3f* result, const XrMatrix4x4f* src);
inline static void XrMatrix4x4f_GetRotation(XrQuaternionf* result, const XrMatrix4x4f* src);
inline static void XrMatrix4x4f_GetScale(XrVector3f* result, const XrMatrix4x4f* src);

inline static void XrMatrix4x4f_Multiply(XrMatrix4x4f* result, const XrMatrix4x4f* a, const XrMatrix4x4f* b);
inline static void XrMatrix4x4f_Transpose(XrMatrix4x4f* result, const XrMatrix4x4f* src);
inline static void XrMatrix4x4f_Invert(XrMatrix4x4f* result, const XrMatrix4x4f* src);
inline static void XrMatrix4x4f_InvertRigidBody(XrMatrix4x4f* result, const XrMatrix4x4f* src);

inline static void XrMatrix4x4f_TransformVector3f(XrVector3f* result, const XrMatrix4x4f* m, const XrVector3f* v);
inline static void XrMatrix4x4f_TransformVector4f(XrVector4f* result, const XrMatrix4x4f* m, const XrVector4f* v);

inline static void XrMatrix4x4f_TransformBounds(XrVector3f* resultMins, XrVector3f* resultMaxs, const XrMatrix4x4f* matrix,
                                                const XrVector3f* mins, const XrVector3f* maxs);
inline static bool XrMatrix4x4f_CullBounds(const XrMatrix4x4f* mvp, const XrVector3f* mins, const XrVector3f* maxs);

================================================================================================
*/

#include <assert.h>
#include <math.h>
#include <stdbool.h>

#define MATH_PI 3.14159265358979323846f

#define DEFAULT_NEAR_Z 0.015625f  // exact floating point representation
#define INFINITE_FAR_Z 0.0f

static const XrColor4f XrColorRed = {1.0f, 0.0f, 0.0f, 1.0f};
static const XrColor4f XrColorGreen = {0.0f, 1.0f, 0.0f, 1.0f};
static const XrColor4f XrColorBlue = {0.0f, 0.0f, 1.0f, 1.0f};
static const XrColor4f XrColorYellow = {1.0f, 1.0f, 0.0f, 1.0f};
static const XrColor4f XrColorPurple = {1.0f, 0.0f, 1.0f, 1.0f};
static const XrColor4f XrColorCyan = {0.0f, 1.0f, 1.0f, 1.0f};
static const XrColor4f XrColorLightGrey = {0.7f, 0.7f, 0.7f, 1.0f};
static const XrColor4f XrColorDarkGrey = {0.3f, 0.3f, 0.3f, 1.0f};

typedef enum GraphicsAPI { GRAPHICS_VULKAN, GRAPHICS_OPENGL, GRAPHICS_OPENGL_ES, GRAPHICS_D3D, GRAPHICS_METAL } GraphicsAPI;

// Column-major, pre-multiplied. This type does not exist in the OpenXR API and is provided for convenience.
typedef struct XrMatrix4x4f {
    float m[16];
} XrMatrix4x4f;

inline static float XrRcpSqrt(const float x) {
    const float SMALLEST_NON_DENORMAL = 1.1754943508222875e-038f;  // ( 1U << 23 )
    const float rcp = (x >= SMALLEST_NON_DENORMAL) ? 1.0f / sqrtf(x) : 1.0f;
    return rcp;
}

inline static float XrVector2f_Length(const XrVector2f* v) { return sqrtf(v->x * v->x + v->y * v->y); }

inline static void XrVector3f_Set(XrVector3f* v, const float value) {
    v->x = value;
    v->y = value;
    v->z = value;
}

inline static void XrVector3f_Add(XrVector3f* result, const XrVector3f* a, const XrVector3f* b) {
    result->x = a->x + b->x;
    result->y = a->y + b->y;
    result->z = a->z + b->z;
}

inline static void XrVector3f_Sub(XrVector3f* result, const XrVector3f* a, const XrVector3f* b) {
    result->x = a->x - b->x;
    result->y = a->y - b->y;
    result->z = a->z - b->z;
}

inline static void XrVector3f_Min(XrVector3f* result, const XrVector3f* a, const XrVector3f* b) {
    result->x = (a->x < b->x) ? a->x : b->x;
    result->y = (a->y < b->y) ? a->y : b->y;
    result->z = (a->z < b->z) ? a->z : b->z;
}

inline static void XrVector3f_Max(XrVector3f* result, const XrVector3f* a, const XrVector3f* b) {
    result->x = (a->x > b->x) ? a->x : b->x;
    result->y = (a->y > b->y) ? a->y : b->y;
    result->z = (a->z > b->z) ? a->z : b->z;
}

inline static void XrVector3f_Decay(XrVector3f* result, const XrVector3f* a, const float value) {
    result->x = (fabsf(a->x) > value) ? ((a->x > 0.0f) ? (a->x - value) : (a->x + value)) : 0.0f;
    result->y = (fabsf(a->y) > value) ? ((a->y > 0.0f) ? (a->y - value) : (a->y + value)) : 0.0f;
    result->z = (fabsf(a->z) > value) ? ((a->z > 0.0f) ? (a->z - value) : (a->z + value)) : 0.0f;
}

inline static void XrVector3f_Lerp(XrVector3f* result, const XrVector3f* a, const XrVector3f* b, const float fraction) {
    result->x = a->x + fraction * (b->x - a->x);
    result->y = a->y + fraction * (b->y - a->y);
    result->z = a->z + fraction * (b->z - a->z);
}

inline static void XrVector3f_Scale(XrVector3f* result, const XrVector3f* a, const float scaleFactor) {
    result->x = a->x * scaleFactor;
    result->y = a->y * scaleFactor;
    result->z = a->z * scaleFactor;
}

inline static float XrVector3f_Dot(const XrVector3f* a, const XrVector3f* b) { return a->x * b->x + a->y * b->y + a->z * b->z; }

// Compute cross product, which generates a normal vector.
// Direction vector can be determined by right-hand rule: Pointing index finder in
// direction a and middle finger in direction b, thumb will point in Cross(a, b).
inline static void XrVector3f_Cross(XrVector3f* result, const XrVector3f* a, const XrVector3f* b) {
    result->x = a->y * b->z - a->z * b->y;
    result->y = a->z * b->x - a->x * b->z;
    result->z = a->x * b->y - a->y * b->x;
}

inline static void XrVector3f_Normalize(XrVector3f* v) {
    const float lengthRcp = XrRcpSqrt(v->x * v->x + v->y * v->y + v->z * v->z);
    v->x *= lengthRcp;
    v->y *= lengthRcp;
    v->z *= lengthRcp;
}

inline static float XrVector3f_Length(const XrVector3f* v) { return sqrtf(v->x * v->x + v->y * v->y + v->z * v->z); }

inline static void XrQuaternionf_CreateIdentity(XrQuaternionf* q) {
    q->x = 0.0f;
    q->y = 0.0f;
    q->z = 0.0f;
    q->w = 1.0f;
}

inline static void XrQuaternionf_CreateFromAxisAngle(XrQuaternionf* result, const XrVector3f* axis, const float angleInRadians) {
    float s = sinf(angleInRadians / 2.0f);
    float lengthRcp = XrRcpSqrt(axis->x * axis->x + axis->y * axis->y + axis->z * axis->z);
    result->x = s * axis->x * lengthRcp;
    result->y = s * axis->y * lengthRcp;
    result->z = s * axis->z * lengthRcp;
    result->w = cosf(angleInRadians / 2.0f);
}

inline static void XrQuaternionf_Lerp(XrQuaternionf* result, const XrQuaternionf* a, const XrQuaternionf* b, const float fraction) {
    const float s = a->x * b->x + a->y * b->y + a->z * b->z + a->w * b->w;
    const float fa = 1.0f - fraction;
    const float fb = (s < 0.0f) ? -fraction : fraction;
    const float x = a->x * fa + b->x * fb;
    const float y = a->y * fa + b->y * fb;
    const float z = a->z * fa + b->z * fb;
    const float w = a->w * fa + b->w * fb;
    const float lengthRcp = XrRcpSqrt(x * x + y * y + z * z + w * w);
    result->x = x * lengthRcp;
    result->y = y * lengthRcp;
    result->z = z * lengthRcp;
    result->w = w * lengthRcp;
}

inline static void XrQuaternionf_Multiply(XrQuaternionf* result, const XrQuaternionf* a, const XrQuaternionf* b) {
    result->x = (b->w * a->x) + (b->x * a->w) + (b->y * a->z) - (b->z * a->y);
    result->y = (b->w * a->y) - (b->x * a->z) + (b->y * a->w) + (b->z * a->x);
    result->z = (b->w * a->z) + (b->x * a->y) - (b->y * a->x) + (b->z * a->w);
    result->w = (b->w * a->w) - (b->x * a->x) - (b->y * a->y) - (b->z * a->z);
}

inline static void XrQuaternionf_Invert(XrQuaternionf* result, const XrQuaternionf* q) {
    result->x = -q->x;
    result->y = -q->y;
    result->z = -q->z;
    result->w = q->w;
}

inline static void XrQuaternionf_Normalize(XrQuaternionf* q) {
    const float lengthRcp = XrRcpSqrt(q->x * q->x + q->y * q->y + q->z * q->z + q->w * q->w);
    q->x *= lengthRcp;
    q->y *= lengthRcp;
    q->z *= lengthRcp;
    q->w *= lengthRcp;
}

inline static void XrQuaternionf_RotateVector3f(XrVector3f* result, const XrQuaternionf* a, const XrVector3f* v) {
    XrQuaternionf q = {v->x, v->y, v->z, 0.0f};
    XrQuaternionf aq;
    XrQuaternionf_Multiply(&aq, &q, a);
    XrQuaternionf aInv;
    XrQuaternionf_Invert(&aInv, a);
    XrQuaternionf aqaInv;
    XrQuaternionf_Multiply(&aqaInv, &aInv, &aq);

    result->x = aqaInv.x;
    result->y = aqaInv.y;
    result->z = aqaInv.z;
}

inline static void XrPosef_CreateIdentity(XrPosef* result) {
    XrQuaternionf_CreateIdentity(&result->orientation);
    XrVector3f_Set(&result->position, 0);
}

inline static void XrPosef_TransformVector3f(XrVector3f* result, const XrPosef* a, const XrVector3f* v) {
    XrVector3f r0;
    XrQuaternionf_RotateVector3f(&r0, &a->orientation, v);
    XrVector3f_Add(result, &r0, &a->position);
}

inline static void XrPosef_Multiply(XrPosef* result, const XrPosef* a, const XrPosef* b) {
    XrQuaternionf_Multiply(&result->orientation, &b->orientation, &a->orientation);
    XrPosef_TransformVector3f(&result->position, a, &b->position);
}

inline static void XrPosef_Invert(XrPosef* result, const XrPosef* a) {
    XrQuaternionf_Invert(&result->orientation, &a->orientation);
    XrVector3f aPosNeg;
    XrVector3f_Scale(&aPosNeg, &a->position, -1.0f);
    XrQuaternionf_RotateVector3f(&result->position, &result->orientation, &aPosNeg);
}

// Use left-multiplication to accumulate transformations.
inline static void XrMatrix4x4f_Multiply(XrMatrix4x4f* result, const XrMatrix4x4f* a, const XrMatrix4x4f* b) {
    result->m[0] = a->m[0] * b->m[0] + a->m[4] * b->m[1] + a->m[8] * b->m[2] + a->m[12] * b->m[3];
    result->m[1] = a->m[1] * b->m[0] + a->m[5] * b->m[1] + a->m[9] * b->m[2] + a->m[13] * b->m[3];
    result->m[2] = a->m[2] * b->m[0] + a->m[6] * b->m[1] + a->m[10] * b->m[2] + a->m[14] * b->m[3];
    result->m[3] = a->m[3] * b->m[0] + a->m[7] * b->m[1] + a->m[11] * b->m[2] + a->m[15] * b->m[3];

    result->m[4] = a->m[0] * b->m[4] + a->m[4] * b->m[5] + a->m[8] * b->m[6] + a->m[12] * b->m[7];
    result->m[5] = a->m[1] * b->m[4] + a->m[5] * b->m[5] + a->m[9] * b->m[6] + a->m[13] * b->m[7];
    result->m[6] = a->m[2] * b->m[4] + a->m[6] * b->m[5] + a->m[10] * b->m[6] + a->m[14] * b->m[7];
    result->m[7] = a->m[3] * b->m[4] + a->m[7] * b->m[5] + a->m[11] * b->m[6] + a->m[15] * b->m[7];

    result->m[8] = a->m[0] * b->m[8] + a->m[4] * b->m[9] + a->m[8] * b->m[10] + a->m[12] * b->m[11];
    result->m[9] = a->m[1] * b->m[8] + a->m[5] * b->m[9] + a->m[9] * b->m[10] + a->m[13] * b->m[11];
    result->m[10] = a->m[2] * b->m[8] + a->m[6] * b->m[9] + a->m[10] * b->m[10] + a->m[14] * b->m[11];
    result->m[11] = a->m[3] * b->m[8] + a->m[7] * b->m[9] + a->m[11] * b->m[10] + a->m[15] * b->m[11];

    result->m[12] = a->m[0] * b->m[12] + a->m[4] * b->m[13] + a->m[8] * b->m[14] + a->m[12] * b->m[15];
    result->m[13] = a->m[1] * b->m[12] + a->m[5] * b->m[13] + a->m[9] * b->m[14] + a->m[13] * b->m[15];
    result->m[14] = a->m[2] * b->m[12] + a->m[6] * b->m[13] + a->m[10] * b->m[14] + a->m[14] * b->m[15];
    result->m[15] = a->m[3] * b->m[12] + a->m[7] * b->m[13] + a->m[11] * b->m[14] + a->m[15] * b->m[15];
}

// Creates the transpose of the given matrix.
inline static void XrMatrix4x4f_Transpose(XrMatrix4x4f* result, const XrMatrix4x4f* src) {
    result->m[0] = src->m[0];
    result->m[1] = src->m[4];
    result->m[2] = src->m[8];
    result->m[3] = src->m[12];

    result->m[4] = src->m[1];
    result->m[5] = src->m[5];
    result->m[6] = src->m[9];
    result->m[7] = src->m[13];

    result->m[8] = src->m[2];
    result->m[9] = src->m[6];
    result->m[10] = src->m[10];
    result->m[11] = src->m[14];

    result->m[12] = src->m[3];
    result->m[13] = src->m[7];
    result->m[14] = src->m[11];
    result->m[15] = src->m[15];
}

// Returns a 3x3 minor of a 4x4 matrix.
inline static float XrMatrix4x4f_Minor(const XrMatrix4x4f* matrix, int r0, int r1, int r2, int c0, int c1, int c2) {
    return matrix->m[4 * r0 + c0] *
               (matrix->m[4 * r1 + c1] * matrix->m[4 * r2 + c2] - matrix->m[4 * r2 + c1] * matrix->m[4 * r1 + c2]) -
           matrix->m[4 * r0 + c1] *
               (matrix->m[4 * r1 + c0] * matrix->m[4 * r2 + c2] - matrix->m[4 * r2 + c0] * matrix->m[4 * r1 + c2]) +
           matrix->m[4 * r0 + c2] *
               (matrix->m[4 * r1 + c0] * matrix->m[4 * r2 + c1] - matrix->m[4 * r2 + c0] * matrix->m[4 * r1 + c1]);
}

// Calculates the inverse of a 4x4 matrix.
inline static void XrMatrix4x4f_Invert(XrMatrix4x4f* result, const XrMatrix4x4f* src) {
    const float rcpDet =
        1.0f / (src->m[0] * XrMatrix4x4f_Minor(src, 1, 2, 3, 1, 2, 3) - src->m[1] * XrMatrix4x4f_Minor(src, 1, 2, 3, 0, 2, 3) +
                src->m[2] * XrMatrix4x4f_Minor(src, 1, 2, 3, 0, 1, 3) - src->m[3] * XrMatrix4x4f_Minor(src, 1, 2, 3, 0, 1, 2));

    result->m[0] = XrMatrix4x4f_Minor(src, 1, 2, 3, 1, 2, 3) * rcpDet;
    result->m[1] = -XrMatrix4x4f_Minor(src, 0, 2, 3, 1, 2, 3) * rcpDet;
    result->m[2] = XrMatrix4x4f_Minor(src, 0, 1, 3, 1, 2, 3) * rcpDet;
    result->m[3] = -XrMatrix4x4f_Minor(src, 0, 1, 2, 1, 2, 3) * rcpDet;
    result->m[4] = -XrMatrix4x4f_Minor(src, 1, 2, 3, 0, 2, 3) * rcpDet;
    result->m[5] = XrMatrix4x4f_Minor(src, 0, 2, 3, 0, 2, 3) * rcpDet;
    result->m[6] = -XrMatrix4x4f_Minor(src, 0, 1, 3, 0, 2, 3) * rcpDet;
    result->m[7] = XrMatrix4x4f_Minor(src, 0, 1, 2, 0, 2, 3) * rcpDet;
    result->m[8] = XrMatrix4x4f_Minor(src, 1, 2, 3, 0, 1, 3) * rcpDet;
    result->m[9] = -XrMatrix4x4f_Minor(src, 0, 2, 3, 0, 1, 3) * rcpDet;
    result->m[10] = XrMatrix4x4f_Minor(src, 0, 1, 3, 0, 1, 3) * rcpDet;
    result->m[11] = -XrMatrix4x4f_Minor(src, 0, 1, 2, 0, 1, 3) * rcpDet;
    result->m[12] = -XrMatrix4x4f_Minor(src, 1, 2, 3, 0, 1, 2) * rcpDet;
    result->m[13] = XrMatrix4x4f_Minor(src, 0, 2, 3, 0, 1, 2) * rcpDet;
    result->m[14] = -XrMatrix4x4f_Minor(src, 0, 1, 3, 0, 1, 2) * rcpDet;
    result->m[15] = XrMatrix4x4f_Minor(src, 0, 1, 2, 0, 1, 2) * rcpDet;
}

// Calculates the inverse of a rigid body transform.
inline static void XrMatrix4x4f_InvertRigidBody(XrMatrix4x4f* result, const XrMatrix4x4f* src) {
    result->m[0] = src->m[0];
    result->m[1] = src->m[4];
    result->m[2] = src->m[8];
    result->m[3] = 0.0f;
    result->m[4] = src->m[1];
    result->m[5] = src->m[5];
    result->m[6] = src->m[9];
    result->m[7] = 0.0f;
    result->m[8] = src->m[2];
    result->m[9] = src->m[6];
    result->m[10] = src->m[10];
    result->m[11] = 0.0f;
    result->m[12] = -(src->m[0] * src->m[12] + src->m[1] * src->m[13] + src->m[2] * src->m[14]);
    result->m[13] = -(src->m[4] * src->m[12] + src->m[5] * src->m[13] + src->m[6] * src->m[14]);
    result->m[14] = -(src->m[8] * src->m[12] + src->m[9] * src->m[13] + src->m[10] * src->m[14]);
    result->m[15] = 1.0f;
}

// Creates an identity matrix.
inline static void XrMatrix4x4f_CreateIdentity(XrMatrix4x4f* result) {
    result->m[0] = 1.0f;
    result->m[1] = 0.0f;
    result->m[2] = 0.0f;
    result->m[3] = 0.0f;
    result->m[4] = 0.0f;
    result->m[5] = 1.0f;
    result->m[6] = 0.0f;
    result->m[7] = 0.0f;
    result->m[8] = 0.0f;
    result->m[9] = 0.0f;
    result->m[10] = 1.0f;
    result->m[11] = 0.0f;
    result->m[12] = 0.0f;
    result->m[13] = 0.0f;
    result->m[14] = 0.0f;
    result->m[15] = 1.0f;
}

// Creates a translation matrix.
inline static void XrMatrix4x4f_CreateTranslation(XrMatrix4x4f* result, const float x, const float y, const float z) {
    result->m[0] = 1.0f;
    result->m[1] = 0.0f;
    result->m[2] = 0.0f;
    result->m[3] = 0.0f;
    result->m[4] = 0.0f;
    result->m[5] = 1.0f;
    result->m[6] = 0.0f;
    result->m[7] = 0.0f;
    result->m[8] = 0.0f;
    result->m[9] = 0.0f;
    result->m[10] = 1.0f;
    result->m[11] = 0.0f;
    result->m[12] = x;
    result->m[13] = y;
    result->m[14] = z;
    result->m[15] = 1.0f;
}

// Creates a rotation matrix.
// If -Z=forward, +Y=up, +X=right, then radiansX=pitch, radiansY=yaw, radiansZ=roll.
inline static void XrMatrix4x4f_CreateRotationRadians(XrMatrix4x4f* result, const float radiansX, const float radiansY,
                                                      const float radiansZ) {
    const float sinX = sinf(radiansX);
    const float cosX = cosf(radiansX);
    const XrMatrix4x4f rotationX = {{1, 0, 0, 0, 0, cosX, sinX, 0, 0, -sinX, cosX, 0, 0, 0, 0, 1}};
    const float sinY = sinf(radiansY);
    const float cosY = cosf(radiansY);
    const XrMatrix4x4f rotationY = {{cosY, 0, -sinY, 0, 0, 1, 0, 0, sinY, 0, cosY, 0, 0, 0, 0, 1}};
    const float sinZ = sinf(radiansZ);
    const float cosZ = cosf(radiansZ);
    const XrMatrix4x4f rotationZ = {{cosZ, sinZ, 0, 0, -sinZ, cosZ, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
    XrMatrix4x4f rotationXY;
    XrMatrix4x4f_Multiply(&rotationXY, &rotationY, &rotationX);
    XrMatrix4x4f_Multiply(result, &rotationZ, &rotationXY);
}

// Creates a rotation matrix.
// If -Z=forward, +Y=up, +X=right, then degreesX=pitch, degreesY=yaw, degreesZ=roll.
inline static void XrMatrix4x4f_CreateRotation(XrMatrix4x4f* result, const float degreesX, const float degreesY,
                                               const float degreesZ) {
    XrMatrix4x4f_CreateRotationRadians(result, degreesX * (MATH_PI / 180.0f), degreesY * (MATH_PI / 180.0f),
                                       degreesZ * (MATH_PI / 180.0f));
}

// Creates a scale matrix.
inline static void XrMatrix4x4f_CreateScale(XrMatrix4x4f* result, const float x, const float y, const float z) {
    result->m[0] = x;
    result->m[1] = 0.0f;
    result->m[2] = 0.0f;
    result->m[3] = 0.0f;
    result->m[4] = 0.0f;
    result->m[5] = y;
    result->m[6] = 0.0f;
    result->m[7] = 0.0f;
    result->m[8] = 0.0f;
    result->m[9] = 0.0f;
    result->m[10] = z;
    result->m[11] = 0.0f;
    result->m[12] = 0.0f;
    result->m[13] = 0.0f;
    result->m[14] = 0.0f;
    result->m[15] = 1.0f;
}

// Creates a matrix from a quaternion.
inline static void XrMatrix4x4f_CreateFromQuaternion(XrMatrix4x4f* result, const XrQuaternionf* quat) {
    const float x2 = quat->x + quat->x;
    const float y2 = quat->y + quat->y;
    const float z2 = quat->z + quat->z;

    const float xx2 = quat->x * x2;
    const float yy2 = quat->y * y2;
    const float zz2 = quat->z * z2;

    const float yz2 = quat->y * z2;
    const float wx2 = quat->w * x2;
    const float xy2 = quat->x * y2;
    const float wz2 = quat->w * z2;
    const float xz2 = quat->x * z2;
    const float wy2 = quat->w * y2;

    result->m[0] = 1.0f - yy2 - zz2;
    result->m[1] = xy2 + wz2;
    result->m[2] = xz2 - wy2;
    result->m[3] = 0.0f;

    result->m[4] = xy2 - wz2;
    result->m[5] = 1.0f - xx2 - zz2;
    result->m[6] = yz2 + wx2;
    result->m[7] = 0.0f;

    result->m[8] = xz2 + wy2;
    result->m[9] = yz2 - wx2;
    result->m[10] = 1.0f - xx2 - yy2;
    result->m[11] = 0.0f;

    result->m[12] = 0.0f;
    result->m[13] = 0.0f;
    result->m[14] = 0.0f;
    result->m[15] = 1.0f;
}

// Creates a combined translation(rotation(scale(object))) matrix.
inline static void XrMatrix4x4f_CreateTranslationRotationScale(XrMatrix4x4f* result, const XrVector3f* translation,
                                                               const XrQuaternionf* rotation, const XrVector3f* scale) {
    XrMatrix4x4f scaleMatrix;
    XrMatrix4x4f_CreateScale(&scaleMatrix, scale->x, scale->y, scale->z);

    XrMatrix4x4f rotationMatrix;
    XrMatrix4x4f_CreateFromQuaternion(&rotationMatrix, rotation);

    XrMatrix4x4f translationMatrix;
    XrMatrix4x4f_CreateTranslation(&translationMatrix, translation->x, translation->y, translation->z);

    XrMatrix4x4f combinedMatrix;
    XrMatrix4x4f_Multiply(&combinedMatrix, &rotationMatrix, &scaleMatrix);
    XrMatrix4x4f_Multiply(result, &translationMatrix, &combinedMatrix);
}

inline static void XrMatrix4x4f_CreateFromRigidTransform(XrMatrix4x4f* result, const XrPosef* s) {
    const XrVector3f identityScale = {1.0f, 1.0f, 1.0f};
    XrMatrix4x4f_CreateTranslationRotationScale(result, &s->position, &s->orientation, &identityScale);
}

// Creates a projection matrix based on the specified dimensions.
// The projection matrix transforms -Z=forward, +Y=up, +X=right to the appropriate clip space for the graphics API.
// The far plane is placed at infinity if farZ <= nearZ.
// An infinite projection matrix is preferred for rasterization because, except for
// things *right* up against the near plane, it always provides better precision:
//              "Tightening the Precision of Perspective Rendering"
//              Paul Upchurch, Mathieu Desbrun
//              Journal of Graphics Tools, Volume 16, Issue 1, 2012
inline static void XrMatrix4x4f_CreateProjection(XrMatrix4x4f* result, GraphicsAPI graphicsApi, const float tanAngleLeft,
                                                 const float tanAngleRight, const float tanAngleUp, float const tanAngleDown,
                                                 const float nearZ, const float farZ) {
    const float tanAngleWidth = tanAngleRight - tanAngleLeft;

    // Set to tanAngleDown - tanAngleUp for a clip space with positive Y down (Vulkan).
    // Set to tanAngleUp - tanAngleDown for a clip space with positive Y up (OpenGL / D3D / Metal).
    const float tanAngleHeight = graphicsApi == GRAPHICS_VULKAN ? (tanAngleDown - tanAngleUp) : (tanAngleUp - tanAngleDown);

    // Set to nearZ for a [-1,1] Z clip space (OpenGL / OpenGL ES).
    // Set to zero for a [0,1] Z clip space (Vulkan / D3D / Metal).
    const float offsetZ = (graphicsApi == GRAPHICS_OPENGL || graphicsApi == GRAPHICS_OPENGL_ES) ? nearZ : 0;

    if (farZ <= nearZ) {
        // place the far plane at infinity
        result->m[0] = 2.0f / tanAngleWidth;
        result->m[4] = 0.0f;
        result->m[8] = (tanAngleRight + tanAngleLeft) / tanAngleWidth;
        result->m[12] = 0.0f;

        result->m[1] = 0.0f;
        result->m[5] = 2.0f / tanAngleHeight;
        result->m[9] = (tanAngleUp + tanAngleDown) / tanAngleHeight;
        result->m[13] = 0.0f;

        result->m[2] = 0.0f;
        result->m[6] = 0.0f;
        result->m[10] = -1.0f;
        result->m[14] = -(nearZ + offsetZ);

        result->m[3] = 0.0f;
        result->m[7] = 0.0f;
        result->m[11] = -1.0f;
        result->m[15] = 0.0f;
    } else {
        // normal projection
        result->m[0] = 2.0f / tanAngleWidth;
        result->m[4] = 0.0f;
        result->m[8] = (tanAngleRight + tanAngleLeft) / tanAngleWidth;
        result->m[12] = 0.0f;

        result->m[1] = 0.0f;
        result->m[5] = 2.0f / tanAngleHeight;
        result->m[9] = (tanAngleUp + tanAngleDown) / tanAngleHeight;
        result->m[13] = 0.0f;

        result->m[2] = 0.0f;
        result->m[6] = 0.0f;
        result->m[10] = -(farZ + offsetZ) / (farZ - nearZ);
        result->m[14] = -(farZ * (nearZ + offsetZ)) / (farZ - nearZ);

        result->m[3] = 0.0f;
        result->m[7] = 0.0f;
        result->m[11] = -1.0f;
        result->m[15] = 0.0f;
    }
}

// Creates a projection matrix based on the specified FOV.
inline static void XrMatrix4x4f_CreateProjectionFov(XrMatrix4x4f* result, GraphicsAPI graphicsApi, const XrFovf fov,
                                                    const float nearZ, const float farZ) {
    const float tanLeft = tanf(fov.angleLeft);
    const float tanRight = tanf(fov.angleRight);

    const float tanDown = tanf(fov.angleDown);
    const float tanUp = tanf(fov.angleUp);

    XrMatrix4x4f_CreateProjection(result, graphicsApi, tanLeft, tanRight, tanUp, tanDown, nearZ, farZ);
}

// Creates a matrix that transforms the -1 to 1 cube to cover the given 'mins' and 'maxs' transformed with the given 'matrix'.
inline static void XrMatrix4x4f_CreateOffsetScaleForBounds(XrMatrix4x4f* result, const XrMatrix4x4f* matrix, const XrVector3f* mins,
                                                           const XrVector3f* maxs) {
    const XrVector3f offset = {(maxs->x + mins->x) * 0.5f, (maxs->y + mins->y) * 0.5f, (maxs->z + mins->z) * 0.5f};
    const XrVector3f scale = {(maxs->x - mins->x) * 0.5f, (maxs->y - mins->y) * 0.5f, (maxs->z - mins->z) * 0.5f};

    result->m[0] = matrix->m[0] * scale.x;
    result->m[1] = matrix->m[1] * scale.x;
    result->m[2] = matrix->m[2] * scale.x;
    result->m[3] = matrix->m[3] * scale.x;

    result->m[4] = matrix->m[4] * scale.y;
    result->m[5] = matrix->m[5] * scale.y;
    result->m[6] = matrix->m[6] * scale.y;
    result->m[7] = matrix->m[7] * scale.y;

    result->m[8] = matrix->m[8] * scale.z;
    result->m[9] = matrix->m[9] * scale.z;
    result->m[10] = matrix->m[10] * scale.z;
    result->m[11] = matrix->m[11] * scale.z;

    result->m[12] = matrix->m[12] + matrix->m[0] * offset.x + matrix->m[4] * offset.y + matrix->m[8] * offset.z;
    result->m[13] = matrix->m[13] + matrix->m[1] * offset.x + matrix->m[5] * offset.y + matrix->m[9] * offset.z;
    result->m[14] = matrix->m[14] + matrix->m[2] * offset.x + matrix->m[6] * offset.y + matrix->m[10] * offset.z;
    result->m[15] = matrix->m[15] + matrix->m[3] * offset.x + matrix->m[7] * offset.y + matrix->m[11] * offset.z;
}

// Returns true if the given matrix is affine.
inline static bool XrMatrix4x4f_IsAffine(const XrMatrix4x4f* matrix, const float epsilon) {
    return fabsf(matrix->m[3]) <= epsilon && fabsf(matrix->m[7]) <= epsilon && fabsf(matrix->m[11]) <= epsilon &&
           fabsf(matrix->m[15] - 1.0f) <= epsilon;
}

// Returns true if the given matrix is orthogonal.
inline static bool XrMatrix4x4f_IsOrthogonal(const XrMatrix4x4f* matrix, const float epsilon) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i != j) {
                if (fabsf(matrix->m[4 * i + 0] * matrix->m[4 * j + 0] + matrix->m[4 * i + 1] * matrix->m[4 * j + 1] +
                          matrix->m[4 * i + 2] * matrix->m[4 * j + 2]) > epsilon) {
                    return false;
                }
                if (fabsf(matrix->m[4 * 0 + i] * matrix->m[4 * 0 + j] + matrix->m[4 * 1 + i] * matrix->m[4 * 1 + j] +
                          matrix->m[4 * 2 + i] * matrix->m[4 * 2 + j]) > epsilon) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Returns true if the given matrix is orthonormal.
inline static bool XrMatrix4x4f_IsOrthonormal(const XrMatrix4x4f* matrix, const float epsilon) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const float kd = (i == j) ? 1.0f : 0.0f;  // Kronecker delta
            if (fabsf(kd - (matrix->m[4 * i + 0] * matrix->m[4 * j + 0] + matrix->m[4 * i + 1] * matrix->m[4 * j + 1] +
                            matrix->m[4 * i + 2] * matrix->m[4 * j + 2])) > epsilon) {
                return false;
            }
            if (fabsf(kd - (matrix->m[4 * 0 + i] * matrix->m[4 * 0 + j] + matrix->m[4 * 1 + i] * matrix->m[4 * 1 + j] +
                            matrix->m[4 * 2 + i] * matrix->m[4 * 2 + j])) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

// Returns true if the given matrix is a rigid body transform.
inline static bool XrMatrix4x4f_IsRigidBody(const XrMatrix4x4f* matrix, const float epsilon) {
    return XrMatrix4x4f_IsAffine(matrix, epsilon) && XrMatrix4x4f_IsOrthonormal(matrix, epsilon);
}

// Get the translation from a combined translation(rotation(scale(object))) matrix.
inline static void XrMatrix4x4f_GetTranslation(XrVector3f* result, const XrMatrix4x4f* src) {
    assert(XrMatrix4x4f_IsAffine(src, 1e-4f));
    assert(XrMatrix4x4f_IsOrthogonal(src, 1e-4f));

    result->x = src->m[12];
    result->y = src->m[13];
    result->z = src->m[14];
}

// Get the rotation from a combined translation(rotation(scale(object))) matrix.
inline static void XrMatrix4x4f_GetRotation(XrQuaternionf* result, const XrMatrix4x4f* src) {
    assert(XrMatrix4x4f_IsAffine(src, 1e-4f));
    assert(XrMatrix4x4f_IsOrthogonal(src, 1e-4f));

    const float rcpScaleX = XrRcpSqrt(src->m[0] * src->m[0] + src->m[1] * src->m[1] + src->m[2] * src->m[2]);
    const float rcpScaleY = XrRcpSqrt(src->m[4] * src->m[4] + src->m[5] * src->m[5] + src->m[6] * src->m[6]);
    const float rcpScaleZ = XrRcpSqrt(src->m[8] * src->m[8] + src->m[9] * src->m[9] + src->m[10] * src->m[10]);
    const float m[9] = {src->m[0] * rcpScaleX, src->m[1] * rcpScaleX, src->m[2] * rcpScaleX,
                        src->m[4] * rcpScaleY, src->m[5] * rcpScaleY, src->m[6] * rcpScaleY,
                        src->m[8] * rcpScaleZ, src->m[9] * rcpScaleZ, src->m[10] * rcpScaleZ};
    if (m[0 * 3 + 0] + m[1 * 3 + 1] + m[2 * 3 + 2] > 0.0f) {
        float t = +m[0 * 3 + 0] + m[1 * 3 + 1] + m[2 * 3 + 2] + 1.0f;
        float s = XrRcpSqrt(t) * 0.5f;
        result->w = s * t;
        result->z = (m[0 * 3 + 1] - m[1 * 3 + 0]) * s;
        result->y = (m[2 * 3 + 0] - m[0 * 3 + 2]) * s;
        result->x = (m[1 * 3 + 2] - m[2 * 3 + 1]) * s;
    } else if (m[0 * 3 + 0] > m[1 * 3 + 1] && m[0 * 3 + 0] > m[2 * 3 + 2]) {
        float t = +m[0 * 3 + 0] - m[1 * 3 + 1] - m[2 * 3 + 2] + 1.0f;
        float s = XrRcpSqrt(t) * 0.5f;
        result->x = s * t;
        result->y = (m[0 * 3 + 1] + m[1 * 3 + 0]) * s;
        result->z = (m[2 * 3 + 0] + m[0 * 3 + 2]) * s;
        result->w = (m[1 * 3 + 2] - m[2 * 3 + 1]) * s;
    } else if (m[1 * 3 + 1] > m[2 * 3 + 2]) {
        float t = -m[0 * 3 + 0] + m[1 * 3 + 1] - m[2 * 3 + 2] + 1.0f;
        float s = XrRcpSqrt(t) * 0.5f;
        result->y = s * t;
        result->x = (m[0 * 3 + 1] + m[1 * 3 + 0]) * s;
        result->w = (m[2 * 3 + 0] - m[0 * 3 + 2]) * s;
        result->z = (m[1 * 3 + 2] + m[2 * 3 + 1]) * s;
    } else {
        float t = -m[0 * 3 + 0] - m[1 * 3 + 1] + m[2 * 3 + 2] + 1.0f;
        float s = XrRcpSqrt(t) * 0.5f;
        result->z = s * t;
        result->w = (m[0 * 3 + 1] - m[1 * 3 + 0]) * s;
        result->x = (m[2 * 3 + 0] + m[0 * 3 + 2]) * s;
        result->y = (m[1 * 3 + 2] + m[2 * 3 + 1]) * s;
    }
}

// Get the scale from a combined translation(rotation(scale(object))) matrix.
inline static void XrMatrix4x4f_GetScale(XrVector3f* result, const XrMatrix4x4f* src) {
    assert(XrMatrix4x4f_IsAffine(src, 1e-4f));
    assert(XrMatrix4x4f_IsOrthogonal(src, 1e-4f));

    result->x = sqrtf(src->m[0] * src->m[0] + src->m[1] * src->m[1] + src->m[2] * src->m[2]);
    result->y = sqrtf(src->m[4] * src->m[4] + src->m[5] * src->m[5] + src->m[6] * src->m[6]);
    result->z = sqrtf(src->m[8] * src->m[8] + src->m[9] * src->m[9] + src->m[10] * src->m[10]);
}

// Transforms a 3D vector.
inline static void XrMatrix4x4f_TransformVector3f(XrVector3f* result, const XrMatrix4x4f* m, const XrVector3f* v) {
    const float w = m->m[3] * v->x + m->m[7] * v->y + m->m[11] * v->z + m->m[15];
    const float rcpW = 1.0f / w;
    result->x = (m->m[0] * v->x + m->m[4] * v->y + m->m[8] * v->z + m->m[12]) * rcpW;
    result->y = (m->m[1] * v->x + m->m[5] * v->y + m->m[9] * v->z + m->m[13]) * rcpW;
    result->z = (m->m[2] * v->x + m->m[6] * v->y + m->m[10] * v->z + m->m[14]) * rcpW;
}

// Transforms a 4D vector.
inline static void XrMatrix4x4f_TransformVector4f(XrVector4f* result, const XrMatrix4x4f* m, const XrVector4f* v) {
    result->x = m->m[0] * v->x + m->m[4] * v->y + m->m[8] * v->z + m->m[12] * v->w;
    result->y = m->m[1] * v->x + m->m[5] * v->y + m->m[9] * v->z + m->m[13] * v->w;
    result->z = m->m[2] * v->x + m->m[6] * v->y + m->m[10] * v->z + m->m[14] * v->w;
    result->w = m->m[3] * v->x + m->m[7] * v->y + m->m[11] * v->z + m->m[15] * v->w;
}

// Transforms the 'mins' and 'maxs' bounds with the given 'matrix'.
inline static void XrMatrix4x4f_TransformBounds(XrVector3f* resultMins, XrVector3f* resultMaxs, const XrMatrix4x4f* matrix,
                                                const XrVector3f* mins, const XrVector3f* maxs) {
    assert(XrMatrix4x4f_IsAffine(matrix, 1e-4f));

    const XrVector3f center = {(mins->x + maxs->x) * 0.5f, (mins->y + maxs->y) * 0.5f, (mins->z + maxs->z) * 0.5f};
    const XrVector3f extents = {maxs->x - center.x, maxs->y - center.y, maxs->z - center.z};
    const XrVector3f newCenter = {matrix->m[0] * center.x + matrix->m[4] * center.y + matrix->m[8] * center.z + matrix->m[12],
                                  matrix->m[1] * center.x + matrix->m[5] * center.y + matrix->m[9] * center.z + matrix->m[13],
                                  matrix->m[2] * center.x + matrix->m[6] * center.y + matrix->m[10] * center.z + matrix->m[14]};
    const XrVector3f newExtents = {
        fabsf(extents.x * matrix->m[0]) + fabsf(extents.y * matrix->m[4]) + fabsf(extents.z * matrix->m[8]),
        fabsf(extents.x * matrix->m[1]) + fabsf(extents.y * matrix->m[5]) + fabsf(extents.z * matrix->m[9]),
        fabsf(extents.x * matrix->m[2]) + fabsf(extents.y * matrix->m[6]) + fabsf(extents.z * matrix->m[10])};
    XrVector3f_Sub(resultMins, &newCenter, &newExtents);
    XrVector3f_Add(resultMaxs, &newCenter, &newExtents);
}

// Returns true if the 'mins' and 'maxs' bounds is completely off to one side of the projection matrix.
inline static bool XrMatrix4x4f_CullBounds(const XrMatrix4x4f* mvp, const XrVector3f* mins, const XrVector3f* maxs) {
    if (maxs->x <= mins->x && maxs->y <= mins->y && maxs->z <= mins->z) {
        return false;
    }

    XrVector4f c[8];
    for (int i = 0; i < 8; i++) {
        const XrVector4f corner = {(i & 1) != 0 ? maxs->x : mins->x, (i & 2) != 0 ? maxs->y : mins->y,
                                   (i & 4) != 0 ? maxs->z : mins->z, 1.0f};
        XrMatrix4x4f_TransformVector4f(&c[i], mvp, &corner);
    }

    int i;
    for (i = 0; i < 8; i++) {
        if (c[i].x > -c[i].w) {
            break;
        }
    }
    if (i == 8) {
        return true;
    }
    for (i = 0; i < 8; i++) {
        if (c[i].x < c[i].w) {
            break;
        }
    }
    if (i == 8) {
        return true;
    }

    for (i = 0; i < 8; i++) {
        if (c[i].y > -c[i].w) {
            break;
        }
    }
    if (i == 8) {
        return true;
    }
    for (i = 0; i < 8; i++) {
        if (c[i].y < c[i].w) {
            break;
        }
    }
    if (i == 8) {
        return true;
    }
    for (i = 0; i < 8; i++) {
        if (c[i].z > -c[i].w) {
            break;
        }
    }
    if (i == 8) {
        return true;
    }
    for (i = 0; i < 8; i++) {
        if (c[i].z < c[i].w) {
            break;
        }
    }
    return i == 8;
}

#endif  // XR_LINEAR_H_
