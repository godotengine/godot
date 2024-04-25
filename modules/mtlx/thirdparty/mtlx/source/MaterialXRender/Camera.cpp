//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/Camera.h>

MATERIALX_NAMESPACE_BEGIN

Matrix44 Camera::createViewMatrix(const Vector3& eye,
                                  const Vector3& target,
                                  const Vector3& up)
{
    Vector3 z = (target - eye).getNormalized();
    Vector3 x = z.cross(up).getNormalized();
    Vector3 y = x.cross(z);

    return Matrix44(
        x[0], y[0], -z[0], 0.0f,
        x[1], y[1], -z[1], 0.0f,
        x[2], y[2], -z[2], 0.0f,
        -x.dot(eye), -y.dot(eye), z.dot(eye), 1.0f);
}

Matrix44 Camera::createPerspectiveMatrixZP(float left, float right,
                                           float bottom, float top,
                                           float nearP, float farP)
{
    return Matrix44(
        (2.0f * nearP) / (right - left), 0.0f, (right + left) / (right - left), 0.0f,
        0.0f, (2.0f * nearP) / (top - bottom), (top + bottom) / (top - bottom), 0.0f,
        0.0f, 0.0f, -1 / (farP - nearP), -1.0f,
        0.0f, 0.0f, -nearP / (farP - nearP), 0.0f);
}

Matrix44 Camera::createOrthographicMatrixZP(float left, float right,
                                            float bottom, float top,
                                            float nearP, float farP)
{
    return Matrix44(
        2.0f / (right - left), 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f / (top - bottom), 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f / (farP - nearP), 0.0f,
        -(right + left) / (right - left), -(top + bottom) / (top - bottom), -nearP / (farP - nearP), 1.0f);
}

Matrix44 Camera::createPerspectiveMatrix(float left, float right,
                                         float bottom, float top,
                                         float nearP, float farP)
{
    return Matrix44(
        (2.0f * nearP) / (right - left), 0.0f, (right + left) / (right - left), 0.0f,
        0.0f, (2.0f * nearP) / (top - bottom), (top + bottom) / (top - bottom), 0.0f,
        0.0f, 0.0f, -(farP + nearP) / (farP - nearP), -1.0f,
        0.0f, 0.0f, -(2.0f * farP * nearP) / (farP - nearP), 0.0f);
}

Matrix44 Camera::createOrthographicMatrix(float left, float right,
                                          float bottom, float top,
                                          float nearP, float farP)
{
    return Matrix44(
        2.0f / (right - left), 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f / (top - bottom), 0.0f, 0.0f,
        0.0f, 0.0f, -2.0f / (farP - nearP), 0.0f,
        -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(farP + nearP) / (farP - nearP), 1.0f);
}

void Camera::arcballButtonEvent(const Vector2& pos, bool pressed)
{
    _arcballActive = pressed;
    _arcballLastPos = pos;
    if (!_arcballActive)
    {
        _arcballQuat = (_arcballDelta * _arcballQuat).getNormalized();
    }
    _arcballDelta = Quaternion::IDENTITY;
}

bool Camera::applyArcballMotion(const Vector2& pos)
{
    if (!_arcballActive)
    {
        return false;
    }

    float w = _viewportSize[0];
    float h = _viewportSize[1];
    float invMinDim = 1.0f / std::min(w, h);

    float ox = (_arcballSpeed * (2.0f * _arcballLastPos[0] - w) + w) - w - 1.0f;
    float tx = (_arcballSpeed * (2.0f * pos[0] - w) + w) - w - 1.0f;
    float oy = (_arcballSpeed * (h - 2.0f * _arcballLastPos[1]) + h) - h - 1.0f;
    float ty = (_arcballSpeed * (h - 2.0f * pos[1]) + h) - h - 1.0f;

    ox *= invMinDim;
    oy *= invMinDim;
    tx *= invMinDim;
    ty *= invMinDim;

    Vector3 v0(ox, oy, 1.0f);
    Vector3 v1(tx, ty, 1.0f);
    if (v0.dot(v0) > 1e-4f && v1.dot(v1) > 1e-4f)
    {
        v0 = v0.getNormalized();
        v1 = v1.getNormalized();
        Vector3 axis = v0.cross(v1);
        float sa = std::sqrt(axis.dot(axis));
        float ca = v0.dot(v1);
        float angle = std::atan2(sa, ca);
        if (tx*tx + ty*ty > 1.0f)
        {
            angle *= 1.0f + 0.2f * (std::sqrt(tx*tx + ty*ty) - 1.0f);
        }
        axis = axis.getNormalized();
        _arcballDelta = Quaternion::createFromAxisAngle(axis, angle);
        if (!std::isfinite(_arcballDelta.getMagnitude()))
        {
            _arcballDelta = Quaternion::IDENTITY;
        }
    }
    return true;
}

MATERIALX_NAMESPACE_END
