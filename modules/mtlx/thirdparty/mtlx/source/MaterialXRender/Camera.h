//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CAMERA_H
#define MATERIALX_CAMERA_H

#include <MaterialXRender/Types.h>

MATERIALX_NAMESPACE_BEGIN

/// Shared pointer to a Camera
using CameraPtr = std::shared_ptr<class Camera>;

/// @class Camera
/// A simple camera class, supporting transform matrices and arcball
/// functionality for object-viewing applications.
class MX_RENDER_API Camera
{
  public:
    Camera() :
        _worldMatrix(Matrix44::IDENTITY),
        _viewMatrix(Matrix44::IDENTITY),
        _projectionMatrix(Matrix44::IDENTITY),
        _arcballActive(false),
        _arcballQuat(Quaternion::IDENTITY),
        _arcballDelta(Quaternion::IDENTITY),
        _arcballSpeed(2.0f)
    {
    }
    ~Camera() { }

    /// Create a new camera.
    static CameraPtr create() { return std::make_shared<Camera>(); }

    /// @name Transform Matrices
    /// @{

    /// Set the world matrix.
    void setWorldMatrix(const Matrix44& mat)
    {
        _worldMatrix = mat;
    }

    /// Return the world matrix.
    const Matrix44& getWorldMatrix() const
    {
        return _worldMatrix;
    }

    /// Set the view matrix.
    void setViewMatrix(const Matrix44& mat)
    {
        _viewMatrix = mat;
    }

    /// Return the view matrix.
    const Matrix44& getViewMatrix() const
    {
        return _viewMatrix;
    }

    /// Set the projection matrix.
    void setProjectionMatrix(const Matrix44& mat)
    {
        _projectionMatrix = mat;
    }

    /// Return the projection matrix.
    const Matrix44& getProjectionMatrix() const
    {
        return _projectionMatrix;
    }

    /// Compute our full model-view-projection matrix.
    Matrix44 getWorldViewProjMatrix() const
    {
        return _worldMatrix * _viewMatrix * _projectionMatrix;
    }

    /// Derive viewer position from the view matrix.
    Vector3 getViewPosition() const
    {
        Matrix44 invView = _viewMatrix.getInverse();
        return Vector3(invView[3][0], invView[3][1], invView[3][2]);
    }

    /// Derive viewer direction from the view matrix.
    Vector3 getViewDirection() const
    {
        Matrix44 invView = _viewMatrix.getInverse();
        return Vector3(invView[2][0], invView[2][1], invView[2][2]);
    }

    /// @}
    /// @name Viewport
    /// @{

    /// Set the size of the viewport window.
    void setViewportSize(const Vector2& size)
    {
        _viewportSize = size;
    }

    /// Return the size of the viewport window.
    const Vector2& getViewportSize() const
    {
        return _viewportSize;
    }

    /// Project a position from object to viewport space.
    Vector3 projectToViewport(Vector3 v)
    {
        v = transformPointPerspective(getWorldViewProjMatrix(), v);
        v = v * 0.5f + Vector3(0.5f);
        v[0] *= _viewportSize[0];
        v[1] *= _viewportSize[1];
        return v;
    }

    /// Unproject a position from viewport to object space.
    Vector3 unprojectFromViewport(Vector3 v)
    {
        v[0] /= _viewportSize[0];
        v[1] /= _viewportSize[1];
        v = v * 2.0f - Vector3(1.0f);
        v = transformPointPerspective(getWorldViewProjMatrix().getInverse(), v);
        return v;
    }

    /// @}
    /// @name Arcball
    /// @{

    /// Indicates a button state change, with pos being the instantaneous location of the mouse.
    void arcballButtonEvent(const Vector2& pos, bool pressed);

    /// Apply mouse motion to the arcball state.
    bool applyArcballMotion(const Vector2& pos);

    /// Return the arcball matrix.
    Matrix44 arcballMatrix() const
    {
        return (_arcballDelta * _arcballQuat).toMatrix();
    }

    /// @}
    /// @name Utilities
    /// @{

    /// Create a view matrix given an eye position, a target position and an up vector.
    static Matrix44 createViewMatrix(const Vector3& eye,
                                     const Vector3& target,
                                     const Vector3& up);

    /// Create a perpective projection matrix given a set of clip planes with [-1,1] projected Z.
    static Matrix44 createPerspectiveMatrix(float left, float right,
                                            float bottom, float top,
                                            float nearP, float farP);

    /// Create an orthographic projection matrix given a set of clip planes with [-1,1] projected Z.
    static Matrix44 createOrthographicMatrix(float left, float right,
                                             float bottom, float top,
                                             float nearP, float farP);

    /// Create a perpective projection matrix given a set of clip planes with [0,1] projected Z.
    static Matrix44 createPerspectiveMatrixZP(float left, float right,
                                              float bottom, float top,
                                              float nearP, float farP);

    /// Create an orthographic projection matrix given a set of clip planes with [0,1] projected Z.
    static Matrix44 createOrthographicMatrixZP(float left, float right,
                                               float bottom, float top,
                                               float nearP, float farP);

    /// Apply a perspective transform to the given 3D point, performing a
    /// homogeneous divide on the transformed result.
    static Vector3 transformPointPerspective(const Matrix44& m, const Vector3& v)
    {
        Vector4 res = m.multiply(Vector4(v[0], v[1], v[2], 1.0f));
        return Vector3(res[0], res[1], res[2]) / res[3];
    }

    /// @}

  protected:
    // Transform matrices
    Matrix44 _worldMatrix;
    Matrix44 _viewMatrix;
    Matrix44 _projectionMatrix;

    // Viewport size
    Vector2 _viewportSize;

    // Arcball properties
    bool _arcballActive;
    Vector2 _arcballLastPos;
    Quaternion _arcballQuat;
    Quaternion _arcballDelta;
    float _arcballSpeed;
};

MATERIALX_NAMESPACE_END

#endif
