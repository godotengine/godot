// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MATH_PLANE_INL
#define NV_MATH_PLANE_INL

#include "Plane.h"
#include "Vector.inl"

namespace nv
{
    inline Plane::Plane() {}
    inline Plane::Plane(float x, float y, float z, float w) : v(x, y, z, w) {}
    inline Plane::Plane(const Vector4 & v) : v(v) {}
    inline Plane::Plane(const Vector3 & v, float d) : v(v, d) {}
    inline Plane::Plane(const Vector3 & normal, const Vector3 & point) : v(normal, -dot(normal, point)) {}
    inline Plane::Plane(const Vector3 & v0, const Vector3 & v1, const Vector3 & v2) {
        Vector3 n = cross(v1-v0, v2-v0);
        float d = -dot(n, v0);
        v = Vector4(n, d);
    }

    inline const Plane & Plane::operator=(const Plane & p) { v = p.v; return *this; }

    inline Vector3 Plane::vector() const { return v.xyz(); }
    inline float Plane::offset() const { return v.w; }
    inline Vector3 Plane::normal() const { return normalize(vector(), 0.0f); }

    // Normalize plane.
    inline Plane normalize(const Plane & plane, float epsilon = NV_EPSILON)
    {
        const float len = length(plane.vector());
        const float inv = isZero(len, epsilon) ? 0 : 1.0f / len;
        return Plane(plane.v * inv);
    }

    // Get the signed distance from the given point to this plane.
    inline float distance(const Plane & plane, const Vector3 & point)
    {
        return dot(plane.vector(), point) + plane.offset();
    }

    inline void Plane::operator*=(float s)
    {
        v *= s;
    }

} // nv namespace

#endif // NV_MATH_PLANE_H
