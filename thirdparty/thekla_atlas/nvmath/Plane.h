// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MATH_PLANE_H
#define NV_MATH_PLANE_H

#include "nvmath.h"
#include "Vector.h"

namespace nv
{
    class Matrix;

    class NVMATH_CLASS Plane
    {
    public:
        Plane();
        Plane(float x, float y, float z, float w);
        Plane(const Vector4 & v);
        Plane(const Vector3 & v, float d);
        Plane(const Vector3 & normal, const Vector3 & point);
        Plane(const Vector3 & v0, const Vector3 & v1, const Vector3 & v2);

        const Plane & operator=(const Plane & v);

        Vector3 vector() const;
        float offset() const;
        Vector3 normal() const;

        void operator*=(float s);

        Vector4 v;
    };

    Plane transformPlane(const Matrix &, const Plane &);

    Vector3 planeIntersection(const Plane & a, const Plane & b, const Plane & c);


} // nv namespace

#endif // NV_MATH_PLANE_H
