// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MATH_SPHERE_H
#define NV_MATH_SPHERE_H

#include "Vector.h"

namespace nv
{
    
    class Sphere
    {
    public:
        Sphere() {}
        Sphere(Vector3::Arg center, float radius) : center(center), radius(radius) {}

        Sphere(Vector3::Arg center) : center(center), radius(0.0f) {}
        Sphere(Vector3::Arg p0, Vector3::Arg p1);
        Sphere(Vector3::Arg p0, Vector3::Arg p1, Vector3::Arg p2);
        Sphere(Vector3::Arg p0, Vector3::Arg p1, Vector3::Arg p2, Vector3::Arg p3);

        Vector3 center;
        float radius;
    };

    // Returns negative values if point is inside.
    float distanceSquared(const Sphere & sphere, const Vector3 &point);


    // Welz's algorithm. Fairly slow, recursive implementation uses large stack.
    Sphere miniBall(const Vector3 * pointArray, uint pointCount);

    Sphere approximateSphere_Ritter(const Vector3 * pointArray, uint pointCount);
    Sphere approximateSphere_AABB(const Vector3 * pointArray, uint pointCount);
    Sphere approximateSphere_EPOS6(const Vector3 * pointArray, uint pointCount);
    Sphere approximateSphere_EPOS14(const Vector3 * pointArray, uint pointCount);


} // nv namespace


#endif // NV_MATH_SPHERE_H
