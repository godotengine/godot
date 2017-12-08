// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MATH_BASIS_H
#define NV_MATH_BASIS_H

#include "nvmath.h"
#include "Vector.inl"
#include "Matrix.h"

namespace nv
{

    /// Basis class to compute tangent space basis, ortogonalizations and to
    /// transform vectors from one space to another.
    class Basis
    {
    public:

        /// Create a null basis.
        Basis() : tangent(0, 0, 0), bitangent(0, 0, 0), normal(0, 0, 0) {}

        /// Create a basis given three vectors.
        Basis(Vector3::Arg n, Vector3::Arg t, Vector3::Arg b) : tangent(t), bitangent(b), normal(n) {}

        /// Create a basis with the given tangent vectors and the handness.
        Basis(Vector3::Arg n, Vector3::Arg t, float sign)
        {
            build(n, t, sign);
        }

        NVMATH_API void normalize(float epsilon = NV_EPSILON);
        NVMATH_API void orthonormalize(float epsilon = NV_EPSILON);
        NVMATH_API void robustOrthonormalize(float epsilon = NV_EPSILON);
        NVMATH_API void buildFrameForDirection(Vector3::Arg d, float angle = 0);

        /// Calculate the determinant [ F G N ] to obtain the handness of the basis. 
        float handness() const
        {
            return determinant() > 0.0f ? 1.0f : -1.0f;
        }

        /// Build a basis from 2 vectors and a handness flag.
        void build(Vector3::Arg n, Vector3::Arg t, float sign)
        {
            normal = n;
            tangent = t;
            bitangent = sign * cross(t, n);
        }

        /// Compute the determinant of this basis.
        float determinant() const
        {
            return 
                tangent.x * bitangent.y * normal.z - tangent.z * bitangent.y * normal.x +
                tangent.y * bitangent.z * normal.x - tangent.y * bitangent.x * normal.z + 
                tangent.z * bitangent.x * normal.y - tangent.x * bitangent.z * normal.y;
        }

        bool isValid() const;

        // Get transform matrix for this basis.
        NVMATH_API Matrix matrix() const;

        // Transform by this basis. (From this basis to object space).
        NVMATH_API Vector3 transform(Vector3::Arg v) const;

        // Transform by the transpose. (From object space to this basis).
        NVMATH_API Vector3 transformT(Vector3::Arg v);

        // Transform by the inverse. (From object space to this basis).
        NVMATH_API Vector3 transformI(Vector3::Arg v) const;


        Vector3 tangent;
        Vector3 bitangent;
        Vector3 normal;
    };

} // nv namespace

#endif // NV_MATH_BASIS_H
