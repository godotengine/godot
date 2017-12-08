// This code is in the public domain -- castano@gmail.com

#pragma once
#ifndef NV_MATH_QUATERNION_H
#define NV_MATH_QUATERNION_H

#include "nvmath/nvmath.h"
#include "nvmath/Vector.inl" // @@ Do not include inl files from header files.
#include "nvmath/Matrix.h"

namespace nv
{

    class NVMATH_CLASS Quaternion
    {
    public:
        typedef Quaternion const & Arg;

        Quaternion();
        explicit Quaternion(float f);
        Quaternion(float x, float y, float z, float w);
        Quaternion(Vector4::Arg v);

        const Quaternion & operator=(Quaternion::Arg v);

        Vector4 asVector() const;

        union {
            struct {
                float x, y, z, w;
            };
            float component[4];
        };
    };

    inline Quaternion::Quaternion() {}
    inline Quaternion::Quaternion(float f) : x(f), y(f), z(f), w(f) {}
    inline Quaternion::Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    inline Quaternion::Quaternion(Vector4::Arg v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    // @@ Move all these to Quaternion.inl!

    inline const Quaternion & Quaternion::operator=(Quaternion::Arg v) { 
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }

    inline Vector4 Quaternion::asVector() const { return Vector4(x, y, z, w); }

    inline Quaternion mul(Quaternion::Arg a, Quaternion::Arg b)
    {
        return Quaternion(
            + a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x,
            - a.x*b.z + a.y*b.w + a.z*b.x + a.w*b.y,
            + a.x*b.y - a.y*b.x + a.z*b.w + a.w*b.z,
            - a.x*b.x - a.y*b.y - a.z*b.z + a.w*b.w);
    }

    inline Quaternion mul(Quaternion::Arg a, Vector3::Arg b)
    {
        return Quaternion(
            + a.y*b.z - a.z*b.y + a.w*b.x,
            - a.x*b.z           + a.z*b.x + a.w*b.y,
            + a.x*b.y - a.y*b.x           + a.w*b.z,
            - a.x*b.x - a.y*b.y - a.z*b.z );
    }

    inline Quaternion mul(Vector3::Arg a, Quaternion::Arg b)
    {
        return Quaternion(
            + a.x*b.w + a.y*b.z - a.z*b.y,
            - a.x*b.z + a.y*b.w + a.z*b.x,
            + a.x*b.y - a.y*b.x + a.z*b.w,
            - a.x*b.x - a.y*b.y - a.z*b.z);
    }

    inline Quaternion operator *(Quaternion::Arg a, Quaternion::Arg b)
    {
        return mul(a, b);
    }

    inline Quaternion operator *(Quaternion::Arg a, Vector3::Arg b)
    {
        return mul(a, b);
    }

    inline Quaternion operator *(Vector3::Arg a, Quaternion::Arg b)
    {
        return mul(a, b);
    }


    inline Quaternion scale(Quaternion::Arg q, float s)
    {
        return scale(q.asVector(), s);
    }
    inline Quaternion operator *(Quaternion::Arg q, float s)
    {
        return scale(q, s);
    }
    inline Quaternion operator *(float s, Quaternion::Arg q)
    {
        return scale(q, s);
    }

    inline Quaternion scale(Quaternion::Arg q, Vector4::Arg s)
    {
        return scale(q.asVector(), s);
    }
    /*inline Quaternion operator *(Quaternion::Arg q, Vector4::Arg s)
    {
    return scale(q, s);
    }
    inline Quaternion operator *(Vector4::Arg s, Quaternion::Arg q)
    {
    return scale(q, s);
    }*/

    inline Quaternion conjugate(Quaternion::Arg q)
    {
        return scale(q, Vector4(-1, -1, -1, 1));
    }

    inline float length(Quaternion::Arg q)
    {
        return length(q.asVector());
    }

    inline bool isNormalized(Quaternion::Arg q, float epsilon = NV_NORMAL_EPSILON)
    {
        return equal(length(q), 1, epsilon);
    }

    inline Quaternion normalize(Quaternion::Arg q, float epsilon = NV_EPSILON)
    {
        float l = length(q);
        nvDebugCheck(!isZero(l, epsilon));
        Quaternion n = scale(q, 1.0f / l);
        nvDebugCheck(isNormalized(n));
        return n;
    }

    inline Quaternion inverse(Quaternion::Arg q)
    {
        return conjugate(normalize(q));
    }

    /// Create a rotation quaternion for @a angle alpha around normal vector @a v.
    inline Quaternion axisAngle(Vector3::Arg v, float alpha)
    {
        float s = sinf(alpha * 0.5f);
        float c = cosf(alpha * 0.5f);
        return Quaternion(Vector4(v * s, c));
    }

    inline Vector3 imag(Quaternion::Arg q)
    {
        return q.asVector().xyz();
    }

    inline float real(Quaternion::Arg q)
    {
        return q.w;
    }


    /// Transform vector.
    inline Vector3 transform(Quaternion::Arg q, Vector3::Arg v)
    {
        //Quaternion t = q * v * conjugate(q);
        //return imag(t);

        // Faster method by Fabian Giesen and others:
        // http://molecularmusings.wordpress.com/2013/05/24/a-faster-quaternion-vector-multiplication/
        // http://mollyrocket.com/forums/viewtopic.php?t=833&sid=3a84e00a70ccb046cfc87ac39881a3d0
        
        Vector3 t = 2 * cross(imag(q), v);
        return v + q.w * t + cross(imag(q), t);
    }

    // @@ Not tested.
    // From Insomniac's Mike Day:
    // http://www.insomniacgames.com/converting-a-rotation-matrix-to-a-quaternion/
    inline Quaternion fromMatrix(const Matrix & m) {
        if (m(2, 2) < 0) {
            if (m(0, 0) < m(1,1)) {
                float t = 1 - m(0, 0) - m(1, 1) - m(2, 2);
                return Quaternion(t, m(0,1)+m(1,0), m(2,0)+m(0,2), m(1,2)-m(2,1));
            }
            else {
                float t = 1 - m(0, 0) + m(1, 1) - m(2, 2);
                return Quaternion(t, m(0,1) + m(1,0), m(1,2) + m(2,1), m(2,0) - m(0,2));
            }
        }
        else {
            if (m(0, 0) < -m(1, 1)) {
                float t = 1 - m(0, 0) - m(1, 1) + m(2, 2);
                return Quaternion(t, m(2,0) + m(0,2), m(1,2) + m(2,1), m(0,1) - m(1,0));
            }
            else {
                float t = 1 + m(0, 0) + m(1, 1) + m(2, 2);
                return Quaternion(t, m(1,2) - m(2,1), m(2,0) - m(0,2), m(0,1) - m(1,0));
            }
        }
    }


} // nv namespace

#endif // NV_MATH_QUATERNION_H
