// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MATH_VECTOR_H
#define NV_MATH_VECTOR_H

#include "nvmath.h"

namespace nv
{
    class NVMATH_CLASS Vector2
    {
    public:
        typedef Vector2 const & Arg;

        Vector2();
        explicit Vector2(float f);
        Vector2(float x, float y);
        Vector2(Vector2::Arg v);

        //template <typename T> explicit Vector2(const T & v) : x(v.x), y(v.y) {}
        //template <typename T> operator T() const { return T(x, y); }

        const Vector2 & operator=(Vector2::Arg v);

        const float * ptr() const;

        void set(float x, float y);

        Vector2 operator-() const;
        void operator+=(Vector2::Arg v);
        void operator-=(Vector2::Arg v);
        void operator*=(float s);
        void operator*=(Vector2::Arg v);

        friend bool operator==(Vector2::Arg a, Vector2::Arg b);
        friend bool operator!=(Vector2::Arg a, Vector2::Arg b);

        union {
            struct {
                float x, y;
            };
            float component[2];
        };
    };

    class NVMATH_CLASS Vector3
    {
    public:
        typedef Vector3 const & Arg;

        Vector3();
        explicit Vector3(float x);
        //explicit Vector3(int x) : x(float(x)), y(float(x)), z(float(x)) {}
        Vector3(float x, float y, float z);
        Vector3(Vector2::Arg v, float z);
        Vector3(Vector3::Arg v);

        //template <typename T> explicit Vector3(const T & v) : x(v.x), y(v.y), z(v.z) {}
        //template <typename T> operator T() const { return T(x, y, z); }

        const Vector3 & operator=(Vector3::Arg v);

        Vector2 xy() const;

        const float * ptr() const;

        void set(float x, float y, float z);

        Vector3 operator-() const;
        void operator+=(Vector3::Arg v);
        void operator-=(Vector3::Arg v);
        void operator*=(float s);
        void operator/=(float s);
        void operator*=(Vector3::Arg v);
        void operator/=(Vector3::Arg v);

        friend bool operator==(Vector3::Arg a, Vector3::Arg b);
        friend bool operator!=(Vector3::Arg a, Vector3::Arg b);

        union {
            struct {
                float x, y, z;
            };
            float component[3];
        };
    };

    class NVMATH_CLASS Vector4
    {
    public:
        typedef Vector4 const & Arg;

        Vector4();
        explicit Vector4(float x);
        Vector4(float x, float y, float z, float w);
        Vector4(Vector2::Arg v, float z, float w);
        Vector4(Vector2::Arg v, Vector2::Arg u);
        Vector4(Vector3::Arg v, float w);
        Vector4(Vector4::Arg v);
        //	Vector4(const Quaternion & v);

        //template <typename T> explicit Vector4(const T & v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
        //template <typename T> operator T() const { return T(x, y, z, w); }

        const Vector4 & operator=(Vector4::Arg v);

        Vector2 xy() const;
        Vector2 zw() const;
        Vector3 xyz() const;

        const float * ptr() const;

        void set(float x, float y, float z, float w);

        Vector4 operator-() const;
        void operator+=(Vector4::Arg v);
        void operator-=(Vector4::Arg v);
        void operator*=(float s);
        void operator/=(float s);
        void operator*=(Vector4::Arg v);
        void operator/=(Vector4::Arg v);

        friend bool operator==(Vector4::Arg a, Vector4::Arg b);
        friend bool operator!=(Vector4::Arg a, Vector4::Arg b);

        union {
            struct {
                float x, y, z, w;
            };
            float component[4];
        };
    };

} // nv namespace

// If we had these functions, they would be ambiguous, the compiler would not know which one to pick:
//template <typename T> Vector2 to(const T & v) { return Vector2(v.x, v.y); }
//template <typename T> Vector3 to(const T & v) { return Vector3(v.x, v.y, v.z); }
//template <typename T> Vector4 to(const T & v) { return Vector4(v.x, v.y, v.z, v.z); }

// We could use a cast operator so that we could infer the expected type, but that doesn't work the same way in all compilers and produces horrible error messages.

// Instead we simply have explicit casts:
template <typename T> T to(const nv::Vector2 & v) { NV_COMPILER_CHECK(sizeof(T) == sizeof(nv::Vector2)); return T(v.x, v.y); }
template <typename T> T to(const nv::Vector3 & v) { NV_COMPILER_CHECK(sizeof(T) == sizeof(nv::Vector3)); return T(v.x, v.y, v.z); }
template <typename T> T to(const nv::Vector4 & v) { NV_COMPILER_CHECK(sizeof(T) == sizeof(nv::Vector4)); return T(v.x, v.y, v.z, v.w); }

#endif // NV_MATH_VECTOR_H
