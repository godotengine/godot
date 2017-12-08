// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MATH_TYPESERIALIZATION_H
#define NV_MATH_TYPESERIALIZATION_H

#include "nvmath.h"

namespace nv
{
    class Stream;

    class Vector2;
    class Vector3;
    class Vector4;

    class Matrix;
    class Quaternion;
    class Basis;
    class Box;
    class Plane;

    NVMATH_API Stream & operator<< (Stream & s, Vector2 & obj);
    NVMATH_API Stream & operator<< (Stream & s, Vector3 & obj);
    NVMATH_API Stream & operator<< (Stream & s, Vector4 & obj);

    NVMATH_API Stream & operator<< (Stream & s, Matrix & obj);
    NVMATH_API Stream & operator<< (Stream & s, Quaternion & obj);
    NVMATH_API Stream & operator<< (Stream & s, Basis & obj);
    NVMATH_API Stream & operator<< (Stream & s, Box & obj);
    NVMATH_API Stream & operator<< (Stream & s, Plane & obj);

} // nv namespace

#endif // NV_MATH_TYPESERIALIZATION_H
