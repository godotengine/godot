// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MATH_CONVEXHULL_H
#define NV_MATH_CONVEXHULL_H

#include "nvmath.h"
#include "nvcore/Array.h"

namespace nv {
    class Vector2;
 
    void convexHull(const Array<Vector2> & input, Array<Vector2> & output, float epsilon = 0);

} // namespace nv

#endif // NV_MATH_CONVEXHULL_H
