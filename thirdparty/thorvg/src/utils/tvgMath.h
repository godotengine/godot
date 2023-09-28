/*
 * Copyright (c) 2021 - 2023 the ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _TVG_MATH_H_
#define _TVG_MATH_H_

 #define _USE_MATH_DEFINES

#include <float.h>
#include <math.h>
#include "tvgCommon.h"

#define MATH_PI  3.14159265358979323846f
#define MATH_PI2 1.57079632679489661923f

#define mathMin(x, y) (((x) < (y)) ? (x) : (y))
#define mathMax(x, y) (((x) > (y)) ? (x) : (y))


static inline bool mathZero(float a)
{
    return (fabsf(a) < FLT_EPSILON) ? true : false;
}


static inline bool mathEqual(float a, float b)
{
    return (fabsf(a - b) < FLT_EPSILON);
}


static inline bool mathEqual(const Matrix& a, const Matrix& b)
{
    if (!mathEqual(a.e11, b.e11) || !mathEqual(a.e12, b.e12) || !mathEqual(a.e13, b.e13) ||
        !mathEqual(a.e21, b.e21) || !mathEqual(a.e22, b.e22) || !mathEqual(a.e23, b.e23) ||
        !mathEqual(a.e31, b.e31) || !mathEqual(a.e32, b.e32) || !mathEqual(a.e33, b.e33)) {
       return false;
    }
    return true;
}


static inline bool mathRightAngle(const Matrix* m)
{
   auto radian = fabsf(atan2f(m->e21, m->e11));
   if (radian < FLT_EPSILON || mathEqual(radian, float(M_PI_2)) || mathEqual(radian, float(M_PI))) return true;
   return false;
}


static inline bool mathSkewed(const Matrix* m)
{
    return (fabsf(m->e21 + m->e12) > FLT_EPSILON);
}


static inline bool mathIdentity(const Matrix* m)
{
    if (!mathEqual(m->e11, 1.0f) || !mathZero(m->e12) || !mathZero(m->e13) ||
        !mathZero(m->e21) || !mathEqual(m->e22, 1.0f) || !mathZero(m->e23) ||
        !mathZero(m->e31) || !mathZero(m->e32) || !mathEqual(m->e33, 1.0f)) {
        return false;
    }
    return true;
}


static inline bool mathInverse(const Matrix* m, Matrix* out)
{
    auto det = m->e11 * (m->e22 * m->e33 - m->e32 * m->e23) -
               m->e12 * (m->e21 * m->e33 - m->e23 * m->e31) +
               m->e13 * (m->e21 * m->e32 - m->e22 * m->e31);

    if (mathZero(det)) return false;

    auto invDet = 1 / det;

    out->e11 = (m->e22 * m->e33 - m->e32 * m->e23) * invDet;
    out->e12 = (m->e13 * m->e32 - m->e12 * m->e33) * invDet;
    out->e13 = (m->e12 * m->e23 - m->e13 * m->e22) * invDet;
    out->e21 = (m->e23 * m->e31 - m->e21 * m->e33) * invDet;
    out->e22 = (m->e11 * m->e33 - m->e13 * m->e31) * invDet;
    out->e23 = (m->e21 * m->e13 - m->e11 * m->e23) * invDet;
    out->e31 = (m->e21 * m->e32 - m->e31 * m->e22) * invDet;
    out->e32 = (m->e31 * m->e12 - m->e11 * m->e32) * invDet;
    out->e33 = (m->e11 * m->e22 - m->e21 * m->e12) * invDet;

    return true;
}


static inline void mathIdentity(Matrix* m)
{
    m->e11 = 1.0f;
    m->e12 = 0.0f;
    m->e13 = 0.0f;
    m->e21 = 0.0f;
    m->e22 = 1.0f;
    m->e23 = 0.0f;
    m->e31 = 0.0f;
    m->e32 = 0.0f;
    m->e33 = 1.0f;
}


static inline void mathTransform(Matrix* transform, Point* coord)
{
    auto x = coord->x;
    auto y = coord->y;
    coord->x = x * transform->e11 + y * transform->e12 + transform->e13;
    coord->y = x * transform->e21 + y * transform->e22 + transform->e23;
}


static inline void mathScale(Matrix* m, float sx, float sy)
{
    m->e11 *= sx;
    m->e22 *= sy;
}


static inline void mathScaleR(Matrix* m, float x, float y)
{
    if (x != 1.0f) {
        m->e11 *= x;
        m->e21 *= x;
    }
    if (y != 1.0f) {
        m->e22 *= y;
        m->e12 *= y;
    }
}


static inline void mathTranslate(Matrix* m, float x, float y)
{
    m->e13 += x;
    m->e23 += y;
}


static inline void mathRotate(Matrix* m, float degree)
{
    auto radian = degree / 180.0f * M_PI;
    auto cosVal = cosf(radian);
    auto sinVal = sinf(radian);

    m->e12 = m->e11 * -sinVal;
    m->e11 *= cosVal;
    m->e21 = m->e22 * sinVal;
    m->e22 *= cosVal;
}


static inline void mathMultiply(Point* pt, const Matrix* transform)
{
    auto tx = pt->x * transform->e11 + pt->y * transform->e12 + transform->e13;
    auto ty = pt->x * transform->e21 + pt->y * transform->e22 + transform->e23;
    pt->x = tx;
    pt->y = ty;
}


static inline Matrix mathMultiply(const Matrix* lhs, const Matrix* rhs)
{
    Matrix m;

    m.e11 = lhs->e11 * rhs->e11 + lhs->e12 * rhs->e21 + lhs->e13 * rhs->e31;
    m.e12 = lhs->e11 * rhs->e12 + lhs->e12 * rhs->e22 + lhs->e13 * rhs->e32;
    m.e13 = lhs->e11 * rhs->e13 + lhs->e12 * rhs->e23 + lhs->e13 * rhs->e33;

    m.e21 = lhs->e21 * rhs->e11 + lhs->e22 * rhs->e21 + lhs->e23 * rhs->e31;
    m.e22 = lhs->e21 * rhs->e12 + lhs->e22 * rhs->e22 + lhs->e23 * rhs->e32;
    m.e23 = lhs->e21 * rhs->e13 + lhs->e22 * rhs->e23 + lhs->e23 * rhs->e33;

    m.e31 = lhs->e31 * rhs->e11 + lhs->e32 * rhs->e21 + lhs->e33 * rhs->e31;
    m.e32 = lhs->e31 * rhs->e12 + lhs->e32 * rhs->e22 + lhs->e33 * rhs->e32;
    m.e33 = lhs->e31 * rhs->e13 + lhs->e32 * rhs->e23 + lhs->e33 * rhs->e33;

    return m;
}


static inline void mathTranslateR(Matrix* m, float x, float y)
{
    if (x == 0.0f && y == 0.0f) return;
    m->e13 += (x * m->e11 + y * m->e12);
    m->e23 += (x * m->e21 + y * m->e22);
}


static inline void mathLog(Matrix* m)
{
    TVGLOG("MATH", "Matrix: [%f %f %f] [%f %f %f] [%f %f %f]", m->e11, m->e12, m->e13, m->e21, m->e22, m->e23, m->e31, m->e32, m->e33);
}


static inline float mathLength(const Point* a, const Point* b)
{
    auto x = b->x - a->x;
    auto y = b->y - a->y;

    if (x < 0) x = -x;
    if (y < 0) y = -y;

    return (x > y) ? (x + 0.375f * y) : (y + 0.375f * x);
}


static inline Point operator-(const Point& lhs, const Point& rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y};
}


static inline Point operator+(const Point& lhs, const Point& rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}


static inline Point operator*(const Point& lhs, float rhs)
{
    return {lhs.x * rhs, lhs.y * rhs};
}


template <typename T>
static inline T mathLerp(const T &start, const T &end, float t)
{
    return static_cast<T>(start + (end - start) * t);
}


#endif //_TVG_MATH_H_
