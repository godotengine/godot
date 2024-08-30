/*
 * Copyright (c) 2021 - 2024 the ThorVG project. All rights reserved.

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
#define FLOAT_EPSILON 1.0e-06f  //1.192092896e-07f
#define PATH_KAPPA 0.552284f

#define mathMin(x, y) (((x) < (y)) ? (x) : (y))
#define mathMax(x, y) (((x) > (y)) ? (x) : (y))


/************************************************************************/
/* General functions                                                    */
/************************************************************************/

float mathAtan2(float y, float x);

static inline float mathDeg2Rad(float degree)
{
     return degree * (MATH_PI / 180.0f);
}


static inline float mathRad2Deg(float radian)
{
    return radian * (180.0f / MATH_PI);
}


static inline bool mathZero(float a)
{
    return (fabsf(a) <= FLOAT_EPSILON) ? true : false;
}


static inline bool mathEqual(float a, float b)
{
    return mathZero(a - b);
}


/************************************************************************/
/* Matrix functions                                                     */
/************************************************************************/

void mathRotate(Matrix* m, float degree);
bool mathInverse(const Matrix* m, Matrix* out);
bool mathIdentity(const Matrix* m);
Matrix operator*(const Matrix& lhs, const Matrix& rhs);
bool operator==(const Matrix& lhs, const Matrix& rhs);

static inline bool mathRightAngle(const Matrix& m)
{
   auto radian = fabsf(mathAtan2(m.e21, m.e11));
   if (radian < FLOAT_EPSILON || mathEqual(radian, MATH_PI2) || mathEqual(radian, MATH_PI)) return true;
   return false;
}


static inline bool mathSkewed(const Matrix& m)
{
    return !mathZero(m.e21 + m.e12);
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


static inline void mathTranslateR(Matrix* m, float x, float y)
{
    if (x == 0.0f && y == 0.0f) return;
    m->e13 += (x * m->e11 + y * m->e12);
    m->e23 += (x * m->e21 + y * m->e22);
}


static inline bool operator!=(const Matrix& lhs, const Matrix& rhs)
{
    return !(lhs == rhs);
}


static inline void operator*=(Matrix& lhs, const Matrix& rhs)
{
    lhs = lhs * rhs;
}


static inline void mathLog(const Matrix& m)
{
    TVGLOG("COMMON", "Matrix: [%f %f %f] [%f %f %f] [%f %f %f]", m.e11, m.e12, m.e13, m.e21, m.e22, m.e23, m.e31, m.e32, m.e33);
}


/************************************************************************/
/* Point functions                                                      */
/************************************************************************/

void operator*=(Point& pt, const Matrix& m);
Point operator*(const Point& pt, const Matrix& m);


static inline bool mathZero(const Point& p)
{
    return mathZero(p.x) && mathZero(p.y);
}


static inline float mathLength(const Point* a, const Point* b)
{
    auto x = b->x - a->x;
    auto y = b->y - a->y;

    if (x < 0) x = -x;
    if (y < 0) y = -y;

    return (x > y) ? (x + 0.375f * y) : (y + 0.375f * x);
}


static inline float mathLength(const Point& a)
{
    return sqrtf(a.x * a.x + a.y * a.y);
}


static inline bool operator==(const Point& lhs, const Point& rhs)
{
    return mathEqual(lhs.x, rhs.x) && mathEqual(lhs.y, rhs.y);
}


static inline bool operator!=(const Point& lhs, const Point& rhs)
{
    return !(lhs == rhs);
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


static inline Point operator*(const float& lhs, const Point& rhs)
{
    return {lhs * rhs.x, lhs * rhs.y};
}


static inline Point operator/(const Point& lhs, const float rhs)
{
    return {lhs.x / rhs, lhs.y / rhs};
}


static inline Point mathNormal(const Point& p1, const Point& p2)
{
    auto dir = p2 - p1;
    auto len = mathLength(dir);
    if (mathZero(len)) return {};

    auto unitDir = dir / len;
    return {-unitDir.y, unitDir.x};
}


static inline void mathLog(const Point& pt)
{
    TVGLOG("COMMON", "Point: [%f %f]", pt.x, pt.y);
}

/************************************************************************/
/* Interpolation functions                                              */
/************************************************************************/

template <typename T>
static inline T mathLerp(const T &start, const T &end, float t)
{
    return static_cast<T>(start + (end - start) * t);
}

uint8_t mathLerp(const uint8_t &start, const uint8_t &end, float t);

#endif //_TVG_MATH_H_
