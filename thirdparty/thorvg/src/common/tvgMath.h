/*
 * Copyright (c) 2021 - 2026 ThorVG project. All rights reserved.

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
#include <cmath>
#include "tvgCommon.h"

namespace tvg
{

#define MATH_PI  3.14159265358979323846f
#define MATH_PI2 1.57079632679489661923f
#define FLOAT_EPSILON 1.0e-06f  //1.192092896e-07f
#define PATH_KAPPA 0.552284f

/************************************************************************/
/* General functions                                                    */
/************************************************************************/

float atan2(float y, float x);
float length(const PathCommand* cmds, uint32_t cmdsCnt, const Point* pts, uint32_t ptsCnt);


static inline float deg2rad(float degree)
{
     return degree * (MATH_PI / 180.0f);
}


static inline float rad2deg(float radian)
{
    return radian * (180.0f / MATH_PI);
}


static inline bool zero(float a)
{
    return (fabsf(a) <= FLOAT_EPSILON) ? true : false;
}


static inline bool equal(float a, float b)
{
    return tvg::zero(a - b);
}


template <typename T>
static inline constexpr const T& clamp(const T& v, const T& min, const T& max)
{
    if (v < min) return min;
    else if (v > max) return max;
    return v;
}

/************************************************************************/
/* Matrix functions                                                     */
/************************************************************************/

void rotate(Matrix* m, float degree);
bool inverse(const Matrix* m, Matrix* out);
bool identity(const Matrix* m);
Matrix operator*(const Matrix& lhs, const Matrix& rhs);
bool operator==(const Matrix& lhs, const Matrix& rhs);


static inline float radian(const Matrix& m)
{
    return fabsf(tvg::atan2(m.e21, m.e11));
}


static inline bool rightAngle(const Matrix& m)
{
   auto radian = tvg::radian(m);
   if (tvg::zero(radian) || tvg::zero(radian - MATH_PI2) || tvg::zero(radian - MATH_PI)) return true;
   return false;
}


static inline bool skewed(const Matrix& m)
{
    return !tvg::zero(m.e21 + m.e12);
}


static inline void identity(Matrix* m)
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


static inline constexpr const Matrix identity()
{
    return {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
}


static inline float scaling(const Matrix& m)
{
    return sqrtf(m.e11 * m.e11 + m.e21 * m.e21);
}


static inline void scale(Matrix* m, const Point& p)
{
    m->e11 *= p.x;
    m->e22 *= p.y;
}


static inline void scaleR(Matrix* m, const Point& p)
{
    if (p.x != 1.0f) {
        m->e11 *= p.x;
        m->e21 *= p.x;
    }
    if (p.y != 1.0f) {
        m->e22 *= p.y;
        m->e12 *= p.y;
    }
}


static inline void translate(Matrix* m, const Point& p)
{
    m->e13 += p.x;
    m->e23 += p.y;
}


static inline void translateR(Matrix* m, const Point& p)
{
    if (p.x == 0.0f && p.y == 0.0f) return;
    m->e13 += (p.x * m->e11 + p.y * m->e12);
    m->e23 += (p.x * m->e21 + p.y * m->e22);
}


static inline bool operator!=(const Matrix& lhs, const Matrix& rhs)
{
    return !(lhs == rhs);
}


static inline void operator*=(Matrix& lhs, const Matrix& rhs)
{
    lhs = lhs * rhs;
}


static inline Matrix operator*(const Matrix* lhs, const Matrix& rhs)
{
    if (lhs) return *lhs * rhs;
    return rhs;
}


static inline void log(const Matrix& m)
{
    TVGLOG("COMMON", "Matrix: [%f %f %f] [%f %f %f] [%f %f %f]", m.e11, m.e12, m.e13, m.e21, m.e22, m.e23, m.e31, m.e32, m.e33);
}


/************************************************************************/
/* Point functions                                                      */
/************************************************************************/

void operator*=(Point& pt, const Matrix& m);
Point operator*(const Point& pt, const Matrix& m);
Point normal(const Point& p1, const Point& p2);
void normalize(Point& pt);

static inline constexpr const Point operator*=(Point& pt, const Matrix* m)
{
    if (m) pt *= *m;
    return pt;
}


static inline Point operator*(const Point& pt, const Matrix* m)
{
    if (m) return pt * *m;
    return pt;
}


static inline Point min(const Point& lhs, const Point& rhs)
{
    return {std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y)};
}


static inline Point max(const Point& lhs, const Point& rhs)
{
    return {std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y)};
}


static inline float dot(const Point& lhs, const Point& rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y;
}


static inline float cross(const Point& lhs, const Point& rhs)
{
    return lhs.x * rhs.y - rhs.x * lhs.y;
}


static inline bool zero(const Point& p)
{
    return tvg::zero(p.x) && tvg::zero(p.y);
}


static inline float length(const Point& a, const Point& b)
{
    auto x = b.x - a.x;
    auto y = b.y - a.y;

    if (x < 0) x = -x;
    if (y < 0) y = -y;

    return (x > y) ? (x + 0.375f * y) : (y + 0.375f * x);
}


static inline float length(const Point& a)
{
    return sqrtf(a.x * a.x + a.y * a.y);
}


static inline float length2(const Point& a)
{
    return a.x * a.x + a.y * a.y;
};


static inline bool operator==(const Point& lhs, const Point& rhs)
{
    return tvg::equal(lhs.x, rhs.x) && tvg::equal(lhs.y, rhs.y);
}


static inline bool operator!=(const Point& lhs, const Point& rhs)
{
    return !(lhs == rhs);
}


static inline Point operator-(const Point& lhs, const Point& rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y};
}


static inline Point operator-(const Point& lhs, const float rhs)
{
    return {lhs.x - rhs, lhs.y - rhs};
}


static inline Point operator+(const Point& lhs, const Point& rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y};
}


static inline Point operator+(const Point& lhs, const float rhs)
{
    return {lhs.x + rhs, lhs.y + rhs};
}


static inline void operator+=(Point& lhs, const Point& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
}


static inline Point operator*(const Point& lhs, const Point& rhs)
{
    return {lhs.x * rhs.x, lhs.y * rhs.y};
}


static inline void operator*=(Point& lhs, const Point& rhs)
{
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
}


static inline Point operator*(const Point& lhs, const float rhs)
{
    return {lhs.x * rhs, lhs.y * rhs};
}


static inline void operator*=(Point& lhs, const float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
}


static inline Point operator*(const float& lhs, const Point& rhs)
{
    return {lhs * rhs.x, lhs * rhs.y};
}


static inline Point operator/(const Point& lhs, const Point& rhs)
{
    return {lhs.x / rhs.x, lhs.y / rhs.y};
}


static inline Point operator/(const Point& lhs, const float rhs)
{
    return {lhs.x / rhs, lhs.y / rhs};
}


static inline Point operator-(const Point& a)
{
    return {-a.x, -a.y};
}

enum class Orientation
{
    Linear = 0,
    Clockwise,
    CounterClockwise,
};


static inline Orientation orientation(const Point& p1, const Point& p2, const Point& p3)
{
    auto val = cross(p2 - p1, p3 - p1);
    if (zero(val)) return Orientation::Linear;
    else return val > 0 ? Orientation::Clockwise : Orientation::CounterClockwise;
}


static inline void log(const Point& pt)
{
    TVGLOG("COMMON", "Point: [%f %f]", pt.x, pt.y);
}


static inline bool closed(const Point& lhs, const Point& rhs, float tolerance)
{
    float dx = lhs.x - rhs.x;
    float dy = lhs.y - rhs.y;
    return (dx * dx + dy * dy) < (tolerance * tolerance);
}


/************************************************************************/
/* Line functions                                                       */
/************************************************************************/

struct Line
{
    Point pt1;
    Point pt2;

    void split(float at, Line& left, Line& right) const;
    float length() const;
};


/************************************************************************/
/* Geometry functions                                                   */
/************************************************************************/

struct BBox
{
    Point min, max;

    void init()
    {
        min = {FLT_MAX, FLT_MAX};
        max = {-FLT_MAX, -FLT_MAX};
    }
};


static inline uint32_t arcSegmentsCnt(float arcAngle, float pixelRadius) 
{
    if (pixelRadius < FLOAT_EPSILON) return 2;
    static constexpr auto PX_TOLERANCE = 0.25f;
    // Sagitta-based formula Approximation: 1 - cos(θ/2) ≈ (θ/2)²/2, so θ ≈ 2 * sqrt(2 * s/r)
    auto segmentAngle = 2.0f * sqrtf(2.0f * PX_TOLERANCE / pixelRadius);
    return static_cast<uint32_t>(ceilf(fabsf(arcAngle) / segmentAngle)) + 1;
}


/************************************************************************/
/* Bezier functions                                                     */
/************************************************************************/

struct Bezier
{
    Point start;
    Point ctrl1;
    Point ctrl2;
    Point end;

    Bezier() {}
    Bezier(const Point& p0, const Point& p1, const Point& p2, const Point& p3):
        start(p0), ctrl1(p1), ctrl2(p2), end(p3) {}
    // Constructor that approximates a quarter-circle segment of arc between 'start' and 'end' points 
    // using a cubic Bezier curve with a given 'radius'.
    Bezier(const Point& start, const Point& end, float radius);

    void split(float t, Bezier& left);
    void split(Bezier& left, Bezier& right) const;
    void split(float at, Bezier& left, Bezier& right) const;
    float length() const;
    float lengthApprox() const;
    float at(float at, float length) const;
    float atApprox(float at, float length) const;
    Point at(float t) const;
    float angle(float t) const;
    bool flatten() const;
    uint32_t segments() const;

    Bezier operator*(const Matrix& m);

    static void bounds(BBox& box, const Point& start, const Point& ctrl1, const Point& ctrl2, const Point& end);
};


/************************************************************************/
/* Interpolation functions                                              */
/************************************************************************/

template <typename T>
static inline T lerp(const T &start, const T &end, float t)
{
    return static_cast<T>(start + (end - start) * t);
}

uint8_t lerp(const uint8_t &start, const uint8_t &end, float t);

}

#endif //_TVG_MATH_H_
