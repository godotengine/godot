/*
 * Copyright (c) 2020 - 2024 the ThorVG project. All rights reserved.

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

#include "tvgMath.h"
#include "tvgLines.h"

#define BEZIER_EPSILON 1e-2f

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static float _lineLengthApprox(const Point& pt1, const Point& pt2)
{
    /* approximate sqrt(x*x + y*y) using alpha max plus beta min algorithm.
       With alpha = 1, beta = 3/8, giving results with the largest error less
       than 7% compared to the exact value. */
    Point diff = {pt2.x - pt1.x, pt2.y - pt1.y};
    if (diff.x < 0) diff.x = -diff.x;
    if (diff.y < 0) diff.y = -diff.y;
    return (diff.x > diff.y) ? (diff.x + diff.y * 0.375f) : (diff.y + diff.x * 0.375f);
}


static float _lineLength(const Point& pt1, const Point& pt2)
{
    Point diff = {pt2.x - pt1.x, pt2.y - pt1.y};
    return sqrtf(diff.x * diff.x + diff.y * diff.y);
}


template<typename LengthFunc>
float _bezLength(const Bezier& cur, LengthFunc lineLengthFunc)
{
    Bezier left, right;
    auto len = lineLengthFunc(cur.start, cur.ctrl1) + lineLengthFunc(cur.ctrl1, cur.ctrl2) + lineLengthFunc(cur.ctrl2, cur.end);
    auto chord = lineLengthFunc(cur.start, cur.end);

    if (fabsf(len - chord) > BEZIER_EPSILON) {
        tvg::bezSplit(cur, left, right);
        return _bezLength(left, lineLengthFunc) + _bezLength(right, lineLengthFunc);
    }
    return len;
}


template<typename LengthFunc>
float _bezAt(const Bezier& bz, float at, float length, LengthFunc lineLengthFunc)
{
    auto biggest = 1.0f;
    auto smallest = 0.0f;
    auto t = 0.5f;

    //just in case to prevent an infinite loop
    if (at <= 0) return 0.0f;
    if (at >= length) return 1.0f;

    while (true) {
        auto right = bz;
        Bezier left;
        bezSplitLeft(right, t, left);
        length = _bezLength(left, lineLengthFunc);
        if (fabsf(length - at) < BEZIER_EPSILON || fabsf(smallest - biggest) < BEZIER_EPSILON) {
            break;
        }
        if (length < at) {
            smallest = t;
            t = (t + biggest) * 0.5f;
        } else {
            biggest = t;
            t = (smallest + t) * 0.5f;
        }
    }
    return t;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

namespace tvg
{

float lineLength(const Point& pt1, const Point& pt2)
{
    return _lineLength(pt1, pt2);
}


void lineSplitAt(const Line& cur, float at, Line& left, Line& right)
{
    auto len = lineLength(cur.pt1, cur.pt2);
    auto dx = ((cur.pt2.x - cur.pt1.x) / len) * at;
    auto dy = ((cur.pt2.y - cur.pt1.y) / len) * at;
    left.pt1 = cur.pt1;
    left.pt2.x = left.pt1.x + dx;
    left.pt2.y = left.pt1.y + dy;
    right.pt1 = left.pt2;
    right.pt2 = cur.pt2;
}


void bezSplit(const Bezier& cur, Bezier& left, Bezier& right)
{
    auto c = (cur.ctrl1.x + cur.ctrl2.x) * 0.5f;
    left.ctrl1.x = (cur.start.x + cur.ctrl1.x) * 0.5f;
    right.ctrl2.x = (cur.ctrl2.x + cur.end.x) * 0.5f;
    left.start.x = cur.start.x;
    right.end.x = cur.end.x;
    left.ctrl2.x = (left.ctrl1.x + c) * 0.5f;
    right.ctrl1.x = (right.ctrl2.x + c) * 0.5f;
    left.end.x = right.start.x = (left.ctrl2.x + right.ctrl1.x) * 0.5f;

    c = (cur.ctrl1.y + cur.ctrl2.y) * 0.5f;
    left.ctrl1.y = (cur.start.y + cur.ctrl1.y) * 0.5f;
    right.ctrl2.y = (cur.ctrl2.y + cur.end.y) * 0.5f;
    left.start.y = cur.start.y;
    right.end.y = cur.end.y;
    left.ctrl2.y = (left.ctrl1.y + c) * 0.5f;
    right.ctrl1.y = (right.ctrl2.y + c) * 0.5f;
    left.end.y = right.start.y = (left.ctrl2.y + right.ctrl1.y) * 0.5f;
}


float bezLength(const Bezier& cur)
{
    return _bezLength(cur, _lineLength);
}


float bezLengthApprox(const Bezier& cur)
{
    return _bezLength(cur, _lineLengthApprox);
}


void bezSplitLeft(Bezier& cur, float at, Bezier& left)
{
    left.start = cur.start;

    left.ctrl1.x = cur.start.x + at * (cur.ctrl1.x - cur.start.x);
    left.ctrl1.y = cur.start.y + at * (cur.ctrl1.y - cur.start.y);

    left.ctrl2.x = cur.ctrl1.x + at * (cur.ctrl2.x - cur.ctrl1.x); //temporary holding spot
    left.ctrl2.y = cur.ctrl1.y + at * (cur.ctrl2.y - cur.ctrl1.y); //temporary holding spot

    cur.ctrl2.x = cur.ctrl2.x + at * (cur.end.x - cur.ctrl2.x);
    cur.ctrl2.y = cur.ctrl2.y + at * (cur.end.y - cur.ctrl2.y);

    cur.ctrl1.x = left.ctrl2.x + at * (cur.ctrl2.x - left.ctrl2.x);
    cur.ctrl1.y = left.ctrl2.y + at * (cur.ctrl2.y - left.ctrl2.y);

    left.ctrl2.x = left.ctrl1.x + at * (left.ctrl2.x - left.ctrl1.x);
    left.ctrl2.y = left.ctrl1.y + at * (left.ctrl2.y - left.ctrl1.y);

    left.end.x = cur.start.x = left.ctrl2.x + at * (cur.ctrl1.x - left.ctrl2.x);
    left.end.y = cur.start.y = left.ctrl2.y + at * (cur.ctrl1.y - left.ctrl2.y);
}


float bezAt(const Bezier& bz, float at, float length)
{
    return _bezAt(bz, at, length, _lineLength);
}


float bezAtApprox(const Bezier& bz, float at, float length)
{
    return _bezAt(bz, at, length, _lineLengthApprox);
}


void bezSplitAt(const Bezier& cur, float at, Bezier& left, Bezier& right)
{
    right = cur;
    auto t = bezAt(right, at, bezLength(right));
    bezSplitLeft(right, t, left);
}


Point bezPointAt(const Bezier& bz, float t)
{
    Point cur;
    auto it = 1.0f - t;

    auto ax = bz.start.x * it + bz.ctrl1.x * t;
    auto bx = bz.ctrl1.x * it + bz.ctrl2.x * t;
    auto cx = bz.ctrl2.x * it + bz.end.x * t;
    ax = ax * it + bx * t;
    bx = bx * it + cx * t;
    cur.x = ax * it + bx * t;

    float ay = bz.start.y * it + bz.ctrl1.y * t;
    float by = bz.ctrl1.y * it + bz.ctrl2.y * t;
    float cy = bz.ctrl2.y * it + bz.end.y * t;
    ay = ay * it + by * t;
    by = by * it + cy * t;
    cur.y = ay * it + by * t;

    return cur;
}


float bezAngleAt(const Bezier& bz, float t)
{
    if (t < 0 || t > 1) return 0;

    //derivate
    // p'(t) = 3 * (-(1-2t+t^2) * p0 + (1 - 4 * t + 3 * t^2) * p1 + (2 * t - 3 *
    // t^2) * p2 + t^2 * p3)
    float mt = 1.0f - t;
    float d = t * t;
    float a = -mt * mt;
    float b = 1 - 4 * t + 3 * d;
    float c = 2 * t - 3 * d;

    Point pt ={a * bz.start.x + b * bz.ctrl1.x + c * bz.ctrl2.x + d * bz.end.x, a * bz.start.y + b * bz.ctrl1.y + c * bz.ctrl2.y + d * bz.end.y};
    pt.x *= 3;
    pt.y *= 3;

    return mathRad2Deg(mathAtan2(pt.y, pt.x));
}


}
