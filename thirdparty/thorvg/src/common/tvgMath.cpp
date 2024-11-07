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

#include "tvgMath.h"

//see: https://en.wikipedia.org/wiki/Remez_algorithm
float mathAtan2(float y, float x)
{
    if (y == 0.0f && x == 0.0f) return 0.0f;

    auto a = std::min(fabsf(x), fabsf(y)) / std::max(fabsf(x), fabsf(y));
    auto s = a * a;
    auto r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (fabsf(y) > fabsf(x)) r = 1.57079637f - r;
    if (x < 0) r = 3.14159274f - r;
    if (y < 0) return -r;
    return r;
}


bool mathInverse(const Matrix* m, Matrix* out)
{
    auto det = m->e11 * (m->e22 * m->e33 - m->e32 * m->e23) -
               m->e12 * (m->e21 * m->e33 - m->e23 * m->e31) +
               m->e13 * (m->e21 * m->e32 - m->e22 * m->e31);

    auto invDet = 1.0f / det;
    if (std::isinf(invDet)) return false;

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


bool mathIdentity(const Matrix* m)
{
    if (m->e11 != 1.0f || m->e12 != 0.0f || m->e13 != 0.0f ||
        m->e21 != 0.0f || m->e22 != 1.0f || m->e23 != 0.0f ||
        m->e31 != 0.0f || m->e32 != 0.0f || m->e33 != 1.0f) {
        return false;
    }
    return true;
}


void mathRotate(Matrix* m, float degree)
{
    if (degree == 0.0f) return;

    auto radian = degree / 180.0f * MATH_PI;
    auto cosVal = cosf(radian);
    auto sinVal = sinf(radian);

    m->e12 = m->e11 * -sinVal;
    m->e11 *= cosVal;
    m->e21 = m->e22 * sinVal;
    m->e22 *= cosVal;
}


Matrix operator*(const Matrix& lhs, const Matrix& rhs)
{
    Matrix m;

    m.e11 = lhs.e11 * rhs.e11 + lhs.e12 * rhs.e21 + lhs.e13 * rhs.e31;
    m.e12 = lhs.e11 * rhs.e12 + lhs.e12 * rhs.e22 + lhs.e13 * rhs.e32;
    m.e13 = lhs.e11 * rhs.e13 + lhs.e12 * rhs.e23 + lhs.e13 * rhs.e33;

    m.e21 = lhs.e21 * rhs.e11 + lhs.e22 * rhs.e21 + lhs.e23 * rhs.e31;
    m.e22 = lhs.e21 * rhs.e12 + lhs.e22 * rhs.e22 + lhs.e23 * rhs.e32;
    m.e23 = lhs.e21 * rhs.e13 + lhs.e22 * rhs.e23 + lhs.e23 * rhs.e33;

    m.e31 = lhs.e31 * rhs.e11 + lhs.e32 * rhs.e21 + lhs.e33 * rhs.e31;
    m.e32 = lhs.e31 * rhs.e12 + lhs.e32 * rhs.e22 + lhs.e33 * rhs.e32;
    m.e33 = lhs.e31 * rhs.e13 + lhs.e32 * rhs.e23 + lhs.e33 * rhs.e33;

    return m;
}


bool operator==(const Matrix& lhs, const Matrix& rhs)
{
    if (!mathEqual(lhs.e11, rhs.e11) || !mathEqual(lhs.e12, rhs.e12) || !mathEqual(lhs.e13, rhs.e13) ||
        !mathEqual(lhs.e21, rhs.e21) || !mathEqual(lhs.e22, rhs.e22) || !mathEqual(lhs.e23, rhs.e23) ||
        !mathEqual(lhs.e31, rhs.e31) || !mathEqual(lhs.e32, rhs.e32) || !mathEqual(lhs.e33, rhs.e33)) {
       return false;
    }
    return true;
}


void operator*=(Point& pt, const Matrix& m)
{
    auto tx = pt.x * m.e11 + pt.y * m.e12 + m.e13;
    auto ty = pt.x * m.e21 + pt.y * m.e22 + m.e23;
    pt.x = tx;
    pt.y = ty;
}


Point operator*(const Point& pt, const Matrix& m)
{
    auto tx = pt.x * m.e11 + pt.y * m.e12 + m.e13;
    auto ty = pt.x * m.e21 + pt.y * m.e22 + m.e23;
    return {tx, ty};
}

uint8_t mathLerp(const uint8_t &start, const uint8_t &end, float t)
{
    auto result = static_cast<int>(start + (end - start) * t);
    mathClamp(result, 0, 255);
    return static_cast<uint8_t>(result);
}
