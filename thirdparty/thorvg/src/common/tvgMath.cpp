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

#include "tvgMath.h"


bool mathInverse(const Matrix* m, Matrix* out)
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


Matrix mathMultiply(const Matrix* lhs, const Matrix* rhs)
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


void mathRotate(Matrix* m, float degree)
{
    if (degree == 0.0f) return;

    auto radian = degree / 180.0f * M_PI;
    auto cosVal = cosf(radian);
    auto sinVal = sinf(radian);

    m->e12 = m->e11 * -sinVal;
    m->e11 *= cosVal;
    m->e21 = m->e22 * sinVal;
    m->e22 *= cosVal;
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


void mathMultiply(Point* pt, const Matrix* transform)
{
    auto tx = pt->x * transform->e11 + pt->y * transform->e12 + transform->e13;
    auto ty = pt->x * transform->e21 + pt->y * transform->e22 + transform->e23;
    pt->x = tx;
    pt->y = ty;
}
