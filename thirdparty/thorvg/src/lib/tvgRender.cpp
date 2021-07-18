/*
 * Copyright (c) 2020-2021 Samsung Electronics Co., Ltd. All rights reserved.

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
#include <float.h>
#include <math.h>
#include "tvgRender.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

void RenderTransform::override(const Matrix& m)
{
    this->m = m;

    if (m.e11 == 0.0f && m.e12 == 0.0f && m.e13 == 0.0f &&
        m.e21 == 0.0f && m.e22 == 0.0f && m.e23 == 0.0f &&
        m.e31 == 0.0f && m.e32 == 0.0f && m.e33 == 0.0f) {
        overriding = false;
    } else overriding = true;
}


bool RenderTransform::update()
{
    constexpr auto PI = 3.141592f;

    if (overriding) return true;

    //Init Status
    if (fabsf(x) <= FLT_EPSILON && fabsf(y) <= FLT_EPSILON &&
        fabsf(degree) <= FLT_EPSILON && fabsf(scale - 1) <= FLT_EPSILON) {
        return false;
    }

    //identity
    m.e11 = 1.0f;
    m.e12 = 0.0f;
    m.e13 = 0.0f;
    m.e21 = 0.0f;
    m.e22 = 1.0f;
    m.e23 = 0.0f;
    m.e31 = 0.0f;
    m.e32 = 0.0f;
    m.e33 = 1.0f;

    //scale
    m.e11 = scale;
    m.e22 = scale;

    //rotation
    if (fabsf(degree) > FLT_EPSILON) {
        auto radian = degree / 180.0f * PI;
        auto cosVal = cosf(radian);
        auto sinVal = sinf(radian);

        m.e12 = m.e11 * -sinVal;
        m.e11 *= cosVal;
        m.e21 = m.e22 * sinVal;
        m.e22 *= cosVal;
    }

    m.e13 = x;
    m.e23 = y;

    return true;
}


RenderTransform::RenderTransform()
{
}


RenderTransform::RenderTransform(const RenderTransform* lhs, const RenderTransform* rhs)
{
    m.e11 = lhs->m.e11 * rhs->m.e11 + lhs->m.e12 * rhs->m.e21 + lhs->m.e13 * rhs->m.e31;
    m.e12 = lhs->m.e11 * rhs->m.e12 + lhs->m.e12 * rhs->m.e22 + lhs->m.e13 * rhs->m.e32;
    m.e13 = lhs->m.e11 * rhs->m.e13 + lhs->m.e12 * rhs->m.e23 + lhs->m.e13 * rhs->m.e33;

    m.e21 = lhs->m.e21 * rhs->m.e11 + lhs->m.e22 * rhs->m.e21 + lhs->m.e23 * rhs->m.e31;
    m.e22 = lhs->m.e21 * rhs->m.e12 + lhs->m.e22 * rhs->m.e22 + lhs->m.e23 * rhs->m.e32;
    m.e23 = lhs->m.e21 * rhs->m.e13 + lhs->m.e22 * rhs->m.e23 + lhs->m.e23 * rhs->m.e33;

    m.e31 = lhs->m.e31 * rhs->m.e11 + lhs->m.e32 * rhs->m.e21 + lhs->m.e33 * rhs->m.e31;
    m.e32 = lhs->m.e31 * rhs->m.e12 + lhs->m.e32 * rhs->m.e22 + lhs->m.e33 * rhs->m.e32;
    m.e33 = lhs->m.e31 * rhs->m.e13 + lhs->m.e32 * rhs->m.e23 + lhs->m.e33 * rhs->m.e33;
}
