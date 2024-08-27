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
#include "tvgRender.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

uint32_t RenderMethod::ref()
{
    ScopedLock lock(key);
    return (++refCnt);
}


uint32_t RenderMethod::unref()
{
    ScopedLock lock(key);
    return (--refCnt);
}


void RenderRegion::intersect(const RenderRegion& rhs)
{
    auto x1 = x + w;
    auto y1 = y + h;
    auto x2 = rhs.x + rhs.w;
    auto y2 = rhs.y + rhs.h;

    x = (x > rhs.x) ? x : rhs.x;
    y = (y > rhs.y) ? y : rhs.y;
    w = ((x1 < x2) ? x1 : x2) - x;
    h = ((y1 < y2) ? y1 : y2) - y;

    if (w < 0) w = 0;
    if (h < 0) h = 0;
}


void RenderRegion::add(const RenderRegion& rhs)
{
    if (rhs.x < x) {
        w += (x - rhs.x);
        x = rhs.x;
    }
    if (rhs.y < y) {
        h += (y - rhs.y);
        y = rhs.y;
    }
    if (rhs.x + rhs.w > x + w) w = (rhs.x + rhs.w) - x;
    if (rhs.y + rhs.h > y + h) h = (rhs.y + rhs.h) - y;
}
