/*
 * Copyright (c) 2026 ThorVG project. All rights reserved.

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

#include "tvgWgGeometry.h"

//***********************************************************************
// WgMeshData
//***********************************************************************

void WgMeshData::bbox(const Point pmin, const Point pmax)
{
    const float vdata[] = {pmin.x, pmin.y, pmax.x, pmin.y, pmax.x, pmax.y, pmin.x, pmax.y};
    const uint32_t idata[] = {0, 1, 2, 0, 2, 3};
    // setup vertex data
    vbuffer.reserve(4);
    vbuffer.count = 4;
    memcpy(vbuffer.data, vdata, sizeof(vdata));
    // setup tex coords data
    tbuffer.clear();
    // setup indexes data
    ibuffer.reserve(6);
    ibuffer.count = 6;
    memcpy(ibuffer.data, idata, sizeof(idata));
}


void WgMeshData::imageBox(float w, float h, const Matrix& transform)
{
    const Point p0 = Point{0.0f, 0.0f} * transform;
    const Point p1 = Point{w,    0.0f} * transform;
    const Point p2 = Point{w,       h} * transform;
    const Point p3 = Point{0.0f,    h} * transform;
    const Point vdata[] = {p0, p1, p2, p3};
    const float tdata[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    const uint32_t idata[] = {0, 1, 2, 0, 2, 3};
    // setup vertex data
    vbuffer.reserve(4);
    vbuffer.count = 4;
    memcpy(vbuffer.data, vdata, sizeof(vdata));
    // setup tex coords data
    tbuffer.reserve(4);
    tbuffer.count = 4;
    memcpy(tbuffer.data, tdata, sizeof(tdata));
    // setup indexes data
    ibuffer.reserve(6);
    ibuffer.count = 6;
    memcpy(ibuffer.data, idata, sizeof(idata));
}


void WgMeshData::blitBox()
{
    const float vdata[] = {-1.0f, +1.0f, +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, -1.0f};
    const float tdata[] = {+0.0f, +0.0f, +1.0f, +0.0f, +1.0f, +1.0f, +0.0f, +1.0f};
    const uint32_t idata[] = { 0, 1, 2, 0, 2, 3 };
    // setup vertex data
    vbuffer.reserve(4);
    vbuffer.count = 4;
    memcpy(vbuffer.data, vdata, sizeof(vdata));
    // setup tex coords data
    tbuffer.reserve(4);
    tbuffer.count = 4;
    memcpy(tbuffer.data, tdata, sizeof(tdata));
    // setup indexes data
    ibuffer.reserve(6);
    ibuffer.count = 6;
    memcpy(ibuffer.data, idata, sizeof(idata));
}


void WgMeshData::clear()
{
    vbuffer.clear();
    tbuffer.clear();
    ibuffer.clear();
    voffset = 0;
    toffset = 0;
    ioffset = 0;
}
