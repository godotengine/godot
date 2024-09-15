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

#ifndef _TVG_LINES_H_
#define _TVG_LINES_H_

#include "tvgCommon.h"

namespace tvg
{

struct Line
{
    Point pt1;
    Point pt2;
};

float lineLength(const Point& pt1, const Point& pt2);
void lineSplitAt(const Line& cur, float at, Line& left, Line& right);


struct Bezier
{
    Point start;
    Point ctrl1;
    Point ctrl2;
    Point end;
};

void bezSplit(const Bezier&cur, Bezier& left, Bezier& right);
float bezLength(const Bezier& cur);
void bezSplitLeft(Bezier& cur, float at, Bezier& left);
float bezAt(const Bezier& bz, float at, float length);
void bezSplitAt(const Bezier& cur, float at, Bezier& left, Bezier& right);
Point bezPointAt(const Bezier& bz, float t);
float bezAngleAt(const Bezier& bz, float t);

float bezLengthApprox(const Bezier& cur);
float bezAtApprox(const Bezier& bz, float at, float length);
}

#endif //_TVG_LINES_H_
