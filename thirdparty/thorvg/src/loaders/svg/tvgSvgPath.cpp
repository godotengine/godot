/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

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

#define _USE_MATH_DEFINES       //Math Constants are not defined in Standard C/C++.

#include <cstring>
#include <ctype.h>
#include "tvgMath.h"
#include "tvgSvgLoaderCommon.h"
#include "tvgSvgPath.h"
#include "tvgStr.h"
#include "tvgSvgUtil.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static char* _skipComma(const char* content)
{
    content = svgUtilSkipWhiteSpace(content, nullptr);
    if (*content == ',') return (char*)content + 1;
    return (char*)content;
}


static bool _parseNumber(char** content, float* number)
{
    char* end = NULL;
    *number = toFloat(*content, &end);
    //If the start of string is not number
    if ((*content) == end) return false;
    //Skip comma if any
    *content = _skipComma(end);
    return true;
}


static bool _parseFlag(char** content, int* number)
{
    char* end = NULL;
    if (*(*content) != '0' && *(*content) != '1') return false;
    *number = *(*content) - '0';
    *content += 1;
    end = *content;
    *content = _skipComma(end);

    return true;
}


//Some helpful stuff is available here:
//http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
void _pathAppendArcTo(RenderPath& out, Point& cur, Point& curCtl, const Point& next, Point radius, float angle, bool largeArc, bool sweep)
{
    auto start = cur;
    auto cosPhi = cosf(angle);
    auto sinPhi = sinf(angle);
    auto d2 = (start - next) * 0.5f;
    auto x1p = cosPhi * d2.x + sinPhi * d2.y;
    auto y1p = cosPhi * d2.y - sinPhi * d2.x;
    auto x1p2 = x1p * x1p;
    auto y1p2 = y1p * y1p;
    auto radius2 = Point{radius.x * radius.x, radius.y * radius.y};
    auto lambda = (x1p2 / radius2.x) + (y1p2 / radius2.y);

    //Correction of out-of-range radii, see F6.6.2 (step 4)
    if (lambda > 1.0f) {
        //See F6.6.3
        radius *= sqrtf(lambda);
        radius2 = {radius.x * radius.x, radius.y * radius.y};
    }

    Point cp, center;
    auto c = (radius2.x * radius2.y) - (radius2.x * y1p2) - (radius2.y * x1p2);

    //Check if there is no possible solution
    //(i.e. we can't do a square root of a negative value)
    if (c < 0.0f) {
        //Scale uniformly until we have a single solution
        //(see F6.2) i.e. when c == 0.0
        radius *= sqrtf(1.0f - c / (radius2.x * radius2.y));
        radius2 = {radius.x * radius.x, radius.y * radius.y};
        //Step 2 (F6.5.2) - simplified since c == 0.0
        cp = {0.0f, 0.0f};
        //Step 3 (F6.5.3 first part) - simplified since cxp and cyp == 0.0
        center = {0.0f, 0.0f};
    } else {
        //Complete c calculation
        c = sqrtf(c / ((radius2.x * y1p2) + (radius2.y * x1p2)));
        //Inverse sign if Fa == Fs
        if (largeArc == sweep) c = -c;
        //Step 2 (F6.5.2)
        cp = c * Point{(radius.x * y1p / radius.y), (-radius.y * x1p / radius.x)};
        //Step 3 (F6.5.3 first part)
        center = {cosPhi * cp.x - sinPhi * cp.y, sinPhi * cp.x + cosPhi * cp.y};
    }

    //Step 3 (F6.5.3 second part) we now have the center point of the ellipse
    center += (start + next) * 0.5f;

    //Step 4 (F6.5.4)
    //We don't use arccos (as per w3c doc), see
    //http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
    //Note: atan2 (0.0, 1.0) == 0.0
    auto at = tvg::atan2(((y1p - cp.y) / radius.y), ((x1p - cp.x) / radius.x));
    auto theta1 = (at < 0.0f) ? 2.0f * MATH_PI + at : at;
    auto nat = tvg::atan2(((-y1p - cp.y) / radius.y), ((-x1p - cp.x) / radius.x));
    auto deltaTheta = (nat < at) ? 2.0f * MATH_PI - at + nat : nat - at;

    if (sweep) {
        //Ensure delta theta < 0 or else add 360 degrees
        if (deltaTheta < 0.0f) deltaTheta += 2.0f * MATH_PI;
    } else {
        //Ensure delta theta > 0 or else subtract 360 degrees
        if (deltaTheta > 0.0f) deltaTheta -= 2.0f * MATH_PI;
    }

    //Add several cubic bezier to approximate the arc
    //(smaller than 90 degrees)
    //We add one extra segment because we want something
    //Smaller than 90deg (i.e. not 90 itself)
    auto segments = int(fabsf(deltaTheta / MATH_PI2) + 1.0f);
    auto delta = deltaTheta / segments;

    //http://www.stillhq.com/ctpfaq/2001/comp.text.pdf-faq-2001-04.txt (section 2.13)
    auto bcp = 4.0f / 3.0f * (1.0f - cosf(delta / 2.0f)) / sinf(delta / 2.0f);
    auto cosPhiR = Point{cosPhi * radius.x, cosPhi * radius.y};
    auto sinPhiR = Point{sinPhi * radius.x, sinPhi * radius.y};
    auto cosTheta1 = cosf(theta1);
    auto sinTheta1 = sinf(theta1);

    for (int i = 0; i < segments; ++i) {
        //End angle (for this segment) = current + delta
        auto theta2 = theta1 + delta;
        auto cosTheta2 = cosf(theta2);
        auto sinTheta2 = sinf(theta2);

        //First control point (based on start point sx,sy)
        auto c1 = start + Point{-bcp * (cosPhiR.x * sinTheta1 + sinPhiR.y * cosTheta1), bcp * (cosPhiR.y * cosTheta1 - sinPhiR.x * sinTheta1)};

        //End point (for this segment)
        auto e = center + Point{cosPhiR.x * cosTheta2 - sinPhiR.y * sinTheta2, sinPhiR.x * cosTheta2 + cosPhiR.y * sinTheta2};

        //Second control point (based on end point ex,ey)
        curCtl = e + Point{bcp * (cosPhiR.x * sinTheta2 + sinPhiR.y * cosTheta2), bcp * (sinPhiR.x * sinTheta2 - cosPhiR.y * cosTheta2)};
        cur = e;
        out.cubicTo(c1, curCtl, cur);

        //Next start point is the current end point (same for angle)
        start = e;
        theta1 = theta2;
        //Avoid recomputations
        cosTheta1 = cosTheta2;
        sinTheta1 = sinTheta2;
    }
}


static int _numberCount(char cmd)
{
    int count = 0;
    switch (cmd) {
        case 'M':
        case 'm':
        case 'L':
        case 'l':
        case 'T':
        case 't': {
            count = 2;
            break;
        }
        case 'C':
        case 'c':
        case 'E':
        case 'e': {
            count = 6;
            break;
        }
        case 'H':
        case 'h':
        case 'V':
        case 'v': {
            count = 1;
            break;
        }
        case 'S':
        case 's':
        case 'Q':
        case 'q': {
            count = 4;
            break;
        }
        case 'A':
        case 'a': {
            count = 7;
            break;
        }
        default: break;
    }
    return count;
}


static bool _processCommand(RenderPath& out, char cmd, float* arr, int count, Point& cur, Point& curCtl, Point& start, bool& quadratic, bool& closed)
{
    switch (cmd) {
        case 'm':
        case 'l':
        case 'c':
        case 's':
        case 'q':
        case 't': {
            for (int i = 0; i < count - 1; i += 2) {
                arr[i] = arr[i] + cur.x;
                arr[i + 1] = arr[i + 1] + cur.y;
            }
            break;
        }
        case 'h': {
            arr[0] = arr[0] + cur.x;
            break;
        }
        case 'v': {
            arr[0] = arr[0] + cur.y;
            break;
        }
        case 'a': {
            arr[5] = arr[5] + cur.x;
            arr[6] = arr[6] + cur.y;
            break;
        }
        default: break;
    }

    switch (cmd) {
        case 'm':
        case 'M': {
            start = cur = {arr[0], arr[1]};
            out.moveTo(cur);
            break;
        }
        case 'l':
        case 'L': {
            cur = {arr[0], arr[1]};
            out.lineTo(cur);
            break;
        }
        case 'c':
        case 'C': {
            curCtl = {arr[2], arr[3]};
            cur = {arr[4], arr[5]};
            out.cubicTo({arr[0], arr[1]}, curCtl, cur);
            quadratic = false;
            break;
        }
        case 's':
        case 'S': {
            Point ctrl;
            if ((out.cmds.count > 1) && (out.cmds.last() == PathCommand::CubicTo) && !quadratic) {
                ctrl = 2 * cur - curCtl;
            } else {
                ctrl = cur;
            }
            curCtl = {arr[0], arr[1]};
            cur = {arr[2], arr[3]};
            out.cubicTo(ctrl, curCtl, cur);
            quadratic = false;
            break;
        }
        case 'q':
        case 'Q': {
            auto ctrl1 = (cur + 2 * Point{arr[0], arr[1]}) * (1.0f / 3.0f);
            auto ctrl2 = (Point{arr[2], arr[3]} + 2 * Point{arr[0], arr[1]}) * (1.0f / 3.0f);
            curCtl = {arr[0], arr[1]};
            cur = {arr[2], arr[3]};
            out.cubicTo(ctrl1, ctrl2, cur);
            quadratic = true;
            break;
        }
        case 't':
        case 'T': {
            Point ctrl;
            if ((out.cmds.count > 1) && (out.cmds.last() == PathCommand::CubicTo) && quadratic) {
                ctrl = 2 * cur - curCtl;
            } else {
                ctrl = cur;
            }
            auto ctrl1 = (cur + 2 * ctrl) * (1.0f / 3.0f);
            auto ctrl2 = (Point{arr[0], arr[1]} + 2 * ctrl) * (1.0f / 3.0f);
            curCtl = {ctrl.x, ctrl.y};
            cur = {arr[0], arr[1]};
            out.cubicTo(ctrl1, ctrl2, cur);
            quadratic = true;
            break;
        }
        case 'h':
        case 'H': {
            out.lineTo({arr[0], cur.y});
            cur.x = arr[0];
            break;
        }
        case 'v':
        case 'V': {
            out.lineTo({cur.x, arr[0]});
            cur.y = arr[0];
            break;
        }
        case 'z':
        case 'Z': {
            out.close();
            cur = start;
            closed = true;
            break;
        }
        case 'a':
        case 'A': {
            if (tvg::zero(arr[0]) || tvg::zero(arr[1])) {
                cur = {arr[5], arr[6]};
                out.lineTo(cur);
            } else if (!tvg::equal(cur.x, arr[5]) || !tvg::equal(cur.y, arr[6])) {
                _pathAppendArcTo(out, cur, curCtl, {arr[5], arr[6]}, {fabsf(arr[0]), fabsf(arr[1])}, deg2rad(arr[2]), arr[3], arr[4]);
                cur = curCtl = {arr[5], arr[6]};
                quadratic = false;
            }
            break;
        }
        default: return false;
    }
    return true;
}


static char* _nextCommand(char* path, char* cmd, float* arr, int* count, bool* closed)
{
    int large, sweep;

    path = _skipComma(path);
    if (isalpha(*path)) {
        *cmd = *path;
        path++;
        *count = _numberCount(*cmd);
    } else {
        if (*cmd == 'm') *cmd = 'l';
        else if (*cmd == 'M') *cmd = 'L';
        else {
          if (*closed) return nullptr;
        }
    }
    if (*count == 7) {
        //Special case for arc command
        if (_parseNumber(&path, &arr[0])) {
            if (_parseNumber(&path, &arr[1])) {
                if (_parseNumber(&path, &arr[2])) {
                    if (_parseFlag(&path, &large)) {
                        if (_parseFlag(&path, &sweep)) {
                            if (_parseNumber(&path, &arr[5])) {
                                if (_parseNumber(&path, &arr[6])) {
                                    arr[3] = (float)large;
                                    arr[4] = (float)sweep;
                                    return path;
                                }
                            }
                        }
                    }
                }
            }
        }
        *count = 0;
        return nullptr;
    }
    for (int i = 0; i < *count; i++) {
        if (!_parseNumber(&path, &arr[i])) {
            *count = 0;
            return nullptr;
        }
        path = _skipComma(path);
    }
    return path;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/


bool svgPathToShape(const char* svgPath, RenderPath& out)
{
    float numberArray[7];
    int numberCount = 0;
    Point cur = {0, 0};
    Point curCtl = {0, 0};
    Point start = {0, 0};
    char cmd = 0;
    auto path = (char*)svgPath;
    auto lastCmds = out.cmds.count;
    auto isQuadratic = false;
    auto closed = false;

    while ((path[0] != '\0')) {
        path = _nextCommand(path, &cmd, numberArray, &numberCount, &closed);
        if (!path) break;
        closed = false;
        if (!_processCommand(out, cmd, numberArray, numberCount, cur, curCtl, start, isQuadratic, closed)) break;
    }

    if (out.cmds.count > lastCmds && out.cmds[lastCmds] != PathCommand::MoveTo) return false;
    return true;
}
