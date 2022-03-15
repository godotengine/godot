/*
 * Copyright (c) 2020 - 2022 Samsung Electronics Co., Ltd. All rights reserved.

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
#include "tvgShapeImpl.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/
constexpr auto PATH_KAPPA = 0.552284f;

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Shape :: Shape() : pImpl(new Impl(this))
{
    Paint::pImpl->id = TVG_CLASS_ID_SHAPE;
    Paint::pImpl->method(new PaintMethod<Shape::Impl>(pImpl));
}


Shape :: ~Shape()
{
    delete(pImpl);
}


unique_ptr<Shape> Shape::gen() noexcept
{
    return unique_ptr<Shape>(new Shape);
}


uint32_t Shape::identifier() noexcept
{
    return TVG_CLASS_ID_SHAPE;
}


Result Shape::reset() noexcept
{
    pImpl->path.reset();
    pImpl->flag = RenderUpdateFlag::Path;

    return Result::Success;
}


uint32_t Shape::pathCommands(const PathCommand** cmds) const noexcept
{
    if (!cmds) return 0;

    *cmds = pImpl->path.cmds;

    return pImpl->path.cmdCnt;
}


uint32_t Shape::pathCoords(const Point** pts) const noexcept
{
    if (!pts) return 0;

    *pts = pImpl->path.pts;

    return pImpl->path.ptsCnt;
}


Result Shape::appendPath(const PathCommand *cmds, uint32_t cmdCnt, const Point* pts, uint32_t ptsCnt) noexcept
{
    if (cmdCnt == 0 || ptsCnt == 0 || !cmds || !pts) return Result::InvalidArguments;

    pImpl->path.grow(cmdCnt, ptsCnt);
    pImpl->path.append(cmds, cmdCnt, pts, ptsCnt);

    pImpl->flag |= RenderUpdateFlag::Path;

    return Result::Success;
}


Result Shape::moveTo(float x, float y) noexcept
{
    pImpl->path.moveTo(x, y);

    pImpl->flag |= RenderUpdateFlag::Path;

    return Result::Success;
}


Result Shape::lineTo(float x, float y) noexcept
{
    pImpl->path.lineTo(x, y);

    pImpl->flag |= RenderUpdateFlag::Path;

    return Result::Success;
}


Result Shape::cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y) noexcept
{
    pImpl->path.cubicTo(cx1, cy1, cx2, cy2, x, y);

    pImpl->flag |= RenderUpdateFlag::Path;

    return Result::Success;
}


Result Shape::close() noexcept
{
    pImpl->path.close();

    pImpl->flag |= RenderUpdateFlag::Path;

    return Result::Success;
}


Result Shape::appendCircle(float cx, float cy, float rx, float ry) noexcept
{
    auto rxKappa = rx * PATH_KAPPA;
    auto ryKappa = ry * PATH_KAPPA;

    pImpl->path.grow(6, 13);
    pImpl->path.moveTo(cx, cy - ry);
    pImpl->path.cubicTo(cx + rxKappa, cy - ry, cx + rx, cy - ryKappa, cx + rx, cy);
    pImpl->path.cubicTo(cx + rx, cy + ryKappa, cx + rxKappa, cy + ry, cx, cy + ry);
    pImpl->path.cubicTo(cx - rxKappa, cy + ry, cx - rx, cy + ryKappa, cx - rx, cy);
    pImpl->path.cubicTo(cx - rx, cy - ryKappa, cx - rxKappa, cy - ry, cx, cy - ry);
    pImpl->path.close();

    pImpl->flag |= RenderUpdateFlag::Path;

    return Result::Success;
}

Result Shape::appendArc(float cx, float cy, float radius, float startAngle, float sweep, bool pie) noexcept
{
    //just circle
    if (sweep >= 360.0f || sweep <= -360.0f) return appendCircle(cx, cy, radius, radius);

    startAngle = (startAngle * M_PI) / 180.0f;
    sweep = sweep * M_PI / 180.0f;

    auto nCurves = ceil(fabsf(sweep / float(M_PI_2)));
    auto sweepSign = (sweep < 0 ? -1 : 1);
    auto fract = fmodf(sweep, float(M_PI_2));
    fract = (mathZero(fract)) ? float(M_PI_2) * sweepSign : fract;

    //Start from here
    Point start = {radius * cosf(startAngle), radius * sinf(startAngle)};

    if (pie) {
        pImpl->path.moveTo(cx, cy);
        pImpl->path.lineTo(start.x + cx, start.y + cy);
    } else {
        pImpl->path.moveTo(start.x + cx, start.y + cy);
    }

    for (int i = 0; i < nCurves; ++i) {
        auto endAngle = startAngle + ((i != nCurves - 1) ? float(M_PI_2) * sweepSign : fract);
        Point end = {radius * cosf(endAngle), radius * sinf(endAngle)};

        //variables needed to calculate bezier control points

        //get bezier control points using article:
        //(http://itc.ktu.lt/index.php/ITC/article/view/11812/6479)
        auto ax = start.x;
        auto ay = start.y;
        auto bx = end.x;
        auto by = end.y;
        auto q1 = ax * ax + ay * ay;
        auto q2 = ax * bx + ay * by + q1;
        auto k2 = (4.0f/3.0f) * ((sqrtf(2 * q1 * q2) - q2) / (ax * by - ay * bx));

        start = end; //Next start point is the current end point

        end.x += cx;
        end.y += cy;

        Point ctrl1 = {ax - k2 * ay + cx, ay + k2 * ax + cy};
        Point ctrl2 = {bx + k2 * by + cx, by - k2 * bx + cy};

        pImpl->path.cubicTo(ctrl1.x, ctrl1.y, ctrl2.x, ctrl2.y, end.x, end.y);

        startAngle = endAngle;
    }

    if (pie) pImpl->path.close();

    pImpl->flag |= RenderUpdateFlag::Path;

    return Result::Success;
}


Result Shape::appendRect(float x, float y, float w, float h, float rx, float ry) noexcept
{
    auto halfW = w * 0.5f;
    auto halfH = h * 0.5f;

    //clamping cornerRadius by minimum size
    if (rx > halfW) rx = halfW;
    if (ry > halfH) ry = halfH;

    //rectangle
    if (rx == 0 && ry == 0) {
        pImpl->path.grow(5, 4);
        pImpl->path.moveTo(x, y);
        pImpl->path.lineTo(x + w, y);
        pImpl->path.lineTo(x + w, y + h);
        pImpl->path.lineTo(x, y + h);
        pImpl->path.close();
    //circle
    } else if (mathEqual(rx, halfW) && mathEqual(ry, halfH)) {
        return appendCircle(x + (w * 0.5f), y + (h * 0.5f), rx, ry);
    } else {
        auto hrx = rx * 0.5f;
        auto hry = ry * 0.5f;
        pImpl->path.grow(10, 17);
        pImpl->path.moveTo(x + rx, y);
        pImpl->path.lineTo(x + w - rx, y);
        pImpl->path.cubicTo(x + w - rx + hrx, y, x + w, y + ry - hry, x + w, y + ry);
        pImpl->path.lineTo(x + w, y + h - ry);
        pImpl->path.cubicTo(x + w, y + h - ry + hry, x + w - rx + hrx, y + h, x + w - rx, y + h);
        pImpl->path.lineTo(x + rx, y + h);
        pImpl->path.cubicTo(x + rx - hrx, y + h, x, y + h - ry + hry, x, y + h - ry);
        pImpl->path.lineTo(x, y + ry);
        pImpl->path.cubicTo(x, y + ry - hry, x + rx - hrx, y, x + rx, y);
        pImpl->path.close();
    }

    pImpl->flag |= RenderUpdateFlag::Path;

    return Result::Success;
}


Result Shape::fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a) noexcept
{
    pImpl->color[0] = r;
    pImpl->color[1] = g;
    pImpl->color[2] = b;
    pImpl->color[3] = a;
    pImpl->flag |= RenderUpdateFlag::Color;

    if (pImpl->fill) {
        delete(pImpl->fill);
        pImpl->fill = nullptr;
        pImpl->flag |= RenderUpdateFlag::Gradient;
    }

    return Result::Success;
}


Result Shape::fill(unique_ptr<Fill> f) noexcept
{
    auto p = f.release();
    if (!p) return Result::MemoryCorruption;

    if (pImpl->fill && pImpl->fill != p) delete(pImpl->fill);
    pImpl->fill = p;
    pImpl->flag |= RenderUpdateFlag::Gradient;

    return Result::Success;
}


Result Shape::fillColor(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) const noexcept
{
    if (r) *r = pImpl->color[0];
    if (g) *g = pImpl->color[1];
    if (b) *b = pImpl->color[2];
    if (a) *a = pImpl->color[3];

    return Result::Success;
}

const Fill* Shape::fill() const noexcept
{
    return pImpl->fill;
}


Result Shape::stroke(float width) noexcept
{
    if (!pImpl->strokeWidth(width)) return Result::FailedAllocation;

    return Result::Success;
}


float Shape::strokeWidth() const noexcept
{
    if (!pImpl->stroke) return 0;
    return pImpl->stroke->width;
}


Result Shape::stroke(uint8_t r, uint8_t g, uint8_t b, uint8_t a) noexcept
{
    if (!pImpl->strokeColor(r, g, b, a)) return Result::FailedAllocation;

    return Result::Success;
}


Result Shape::strokeColor(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) const noexcept
{
    if (!pImpl->stroke) return Result::InsufficientCondition;

    if (r) *r = pImpl->stroke->color[0];
    if (g) *g = pImpl->stroke->color[1];
    if (b) *b = pImpl->stroke->color[2];
    if (a) *a = pImpl->stroke->color[3];

    return Result::Success;
}


Result Shape::stroke(unique_ptr<Fill> f) noexcept
{
    return pImpl->strokeFill(move(f));
}


const Fill* Shape::strokeFill() const noexcept
{
    if (!pImpl->stroke) return nullptr;

    return pImpl->stroke->fill;
}


Result Shape::stroke(const float* dashPattern, uint32_t cnt) noexcept
{
    if ((cnt == 1) || (!dashPattern && cnt > 0) || (dashPattern && cnt == 0)) {
        return Result::InvalidArguments;
    }

    for (uint32_t i = 0; i < cnt; i++)
        if (dashPattern[i] < FLT_EPSILON) return Result::InvalidArguments;

    if (!pImpl->strokeDash(dashPattern, cnt)) return Result::FailedAllocation;

    return Result::Success;
}


uint32_t Shape::strokeDash(const float** dashPattern) const noexcept
{
    if (!pImpl->stroke) return 0;

    if (dashPattern) *dashPattern = pImpl->stroke->dashPattern;

    return pImpl->stroke->dashCnt;
}


Result Shape::stroke(StrokeCap cap) noexcept
{
    if (!pImpl->strokeCap(cap)) return Result::FailedAllocation;

    return Result::Success;
}


Result Shape::stroke(StrokeJoin join) noexcept
{
    if (!pImpl->strokeJoin(join)) return Result::FailedAllocation;

    return Result::Success;
}


StrokeCap Shape::strokeCap() const noexcept
{
    if (!pImpl->stroke) return StrokeCap::Square;

    return pImpl->stroke->cap;
}


StrokeJoin Shape::strokeJoin() const noexcept
{
    if (!pImpl->stroke) return StrokeJoin::Bevel;

    return pImpl->stroke->join;
}


Result Shape::fill(FillRule r) noexcept
{
    pImpl->rule = r;

    return Result::Success;
}


FillRule Shape::fillRule() const noexcept
{
    return pImpl->rule;
}