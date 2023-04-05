/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

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

#ifndef _TVG_SHAPE_IMPL_H_
#define _TVG_SHAPE_IMPL_H_

#include <memory.h>
#include "tvgPaint.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct Shape::Impl
{
    RenderShape rs;                     //shape data
    RenderData rd = nullptr;            //engine data
    uint32_t flag = RenderUpdateFlag::None;

    bool dispose(RenderMethod& renderer)
    {
        auto ret = renderer.dispose(rd);
        rd = nullptr;
        return ret;
    }

    bool render(RenderMethod& renderer)
    {
        return renderer.renderShape(rd);
    }

    void* update(RenderMethod& renderer, const RenderTransform* transform, uint32_t opacity, Array<RenderData>& clips, RenderUpdateFlag pFlag, bool clipper)
    {
        rd = renderer.prepare(rs, rd, transform, opacity, clips, static_cast<RenderUpdateFlag>(pFlag | flag), clipper);
        flag = RenderUpdateFlag::None;
        return rd;
    }

    RenderRegion bounds(RenderMethod& renderer)
    {
        return renderer.region(rd);
    }

    bool bounds(float* x, float* y, float* w, float* h)
    {
        //Path bounding size
        if (rs.path.ptsCnt > 0 ) {
            Point min = { rs.path.pts[0].x, rs.path.pts[0].y };
            Point max = { rs.path.pts[0].x, rs.path.pts[0].y };

            for (uint32_t i = 1; i < rs.path.ptsCnt; ++i) {
                if (rs.path.pts[i].x < min.x) min.x = rs.path.pts[i].x;
                if (rs.path.pts[i].y < min.y) min.y = rs.path.pts[i].y;
                if (rs.path.pts[i].x > max.x) max.x = rs.path.pts[i].x;
                if (rs.path.pts[i].y > max.y) max.y = rs.path.pts[i].y;
            }

            if (x) *x = min.x;
            if (y) *y = min.y;
            if (w) *w = max.x - min.x;
            if (h) *h = max.y - min.y;
        }

        //Stroke feathering
        if (rs.stroke) {
            if (x) *x -= rs.stroke->width * 0.5f;
            if (y) *y -= rs.stroke->width * 0.5f;
            if (w) *w += rs.stroke->width;
            if (h) *h += rs.stroke->width;
        }
        return rs.path.ptsCnt > 0 ? true : false;
    }

    void reserveCmd(uint32_t cmdCnt)
    {
        if (cmdCnt <= rs.path.reservedCmdCnt) return;
        rs.path.reservedCmdCnt = cmdCnt;
        rs.path.cmds = static_cast<PathCommand*>(realloc(rs.path.cmds, sizeof(PathCommand) * rs.path.reservedCmdCnt));
    }

    void reservePts(uint32_t ptsCnt)
    {
        if (ptsCnt <= rs.path.reservedPtsCnt) return;
        rs.path.reservedPtsCnt = ptsCnt;
        rs.path.pts = static_cast<Point*>(realloc(rs.path.pts, sizeof(Point) * rs.path.reservedPtsCnt));
    }

    void grow(uint32_t cmdCnt, uint32_t ptsCnt)
    {
        reserveCmd(rs.path.cmdCnt + cmdCnt);
        reservePts(rs.path.ptsCnt + ptsCnt);
    }

    void reset()
    {
        rs.path.cmdCnt = 0;
        rs.path.ptsCnt = 0;

        flag = RenderUpdateFlag::Path;
    }

    void append(const PathCommand* cmds, uint32_t cmdCnt, const Point* pts, uint32_t ptsCnt)
    {
        memcpy(rs.path.cmds + rs.path.cmdCnt, cmds, sizeof(PathCommand) * cmdCnt);
        memcpy(rs.path.pts + rs.path.ptsCnt, pts, sizeof(Point) * ptsCnt);
        rs.path.cmdCnt += cmdCnt;
        rs.path.ptsCnt += ptsCnt;

        flag |= RenderUpdateFlag::Path;
    }

    void moveTo(float x, float y)
    {
        if (rs.path.cmdCnt + 1 > rs.path.reservedCmdCnt) reserveCmd((rs.path.cmdCnt + 1) * 2);
        if (rs.path.ptsCnt + 2 > rs.path.reservedPtsCnt) reservePts((rs.path.ptsCnt + 2) * 2);

        rs.path.cmds[rs.path.cmdCnt++] = PathCommand::MoveTo;
        rs.path.pts[rs.path.ptsCnt++] = {x, y};

        flag |= RenderUpdateFlag::Path;
    }

    void lineTo(float x, float y)
    {
        if (rs.path.cmdCnt + 1 > rs.path.reservedCmdCnt) reserveCmd((rs.path.cmdCnt + 1) * 2);
        if (rs.path.ptsCnt + 2 > rs.path.reservedPtsCnt) reservePts((rs.path.ptsCnt + 2) * 2);

        rs.path.cmds[rs.path.cmdCnt++] = PathCommand::LineTo;
        rs.path.pts[rs.path.ptsCnt++] = {x, y};

        flag |= RenderUpdateFlag::Path;
    }

    void cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y)
    {
        if (rs.path.cmdCnt + 1 > rs.path.reservedCmdCnt) reserveCmd((rs.path.cmdCnt + 1) * 2);
        if (rs.path.ptsCnt + 3 > rs.path.reservedPtsCnt) reservePts((rs.path.ptsCnt + 3) * 2);

        rs.path.cmds[rs.path.cmdCnt++] = PathCommand::CubicTo;
        rs.path.pts[rs.path.ptsCnt++] = {cx1, cy1};
        rs.path.pts[rs.path.ptsCnt++] = {cx2, cy2};
        rs.path.pts[rs.path.ptsCnt++] = {x, y};

        flag |= RenderUpdateFlag::Path;
    }

    void close()
    {
        if (rs.path.cmdCnt > 0 && rs.path.cmds[rs.path.cmdCnt - 1] == PathCommand::Close) return;

        if (rs.path.cmdCnt + 1 > rs.path.reservedCmdCnt) reserveCmd((rs.path.cmdCnt + 1) * 2);
        rs.path.cmds[rs.path.cmdCnt++] = PathCommand::Close;

        flag |= RenderUpdateFlag::Path;
    }

    bool strokeWidth(float width)
    {
        //TODO: Size Exception?

        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->width = width;
        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    bool strokeCap(StrokeCap cap)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->cap = cap;
        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    bool strokeJoin(StrokeJoin join)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->join = join;
        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    bool strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        if (rs.stroke->fill) {
            delete(rs.stroke->fill);
            rs.stroke->fill = nullptr;
            flag |= RenderUpdateFlag::GradientStroke;
        }

        rs.stroke->color[0] = r;
        rs.stroke->color[1] = g;
        rs.stroke->color[2] = b;
        rs.stroke->color[3] = a;

        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    Result strokeFill(unique_ptr<Fill> f)
    {
        auto p = f.release();
        if (!p) return Result::MemoryCorruption;

        if (!rs.stroke) rs.stroke = new RenderStroke();
        if (rs.stroke->fill && rs.stroke->fill != p) delete(rs.stroke->fill);
        rs.stroke->fill = p;

        flag |= RenderUpdateFlag::Stroke;
        flag |= RenderUpdateFlag::GradientStroke;

        return Result::Success;
    }

    bool strokeDash(const float* pattern, uint32_t cnt)
    {
        //Reset dash
        if (!pattern && cnt == 0) {
            free(rs.stroke->dashPattern);
            rs.stroke->dashPattern = nullptr;
        } else {
            if (!rs.stroke) rs.stroke = new RenderStroke();
            if (rs.stroke->dashCnt != cnt) {
                free(rs.stroke->dashPattern);
                rs.stroke->dashPattern = nullptr;
            }
            if (!rs.stroke->dashPattern) {
                rs.stroke->dashPattern = static_cast<float*>(malloc(sizeof(float) * cnt));
                if (!rs.stroke->dashPattern) return false;
            }
            for (uint32_t i = 0; i < cnt; ++i) {
                rs.stroke->dashPattern[i] = pattern[i];
            }
        }
        rs.stroke->dashCnt = cnt;
        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    Paint* duplicate()
    {
        auto ret = Shape::gen();

        auto dup = ret.get()->pImpl;
        dup->rs.rule = rs.rule;

        //Color
        memcpy(dup->rs.color, rs.color, sizeof(rs.color));
        dup->flag = RenderUpdateFlag::Color;

        //Path
        if (rs.path.cmdCnt > 0 && rs.path.ptsCnt > 0) {
            dup->rs.path.cmdCnt = rs.path.cmdCnt;
            dup->rs.path.reservedCmdCnt = rs.path.reservedCmdCnt;
            dup->rs.path.ptsCnt = rs.path.ptsCnt;
            dup->rs.path.reservedPtsCnt = rs.path.reservedPtsCnt;

            dup->rs.path.cmds = static_cast<PathCommand*>(malloc(sizeof(PathCommand) * dup->rs.path.reservedCmdCnt));
            if (dup->rs.path.cmds) memcpy(dup->rs.path.cmds, rs.path.cmds, sizeof(PathCommand) * dup->rs.path.cmdCnt);

            dup->rs.path.pts = static_cast<Point*>(malloc(sizeof(Point) * dup->rs.path.reservedPtsCnt));
            if (dup->rs.path.pts) memcpy(dup->rs.path.pts, rs.path.pts, sizeof(Point) * dup->rs.path.ptsCnt);
        }
        dup->flag |= RenderUpdateFlag::Path;

        //Stroke
        if (rs.stroke) {
            dup->rs.stroke = new RenderStroke();
            dup->rs.stroke->width = rs.stroke->width;
            dup->rs.stroke->dashCnt = rs.stroke->dashCnt;
            dup->rs.stroke->cap = rs.stroke->cap;
            dup->rs.stroke->join = rs.stroke->join;
            memcpy(dup->rs.stroke->color, rs.stroke->color, sizeof(rs.stroke->color));

            if (rs.stroke->dashCnt > 0) {
                dup->rs.stroke->dashPattern = static_cast<float*>(malloc(sizeof(float) * rs.stroke->dashCnt));
                memcpy(dup->rs.stroke->dashPattern, rs.stroke->dashPattern, sizeof(float) * rs.stroke->dashCnt);
            }

            dup->flag |= RenderUpdateFlag::Stroke;

            if (rs.stroke->fill) {
                dup->rs.stroke->fill = rs.stroke->fill->duplicate();
                dup->flag |= RenderUpdateFlag::GradientStroke;
            }
        }

        //Fill
        if (rs.fill) {
            dup->rs.fill = rs.fill->duplicate();
            dup->flag |= RenderUpdateFlag::Gradient;
        }

        return ret.release();
    }

    Iterator* iterator()
    {
        return nullptr;
    }
};

#endif //_TVG_SHAPE_IMPL_H_
