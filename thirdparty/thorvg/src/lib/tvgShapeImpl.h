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
#ifndef _TVG_SHAPE_IMPL_H_
#define _TVG_SHAPE_IMPL_H_

#include <memory.h>
#include "tvgPaint.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct ShapeStroke
{
    float width;
    uint8_t color[4];
    Fill *fill;
    float* dashPattern;
    uint32_t dashCnt;
    StrokeCap cap;
    StrokeJoin join;

    void copy(const ShapeStroke* src)
    {
       width = src->width;
       dashCnt = src->dashCnt;
       cap = src->cap;
       join = src->join;

        memcpy(color, src->color, sizeof(color));
        if (dashCnt > 0) {
            dashPattern = static_cast<float*>(malloc(sizeof(float) * dashCnt));
            memcpy(dashPattern, src->dashPattern, sizeof(float) * dashCnt);
        }
        if (src->fill) fill = src->fill->duplicate();
    }

    void clear()
    {
        if (dashPattern) free(dashPattern);
        if (fill) delete(fill);
    }
};


struct ShapePath
{
    PathCommand* cmds = nullptr;
    uint32_t cmdCnt = 0;
    uint32_t reservedCmdCnt = 0;

    Point *pts = nullptr;
    uint32_t ptsCnt = 0;
    uint32_t reservedPtsCnt = 0;

    ~ShapePath()
    {
        if (cmds) free(cmds);
        if (pts) free(pts);
    }

    ShapePath()
    {
    }

    void duplicate(const ShapePath* src)
    {
        if (src->cmdCnt == 0 || src->ptsCnt == 0) return;

        cmdCnt = src->cmdCnt;
        reservedCmdCnt = src->reservedCmdCnt;
        ptsCnt = src->ptsCnt;
        reservedPtsCnt = src->reservedPtsCnt;

        cmds = static_cast<PathCommand*>(malloc(sizeof(PathCommand) * reservedCmdCnt));
        if (!cmds) return;
        memcpy(cmds, src->cmds, sizeof(PathCommand) * cmdCnt);

        pts = static_cast<Point*>(malloc(sizeof(Point) * reservedPtsCnt));
        if (!pts) {
            free(cmds);
            return;
        }
        memcpy(pts, src->pts, sizeof(Point) * ptsCnt);
    }

    void reserveCmd(uint32_t cmdCnt)
    {
        if (cmdCnt <= reservedCmdCnt) return;
        reservedCmdCnt = cmdCnt;
        cmds = static_cast<PathCommand*>(realloc(cmds, sizeof(PathCommand) * reservedCmdCnt));
    }

    void reservePts(uint32_t ptsCnt)
    {
        if (ptsCnt <= reservedPtsCnt) return;
        reservedPtsCnt = ptsCnt;
        pts = static_cast<Point*>(realloc(pts, sizeof(Point) * reservedPtsCnt));
    }

    void grow(uint32_t cmdCnt, uint32_t ptsCnt)
    {
        reserveCmd(this->cmdCnt + cmdCnt);
        reservePts(this->ptsCnt + ptsCnt);
    }

    void reset()
    {
        cmdCnt = 0;
        ptsCnt = 0;
    }

    void append(const PathCommand* cmds, uint32_t cmdCnt, const Point* pts, uint32_t ptsCnt)
    {
        memcpy(this->cmds + this->cmdCnt, cmds, sizeof(PathCommand) * cmdCnt);
        memcpy(this->pts + this->ptsCnt, pts, sizeof(Point) * ptsCnt);
        this->cmdCnt += cmdCnt;
        this->ptsCnt += ptsCnt;
    }

    void moveTo(float x, float y)
    {
        if (cmdCnt + 1 > reservedCmdCnt) reserveCmd((cmdCnt + 1) * 2);
        if (ptsCnt + 2 > reservedPtsCnt) reservePts((ptsCnt + 2) * 2);

        cmds[cmdCnt++] = PathCommand::MoveTo;
        pts[ptsCnt++] = {x, y};
    }

    void lineTo(float x, float y)
    {
        if (cmdCnt + 1 > reservedCmdCnt) reserveCmd((cmdCnt + 1) * 2);
        if (ptsCnt + 2 > reservedPtsCnt) reservePts((ptsCnt + 2) * 2);

        cmds[cmdCnt++] = PathCommand::LineTo;
        pts[ptsCnt++] = {x, y};
    }

    void cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y)
    {
        if (cmdCnt + 1 > reservedCmdCnt) reserveCmd((cmdCnt + 1) * 2);
        if (ptsCnt + 3 > reservedPtsCnt) reservePts((ptsCnt + 3) * 2);

        cmds[cmdCnt++] = PathCommand::CubicTo;
        pts[ptsCnt++] = {cx1, cy1};
        pts[ptsCnt++] = {cx2, cy2};
        pts[ptsCnt++] = {x, y};
    }

    void close()
    {
        if (cmdCnt > 0 && cmds[cmdCnt - 1] == PathCommand::Close) return;

        if (cmdCnt + 1 > reservedCmdCnt) reserveCmd((cmdCnt + 1) * 2);
        cmds[cmdCnt++] = PathCommand::Close;
    }

    bool bounds(float* x, float* y, float* w, float* h) const
    {
        if (ptsCnt == 0) return false;

        Point min = { pts[0].x, pts[0].y };
        Point max = { pts[0].x, pts[0].y };

        for (uint32_t i = 1; i < ptsCnt; ++i) {
            if (pts[i].x < min.x) min.x = pts[i].x;
            if (pts[i].y < min.y) min.y = pts[i].y;
            if (pts[i].x > max.x) max.x = pts[i].x;
            if (pts[i].y > max.y) max.y = pts[i].y;
        }

        if (x) *x = min.x;
        if (y) *y = min.y;
        if (w) *w = max.x - min.x;
        if (h) *h = max.y - min.y;

        return true;
    }
};


struct Shape::Impl
{
    ShapePath path;
    Fill *fill = nullptr;
    ShapeStroke *stroke = nullptr;
    uint8_t color[4] = {0, 0, 0, 0};    //r, g, b, a
    FillRule rule = FillRule::Winding;
    RenderData rdata = nullptr;         //engine data
    Shape *shape = nullptr;
    uint32_t flag = RenderUpdateFlag::None;

    Impl(Shape* s) : shape(s)
    {
    }

    ~Impl()
    {
        if (fill) delete(fill);
        if (stroke) {
            stroke->clear();
            free (stroke);
        }
    }

    bool dispose(RenderMethod& renderer)
    {
        auto ret = renderer.dispose(rdata);
        rdata = nullptr;
        return ret;
    }

    bool render(RenderMethod& renderer)
    {
        return renderer.renderShape(rdata);
    }

    void* update(RenderMethod& renderer, const RenderTransform* transform, uint32_t opacity, Array<RenderData>& clips, RenderUpdateFlag pFlag)
    {
        this->rdata = renderer.prepare(*shape, this->rdata, transform, opacity, clips, static_cast<RenderUpdateFlag>(pFlag | flag));
        flag = RenderUpdateFlag::None;
        return this->rdata;
    }

    RenderRegion bounds(RenderMethod& renderer)
    {
        return renderer.region(rdata);
    }

    bool bounds(float* x, float* y, float* w, float* h)
    {
        auto ret = path.bounds(x, y, w, h);

        //Stroke feathering
        if (stroke) {
            if (x) *x -= stroke->width * 0.5f;
            if (y) *y -= stroke->width * 0.5f;
            if (w) *w += stroke->width;
            if (h) *h += stroke->width;
        }
        return ret;
    }

    bool strokeWidth(float width)
    {
        //TODO: Size Exception?

        if (!stroke) stroke = static_cast<ShapeStroke*>(calloc(sizeof(ShapeStroke), 1));
        stroke->width = width;
        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    bool strokeCap(StrokeCap cap)
    {
        if (!stroke) stroke = static_cast<ShapeStroke*>(calloc(sizeof(ShapeStroke), 1));
        stroke->cap = cap;
        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    bool strokeJoin(StrokeJoin join)
    {
        if (!stroke) stroke = static_cast<ShapeStroke*>(calloc(sizeof(ShapeStroke), 1));
        stroke->join = join;
        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    bool strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        if (!stroke) stroke = static_cast<ShapeStroke*>(calloc(sizeof(ShapeStroke), 1));
        if (stroke->fill) {
            delete(stroke->fill);
            stroke->fill = nullptr;
            flag |= RenderUpdateFlag::GradientStroke;
        }

        stroke->color[0] = r;
        stroke->color[1] = g;
        stroke->color[2] = b;
        stroke->color[3] = a;

        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    Result strokeFill(unique_ptr<Fill> f)
    {
        auto p = f.release();
        if (!p) return Result::MemoryCorruption;

        if (!stroke) stroke = static_cast<ShapeStroke*>(calloc(sizeof(ShapeStroke), 1));
        if (stroke->fill && stroke->fill != p) delete(stroke->fill);
        stroke->fill = p;

        flag |= RenderUpdateFlag::Stroke;
        flag |= RenderUpdateFlag::GradientStroke;

        return Result::Success;
    }

    bool strokeDash(const float* pattern, uint32_t cnt)
    {
        //Reset dash
        if (!pattern && cnt == 0) {
            free(stroke->dashPattern);
            stroke->dashPattern = nullptr;
        } else {
            if (!stroke) stroke = static_cast<ShapeStroke*>(calloc(sizeof(ShapeStroke), 1));
            if (stroke->dashCnt != cnt) {
                free(stroke->dashPattern);
                stroke->dashPattern = nullptr;
            }
            if (!stroke->dashPattern) {
                stroke->dashPattern = static_cast<float*>(malloc(sizeof(float) * cnt));
                if (!stroke->dashPattern) return false;
            }
            for (uint32_t i = 0; i < cnt; ++i) {
                stroke->dashPattern[i] = pattern[i];
            }
        }
        stroke->dashCnt = cnt;
        flag |= RenderUpdateFlag::Stroke;

        return true;
    }

    Paint* duplicate()
    {
        auto ret = Shape::gen();

        auto dup = ret.get()->pImpl;
        dup->rule = rule;

        //Color
        memcpy(dup->color, color, sizeof(color));
        dup->flag = RenderUpdateFlag::Color;

        //Path
        dup->path.duplicate(&path);
        dup->flag |= RenderUpdateFlag::Path;

        //Stroke
        if (stroke) {
            dup->stroke = static_cast<ShapeStroke*>(calloc(sizeof(ShapeStroke), 1));
            dup->stroke->copy(stroke);
            dup->flag |= RenderUpdateFlag::Stroke;

            if (stroke->fill)
                dup->flag |= RenderUpdateFlag::GradientStroke;
        }

        //Fill
        if (fill) {
            dup->fill = fill->duplicate();
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
