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

#ifndef _TVG_SHAPE_H_
#define _TVG_SHAPE_H_

#include <memory.h>
#include "tvgMath.h"
#include "tvgPaint.h"


struct Shape::Impl
{
    RenderShape rs;                     //shape data
    RenderData rd = nullptr;            //engine data
    Shape* shape;
    uint8_t flag = RenderUpdateFlag::None;
    uint8_t opacity;                    //for composition
    bool needComp = false;              //composite or not

    Impl(Shape* s) : shape(s)
    {
    }

    ~Impl()
    {
        if (auto renderer = PP(shape)->renderer) {
            renderer->dispose(rd);
        }
    }

    bool render(RenderMethod* renderer)
    {
        Compositor* cmp = nullptr;
        bool ret;

        if (needComp) {
            cmp = renderer->target(bounds(renderer), renderer->colorSpace());
            renderer->beginComposite(cmp, CompositeMethod::None, opacity);
        }
        ret = renderer->renderShape(rd);
        if (cmp) renderer->endComposite(cmp);
        return ret;
    }

    bool needComposition(uint8_t opacity)
    {
        if (opacity == 0) return false;

        //Shape composition is only necessary when stroking & fill are valid.
        if (!rs.stroke || rs.stroke->width < FLOAT_EPSILON || (!rs.stroke->fill && rs.stroke->color[3] == 0)) return false;
        if (!rs.fill && rs.color[3] == 0) return false;

        //translucent fill & stroke
        if (opacity < 255) return true;

        //Composition test
        const Paint* target;
        auto method = shape->composite(&target);
        if (!target || method == CompositeMethod::ClipPath) return false;
        if (target->pImpl->opacity == 255 || target->pImpl->opacity == 0) {
            if (target->identifier() == TVG_CLASS_ID_SHAPE) {
                auto shape = static_cast<const Shape*>(target);
                if (!shape->fill()) {
                    uint8_t r, g, b, a;
                    shape->fillColor(&r, &g, &b, &a);
                    if (a == 0 || a == 255) {
                        if (method == CompositeMethod::LumaMask || method == CompositeMethod::InvLumaMask) {
                            if ((r == 255 && g == 255 && b == 255) || (r == 0 && g == 0 && b == 0)) return false;
                        } else return false;
                    }
                }
            }
        }

        return true;
    }

    RenderData update(RenderMethod* renderer, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag pFlag, bool clipper)
    {     
        if ((needComp = needComposition(opacity))) {
            /* Overriding opacity value. If this scene is half-translucent,
               It must do intermeidate composition with that opacity value. */ 
            this->opacity = opacity;
            opacity = 255;
        }

        rd = renderer->prepare(rs, rd, transform, clips, opacity, static_cast<RenderUpdateFlag>(pFlag | flag), clipper);
        flag = RenderUpdateFlag::None;
        return rd;
    }

    RenderRegion bounds(RenderMethod* renderer)
    {
        return renderer->region(rd);
    }

    bool bounds(float* x, float* y, float* w, float* h, bool stroking)
    {
        //Path bounding size
        if (rs.path.pts.count > 0 ) {
            auto pts = rs.path.pts.begin();
            Point min = { pts->x, pts->y };
            Point max = { pts->x, pts->y };

            for (auto pts2 = pts + 1; pts2 < rs.path.pts.end(); ++pts2) {
                if (pts2->x < min.x) min.x = pts2->x;
                if (pts2->y < min.y) min.y = pts2->y;
                if (pts2->x > max.x) max.x = pts2->x;
                if (pts2->y > max.y) max.y = pts2->y;
            }

            if (x) *x = min.x;
            if (y) *y = min.y;
            if (w) *w = max.x - min.x;
            if (h) *h = max.y - min.y;
        }

        //Stroke feathering
        if (stroking && rs.stroke) {
            if (x) *x -= rs.stroke->width * 0.5f;
            if (y) *y -= rs.stroke->width * 0.5f;
            if (w) *w += rs.stroke->width;
            if (h) *h += rs.stroke->width;
        }
        return rs.path.pts.count > 0 ? true : false;
    }

    void reserveCmd(uint32_t cmdCnt)
    {
        rs.path.cmds.reserve(cmdCnt);
    }

    void reservePts(uint32_t ptsCnt)
    {
        rs.path.pts.reserve(ptsCnt);
    }

    void grow(uint32_t cmdCnt, uint32_t ptsCnt)
    {
        rs.path.cmds.grow(cmdCnt);
        rs.path.pts.grow(ptsCnt);
    }

    void append(const PathCommand* cmds, uint32_t cmdCnt, const Point* pts, uint32_t ptsCnt)
    {
        memcpy(rs.path.cmds.end(), cmds, sizeof(PathCommand) * cmdCnt);
        memcpy(rs.path.pts.end(), pts, sizeof(Point) * ptsCnt);
        rs.path.cmds.count += cmdCnt;
        rs.path.pts.count += ptsCnt;

        flag |= RenderUpdateFlag::Path;
    }

    void moveTo(float x, float y)
    {
        rs.path.cmds.push(PathCommand::MoveTo);
        rs.path.pts.push({x, y});

        flag |= RenderUpdateFlag::Path;
    }

    void lineTo(float x, float y)
    {
        rs.path.cmds.push(PathCommand::LineTo);
        rs.path.pts.push({x, y});

        flag |= RenderUpdateFlag::Path;
    }

    void cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y)
    {
        rs.path.cmds.push(PathCommand::CubicTo);
        rs.path.pts.push({cx1, cy1});
        rs.path.pts.push({cx2, cy2});
        rs.path.pts.push({x, y});

        flag |= RenderUpdateFlag::Path;
    }

    void close()
    {
        //Don't close multiple times.
        if (rs.path.cmds.count > 0 && rs.path.cmds.last() == PathCommand::Close) return;

        rs.path.cmds.push(PathCommand::Close);

        flag |= RenderUpdateFlag::Path;
    }

    void strokeWidth(float width)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->width = width;
        flag |= RenderUpdateFlag::Stroke;
    }

    void strokeTrim(float begin, float end, bool simultaneous)
    {
        if (!rs.stroke) {
            if (begin == 0.0f && end == 1.0f) return;
            rs.stroke = new RenderStroke();
        }

        if (mathEqual(rs.stroke->trim.begin, begin) && mathEqual(rs.stroke->trim.end, end) &&
            rs.stroke->trim.simultaneous == simultaneous) return;

        auto loop = true;

        if (begin > 1.0f && end > 1.0f) loop = false;
        if (begin < 0.0f && end < 0.0f) loop = false;
        if (begin >= 0.0f && begin <= 1.0f && end >= 0.0f  && end <= 1.0f) loop = false;

        if (begin > 1.0f) begin -= 1.0f;
        if (begin < 0.0f) begin += 1.0f;
        if (end > 1.0f) end -= 1.0f;
        if (end < 0.0f) end += 1.0f;

        if ((loop && begin < end) || (!loop && begin > end)) {
            auto tmp = begin;
            begin = end;
            end = tmp;
        }

        rs.stroke->trim.begin = begin;
        rs.stroke->trim.end = end;
        rs.stroke->trim.simultaneous = simultaneous;
        flag |= RenderUpdateFlag::Stroke;
    }

    bool strokeTrim(float* begin, float* end)
    {
        if (rs.stroke) {
            if (begin) *begin = rs.stroke->trim.begin;
            if (end) *end = rs.stroke->trim.end;
            return rs.stroke->trim.simultaneous;
        } else {
            if (begin) *begin = 0.0f;
            if (end) *end = 1.0f;
            return false;
        }
    }

    void strokeCap(StrokeCap cap)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->cap = cap;
        flag |= RenderUpdateFlag::Stroke;
    }

    void strokeJoin(StrokeJoin join)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->join = join;
        flag |= RenderUpdateFlag::Stroke;
    }

    void strokeMiterlimit(float miterlimit)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->miterlimit = miterlimit;
        flag |= RenderUpdateFlag::Stroke;
    }

    void strokeColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
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

    Result strokeDash(const float* pattern, uint32_t cnt, float offset)
    {
        if ((cnt == 1) || (!pattern && cnt > 0) || (pattern && cnt == 0)) {
            return Result::InvalidArguments;
        }

        for (uint32_t i = 0; i < cnt; i++) {
            if (pattern[i] < FLOAT_EPSILON) return Result::InvalidArguments;
        }

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
                if (!rs.stroke->dashPattern) return Result::FailedAllocation;
            }
            for (uint32_t i = 0; i < cnt; ++i) {
                rs.stroke->dashPattern[i] = pattern[i];
            }
        }
        rs.stroke->dashCnt = cnt;
        rs.stroke->dashOffset = offset;
        flag |= RenderUpdateFlag::Stroke;

        return Result::Success;
    }

    bool strokeFirst()
    {
        if (!rs.stroke) return true;
        return rs.stroke->strokeFirst;
    }

    void strokeFirst(bool strokeFirst)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->strokeFirst = strokeFirst;
        flag |= RenderUpdateFlag::Stroke;
    }

    void update(RenderUpdateFlag flag)
    {
        this->flag |= flag;
    }

    Paint* duplicate()
    {
        auto ret = Shape::gen().release();
        auto dup = ret->pImpl;

        dup->rs.rule = rs.rule;

        //Color
        memcpy(dup->rs.color, rs.color, sizeof(rs.color));
        dup->flag = RenderUpdateFlag::Color;

        //Path
        if (rs.path.cmds.count > 0 && rs.path.pts.count > 0) {
            dup->rs.path.cmds = rs.path.cmds;
            dup->rs.path.pts = rs.path.pts;
            dup->flag |= RenderUpdateFlag::Path;
        }

        //Stroke
        if (rs.stroke) {
            dup->rs.stroke = new RenderStroke();
            *dup->rs.stroke = *rs.stroke;
            memcpy(dup->rs.stroke->color, rs.stroke->color, sizeof(rs.stroke->color));
            if (rs.stroke->dashCnt > 0) {
                dup->rs.stroke->dashPattern = static_cast<float*>(malloc(sizeof(float) * rs.stroke->dashCnt));
                memcpy(dup->rs.stroke->dashPattern, rs.stroke->dashPattern, sizeof(float) * rs.stroke->dashCnt);
            }
            if (rs.stroke->fill) {
                dup->rs.stroke->fill = rs.stroke->fill->duplicate();
                dup->flag |= RenderUpdateFlag::GradientStroke;
            }
            dup->flag |= RenderUpdateFlag::Stroke;
        }

        //Fill
        if (rs.fill) {
            dup->rs.fill = rs.fill->duplicate();
            dup->flag |= RenderUpdateFlag::Gradient;
        }

        return ret;
    }

    Iterator* iterator()
    {
        return nullptr;
    }
};

#endif //_TVG_SHAPE_H_
