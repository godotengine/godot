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

#ifndef _TVG_SHAPE_H_
#define _TVG_SHAPE_H_

#include "tvgCommon.h"
#include "tvgMath.h"
#include "tvgPaint.h"

namespace tvg
{

struct ShapeImpl : Shape
{
    Paint::Impl impl;
    RenderShape rs;
    uint8_t opacity;    //for composition

    ShapeImpl() : impl(Paint::Impl(this))
    {
    }

    bool render(RenderMethod* renderer, TVG_UNUSED CompositionFlag flag)
    {
        if (!impl.rd) return false;

        RenderCompositor* cmp = nullptr;

        renderer->blend(impl.blendMethod);

        if (impl.cmpFlag) {
            cmp = renderer->target(bounds(), renderer->colorSpace(), impl.cmpFlag);
            renderer->beginComposite(cmp, MaskMethod::None, opacity);
        }

        auto ret = renderer->renderShape(impl.rd);
        if (cmp) renderer->endComposite(cmp);
        return ret;
    }

    bool needComposition(uint8_t opacity)
    {
        if (opacity == 0) return false;

        //Shape composition is only necessary when stroking & fill are valid.
        if (!rs.stroke || rs.stroke->width < FLOAT_EPSILON || (!rs.stroke->fill && rs.stroke->color.a == 0)) return false;
        if (!rs.fill && rs.color.a == 0) return false;

        //translucent fill & stroke
        if (opacity < 255) {
            impl.mark(CompositionFlag::Opacity);
            return true;
        }

        //Composition test
        const Paint* target;
        auto method = PAINT(this)->mask(&target);
        if (!target) return false;

        if ((target->pImpl->opacity == 255 || target->pImpl->opacity == 0) && target->type() == tvg::Type::Shape) {
            auto shape = static_cast<const Shape*>(target);
            if (!shape->fill()) {
                uint8_t r, g, b, a;
                shape->fill(&r, &g, &b, &a);
                if (a == 0 || a == 255) {
                    if (method == MaskMethod::Luma || method == MaskMethod::InvLuma) {
                        if ((r == 255 && g == 255 && b == 255) || (r == 0 && g == 0 && b == 0)) return false;
                    } else return false;
                }
            }
        }

        impl.mark(CompositionFlag::Masking);
        return true;
    }

    bool skip(RenderUpdateFlag flag)
    {
        if (flag == RenderUpdateFlag::None) return true;
        return false;
    }

    bool update(RenderMethod* renderer, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flag, bool clipper)
    {
        if (needComposition(opacity)) {
            /* Overriding opacity value. If this scene is half-translucent,
               It must do intermediate composition with that opacity value. */ 
            this->opacity = opacity;
            opacity = 255;
        }

        impl.rd = renderer->prepare(rs, impl.rd, transform, clips, opacity, flag, clipper);
        return true;
    }

    RenderRegion bounds()
    {
        return impl.renderer->region(impl.rd);
    }

    bool bounds(Point* pt4, const Matrix& m, bool obb)
    {
        auto fallback = true;  //TODO: remove this when all backend engines support bounds()

        if (impl.renderer && rs.strokeWidth() > 0.0f) {
            if (impl.renderer->bounds(impl.rd, pt4, obb ? tvg::identity() : m)) {
                fallback = false;
            }
        }
        //Keep this for legacy. loaders still depend on this logic, remove it if possible.
        if (fallback) {
            BBox box = {{FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX}};
            if (!rs.path.bounds(obb ? nullptr : &m, box)) return false;
            if (rs.stroke) {
                //Use geometric mean for feathering.
                //Join, Cap wouldn't be considered. Generate stroke outline and compute bbox for accurate size?
                auto sx = sqrt(m.e11 * m.e11 + m.e21 * m.e21);
                auto sy = sqrt(m.e12 * m.e12 + m.e22 * m.e22);
                auto feather = rs.stroke->width * sqrt(sx * sy);
                box.min.x -= feather * 0.5f;
                box.min.y -= feather * 0.5f;
                box.max.x += feather * 0.5f;
                box.max.y += feather * 0.5f;
            }
            pt4[0] = box.min;
            pt4[1] = {box.max.x, box.min.y};
            pt4[2] = box.max;
            pt4[3] = {box.min.x, box.max.y};
        }

        if (obb) {
            pt4[0] *= m;
            pt4[1] *= m;
            pt4[2] *= m;
            pt4[3] *= m;
        }

        return true;
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
    }

    void strokeWidth(float width)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->width = width;
        impl.mark(RenderUpdateFlag::Stroke);
    }

    void trimpath(const RenderTrimPath& trim)
    {
        if (!rs.stroke) {
            if (trim.begin == 0.0f && trim.end == 1.0f) return;
            rs.stroke = new RenderStroke();
        }

        if (tvg::equal(rs.stroke->trim.begin, trim.begin) && tvg::equal(rs.stroke->trim.end, trim.end) && rs.stroke->trim.simultaneous == trim.simultaneous) return;

        rs.stroke->trim = trim;
        impl.mark(RenderUpdateFlag::Path);
    }

    bool trimpath(float* begin, float* end)
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
        impl.mark(RenderUpdateFlag::Stroke);
    }

    void strokeJoin(StrokeJoin join)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->join = join;
        impl.mark(RenderUpdateFlag::Stroke);
    }

    Result strokeMiterlimit(float miterlimit)
    {
        // https://www.w3.org/TR/SVG2/painting.html#LineJoin
        // - A negative value for stroke-miterlimit must be treated as an illegal value.
        if (miterlimit < 0.0f) return Result::InvalidArguments;
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->miterlimit = miterlimit;
        impl.mark(RenderUpdateFlag::Stroke);

        return Result::Success;
    }

    bool intersects(const RenderRegion& region)
    {
        if (!impl.rd || !impl.renderer) return false;
        return impl.renderer->intersectsShape(impl.rd, region);
    }

    void strokeFill(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        if (rs.stroke->fill) {
            delete(rs.stroke->fill);
            rs.stroke->fill = nullptr;
            impl.mark(RenderUpdateFlag::GradientStroke);
        }

        rs.stroke->color = {r, g, b, a};

        impl.mark(RenderUpdateFlag::Stroke);
    }

    Result strokeFill(Fill* f)
    {
        if (!f) return Result::InvalidArguments;

        if (!rs.stroke) rs.stroke = new RenderStroke();
        if (rs.stroke->fill && rs.stroke->fill != f) delete(rs.stroke->fill);
        rs.stroke->fill = f;
        rs.stroke->color.a = 0;

        impl.mark(RenderUpdateFlag::Stroke | RenderUpdateFlag::GradientStroke);

        return Result::Success;
    }

    Result strokeDash(const float* pattern, uint32_t cnt, float offset)
    {
        if ((!pattern && cnt > 0) || (pattern && cnt == 0)) return Result::InvalidArguments;
        if (!rs.stroke) rs.stroke = new RenderStroke;
        //Reset dash
        auto& dash = rs.stroke->dash;
        if (dash.count != cnt) {
            tvg::free(dash.pattern);
            dash.pattern = nullptr;
        }
        if (cnt > 0) {
            if (!dash.pattern) dash.pattern = tvg::malloc<float>(sizeof(float) * cnt);
            dash.length = 0.0f;
            for (uint32_t i = 0; i < cnt; ++i) {
                dash.pattern[i] = pattern[i] < 0.0f ? 0.0f : pattern[i];
                dash.length += dash.pattern[i];
            }
        }
        rs.stroke->dash.count = cnt;
        rs.stroke->dash.offset = offset;
        impl.mark(RenderUpdateFlag::Stroke);

        return Result::Success;
    }

    bool strokeFirst()
    {
        if (!rs.stroke) return true;
        return rs.stroke->first;
    }

    void strokeFirst(bool first)
    {
        if (!rs.stroke) rs.stroke = new RenderStroke();
        rs.stroke->first = first;
        impl.mark(RenderUpdateFlag::Stroke);
    }

    Result fill(Fill* f)
    {
        if (!f) return Result::InvalidArguments;

        if (rs.fill && rs.fill != f) delete(rs.fill);
        rs.fill = f;
        impl.mark(RenderUpdateFlag::Gradient);

        return Result::Success;
    }

    void fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        if (rs.fill) {
            delete(rs.fill);
            rs.fill = nullptr;
            impl.mark(RenderUpdateFlag::Gradient);
        }

        if (r == rs.color.r && g == rs.color.g && b == rs.color.b && a == rs.color.a) return;

        rs.color = {r, g, b, a};
        impl.mark(RenderUpdateFlag::Color);
    }

    void resetPath()
    {
        rs.path.cmds.clear();
        rs.path.pts.clear();
        impl.mark(RenderUpdateFlag::Path);
    }

    Result addPath(const PathCommand* cmds, uint32_t cmdCnt, const Point* pts, uint32_t ptsCnt)
    {
        if (cmdCnt == 0 || ptsCnt == 0 || !cmds || !pts) return Result::InvalidArguments;

        grow(cmdCnt, ptsCnt);
        append(cmds, cmdCnt, pts, ptsCnt);
        impl.mark(RenderUpdateFlag::Path);

        return Result::Success;
    }

    void addCircle(float cx, float cy, float rx, float ry, bool cw)
    {
        rs.path.addCircle(cx, cy, rx, ry, cw);
        impl.mark(RenderUpdateFlag::Path);
    }

    void addRect(float x, float y, float w, float h, float rx, float ry, bool cw)
    {
        rs.path.addRect(x, y, w, h, rx, ry, cw);
        impl.mark(RenderUpdateFlag::Path);
    }

    Paint* duplicate(Paint* ret)
    {
        auto shape = static_cast<Shape*>(ret);
        if (!shape) shape = Shape::gen();
        auto dup = to<ShapeImpl>(shape);

        //Path
        dup->rs.path.clear();
        dup->rs.path.cmds.push(rs.path.cmds);
        dup->rs.path.pts.push(rs.path.pts);

        //Fill
        delete(dup->rs.fill);
        if (rs.fill) dup->rs.fill = rs.fill->duplicate();
        else dup->rs.fill = nullptr;

        //Stroke
        if (rs.stroke) {
            if (!dup->rs.stroke) dup->rs.stroke = new RenderStroke;
            *dup->rs.stroke = *rs.stroke;
        } else {
            delete(dup->rs.stroke);
            dup->rs.stroke = nullptr;
        }

        dup->rs.color = rs.color;
        dup->rs.rule = rs.rule;

        return shape;
    }

    void reset()
    {
        PAINT(this)->reset();
        rs.path.cmds.clear();
        rs.path.pts.clear();

        rs.color.a = 0;
        rs.rule = FillRule::NonZero;

        delete(rs.stroke);
        rs.stroke = nullptr;

        delete(rs.fill);
        rs.fill = nullptr;
    }

    Iterator* iterator()
    {
        return nullptr;
    }
};

}

#endif //_TVG_SHAPE_H_
