/*
 * Copyright (c) 2023 - 2026 ThorVG project. All rights reserved.

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

#ifndef _TVG_TEXT_H
#define _TVG_TEXT_H

#include "tvgStr.h"
#include "tvgMath.h"
#include "tvgShape.h"
#include "tvgFill.h"
#include "tvgLoader.h"

namespace tvg
{

struct TextImpl : Text
{
    Paint::Impl impl;
    Shape* shape;   //text shape
    FontLoader* loader = nullptr;
    FontMetrics fm;
    char* utf8 = nullptr;
    float outlineWidth = 0.0f;
    float italicShear = 0.0f;
    bool updated = false;

    TextImpl() : impl(Paint::Impl(this)), shape(Shape::gen())
    {
        PAINT(shape)->parent = this;
        shape->strokeJoin(StrokeJoin::Round);
    }

    ~TextImpl()
    {
        tvg::free(utf8);
        if (loader) {
            loader->release(fm);
            LoaderMgr::retrieve(loader);
        }
        Paint::rel(shape);
    }

    Result text(const char* utf8)
    {
        tvg::free(this->utf8);
        if (utf8) this->utf8 = tvg::duplicate(utf8);
        else this->utf8 = nullptr;
        updated = true;
        impl.mark(RenderUpdateFlag::Path);

        return Result::Success;
    }

    Result font(const char* name)
    {
        auto loader = static_cast<FontLoader*>(name ? LoaderMgr::font(name) : LoaderMgr::anyfont());
        if (!loader) return Result::InsufficientCondition;

        //Same resource has been loaded.
        if (this->loader == loader) {
            this->loader->sharing--;  //make it sure the reference counting.
            return Result::Success;
        } else if (this->loader) {
            this->loader->release(fm);
            LoaderMgr::retrieve(this->loader);
        }
        this->loader = loader;
        updated = true;

        return Result::Success;
    }

    Result size(float fontSize)
    {
        if (fontSize > 0.0f) {
            if (fm.fontSize != fontSize) {
                fm.fontSize = fontSize;
                updated = true;
            }
            return Result::Success;
        }
        return Result::InvalidArguments;
    }

    RenderRegion bounds()
    {
        if (!load()) return {};
        return to<ShapeImpl>(shape)->bounds();
    }

    bool render(RenderMethod* renderer)
    {
        if (!loader || !fm.engine) return true;
        renderer->blend(impl.blendMethod);
        return PAINT(shape)->render(renderer);
    }

    bool load()
    {
        if (!loader) return false;
        if (updated) {
            if (loader->get(fm, utf8, to<ShapeImpl>(shape)->rs.path)) {
                loader->transform(shape, fm, italicShear);
            }
            updated = false;
        }
        return true;
    }

    Result metrics(TextMetrics& metrics)
    {
        if (!loader || fm.fontSize <= 0.0f) return Result::InsufficientCondition;
        loader->metrics(fm, metrics);
        return Result::Success;
    }

    bool skip(RenderUpdateFlag flag)
    {
        if (flag == RenderUpdateFlag::None) return true;
        return false;
    }

    void wrapping(TextWrap mode)
    {
        if (fm.wrap == mode) return;
        fm.wrap = mode;
        updated = true;
        impl.mark(RenderUpdateFlag::Path);
    }

    void layout(float w, float h)
    {
        fm.box = {w, h};
        updated = true;
    }

    Result spacing(float letter, float line)
    {
        if (letter < 0.0f || line < 0.0f) return Result::InvalidArguments;

        fm.spacing = {letter, line};
        updated = true;

        return Result::Success;
    }

    bool update(RenderMethod* renderer, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flag, TVG_UNUSED bool clipper)
    {
        if (!load()) return true;

        auto scale = fm.scale;

        //transform the gradient coordinates based on the final scaled font.
        auto fill = to<ShapeImpl>(shape)->rs.fill;
        if (fill && to<ShapeImpl>(shape)->impl.marked(RenderUpdateFlag::Gradient)) {
            if (fill->type() == Type::LinearGradient) {
                LINEAR(fill)->p1 *= scale;
                LINEAR(fill)->p2 *= scale;
            } else {
                RADIAL(fill)->center *= scale;
                RADIAL(fill)->r *= scale;
                RADIAL(fill)->focal *= scale;
                RADIAL(fill)->fr *= scale;
            }
        }

        if (outlineWidth > 0.0f && impl.marked(RenderUpdateFlag::Stroke)) shape->strokeWidth(outlineWidth * scale);

        PAINT(shape)->update(renderer, transform, clips, opacity, flag, false);
        return true;
    }

    bool intersects(const RenderRegion& region)
    {
        if (!load()) return false;
        return to<ShapeImpl>(shape)->intersects(region);
    }

    bool bounds(Point* pt4, const Matrix& m, bool obb)
    {
        if (!load()) return true;
        return PAINT(shape)->bounds(pt4, &const_cast<Matrix&>(m), obb);
    }

    Paint* duplicate(Paint* ret)
    {
        if (ret) TVGERR("RENDERER", "TODO: duplicate()");

        load();

        auto text = Text::gen();
        auto dup = to<TextImpl>(text);

        to<ShapeImpl>(shape)->duplicate(dup->shape);

        if (loader) {
            dup->loader = loader;
            ++dup->loader->sharing;
            loader->copy(fm, dup->fm);
        }

        dup->utf8 = tvg::duplicate(utf8);
        dup->italicShear = italicShear;
        dup->outlineWidth = outlineWidth;
        dup->updated = true;

        return text;
    }

    Iterator* iterator()
    {
        return nullptr;
    }
};

}

#endif //_TVG_TEXT_H
