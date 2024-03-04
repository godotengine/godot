/*
 * Copyright (c) 2023 - 2024 the ThorVG project. All rights reserved.

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

#include <cstring>
#include "tvgShape.h"
#include "tvgFill.h"

#ifdef THORVG_TTF_LOADER_SUPPORT
    #include "tvgTtfLoader.h"
#else
    #include "tvgLoader.h"
#endif

struct Text::Impl
{
    FontLoader* loader = nullptr;
    Shape* paint = nullptr;
    char* utf8 = nullptr;
    float fontSize;
    bool italic = false;
    bool changed = false;

    ~Impl()
    {
        free(utf8);
        LoaderMgr::retrieve(loader);
        delete(paint);
    }

    Result fill(uint8_t r, uint8_t g, uint8_t b)
    {
        return paint->fill(r, g, b);
    }

    Result fill(Fill* f)
    {
        return paint->fill(cast<Fill>(f));
    }

    Result text(const char* utf8)
    {
        free(this->utf8);
        if (utf8) this->utf8 = strdup(utf8);
        else this->utf8 = nullptr;
        changed = true;

        return Result::Success;
    }

    Result font(const char* name, float size, const char* style)
    {
        auto loader = LoaderMgr::loader(name);
        if (!loader) return Result::InsufficientCondition;

        //Same resource has been loaded.
        if (this->loader == loader) {
            this->loader->sharing--;  //make it sure the reference counting.
            return Result::Success;
        } else if (this->loader) {
            LoaderMgr::retrieve(this->loader);
        }
        this->loader = static_cast<FontLoader*>(loader);

        if (!paint) paint = Shape::gen().release();

        fontSize = size;
        if (style && strstr(style, "italic")) italic = true;
        changed = true;
        return Result::Success;
    }

    RenderRegion bounds(RenderMethod* renderer)
    {
        if (paint) return P(paint)->bounds(renderer);
        else return {0, 0, 0, 0};
    }

    bool render(RenderMethod* renderer)
    {
        if (paint) return PP(paint)->render(renderer);
        return false;
    }

    bool load()
    {
        if (!loader) return false;

        //reload
        if (changed) {
            loader->request(paint, utf8, italic);
            loader->read();
            changed = false;
        }
        if (paint) {
            loader->resize(paint, fontSize, fontSize);
            return true;
        }
        return false;
    }

    RenderData update(RenderMethod* renderer, const RenderTransform* transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag pFlag, bool clipper)
    {
        if (!load()) return nullptr;

        //transform the gradient coordinates based on the final scaled font.
        if (P(paint)->flag & RenderUpdateFlag::Gradient) {
            auto fill = P(paint)->rs.fill;
            auto scale = 1.0f / loader->scale;
            if (fill->identifier() == TVG_CLASS_ID_LINEAR) {
                P(static_cast<LinearGradient*>(fill))->x1 *= scale;
                P(static_cast<LinearGradient*>(fill))->y1 *= scale;
                P(static_cast<LinearGradient*>(fill))->x2 *= scale;
                P(static_cast<LinearGradient*>(fill))->y2 *= scale;
            } else {
                P(static_cast<RadialGradient*>(fill))->cx *= scale;
                P(static_cast<RadialGradient*>(fill))->cy *= scale;
                P(static_cast<RadialGradient*>(fill))->r *= scale;
                P(static_cast<RadialGradient*>(fill))->fx *= scale;
                P(static_cast<RadialGradient*>(fill))->fy *= scale;
                P(static_cast<RadialGradient*>(fill))->fr *= scale;
            }
        }
        return PP(paint)->update(renderer, transform, clips, opacity, pFlag, clipper);
    }

    bool bounds(float* x, float* y, float* w, float* h, TVG_UNUSED bool stroking)
    {
        if (!load() || !paint) return false;
        paint->bounds(x, y, w, h, true);
        return true;
    }

    Paint* duplicate()
    {
        load();

        auto ret = Text::gen().release();
        auto dup = ret->pImpl;
        if (paint) dup->paint = static_cast<Shape*>(paint->duplicate());

        if (loader) {
            dup->loader = loader;
            ++dup->loader->sharing;
        }

        dup->utf8 = strdup(utf8);
        dup->italic = italic;
        dup->fontSize = fontSize;

        return ret;
    }

    Iterator* iterator()
    {
        return nullptr;
    }
};



#endif //_TVG_TEXT_H
