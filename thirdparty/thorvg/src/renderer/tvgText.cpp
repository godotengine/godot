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


#include "tvgText.h"


Text::Text() = default;


Result Text::text(const char* text) noexcept
{
    return to<TextImpl>(this)->text(text);
}


Result Text::font(const char* name) noexcept
{
    return to<TextImpl>(this)->font(name);
}


Result Text::size(float size) noexcept
{
    return to<TextImpl>(this)->size(size);
}


Result Text::load(const char* filename) noexcept
{
#ifdef THORVG_FILE_IO_SUPPORT
    bool invalid; //invalid path
    auto loader = LoaderMgr::loader(filename, &invalid);
    if (loader) {
        if (loader->sharing > 0) --loader->sharing;   //font loading doesn't mean sharing.
        return Result::Success;
    } else {
        if (invalid) return Result::InvalidArguments;
        else return Result::NonSupport;
    }
#else
    TVGLOG("RENDERER", "FILE IO is disabled!");
    return Result::NonSupport;
#endif
}


Result Text::load(const char* name, const char* data, uint32_t size, const char* mimeType, bool copy) noexcept
{
    if (!name || (size == 0 && data)) return Result::InvalidArguments;

    //unload font
    if (!data) {
        if (LoaderMgr::retrieve(LoaderMgr::font(name))) return Result::Success;
        return Result::InsufficientCondition;
    }

    if (!LoaderMgr::loader(name, data, size, mimeType, copy)) return Result::NonSupport;
    return Result::Success;
}


Result Text::unload(const char* filename) noexcept
{
#ifdef THORVG_FILE_IO_SUPPORT
    if (LoaderMgr::retrieve(filename)) return Result::Success;
    return Result::InsufficientCondition;
#else
    TVGLOG("RENDERER", "FILE IO is disabled!");
    return Result::NonSupport;
#endif
}


Result Text::align(float x, float y) noexcept
{
    to<TextImpl>(this)->fm.align = {x, y};
    PAINT(this)->mark(RenderUpdateFlag::Transform);
    return Result::Success;
}


Result Text::layout(float w, float h) noexcept
{
    to<TextImpl>(this)->layout(w, h);
    return Result::Success;
}


Result Text::fill(uint8_t r, uint8_t g, uint8_t b) noexcept
{
    return to<TextImpl>(this)->shape->fill(r, g, b);
}


Result Text::outline(float width, uint8_t r, uint8_t g, uint8_t b) noexcept
{
    to<TextImpl>(this)->outlineWidth = width;
    to<TextImpl>(this)->shape->strokeFill(r, g, b);
    PAINT(this)->mark(RenderUpdateFlag::Stroke);
    return Result::Success;
}


Result Text::fill(Fill* f) noexcept
{
    return to<TextImpl>(this)->shape->fill(f);
}


Result Text::italic(float shear) noexcept
{
    if (shear < 0.0f) shear = 0.0f;
    else if (shear > 0.5f) shear = 0.5f;
    to<TextImpl>(this)->italicShear = shear;
    to<TextImpl>(this)->updated = true;
    return Result::Success;
}


Result Text::spacing(float letter, float line) noexcept
{
    return to<TextImpl>(this)->spacing(letter, line);
}


Result Text::wrap(TextWrap mode) noexcept
{
    to<TextImpl>(this)->wrapping(mode);
    return Result::Success;
}


Result Text::metrics(TextMetrics& metrics) const noexcept
{
    return to<TextImpl>(this)->metrics(metrics);
}


Text* Text::gen() noexcept
{
    return new TextImpl;
}


Type Text::type() const noexcept
{
    return Type::Text;
}
