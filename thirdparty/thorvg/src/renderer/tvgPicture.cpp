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

#include "tvgPaint.h"
#include "tvgPicture.h"

Picture::Picture() = default;


Picture* Picture::gen() noexcept
{
    return new PictureImpl;
}


Type Picture::type() const noexcept
{
    return Type::Picture;
}


Result Picture::load(const char* filename) noexcept
{
#ifdef THORVG_FILE_IO_SUPPORT
    if (!filename) return Result::InvalidArguments;
    return to<PictureImpl>(this)->load(filename);
#else
    TVGLOG("RENDERER", "FILE IO is disabled!");
    return Result::NonSupport;
#endif
}


Result Picture::load(const char* data, uint32_t size, const char* mimeType, const char* rpath, bool copy) noexcept
{
    return to<PictureImpl>(this)->load(data, size, mimeType, rpath, copy);
}


Result Picture::load(const uint32_t* data, uint32_t w, uint32_t h, ColorSpace cs, bool copy) noexcept
{
    return to<PictureImpl>(this)->load(data, w, h, cs, copy);
}


Result Picture::resolver(std::function<bool(Paint* paint, const char* src, void* data)> func, void* data) noexcept
{
    return to<PictureImpl>(this)->set(func, data);
}


Result Picture::size(float w, float h) noexcept
{
    to<PictureImpl>(this)->size(w, h);
    return Result::Success;
}


Result Picture::size(float* w, float* h) const noexcept
{
    return to<PictureImpl>(this)->size(w, h);
}


Result Picture::origin(float x, float y) noexcept
{
    to<PictureImpl>(this)->origin = {x, y};
    PAINT(this)->mark(RenderUpdateFlag::Transform);
    return Result::Success;
}


Result Picture::origin(float* x, float* y) const noexcept
{
    if (x) *x = to<PictureImpl>(this)->origin.x;
    if (y) *y = to<PictureImpl>(this)->origin.y;
    return Result::Success;
}


const Paint* Picture::paint(uint32_t id) noexcept
{
    struct Value
    {
        uint32_t id;
        const Paint* ret;
    } value = {id, nullptr};

    auto cb = [](const tvg::Paint* paint, void* data) -> bool
    {
        auto p = static_cast<Value*>(data);
        if (p->id == paint->id) {
            p->ret = paint;
            return false;
        }
        return true;
    };

    auto accessor = tvg::Accessor::gen();
    accessor->set(this, cb, &value);
    delete(accessor);

    return value.ret;
}
