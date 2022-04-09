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

#include "tvgPictureImpl.h"

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Picture::Picture() : pImpl(new Impl)
{
    Paint::pImpl->id = TVG_CLASS_ID_PICTURE;
    Paint::pImpl->method(new PaintMethod<Picture::Impl>(pImpl));
}


Picture::~Picture()
{
    delete(pImpl);
}


unique_ptr<Picture> Picture::gen() noexcept
{
    return unique_ptr<Picture>(new Picture);
}


uint32_t Picture::identifier() noexcept
{
    return TVG_CLASS_ID_PICTURE;
}


Result Picture::load(const std::string& path) noexcept
{
    if (path.empty()) return Result::InvalidArguments;

    return pImpl->load(path);
}


Result Picture::load(const char* data, uint32_t size, const string& mimeType, bool copy) noexcept
{
    if (!data || size <= 0) return Result::InvalidArguments;

    return pImpl->load(data, size, mimeType, copy);
}


TVG_DEPRECATED Result Picture::load(const char* data, uint32_t size, bool copy) noexcept
{
    return load(data, size, "", copy);
}


Result Picture::load(uint32_t* data, uint32_t w, uint32_t h, bool copy) noexcept
{
    if (!data || w <= 0 || h <= 0) return Result::InvalidArguments;

    return pImpl->load(data, w, h, copy);
}


Result Picture::viewbox(float* x, float* y, float* w, float* h) const noexcept
{
    if (pImpl->viewbox(x, y, w, h)) return Result::Success;
    return Result::InsufficientCondition;
}


Result Picture::size(float w, float h) noexcept
{
    if (pImpl->size(w, h)) return Result::Success;
    return Result::InsufficientCondition;
}


Result Picture::size(float* w, float* h) const noexcept
{
    if (!pImpl->loader) return Result::InsufficientCondition;
    if (w) *w = pImpl->w;
    if (h) *h = pImpl->h;
    return Result::Success;
}


const uint32_t* Picture::data(uint32_t* w, uint32_t* h) const noexcept
{
    //Try it, If not loaded yet.
    pImpl->reload();

    if (pImpl->loader) {
        if (w) *w = static_cast<uint32_t>(pImpl->loader->w);
        if (h) *h = static_cast<uint32_t>(pImpl->loader->h);
    } else {
        if (w) *w = 0;
        if (h) *h = 0;
    }
    if (pImpl->surface) return pImpl->surface->buffer;
    else return nullptr;
}