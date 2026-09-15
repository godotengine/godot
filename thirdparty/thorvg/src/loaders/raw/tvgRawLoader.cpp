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

#include "tvgLoader.h"
#include "tvgRawLoader.h"

RawLoader::~RawLoader()
{
    if (owner != Ownership::Borrow) tvg::free(surface.buf32);
}

bool RawLoader::open(const uint32_t* data, uint32_t w, uint32_t h, ColorSpace cs, Ownership owner)
{
    if (!Loader::read()) return true;

    if (!data || w == 0 || h == 0) return false;

    this->w = (float)w;
    this->h = (float)h;
    this->owner = owner;

    if (owner == Ownership::Copy) {
        surface.buf32 = tvg::malloc<uint32_t>(sizeof(uint32_t) * w * h);
        memcpy((void*)surface.buf32, data, sizeof(uint32_t) * w * h);
    } else {
        surface.buf32 = const_cast<uint32_t*>(data);
    }
    surface.setup(surface.buf32, w, w, h, sizeof(uint32_t), cs);
    return true;
}


bool RawLoader::read()
{
    Loader::read();

    return true;
}
