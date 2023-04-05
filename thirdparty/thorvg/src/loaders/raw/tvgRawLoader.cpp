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

#include <fstream>
#include <string.h>
#include "tvgLoader.h"
#include "tvgRawLoader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static inline uint32_t CHANGE_COLORSPACE(uint32_t c)
{
    return (c & 0xff000000) + ((c & 0x00ff0000)>>16) + (c & 0x0000ff00) + ((c & 0x000000ff)<<16);
}


static void _changeColorSpace(uint32_t* data, uint32_t w, uint32_t h)
{
    auto buffer = data;
    for (uint32_t y = 0; y < h; ++y, buffer += w) {
        auto src = buffer;
        for (uint32_t x = 0; x < w; ++x, ++src) {
            *src = CHANGE_COLORSPACE(*src);
        }
    }
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

RawLoader::~RawLoader()
{
    if (copy && content) {
        free((void*)content);
        content = nullptr;
    }
}


bool RawLoader::open(const uint32_t* data, uint32_t w, uint32_t h, bool copy)
{
    if (!data || w == 0 || h == 0) return false;

    this->w = (float)w;
    this->h = (float)h;
    this->copy = copy;

    if (copy) {
        content = (uint32_t*)malloc(sizeof(uint32_t) * w * h);
        if (!content) return false;
        memcpy((void*)content, data, sizeof(uint32_t) * w * h);
    }
    else content = const_cast<uint32_t*>(data);

    return true;
}


bool RawLoader::read()
{
    return true;
}


bool RawLoader::close()
{
    return true;
}


unique_ptr<Surface> RawLoader::bitmap(uint32_t colorSpace)
{
    if (!content) return nullptr;
    if (this->colorSpace != colorSpace) {
        this->colorSpace = colorSpace;
        _changeColorSpace(content, w, h);
    }

    auto surface = static_cast<Surface*>(malloc(sizeof(Surface)));
    surface->buffer = content;
    surface->stride = static_cast<uint32_t>(w);
    surface->w = static_cast<uint32_t>(w);
    surface->h = static_cast<uint32_t>(h);
    surface->cs = colorSpace;

    return unique_ptr<Surface>(surface);
}
