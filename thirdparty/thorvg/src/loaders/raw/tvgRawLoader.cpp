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

#include <fstream>
#include <string.h>
#include "tvgLoader.h"
#include "tvgRawLoader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

RawLoader::RawLoader() : ImageLoader(FileType::Raw)
{
}


RawLoader::~RawLoader()
{
    if (copy) free(surface.buf32);
}


bool RawLoader::open(const uint32_t* data, uint32_t w, uint32_t h, bool copy)
{
    if (!LoadModule::read()) return true;

    if (!data || w == 0 || h == 0) return false;

    this->w = (float)w;
    this->h = (float)h;
    this->copy = copy;

    if (copy) {
        surface.buf32 = (uint32_t*)malloc(sizeof(uint32_t) * w * h);
        if (!surface.buf32) return false;
        memcpy((void*)surface.buf32, data, sizeof(uint32_t) * w * h);
    }
    else surface.buf32 = const_cast<uint32_t*>(data);

    //setup the surface
    surface.stride = w;
    surface.w = w;
    surface.h = h;
    surface.cs = ColorSpace::ARGB8888;
    surface.channelSize = sizeof(uint32_t);
    surface.premultiplied = true;

    return true;
}


bool RawLoader::read()
{
    LoadModule::read();

    return true;
}
