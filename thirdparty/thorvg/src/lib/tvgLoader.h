/*
 * Copyright (c) 2020-2021 Samsung Electronics Co., Ltd. All rights reserved.

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
#ifndef _TVG_LOADER_H_
#define _TVG_LOADER_H_

#include "tvgCommon.h"

namespace tvg
{

class Loader
{
public:
    //default view box, if any.
    float vx = 0;
    float vy = 0;
    float vw = 0;
    float vh = 0;
    float w = 0, h = 0;         //default image size
    bool preserveAspect = true; //keep aspect ratio by default.

    virtual ~Loader() {}

    virtual bool open(const string& path) { /* Not supported */ return false; };
    virtual bool open(const char* data, uint32_t size, bool copy) { /* Not supported */ return false; };
    virtual bool open(const uint32_t* data, uint32_t w, uint32_t h, bool copy) { /* Not supported */ return false; };
    virtual bool read() = 0;
    virtual bool close() = 0;
    virtual const uint32_t* pixels() { return nullptr; };
    virtual unique_ptr<Scene> scene() { return nullptr; };
};

}

#endif //_TVG_LOADER_H_
