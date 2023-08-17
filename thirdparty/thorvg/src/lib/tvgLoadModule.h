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

#ifndef _TVG_LOAD_MODULE_H_
#define _TVG_LOAD_MODULE_H_

#include "tvgRender.h"

namespace tvg
{

class LoadModule
{
public:
    float w = 0, h = 0;                             //default image size
    ColorSpace cs = ColorSpace::Unsupported;        //must be clarified at open()

    virtual ~LoadModule() {}

    virtual bool open(const string& path) { return false; }
    virtual bool open(const char* data, uint32_t size, bool copy) { return false; }
    virtual bool open(const uint32_t* data, uint32_t w, uint32_t h, bool copy) { return false; }

    //Override this if the vector-format has own resizing policy.
    virtual bool resize(Paint* paint, float w, float h) { return false; }

    virtual bool animatable() { return false; }  //true if this loader supports animation.
    virtual void sync() {};  //finish immediately if any async update jobs.

    virtual bool read() = 0;
    virtual bool close() = 0;

    virtual unique_ptr<Surface> bitmap() { return nullptr; }
    virtual unique_ptr<Paint> paint() { return nullptr; }
};

}

#endif //_TVG_LOAD_MODULE_H_
