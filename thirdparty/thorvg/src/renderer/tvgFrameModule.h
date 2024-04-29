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

#ifndef _TVG_FRAME_MODULE_H_
#define _TVG_FRAME_MODULE_H_

#include "tvgLoadModule.h"

namespace tvg
{

class FrameModule: public ImageLoader
{
public:
    FrameModule(FileType type) : ImageLoader(type) {}
    virtual ~FrameModule() {}

    virtual bool frame(float no) = 0;       //set the current frame number
    virtual float totalFrame() = 0;         //return the total frame count
    virtual float curFrame() = 0;           //return the current frame number
    virtual float duration() = 0;           //return the animation duration in seconds

    virtual bool animatable() override { return true; }
};

}

#endif //_TVG_FRAME_MODULE_H_
