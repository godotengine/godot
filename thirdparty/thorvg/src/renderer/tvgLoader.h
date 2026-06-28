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

#ifndef _TVG_LOADER_H_
#define _TVG_LOADER_H_

#include "tvgLoadModule.h"

namespace tvg
{

struct LoaderMgr
{
    static bool init();
    static bool term();
    static LoadModule* loader(const char* filename, bool* invalid);
    static LoadModule* loader(const char* data, uint32_t size, const char* mimeType, const char* rpath, bool copy);
    static LoadModule* loader(const uint32_t* data, uint32_t w, uint32_t h, ColorSpace cs, bool copy);
    static LoadModule* loader(const char* name, const char* data, uint32_t size, const char* mimeType, bool copy);
    static LoadModule* font(const char* name);
    static LoadModule* anyfont();
    static bool retrieve(const char* filename);
    static bool retrieve(LoadModule* loader);
};

}

#endif //_TVG_LOADER_H_
