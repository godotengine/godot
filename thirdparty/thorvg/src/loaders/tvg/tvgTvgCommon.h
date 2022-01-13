/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All rights reserved.

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

#ifndef _TVG_TVG_COMMON_H_
#define _TVG_TVG_COMMON_H_

#include "tvgCommon.h"
#include "tvgBinaryDesc.h"

#define SIZE(A) sizeof(A)
#define READ_UI32(dst, src) memcpy(dst, (src), sizeof(uint32_t))
#define READ_FLOAT(dst, src) memcpy(dst, (src), sizeof(float))


/* Interface for Tvg Binary Interpreter */
class TvgBinInterpreterBase
{
public:
    virtual ~TvgBinInterpreterBase() {}

    /* ptr: points the tvg binary body (after header)
       end: end of the tvg binary data */
    virtual unique_ptr<Scene> run(const char* ptr, const char* end) = 0;
};


/* Version 0 */
class TvgBinInterpreter : public TvgBinInterpreterBase
{
public:
    unique_ptr<Scene> run(const char* ptr, const char* end) override;
};


#endif  //_TVG_TVG_COMMON_H_