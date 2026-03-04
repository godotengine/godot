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

#ifndef _TVG_STR_H_
#define _TVG_STR_H_

#include "tvgCommon.h"

namespace tvg
{

static inline bool equal(const char* a, const char* b)
{
    return !strcmp(a, b) && strlen(a) == strlen(b);
}

char* concat(const char* a, const char* b);
float toFloat(const char *str, char **end);                    //convert to float
char* duplicate(const char *str, size_t n = SIZE_MAX);         //copy the string
char* append(char* lhs, const char* rhs, size_t n);            //append the rhs to the lhs
char* dirname(const char* path);                               //return the full directory name
char* filename(const char* path);                              //return the file name without extension
const char* fileext(const char* path);                         //return the file extension name

}
#endif //_TVG_STR_H_
