/*
 * Copyright (c) 2026 ThorVG project. All rights reserved.

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

#ifndef _TVG_ALLOCATOR_H_
#define _TVG_ALLOCATOR_H_

#include <cstdlib>
#include <cstddef>

//separate memory alloators for clean customization
namespace tvg
{
    template<typename T = void>
    static inline T* malloc(size_t size)
    {
        return static_cast<T*>(std::malloc(size));
    }

    template<typename T = void>
    static inline T* calloc(size_t nmem, size_t size)
    {
        return static_cast<T*>(std::calloc(nmem, size));
    }

    template<typename T = void>
    static inline T* realloc(T* ptr, size_t size)
    {
        return static_cast<T*>(std::realloc(ptr, size));
    }

    template<typename T = void>
    static inline void free(T* ptr)
    {
        std::free(ptr);
    }
}

#endif //_TVG_ALLOCATOR_H_
