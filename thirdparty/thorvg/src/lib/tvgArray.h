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
#ifndef _TVG_ARRAY_H_
#define _TVG_ARRAY_H_

#include <memory.h>

namespace tvg
{

template<class T>
struct Array
{
    T* data = nullptr;
    uint32_t count = 0;
    uint32_t reserved = 0;

    void push(T element)
    {
        if (count + 1 > reserved) {
            reserved = (count + 1) * 2;
            auto p  = data;
            data = static_cast<T*>(realloc(data, sizeof(T) * reserved));
            if (!data) {
                data = p;
                return;
            }
        }
        data[count++] = element;
    }

    bool reserve(uint32_t size)
    {
        if (size > reserved) {
            reserved = size;
            auto p = data;
            data = static_cast<T*>(realloc(data, sizeof(T) * reserved));
            if (!data) {
                data = p;
                return false;
            }
        }
        return true;
    }

    bool grow(uint32_t size)
    {
        return reserve(count + size);
    }

    T* ptr()
    {
        return data + count;
    }

    void pop()
    {
        if (count > 0) --count;
    }

    void reset()
    {
        if (data) {
            free(data);
            data = nullptr;
        }
        count = reserved = 0;
    }

    void clear()
    {
        count = 0;
    }

    void operator=(const Array& rhs)
    {
        reserve(rhs.count);
        if (rhs.count > 0) memcpy(data, rhs.data, sizeof(T) * reserved);
        count = rhs.count;
    }

    ~Array()
    {
        if (data) free(data);
    }
};

}

#endif //_TVG_ARRAY_H_
