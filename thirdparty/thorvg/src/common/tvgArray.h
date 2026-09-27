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

#ifndef _TVG_ARRAY_H_
#define _TVG_ARRAY_H_

#include "tvgCommon.h"

#define ARRAY_FOREACH(A, B) \
    for (auto A = (B).begin(); A < (B).end(); ++A)

#define ARRAY_REVERSE_FOREACH(A, B) \
    for (auto A = (B).end() - 1; A >= (B).begin(); --A)

namespace tvg
{

template<class T>
struct Array
{
    T* data = nullptr;
    uint32_t count = 0;
    uint32_t reserved = 0;

    Array() = default;

    Array(int32_t size)
    {
        reserve(size);
    }

    Array(const Array& rhs)
    {
        reset();
        *this = rhs;
    }

    void push(T element)
    {
        if (count + 1 > reserved) {
            reserved = count + (count + 2) / 2;
            data = tvg::realloc<T>(data, sizeof(T) * reserved);
        }
        data[count++] = element;
    }

    void push(const Array<T>& rhs)
    {
        if (rhs.count == 0) return;
        grow(rhs.count);
        memcpy(data + count, rhs.data, rhs.count * sizeof(T));
        count += rhs.count;
    }

    bool reserve(uint32_t size)
    {
        if (size > reserved) {
            reserved = size;
            data = tvg::realloc<T>(data, sizeof(T) * reserved);
        }
        return true;
    }

    bool grow(uint32_t size)
    {
        return reserve(count + size);
    }

    const T& operator[](size_t idx) const
    {
        return data[idx];
    }

    T& operator[](size_t idx)
    {
        return data[idx];
    }

    void operator=(const Array& rhs)
    {
        reserve(rhs.count);
        if (rhs.count > 0) memcpy(data, rhs.data, sizeof(T) * rhs.count);
        count = rhs.count;
    }

    void move(Array& to)
    {
        to.reset();
        to.data = data;
        to.count = count;
        to.reserved = reserved;

        data = nullptr;
        count = reserved = 0;
    }

    const T* begin() const
    {
        return data;
    }

    T* begin()
    {
        return data;
    }

    T* end()
    {
        return data + count;
    }

    const T* end() const
    {
        return data + count;
    }

    const T& last() const
    {
        return data[count - 1];
    }

    const T& first() const
    {
        return data[0];
    }

    T& last()
    {
        return data[count - 1];
    }

    T& next()
    {
        if (full()) grow(count + 1);
        return data[count++];
    }

    T& first()
    {
        return data[0];
    }

    void pop()
    {
        if (count > 0) --count;
    }

    void reset()
    {
        tvg::free(data);
        data = nullptr;
        count = reserved = 0;
    }

    void clear()
    {
        count = 0;
    }

    bool empty() const
    {
        return count == 0;
    }

    bool full()
    {
        return count == reserved;
    }

    ~Array()
    {
        tvg::free(data);
    }
};

}

#endif //_TVG_ARRAY_H_
