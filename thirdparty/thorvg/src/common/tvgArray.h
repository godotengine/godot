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

#ifndef _TVG_ARRAY_H_
#define _TVG_ARRAY_H_

#include <memory.h>
#include <cstdint>
#include <cstdlib>

namespace tvg
{

template<class T>
struct Array
{
    T* data = nullptr;
    uint32_t count = 0;
    uint32_t reserved = 0;

    Array(){}

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
            data = static_cast<T*>(realloc(data, sizeof(T) * reserved));
        }
        data[count++] = element;
    }

    void push(Array<T>& rhs)
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
            data = static_cast<T*>(realloc(data, sizeof(T) * reserved));
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
        free(data);
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

    template<class COMPARE>
    void sort()
    {
        qsort<COMPARE>(data, 0, static_cast<int32_t>(count) - 1);
    }

    void operator=(const Array& rhs)
    {
        reserve(rhs.count);
        if (rhs.count > 0) memcpy(data, rhs.data, sizeof(T) * rhs.count);
        count = rhs.count;
    }

    ~Array()
    {
        free(data);
    }

private:
    template<class COMPARE>
    void qsort(T* arr, int32_t low, int32_t high)
    {
        if (low < high) {
            int32_t i = low;
            int32_t j = high;
            T tmp = arr[low];
            while (i < j) {
                while (i < j && !COMPARE{}(arr[j], tmp)) --j;
                if (i < j) {
                    arr[i] = arr[j];
                    ++i;
                }
                while (i < j && COMPARE{}(arr[i], tmp)) ++i;
                if (i < j) {
                    arr[j] = arr[i];
                    --j;
                }
            }
            arr[i] = tmp;
            qsort<COMPARE>(arr, low, i - 1);
            qsort<COMPARE>(arr, i + 1, high);
        }
    }
};

}

#endif //_TVG_ARRAY_H_
