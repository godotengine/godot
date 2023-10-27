/*
 * Copyright (c) 2023 the ThorVG project. All rights reserved.

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

#ifndef _TVG_LIST_H_
#define _TVG_LIST_H_

namespace tvg {

template<typename T>
struct LinkedList
{
    T *head = nullptr;
    T *tail = nullptr;

    LinkedList() = default;
    LinkedList(T *head, T *tail) : head(head), tail(tail)
    {
    }

    template<T *T::*Prev, T *T::*Next>
    static void insert(T *t, T *prev, T *next, T **head, T **tail)
    {
        t->*Prev = prev;
        t->*Next = next;

        if (prev) {
            prev->*Next = t;
        } else if (head) {
            *head = t;
        }

        if (next) {
            next->*Prev = t;
        } else if (tail) {
            *tail = t;
        }
    }

    template<T *T::*Prev, T *T::*Next>
    static void remove(T *t, T **head, T **tail)
    {
        if (t->*Prev) {
            t->*Prev->*Next = t->*Next;
        } else if (head) {
            *head = t->*Next;
        }

        if (t->*Next) {
            t->*Next->*Prev = t->*Prev;
        } else if (tail) {
            *tail = t->*Prev;
        }

        t->*Prev = t->*Next = nullptr;
    }

    template <T* T::*Next>
    static bool contains(T *t, T **head, T **tail) {
        for (T *it = *head; it; it = it->*Next) {
            if (it == t) {
                return true;
            }
        }

        return false;
    }
};

}

#endif // _TVG_LIST_H_
