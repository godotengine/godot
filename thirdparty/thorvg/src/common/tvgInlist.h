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

#ifndef _TVG_INLIST_H_
#define _TVG_INLIST_H_

namespace tvg {

//NOTE: declare this in your list item
#define INLIST_ITEM(T) \
    T* prev; \
    T* next

template<typename T>
struct Inlist
{
    T* head = nullptr;
    T* tail = nullptr;

    void free()
    {
        while (head) {
            auto t = head;
            head = t->next;
            delete(t);
        }
        head = tail = nullptr;
    }

    void back(T* element)
    {
        if (tail) {
            tail->next = element;
            element->prev = tail;
            element->next = nullptr;
            tail = element;
        } else {
            head = tail = element;
            element->prev = nullptr;
            element->next = nullptr;
        }
    }

    void front(T* element)
    {
        if (head) {
            head->prev = element;
            element->prev = nullptr;
            element->next = head;
            head = element;
        } else {
            head = tail = element;
            element->prev = nullptr;
            element->next = nullptr;
        }
    }

    T* back()
    {
        if (!tail) return nullptr;
        auto t = tail;
        tail = t->prev;
        if (!tail) head = nullptr;
        return t;
    }

    T* front()
    {
        if (!head) return nullptr;
        auto t = head;
        head = t->next;
        if (!head) tail = nullptr;
        return t;
    }

    void remove(T* element)
    {
        if (element->prev) element->prev->next = element->next;
        if (element->next) element->next->prev = element->prev;
        if (element == head) head = element->next;
        if (element == tail) tail = element->prev;
    }

    bool empty() const
    {
        return head ? false : true;
    }
};

}

#endif // _TVG_INLIST_H_
