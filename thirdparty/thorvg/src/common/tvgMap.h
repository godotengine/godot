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

#ifndef _TVG_MAP_H_
#define _TVG_MAP_H_

#include "tvgCommon.h"
#include "tvgInlist.h"

namespace tvg
{

template<typename K, typename V>
struct Map
{
    struct Item
    {
        INLIST_ITEM(Item);
        K key;
        V val;

        Item(const K& key) : key(key), val{} {}
    };

    Item* find(const K& key)
    {
        auto& bucket = buckets[hash(key)];
        INLIST_FOREACH(bucket, item) {
            if (item->key == key) return item;
        }
        return nullptr;
    }

    void remove(const K& key)
    {
        auto& bucket = buckets[hash(key)];
        INLIST_FOREACH(bucket, item) {
            if (item->key == key) {
                bucket.remove(item);
                delete (item);
                return;
            }
        }
    }

    void clear()
    {
        for (size_t i = 0; i < size; ++i) {
            buckets[i].free();
        }
    }

    V& operator[](const K& key)
    {
        if (auto item = find(key)) return item->val;

        auto& bucket = buckets[hash(key)];
        auto item = new Item(key);
        bucket.back(item);
        return item->val;
    }

    Map(size_t size = 32)
    {
        if (size < 1) size = 1;
        buckets = new Inlist<Item>[size];
        this->size = size;
    }

    ~Map()
    {
        delete[] buckets;
    }

    Map(const Map&) = delete;
    Map& operator=(const Map&) = delete;

    Inlist<Item>* buckets = nullptr;
    size_t size = 0;

private:
    uintptr_t shuffle(uintptr_t x) const
    {
        x ^= x >> (sizeof(uintptr_t) * 4);
        x *= 0x9e3779b1u;
        x ^= x >> (sizeof(uintptr_t) * 4);
        return x;
    }

    template<typename T>
    uintptr_t hash(T v) const
    {
        return shuffle(static_cast<uintptr_t>(v)) % size;
    }

    template<typename T>
    uintptr_t hash(T* p) const
    {
        return shuffle(reinterpret_cast<uintptr_t>(p)) % size;
    }
};

}  // namespace tvg

#endif  //_TVG_MAP_H_
