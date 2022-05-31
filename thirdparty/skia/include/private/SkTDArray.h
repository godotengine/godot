/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */


#ifndef SkTDArray_DEFINED
#define SkTDArray_DEFINED

#include "include/core/SkTypes.h"
#include "include/private/SkMalloc.h"
#include "include/private/SkTo.h"

#include <algorithm>
#include <initializer_list>
#include <utility>

/** SkTDArray<T> implements a std::vector-like array for raw data-only objects that do not require
    construction or destruction. The constructor and destructor for T will not be called; T objects
    will always be moved via raw memcpy. Newly created T objects will contain uninitialized memory.

    In most cases, std::vector<T> can provide a similar level of performance for POD objects when
    used with appropriate care. In new code, consider std::vector<T> instead.
*/
template <typename T> class SkTDArray {
public:
    SkTDArray() : fArray(nullptr), fReserve(0), fCount(0) {}
    SkTDArray(const T src[], int count) {
        SkASSERT(src || count == 0);

        fReserve = fCount = 0;
        fArray = nullptr;
        if (count) {
            fArray = (T*)sk_malloc_throw(SkToSizeT(count) * sizeof(T));
            memcpy(fArray, src, sizeof(T) * SkToSizeT(count));
            fReserve = fCount = count;
        }
    }
    SkTDArray(const std::initializer_list<T>& list) : SkTDArray(list.begin(), list.size()) {}
    SkTDArray(const SkTDArray<T>& src) : fArray(nullptr), fReserve(0), fCount(0) {
        SkTDArray<T> tmp(src.fArray, src.fCount);
        this->swap(tmp);
    }
    SkTDArray(SkTDArray<T>&& src) : fArray(nullptr), fReserve(0), fCount(0) {
        this->swap(src);
    }
    ~SkTDArray() {
        sk_free(fArray);
    }

    SkTDArray<T>& operator=(const SkTDArray<T>& src) {
        if (this != &src) {
            if (src.fCount > fReserve) {
                SkTDArray<T> tmp(src.fArray, src.fCount);
                this->swap(tmp);
            } else {
                sk_careful_memcpy(fArray, src.fArray, sizeof(T) * SkToSizeT(src.fCount));
                fCount = src.fCount;
            }
        }
        return *this;
    }
    SkTDArray<T>& operator=(SkTDArray<T>&& src) {
        if (this != &src) {
            this->swap(src);
            src.reset();
        }
        return *this;
    }

    friend bool operator==(const SkTDArray<T>& a, const SkTDArray<T>& b) {
        return  a.fCount == b.fCount &&
                (a.fCount == 0 ||
                 !memcmp(a.fArray, b.fArray, SkToSizeT(a.fCount) * sizeof(T)));
    }
    friend bool operator!=(const SkTDArray<T>& a, const SkTDArray<T>& b) {
        return !(a == b);
    }

    void swap(SkTDArray<T>& that) {
        using std::swap;
        swap(fArray, that.fArray);
        swap(fReserve, that.fReserve);
        swap(fCount, that.fCount);
    }

    bool isEmpty() const { return fCount == 0; }
    bool empty() const { return this->isEmpty(); }

    /**
     *  Return the number of elements in the array
     */
    int count() const { return fCount; }
    size_t size() const { return fCount; }

    /**
     *  Return the total number of elements allocated.
     *  reserved() - count() gives you the number of elements you can add
     *  without causing an allocation.
     */
    int reserved() const { return fReserve; }

    /**
     *  return the number of bytes in the array: count * sizeof(T)
     */
    size_t bytes() const { return fCount * sizeof(T); }

    T*        begin() { return fArray; }
    const T*  begin() const { return fArray; }
    T*        end() { return fArray ? fArray + fCount : nullptr; }
    const T*  end() const { return fArray ? fArray + fCount : nullptr; }

    T&  operator[](int index) {
        SkASSERT(index < fCount);
        return fArray[index];
    }
    const T&  operator[](int index) const {
        SkASSERT(index < fCount);
        return fArray[index];
    }

    T&  getAt(int index)  {
        return (*this)[index];
    }

    const T& back() const { SkASSERT(fCount > 0); return fArray[fCount-1]; }
          T& back()       { SkASSERT(fCount > 0); return fArray[fCount-1]; }

    void reset() {
        if (fArray) {
            sk_free(fArray);
            fArray = nullptr;
            fReserve = fCount = 0;
        } else {
            SkASSERT(fReserve == 0 && fCount == 0);
        }
    }

    void rewind() {
        // same as setCount(0)
        fCount = 0;
    }

    /**
     *  Sets the number of elements in the array.
     *  If the array does not have space for count elements, it will increase
     *  the storage allocated to some amount greater than that required.
     *  It will never shrink the storage.
     */
    void setCount(int count) {
        SkASSERT(count >= 0);
        if (count > fReserve) {
            this->resizeStorageToAtLeast(count);
        }
        fCount = count;
    }

    void setReserve(int reserve) {
        SkASSERT(reserve >= 0);
        if (reserve > fReserve) {
            this->resizeStorageToAtLeast(reserve);
        }
    }
    void reserve(size_t n) {
        SkASSERT_RELEASE(SkTFitsIn<int>(n));
        this->setReserve(SkToInt(n));
    }

    T* prepend() {
        this->adjustCount(1);
        memmove(fArray + 1, fArray, (fCount - 1) * sizeof(T));
        return fArray;
    }

    T* append() {
        return this->append(1, nullptr);
    }
    T* append(int count, const T* src = nullptr) {
        int oldCount = fCount;
        if (count)  {
            SkASSERT(src == nullptr || fArray == nullptr ||
                    src + count <= fArray || fArray + oldCount <= src);

            this->adjustCount(count);
            if (src) {
                memcpy(fArray + oldCount, src, sizeof(T) * count);
            }
        }
        return fArray + oldCount;
    }

    T* insert(int index) {
        return this->insert(index, 1, nullptr);
    }
    T* insert(int index, int count, const T* src = nullptr) {
        SkASSERT(count);
        SkASSERT(index <= fCount);
        size_t oldCount = fCount;
        this->adjustCount(count);
        T* dst = fArray + index;
        memmove(dst + count, dst, sizeof(T) * (oldCount - index));
        if (src) {
            memcpy(dst, src, sizeof(T) * count);
        }
        return dst;
    }

    void remove(int index, int count = 1) {
        SkASSERT(index + count <= fCount);
        fCount = fCount - count;
        memmove(fArray + index, fArray + index + count, sizeof(T) * (fCount - index));
    }

    void removeShuffle(int index) {
        SkASSERT(index < fCount);
        int newCount = fCount - 1;
        fCount = newCount;
        if (index != newCount) {
            memcpy(fArray + index, fArray + newCount, sizeof(T));
        }
    }

    int find(const T& elem) const {
        const T* iter = fArray;
        const T* stop = fArray + fCount;

        for (; iter < stop; iter++) {
            if (*iter == elem) {
                return SkToInt(iter - fArray);
            }
        }
        return -1;
    }

    int rfind(const T& elem) const {
        const T* iter = fArray + fCount;
        const T* stop = fArray;

        while (iter > stop) {
            if (*--iter == elem) {
                return SkToInt(iter - stop);
            }
        }
        return -1;
    }

    /**
     * Returns true iff the array contains this element.
     */
    bool contains(const T& elem) const {
        return (this->find(elem) >= 0);
    }

    /**
     * Copies up to max elements into dst. The number of items copied is
     * capped by count - index. The actual number copied is returned.
     */
    int copyRange(T* dst, int index, int max) const {
        SkASSERT(max >= 0);
        SkASSERT(!max || dst);
        if (index >= fCount) {
            return 0;
        }
        int count = std::min(max, fCount - index);
        memcpy(dst, fArray + index, sizeof(T) * count);
        return count;
    }

    void copy(T* dst) const {
        this->copyRange(dst, 0, fCount);
    }

    // routines to treat the array like a stack
    void push_back(const T& v) { *this->append() = v; }
    T*      push() { return this->append(); }
    const T& top() const { return (*this)[fCount - 1]; }
    T&       top() { return (*this)[fCount - 1]; }
    void     pop(T* elem) { SkASSERT(fCount > 0); if (elem) *elem = (*this)[fCount - 1]; --fCount; }
    void     pop() { SkASSERT(fCount > 0); --fCount; }

    void deleteAll() {
        T*  iter = fArray;
        T*  stop = fArray + fCount;
        while (iter < stop) {
            delete *iter;
            iter += 1;
        }
        this->reset();
    }

    void freeAll() {
        T*  iter = fArray;
        T*  stop = fArray + fCount;
        while (iter < stop) {
            sk_free(*iter);
            iter += 1;
        }
        this->reset();
    }

    void unrefAll() {
        T*  iter = fArray;
        T*  stop = fArray + fCount;
        while (iter < stop) {
            (*iter)->unref();
            iter += 1;
        }
        this->reset();
    }

    void safeUnrefAll() {
        T*  iter = fArray;
        T*  stop = fArray + fCount;
        while (iter < stop) {
            SkSafeUnref(*iter);
            iter += 1;
        }
        this->reset();
    }

#ifdef SK_DEBUG
    void validate() const {
        SkASSERT((fReserve == 0 && fArray == nullptr) ||
                 (fReserve > 0 && fArray != nullptr));
        SkASSERT(fCount <= fReserve);
    }
#endif

    void shrinkToFit() {
        if (fReserve != fCount) {
            SkASSERT(fReserve > fCount);
            fReserve = fCount;
            fArray = (T*)sk_realloc_throw(fArray, fReserve * sizeof(T));
        }
    }

private:
    T*      fArray;
    int     fReserve;   // size of the allocation in fArray (#elements)
    int     fCount;     // logical number of elements (fCount <= fReserve)

    /**
     *  Adjusts the number of elements in the array.
     *  This is the same as calling setCount(count() + delta).
     */
    void adjustCount(int delta) {
        SkASSERT(delta > 0);

        // We take care to avoid overflow here.
        // The sum of fCount and delta is at most 4294967294, which fits fine in uint32_t.
        uint32_t count = (uint32_t)fCount + (uint32_t)delta;
        SkASSERT_RELEASE( SkTFitsIn<int>(count) );

        this->setCount(SkTo<int>(count));
    }

    /**
     *  Increase the storage allocation such that it can hold (fCount + extra)
     *  elements.
     *  It never shrinks the allocation, and it may increase the allocation by
     *  more than is strictly required, based on a private growth heuristic.
     *
     *  note: does NOT modify fCount
     */
    void resizeStorageToAtLeast(int count) {
        SkASSERT(count > fReserve);

        // We take care to avoid overflow here.
        // The maximum value we can get for reserve here is 2684354563, which fits in uint32_t.
        uint32_t reserve = (uint32_t)count + 4;
        reserve += reserve / 4;
        SkASSERT_RELEASE( SkTFitsIn<int>(reserve) );

        fReserve = SkTo<int>(reserve);
        fArray = (T*)sk_realloc_throw(fArray, (size_t)fReserve * sizeof(T));
    }
};

template <typename T> static inline void swap(SkTDArray<T>& a, SkTDArray<T>& b) {
    a.swap(b);
}

#endif
