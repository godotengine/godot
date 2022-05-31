/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkTSort_DEFINED
#define SkTSort_DEFINED

#include "include/core/SkTypes.h"
#include "include/private/SkTo.h"
#include "src/core/SkMathPriv.h"

#include <utility>

///////////////////////////////////////////////////////////////////////////////

/*  Sifts a broken heap. The input array is a heap from root to bottom
 *  except that the root entry may be out of place.
 *
 *  Sinks a hole from array[root] to leaf and then sifts the original array[root] element
 *  from the leaf level up.
 *
 *  This version does extra work, in that it copies child to parent on the way down,
 *  then copies parent to child on the way back up. When copies are inexpensive,
 *  this is an optimization as this sift variant should only be used when
 *  the potentially out of place root entry value is expected to be small.
 *
 *  @param root the one based index into array of the out-of-place root of the heap.
 *  @param bottom the one based index in the array of the last entry in the heap.
 */
template <typename T, typename C>
void SkTHeapSort_SiftUp(T array[], size_t root, size_t bottom, const C& lessThan) {
    T x = array[root-1];
    size_t start = root;
    size_t j = root << 1;
    while (j <= bottom) {
        if (j < bottom && lessThan(array[j-1], array[j])) {
            ++j;
        }
        array[root-1] = array[j-1];
        root = j;
        j = root << 1;
    }
    j = root >> 1;
    while (j >= start) {
        if (lessThan(array[j-1], x)) {
            array[root-1] = array[j-1];
            root = j;
            j = root >> 1;
        } else {
            break;
        }
    }
    array[root-1] = x;
}

/*  Sifts a broken heap. The input array is a heap from root to bottom
 *  except that the root entry may be out of place.
 *
 *  Sifts the array[root] element from the root down.
 *
 *  @param root the one based index into array of the out-of-place root of the heap.
 *  @param bottom the one based index in the array of the last entry in the heap.
 */
template <typename T, typename C>
void SkTHeapSort_SiftDown(T array[], size_t root, size_t bottom, const C& lessThan) {
    T x = array[root-1];
    size_t child = root << 1;
    while (child <= bottom) {
        if (child < bottom && lessThan(array[child-1], array[child])) {
            ++child;
        }
        if (lessThan(x, array[child-1])) {
            array[root-1] = array[child-1];
            root = child;
            child = root << 1;
        } else {
            break;
        }
    }
    array[root-1] = x;
}

/** Sorts the array of size count using comparator lessThan using a Heap Sort algorithm. Be sure to
 *  specialize swap if T has an efficient swap operation.
 *
 *  @param array the array to be sorted.
 *  @param count the number of elements in the array.
 *  @param lessThan a functor with bool operator()(T a, T b) which returns true if a comes before b.
 */
template <typename T, typename C> void SkTHeapSort(T array[], size_t count, const C& lessThan) {
    for (size_t i = count >> 1; i > 0; --i) {
        SkTHeapSort_SiftDown(array, i, count, lessThan);
    }

    for (size_t i = count - 1; i > 0; --i) {
        using std::swap;
        swap(array[0], array[i]);
        SkTHeapSort_SiftUp(array, 1, i, lessThan);
    }
}

/** Sorts the array of size count using comparator '<' using a Heap Sort algorithm. */
template <typename T> void SkTHeapSort(T array[], size_t count) {
    SkTHeapSort(array, count, [](const T& a, const T& b) { return a < b; });
}

///////////////////////////////////////////////////////////////////////////////

/** Sorts the array of size count using comparator lessThan using an Insertion Sort algorithm. */
template <typename T, typename C>
void SkTInsertionSort(T* left, int count, const C& lessThan) {
    T* right = left + count - 1;
    for (T* next = left + 1; next <= right; ++next) {
        if (!lessThan(*next, *(next - 1))) {
            continue;
        }
        T insert = std::move(*next);
        T* hole = next;
        do {
            *hole = std::move(*(hole - 1));
            --hole;
        } while (left < hole && lessThan(insert, *(hole - 1)));
        *hole = std::move(insert);
    }
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename C>
T* SkTQSort_Partition(T* left, int count, T* pivot, const C& lessThan) {
    T* right = left + count - 1;
    using std::swap;
    T pivotValue = *pivot;
    swap(*pivot, *right);
    T* newPivot = left;
    while (left < right) {
        if (lessThan(*left, pivotValue)) {
            swap(*left, *newPivot);
            newPivot += 1;
        }
        left += 1;
    }
    swap(*newPivot, *right);
    return newPivot;
}

/*  Introsort is a modified Quicksort.
 *  When the region to be sorted is a small constant size, it uses Insertion Sort.
 *  When depth becomes zero, it switches over to Heap Sort.
 *  This implementation recurses on the left region after pivoting and loops on the right,
 *    we already limit the stack depth by switching to heap sort,
 *    and cache locality on the data appears more important than saving a few stack frames.
 *
 *  @param depth at this recursion depth, switch to Heap Sort.
 *  @param left points to the beginning of the region to be sorted
 *  @param count number of items to be sorted
 *  @param lessThan  a functor/lambda which returns true if a comes before b.
 */
template <typename T, typename C>
void SkTIntroSort(int depth, T* left, int count, const C& lessThan) {
    for (;;) {
        if (count <= 32) {
            SkTInsertionSort(left, count, lessThan);
            return;
        }

        if (depth == 0) {
            SkTHeapSort<T>(left, count, lessThan);
            return;
        }
        --depth;

        T* middle = left + ((count - 1) >> 1);
        T* pivot = SkTQSort_Partition(left, count, middle, lessThan);
        int pivotCount = pivot - left;

        SkTIntroSort(depth, left, pivotCount, lessThan);
        left += pivotCount + 1;
        count -= pivotCount + 1;
    }
}

/** Sorts the region from left to right using comparator lessThan using Introsort.
 *  Be sure to specialize `swap` if T has an efficient swap operation.
 *
 *  @param begin points to the beginning of the region to be sorted
 *  @param end points past the end of the region to be sorted
 *  @param lessThan a functor/lambda which returns true if a comes before b.
 */
template <typename T, typename C>
void SkTQSort(T* begin, T* end, const C& lessThan) {
    int n = SkToInt(end - begin);
    if (n <= 1) {
        return;
    }
    // Limit Introsort recursion depth to no more than 2 * ceil(log2(n-1)).
    int depth = 2 * SkNextLog2(n - 1);
    SkTIntroSort(depth, begin, n, lessThan);
}

/** Sorts the region from left to right using comparator 'a < b' using Introsort. */
template <typename T> void SkTQSort(T* begin, T* end) {
    SkTQSort(begin, end, [](const T& a, const T& b) { return a < b; });
}

/** Sorts the region from left to right using comparator '*a < *b' using Introsort. */
template <typename T> void SkTQSort(T** begin, T** end) {
    SkTQSort(begin, end, [](const T* a, const T* b) { return *a < *b; });
}

#endif
