#pragma once
#ifndef NV_CORE_RADIXSORT_H
#define NV_CORE_RADIXSORT_H

// Based on Pierre Terdiman's and Michael Herf's source code.
// http://www.codercorner.com/RadixSortRevisited.htm
// http://www.stereopsis.com/radix.html

#include "nvcore.h"
#include "Array.h"

namespace nv
{

    class NVCORE_CLASS RadixSort
    {
        NV_FORBID_COPY(RadixSort);
    public:
        // Constructor/Destructor
        RadixSort();
        RadixSort(uint reserve_count);
        ~RadixSort();

        // Invalidate ranks.
        RadixSort & reset() { m_validRanks = false; return *this; }

        // Sorting methods.
        RadixSort & sort(const uint32 * input, uint count);
        RadixSort & sort(const uint64 * input, uint count);
        RadixSort & sort(const float * input, uint count);

        // Helpers.
        RadixSort & sort(const Array<uint32> & input);
        RadixSort & sort(const Array<uint64> & input);
        RadixSort & sort(const Array<float> & input);

        // Access to results. m_ranks is a list of indices in sorted order, i.e. in the order you may further process your data
        inline const uint * ranks() const { nvDebugCheck(m_validRanks); return m_ranks; }
        inline uint * ranks() { nvDebugCheck(m_validRanks); return m_ranks; }
        inline uint rank(uint i) const { nvDebugCheck(m_validRanks); return m_ranks[i]; }

        // query whether the sort has been performed
        inline bool valid() const { return m_validRanks; }

    private:
        uint m_size;
        uint * m_ranks;
        uint * m_ranks2;
        bool m_validRanks;

        // Internal methods
        template <typename T> void insertionSort(const T * input, uint count);
        template <typename T> void radixSort(const T * input, uint count);

        void checkResize(uint nb);
        void resize(uint nb);
    };

    inline RadixSort & RadixSort::sort(const Array<uint32> & input) {
        return sort(input.buffer(), input.count());
    }

    inline RadixSort & RadixSort::sort(const Array<uint64> & input) {
        return sort(input.buffer(), input.count());
    }

    inline RadixSort & RadixSort::sort(const Array<float> & input) {
        return sort(input.buffer(), input.count());
    }

} // nv namespace



#endif // NV_CORE_RADIXSORT_H
