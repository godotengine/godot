// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "RadixSort.h"

#include "Utils.h"

#include <string.h> // memset

using namespace nv;

static inline void FloatFlip(uint32 & f) {
    //uint32 mask = -int32(f >> 31) | 0x80000000; // Michael Herf.
    int32 mask = (int32(f) >> 31) | 0x80000000; // Warren Hunt, Manchor Ko.
    f ^= mask;
}

static inline void IFloatFlip(uint32 & f) {
    uint32 mask = ((f >> 31) - 1) | 0x80000000; // Michael Herf.
    //uint32 mask = (int32(f ^ 0x80000000) >> 31) | 0x80000000; // Warren Hunt, Manchor Ko. @@ Correct, but fails in release on gcc-4.2.1
    f ^= mask;
}


template<typename T> 
void createHistograms(const T * buffer, uint count, uint * histogram)
{
    const uint bucketCount = sizeof(T); // (8 * sizeof(T)) / log2(radix)

    // Init bucket pointers.
    uint * h[bucketCount];
    for (uint i = 0; i < bucketCount; i++) {
#if NV_BIG_ENDIAN
        h[sizeof(T)-1-i] = histogram + 256 * i;
#else
        h[i] = histogram + 256 * i;
#endif
    }

    // Clear histograms.
    memset(histogram, 0, 256 * bucketCount * sizeof(uint));

    // @@ Add support for signed integers.

    // Build histograms.
    const uint8 * p = (const uint8 *)buffer; // @@ Does this break aliasing rules?
    const uint8 * pe = p + count * sizeof(T);

    while (p != pe) {
        h[0][*p++]++, h[1][*p++]++, h[2][*p++]++, h[3][*p++]++;
        if (bucketCount == 8) h[4][*p++]++, h[5][*p++]++, h[6][*p++]++, h[7][*p++]++;
    }
}

/*
template <>
void createHistograms<float>(const float * buffer, uint count, uint * histogram)
{
    // Init bucket pointers.
    uint32 * h[4];
    for (uint i = 0; i < 4; i++) {
#if NV_BIG_ENDIAN
        h[3-i] = histogram + 256 * i;
#else
        h[i] = histogram + 256 * i;
#endif
    }

    // Clear histograms.
    memset(histogram, 0, 256 * 4 * sizeof(uint32));

    // Build histograms.
    for (uint i = 0; i < count; i++) {
        uint32 fi = FloatFlip(buffer[i]);

        h[0][fi & 0xFF]++;
        h[1][(fi >> 8) & 0xFF]++;
        h[2][(fi >> 16) & 0xFF]++;
        h[3][fi >> 24]++;
    }
}
*/

RadixSort::RadixSort() : m_size(0), m_ranks(NULL), m_ranks2(NULL), m_validRanks(false)
{
}

RadixSort::RadixSort(uint reserve_count) : m_size(0), m_ranks(NULL), m_ranks2(NULL), m_validRanks(false)
{
    checkResize(reserve_count);
}

RadixSort::~RadixSort()
{
    // Release everything
    free(m_ranks2);
    free(m_ranks);
}


void RadixSort::resize(uint count)
{
    m_ranks2 = realloc<uint>(m_ranks2, count);
    m_ranks = realloc<uint>(m_ranks, count);
}

inline void RadixSort::checkResize(uint count)
{
    if (count != m_size)
    {
        if (count > m_size) resize(count);
        m_size = count;
        m_validRanks = false;
    }
}

template <typename T> inline void RadixSort::insertionSort(const T * input, uint count)
{
    if (!m_validRanks) {
        /*for (uint i = 0; i < count; i++) {
            m_ranks[i] = i;
        }*/

        m_ranks[0] = 0;
        for (uint i = 1; i != count; ++i)
        {
            int rank = m_ranks[i] = i;

            uint j = i;
            while (j != 0 && input[rank] < input[m_ranks[j-1]])
            {
                m_ranks[j] = m_ranks[j-1];
                --j;
            }
            if (i != j)
            {
                m_ranks[j] = rank;
            }
        }

        m_validRanks = true;
    }
    else {
        for (uint i = 1; i != count; ++i)
        {
            int rank = m_ranks[i];

            uint j = i;
            while (j != 0 && input[rank] < input[m_ranks[j-1]])
            {
                m_ranks[j] = m_ranks[j-1];
                --j;
            }
            if (i != j)
            {
                m_ranks[j] = rank;
            }
        }
    }
}

template <typename T> inline void RadixSort::radixSort(const T * input, uint count)
{
    const uint P = sizeof(T); // pass count

    // Allocate histograms & offsets on the stack
    uint histogram[256 * P];
    uint * link[256];

    createHistograms(input, count, histogram);

    // Radix sort, j is the pass number (0=LSB, P=MSB)
    for (uint j = 0; j < P; j++)
    {
        // Pointer to this bucket.
        const uint * h = &histogram[j * 256];

        const uint8 * inputBytes = (const uint8*)input; // @@ Is this aliasing legal?

#if NV_BIG_ENDIAN
        inputBytes += P - 1 - j;
#else
        inputBytes += j;
#endif

        if (h[inputBytes[0]] == count) {
            // Skip this pass, all values are the same.
            continue;
        }

        // Create offsets
        link[0] = m_ranks2;
        for (uint i = 1; i < 256; i++) link[i] = link[i-1] + h[i-1];

        // Perform Radix Sort
        if (!m_validRanks)
        {
            for (uint i = 0; i < count; i++)
            {
                *link[inputBytes[i*P]]++ = i;
            }
            m_validRanks = true;
        }
        else
        {
            for (uint i = 0; i < count; i++)
            {
                const uint idx = m_ranks[i];
                *link[inputBytes[idx*P]]++ = idx;
            }
        }

        // Swap pointers for next pass. Valid indices - the most recent ones - are in m_ranks after the swap.
        swap(m_ranks, m_ranks2);
    }

    // All values were equal, generate linear ranks.
    if (!m_validRanks)
    {
        for (uint i = 0; i < count; i++)
        {
            m_ranks[i] = i;
        }
        m_validRanks = true;
    }
}


RadixSort & RadixSort::sort(const uint32 * input, uint count)
{
    if (input == NULL || count == 0) return *this;

    // Resize lists if needed
    checkResize(count);

    if (count < 32) {
        insertionSort(input, count);
    }
    else {
        radixSort<uint32>(input, count);
    }
    return *this;
}


RadixSort & RadixSort::sort(const uint64 * input, uint count)
{
    if (input == NULL || count == 0) return *this;

    // Resize lists if needed
    checkResize(count);

    if (count < 64) {
        insertionSort(input, count);
    }
    else {
        radixSort(input, count);
    }
    return *this;
}

RadixSort& RadixSort::sort(const float * input, uint count)
{
    if (input == NULL || count == 0) return *this;

    // Resize lists if needed
    checkResize(count);

    if (count < 32) {
        insertionSort(input, count);
    }
    else {
        // @@ Avoid touching the input multiple times.
        for (uint i = 0; i < count; i++) {
            FloatFlip((uint32 &)input[i]);
        }

        radixSort<uint32>((const uint32 *)input, count);

        for (uint i = 0; i < count; i++) {
            IFloatFlip((uint32 &)input[i]);
        }
    }

    return *this;
}
