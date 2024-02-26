/*
 * Copyright 2017 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkSafeMath_DEFINED
#define SkSafeMath_DEFINED

#include "include/core/SkTypes.h"
#include "include/private/SkTFitsIn.h"
#include <limits>

// SkSafeMath always check that a series of operations do not overflow.
// This must be correct for all platforms, because this is a check for safety at runtime.

class SkSafeMath {
public:
    SkSafeMath() = default;

    bool ok() const { return fOK; }
    explicit operator bool() const { return fOK; }

    size_t mul(size_t x, size_t y) {
        return sizeof(size_t) == sizeof(uint64_t) ? mul64(x, y) : mul32(x, y);
    }

    size_t add(size_t x, size_t y) {
        size_t result = x + y;
        fOK &= result >= x;
        return result;
    }

    /**
     *  Return a + b, unless this result is an overflow/underflow. In those cases, fOK will
     *  be set to false, and it is undefined what this returns.
     */
    int addInt(int a, int b) {
        if (b < 0 && a < std::numeric_limits<int>::min() - b) {
            fOK = false;
            return a;
        } else if (b > 0 && a > std::numeric_limits<int>::max() - b) {
            fOK = false;
            return a;
        }
        return a + b;
    }

    size_t alignUp(size_t x, size_t alignment) {
        SkASSERT(alignment && !(alignment & (alignment - 1)));
        return add(x, alignment - 1) & ~(alignment - 1);
    }

    template <typename T> T castTo(size_t value) {
        if (!SkTFitsIn<T>(value)) {
            fOK = false;
        }
        return static_cast<T>(value);
    }

    // These saturate to their results
    static size_t Add(size_t x, size_t y);
    static size_t Mul(size_t x, size_t y);
    static size_t Align4(size_t x) {
        SkSafeMath safe;
        return safe.alignUp(x, 4);
    }

private:
    uint32_t mul32(uint32_t x, uint32_t y) {
        uint64_t bx = x;
        uint64_t by = y;
        uint64_t result = bx * by;
        fOK &= result >> 32 == 0;
        return result;
    }

    uint64_t mul64(uint64_t x, uint64_t y) {
        if (x <= std::numeric_limits<uint64_t>::max() >> 32
            && y <= std::numeric_limits<uint64_t>::max() >> 32) {
            return x * y;
        } else {
            auto hi = [](uint64_t x) { return x >> 32; };
            auto lo = [](uint64_t x) { return x & 0xFFFFFFFF; };

            uint64_t lx_ly = lo(x) * lo(y);
            uint64_t hx_ly = hi(x) * lo(y);
            uint64_t lx_hy = lo(x) * hi(y);
            uint64_t hx_hy = hi(x) * hi(y);
            uint64_t result = 0;
            result = this->add(lx_ly, (hx_ly << 32));
            result = this->add(result, (lx_hy << 32));
            fOK &= (hx_hy + (hx_ly >> 32) + (lx_hy >> 32)) == 0;

            #if defined(SK_DEBUG) && defined(__clang__) && defined(__x86_64__)
                auto double_check = (unsigned __int128)x * y;
                SkASSERT(result == (double_check & 0xFFFFFFFFFFFFFFFF));
                SkASSERT(!fOK || (double_check >> 64 == 0));
            #endif

            return result;
        }
    }
    bool fOK = true;
};

#endif//SkSafeMath_DEFINED
