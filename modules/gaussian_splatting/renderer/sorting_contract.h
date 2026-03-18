#ifndef GS_SORTING_CONTRACT_H
#define GS_SORTING_CONTRACT_H

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include <cfloat>
#include <cstdint>
#include <cstring>

namespace GaussianSplatting {

static constexpr uint32_t kSortWorkgroupSize = 256;
static constexpr float kSortPadDepth = FLT_MAX;
static_assert(kSortPadDepth == FLT_MAX, "kSortPadDepth must remain aligned with the shader pad depth.");

struct SortPaddingInfo {
    uint32_t count = 0;
    uint32_t padded_elements = 0;
    uint32_t dispatch_elements = 0;
};

struct SortKey64 {
    uint32_t lo = 0; // tie-break (low bits)
    uint32_t hi = 0; // sortable depth (high bits)
};

inline uint32_t align_up(uint32_t value, uint32_t alignment) {
    if (alignment == 0) {
        return value;
    }
    uint32_t rounded = value + alignment - 1;
    if (rounded < value) {
        return value;
    }
    return (rounded / alignment) * alignment;
}

inline SortPaddingInfo get_sort_padding(uint32_t count) {
    SortPaddingInfo info;
    info.count = count;

    uint32_t padded_elements = next_power_of_2(MAX(count, (uint32_t)1));
    uint32_t aligned = align_up(count, kSortWorkgroupSize);
    if (aligned > 0) {
        uint32_t aligned_pow2 = next_power_of_2(aligned);
        if (aligned_pow2 == 0) {
            aligned_pow2 = aligned;
        }
        padded_elements = MAX(padded_elements, aligned_pow2);
    }
    if (padded_elements == 0) {
        padded_elements = 1;
    }

    info.padded_elements = padded_elements;
    info.dispatch_elements = padded_elements;
    return info;
}

inline uint32_t float_to_sortable_uint(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t mask = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return bits ^ mask;
}

// 64-bit key layout: hi = sortable depth, lo = tie-break.
inline SortKey64 pack_sort_key_64(float depth, uint32_t tie_break) {
    SortKey64 key;
    key.lo = tie_break;
    key.hi = float_to_sortable_uint(depth);
    return key;
}

inline void fill_sort_padding(float *keys, uint32_t *values, uint32_t count, uint32_t total) {
    uint32_t pad_index = count > 0 ? count - 1 : 0;
    for (uint32_t i = count; i < total; ++i) {
        keys[i] = kSortPadDepth;
        values[i] = pad_index;
    }
}

} // namespace GaussianSplatting

#endif // GS_SORTING_CONTRACT_H
