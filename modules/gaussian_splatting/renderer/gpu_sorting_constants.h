#ifndef GPU_SORTING_CONSTANTS_H
#define GPU_SORTING_CONSTANTS_H

#include <cstdint>

namespace GPUSortingConstants {

static constexpr uint32_t DEFAULT_WORKGROUP_SIZE = 256;
static constexpr uint32_t DEFAULT_RADIX_BITS = 4;
static constexpr uint32_t RADIX_BITS = 8;
static constexpr uint32_t RADIX_SIZE = 1u << RADIX_BITS;
static constexpr uint32_t MAX_WORKGROUP_SIZE = 1024;
static constexpr uint32_t HISTOGRAM_BINS = RADIX_SIZE;

} // namespace GPUSortingConstants

#endif // GPU_SORTING_CONSTANTS_H
