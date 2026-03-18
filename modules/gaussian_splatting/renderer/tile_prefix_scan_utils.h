#ifndef GS_TILE_PREFIX_SCAN_UTILS_H
#define GS_TILE_PREFIX_SCAN_UTILS_H

#include "pipeline_io_contracts.h"

#include "core/error/error_macros.h"
#include "core/string/ustring.h"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace GaussianSplatting {

static constexpr uint32_t kTilePrefixPassLocalSize = 256u;

// Host/shader ABI contract for tile_prefix_scan.glsl PrefixParams (set=1 binding=0, std140).
struct TilePrefixParamsLayout {
    uint32_t total_tiles = 0;
    uint32_t workgroup_stride = kTilePrefixPassLocalSize;
    uint32_t total_workgroups = 0;
    uint32_t global_sort_capacity = 0;
};

static_assert(offsetof(TilePrefixParamsLayout, total_tiles) == 0, "TilePrefixParamsLayout::total_tiles offset mismatch");
static_assert(offsetof(TilePrefixParamsLayout, workgroup_stride) == 4, "TilePrefixParamsLayout::workgroup_stride offset mismatch");
static_assert(offsetof(TilePrefixParamsLayout, total_workgroups) == 8, "TilePrefixParamsLayout::total_workgroups offset mismatch");
static_assert(offsetof(TilePrefixParamsLayout, global_sort_capacity) == 12, "TilePrefixParamsLayout::global_sort_capacity offset mismatch");
static_assert(sizeof(TilePrefixParamsLayout) == sizeof(uint32_t) * 4, "TilePrefixParamsLayout size mismatch");

enum TilePrefixPass2Operation : uint32_t {
    TILE_PREFIX_PASS2_OP_INCLUSIVE_STEP = 0u,
    TILE_PREFIX_PASS2_OP_EXCLUSIVE_SHIFT = 1u,
    TILE_PREFIX_PASS2_OP_COPY = 2u,
};

enum TilePrefixPass2SourceBuffer : uint32_t {
    TILE_PREFIX_PASS2_SOURCE_WG_SUMS = 0u,
    TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS = 1u,
};

// Host/shader ABI contract for tile_prefix_scan.glsl pass2 push constants.
struct TilePrefixPass2ControlLayout {
    uint32_t operation = TILE_PREFIX_PASS2_OP_INCLUSIVE_STEP;
    uint32_t source_buffer = TILE_PREFIX_PASS2_SOURCE_WG_SUMS;
    uint32_t stride = 1u;
    uint32_t reserved = 0u;
};

static_assert(offsetof(TilePrefixPass2ControlLayout, operation) == 0, "TilePrefixPass2ControlLayout::operation offset mismatch");
static_assert(offsetof(TilePrefixPass2ControlLayout, source_buffer) == 4, "TilePrefixPass2ControlLayout::source_buffer offset mismatch");
static_assert(offsetof(TilePrefixPass2ControlLayout, stride) == 8, "TilePrefixPass2ControlLayout::stride offset mismatch");
static_assert(offsetof(TilePrefixPass2ControlLayout, reserved) == 12, "TilePrefixPass2ControlLayout::reserved offset mismatch");
static_assert(sizeof(TilePrefixPass2ControlLayout) == sizeof(uint32_t) * 4, "TilePrefixPass2ControlLayout size mismatch");

inline uint32_t tile_prefix_compute_total_workgroups(uint32_t p_total_tiles) {
    return (p_total_tiles + kTilePrefixPassLocalSize - 1u) / kTilePrefixPassLocalSize;
}

struct TilePrefixDispatchCounts {
    uint32_t pass1_dispatch_x = 0u;
    uint32_t pass2_dispatch_x = 0u;
    uint32_t pass3_dispatch_x = 0u;
};

inline uint32_t tile_prefix_compute_dispatch_groups(uint32_t p_total_workgroups) {
    if (p_total_workgroups == 0u) {
        return 0u;
    }
    return (p_total_workgroups + kTilePrefixPassLocalSize - 1u) / kTilePrefixPassLocalSize;
}

inline TilePrefixDispatchCounts tile_prefix_compute_dispatch_counts(uint32_t p_total_workgroups) {
    TilePrefixDispatchCounts counts;
    counts.pass1_dispatch_x = p_total_workgroups;
    counts.pass2_dispatch_x = tile_prefix_compute_dispatch_groups(p_total_workgroups);
    counts.pass3_dispatch_x = p_total_workgroups;
    return counts;
}

inline uint32_t tile_prefix_compute_pass2_levels(uint32_t p_total_workgroups) {
    uint32_t levels = 0u;
    for (uint64_t stride = 1u; stride < uint64_t(p_total_workgroups); stride <<= 1u) {
        levels++;
    }
    return levels;
}

struct TilePrefixCpuScanResult {
    uint32_t raw_total = 0;
    bool raw_total_saturated = false;
    IndirectDispatchLayout indirect_dispatch{};
};

inline bool tile_prefix_dispatch_exceeds_limit(uint32_t p_dispatch_groups_x, uint32_t p_max_dispatch_groups_x) {
    if (p_dispatch_groups_x == 0u || p_max_dispatch_groups_x == 0u) {
        return false;
    }
    return p_dispatch_groups_x > p_max_dispatch_groups_x;
}

inline bool tile_prefix_pass1_requires_cpu_fallback(uint32_t p_total_workgroups, uint32_t p_max_dispatch_groups_x) {
    return tile_prefix_dispatch_exceeds_limit(p_total_workgroups, p_max_dispatch_groups_x);
}

inline bool tile_prefix_pass2_requires_cpu_fallback(uint32_t p_total_workgroups, uint32_t p_max_dispatch_groups_x) {
    return tile_prefix_dispatch_exceeds_limit(tile_prefix_compute_dispatch_groups(p_total_workgroups), p_max_dispatch_groups_x);
}

inline bool tile_prefix_pass3_requires_cpu_fallback(uint32_t p_total_workgroups, uint32_t p_max_dispatch_groups_x) {
    return tile_prefix_dispatch_exceeds_limit(p_total_workgroups, p_max_dispatch_groups_x);
}

inline bool tile_prefix_any_pass_requires_cpu_fallback(uint32_t p_total_workgroups, uint32_t p_max_dispatch_groups_x) {
    return tile_prefix_pass1_requires_cpu_fallback(p_total_workgroups, p_max_dispatch_groups_x) ||
            tile_prefix_pass2_requires_cpu_fallback(p_total_workgroups, p_max_dispatch_groups_x) ||
            tile_prefix_pass3_requires_cpu_fallback(p_total_workgroups, p_max_dispatch_groups_x);
}

// Builds exclusive tile ranges and indirect dispatch values from tile counts.
// `p_out_ranges_words` must have `p_total_tiles * 2` uint32 slots.
inline bool compute_tile_prefix_cpu(const uint32_t *p_tile_counts, uint32_t p_total_tiles,
        uint32_t *p_out_ranges_words, uint32_t p_global_sort_capacity, TilePrefixCpuScanResult &r_result) {
    if (p_tile_counts == nullptr || p_out_ranges_words == nullptr) {
        return false;
    }

    uint64_t prefix64 = 0;
    const uint64_t max_u32 = uint64_t(std::numeric_limits<uint32_t>::max());

    for (uint32_t i = 0; i < p_total_tiles; i++) {
        const uint32_t prefix_u32 = prefix64 > max_u32 ? std::numeric_limits<uint32_t>::max() : uint32_t(prefix64);
        p_out_ranges_words[i * 2u + 0u] = prefix_u32;
        p_out_ranges_words[i * 2u + 1u] = p_tile_counts[i];
        prefix64 += p_tile_counts[i];
    }

    r_result.raw_total_saturated = prefix64 > max_u32;
    r_result.raw_total = r_result.raw_total_saturated ? std::numeric_limits<uint32_t>::max() : uint32_t(prefix64);

    const bool overflow = (p_global_sort_capacity > 0u) && (r_result.raw_total > p_global_sort_capacity);
    if (overflow) {
        static uint32_t prefix_overflow_throttle = 0;
        if (++prefix_overflow_throttle % 300 == 1) {
            WARN_PRINT(vformat("[TilePrefixScan] Overlap record count %d exceeds global sort capacity %d; "
                    "%d records dropped. Consider increasing max_overlap_records.",
                    r_result.raw_total, p_global_sort_capacity,
                    r_result.raw_total - p_global_sort_capacity));
        }
    }
    const uint32_t clamped_total = overflow ? p_global_sort_capacity : r_result.raw_total;
    uint64_t groups_x64 = (uint64_t(clamped_total) + uint64_t(kTilePrefixPassLocalSize) - 1u) / uint64_t(kTilePrefixPassLocalSize);
    if (groups_x64 == 0u) {
        groups_x64 = 1u;
    }
    const uint32_t groups_x = groups_x64 > max_u32 ? std::numeric_limits<uint32_t>::max() : uint32_t(groups_x64);

    r_result.indirect_dispatch.dispatch_x = groups_x;
    r_result.indirect_dispatch.dispatch_y = 1u;
    r_result.indirect_dispatch.dispatch_z = 1u;
    r_result.indirect_dispatch.element_count = clamped_total;
    r_result.indirect_dispatch.overflow_flag = overflow ? 1u : 0u;
    r_result.indirect_dispatch.unclamped_total = r_result.raw_total;
    return true;
}

} // namespace GaussianSplatting

#endif // GS_TILE_PREFIX_SCAN_UTILS_H
