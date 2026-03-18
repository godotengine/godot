#[compute]

#version 450

#include "includes/platform_compat.glsl"

#ifndef TILE_SIZE
#define TILE_SIZE GS_TILE_SIZE
#endif

#ifndef GS_PREFIX_LOCAL_SIZE
#define GS_PREFIX_LOCAL_SIZE 256
#endif

#ifndef GS_TILE_PREFIX_PASS2_OP_INCLUSIVE_STEP
#define GS_TILE_PREFIX_PASS2_OP_INCLUSIVE_STEP 0
#endif

#ifndef GS_TILE_PREFIX_PASS2_OP_EXCLUSIVE_SHIFT
#define GS_TILE_PREFIX_PASS2_OP_EXCLUSIVE_SHIFT 1
#endif

#ifndef GS_TILE_PREFIX_PASS2_OP_COPY
#define GS_TILE_PREFIX_PASS2_OP_COPY 2
#endif

#ifndef GS_TILE_PREFIX_PASS2_SOURCE_WG_SUMS
#define GS_TILE_PREFIX_PASS2_SOURCE_WG_SUMS 0
#endif

#ifndef GS_TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS
#define GS_TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS 1
#endif

layout(local_size_x = GS_PREFIX_LOCAL_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer TileCounts {
    uint counts[];
} tile_counts;

layout(set = 0, binding = 1, std430) buffer TileRanges {
    uvec2 ranges[];
} tile_ranges;

// Workgroup partial sums (one uint per workgroup)
layout(set = 0, binding = 2, std430) buffer WorkgroupSums {
    uint wg_sums[];
} wg_sums;

// Scanned workgroup offsets (same length as wg_sums)
layout(set = 0, binding = 3, std430) buffer WorkgroupOffsets {
    uint wg_offsets[];
} wg_offsets;

layout(set = 0, binding = 4, std430) buffer PrefixTotal {
    uint total;
} prefix_total;

// GPU-driven dispatch/count with overflow detection (matches IndirectDispatchLayout).
layout(set = 0, binding = 5, std430) buffer IndirectDispatch {
    uint dispatch_xyz[3];   // offset 0: dispatch groups for downstream stages
    uint element_count;     // offset 12: clamped to capacity
    uint overflow_flag;     // offset 16: 1 if total > capacity, 0 otherwise
    uint unclamped_total;   // offset 20: raw total before clamping
} indirect_dispatch;

layout(set = 1, binding = 0, std140) uniform PrefixParams {
    uint total_tiles;
    uint workgroup_stride;
    uint total_workgroups;
    uint global_sort_capacity;  // Used for overflow detection in Pass 3
} params;

#ifdef GS_TILE_PREFIX_PASS_1
// Pass 1: local exclusive scan + write wg_sums and base ranges.
shared uint scratch[GS_PREFIX_LOCAL_SIZE];
// PERF-7 (#678): Shared flag to track if any thread in workgroup has non-zero count
shared uint wg_has_nonzero;

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    uint global_idx = gid * GS_PREFIX_LOCAL_SIZE + lid;

    uint count = (global_idx < params.total_tiles) ? tile_counts.counts[global_idx] : 0;
    scratch[lid] = count;

    // PERF-7 (#678): Early-out optimization for empty workgroups
    // Initialize shared flag and check if all tiles in workgroup are empty
    if (lid == 0u) {
        wg_has_nonzero = 0u;
    }
    barrier();

    // Atomically set flag if this thread has non-zero count
    if (count > 0u) {
        atomicOr(wg_has_nonzero, 1u);
    }
    barrier();

    // If entire workgroup is empty, write zeros and skip expensive scan
    if (wg_has_nonzero == 0u) {
        if (lid == 0u) {
            wg_sums.wg_sums[gid] = 0u;
        }
        if (global_idx < params.total_tiles) {
            tile_ranges.ranges[global_idx] = uvec2(0u, 0u);
        }
        return;
    }

    // Upsweep
    for (uint stride = 1u; stride < GS_PREFIX_LOCAL_SIZE; stride <<= 1u) {
        uint idx = (lid + 1u) * stride * 2u - 1u;
        if (idx < GS_PREFIX_LOCAL_SIZE) {
            scratch[idx] += scratch[idx - stride];
        }
        barrier();
    }

    if (lid == 0u) {
        wg_sums.wg_sums[gid] = scratch[GS_PREFIX_LOCAL_SIZE - 1u];
        scratch[GS_PREFIX_LOCAL_SIZE - 1u] = 0u; // clear for downsweep
    }
    barrier();

    // Downsweep
    for (uint stride = GS_PREFIX_LOCAL_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        uint idx = (lid + 1u) * stride * 2u - 1u;
        if (idx < GS_PREFIX_LOCAL_SIZE) {
            uint tmp = scratch[idx - stride];
            scratch[idx - stride] = scratch[idx];
            scratch[idx] += tmp;
        }
        barrier();
    }

    if (global_idx < params.total_tiles) {
        tile_ranges.ranges[global_idx] = uvec2(scratch[lid], count);
    }
}
#endif // GS_TILE_PREFIX_PASS_1

#ifdef GS_TILE_PREFIX_PASS_2
// Pass 2: scalable multi-level scan over wg_sums/wg_offsets.
layout(push_constant, std430) uniform PrefixPass2Control {
    uint operation;
    uint source_buffer;
    uint stride;
    uint reserved;
} pass2_control;

uint read_pass2_source(uint idx) {
    if (pass2_control.source_buffer == GS_TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS) {
        return wg_offsets.wg_offsets[idx];
    }
    return wg_sums.wg_sums[idx];
}

void write_pass2_dest(uint idx, uint value) {
    // Destination is the opposite buffer of source_buffer to keep ping-pong deterministic.
    if (pass2_control.source_buffer == GS_TILE_PREFIX_PASS2_SOURCE_WG_OFFSETS) {
        wg_sums.wg_sums[idx] = value;
    } else {
        wg_offsets.wg_offsets[idx] = value;
    }
}

void main() {
    uint global_idx = gl_GlobalInvocationID.x;
    if (global_idx >= params.total_workgroups) {
        return;
    }

    uint result = 0u;
    if (pass2_control.operation == GS_TILE_PREFIX_PASS2_OP_INCLUSIVE_STEP) {
        uint value = read_pass2_source(global_idx);
        uint addend = 0u;
        if (pass2_control.stride > 0u && global_idx >= pass2_control.stride) {
            addend = read_pass2_source(global_idx - pass2_control.stride);
        }
        result = value + addend;
    } else if (pass2_control.operation == GS_TILE_PREFIX_PASS2_OP_EXCLUSIVE_SHIFT) {
        result = (global_idx == 0u) ? 0u : read_pass2_source(global_idx - 1u);
    } else { // GS_TILE_PREFIX_PASS2_OP_COPY
        result = read_pass2_source(global_idx);
    }

    write_pass2_dest(global_idx, result);
}
#endif // GS_TILE_PREFIX_PASS_2

#ifdef GS_TILE_PREFIX_PASS_3
// Pass 3: add workgroup offsets into base ranges.
void main() {
    uint lid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    uint global_idx = gid * GS_PREFIX_LOCAL_SIZE + lid;

    if (global_idx >= params.total_tiles) {
        return;
    }

    uint base_offset = (gid < params.total_workgroups) ? wg_offsets.wg_offsets[gid] : 0u;
    uvec2 range = tile_ranges.ranges[global_idx];
    range.x += base_offset;
    tile_ranges.ranges[global_idx] = range;

    // Last tile writes the total overlap count and performs overflow detection.
    if (global_idx == params.total_tiles - 1u) {
        uint total = range.x + range.y;
        prefix_total.total = total;

        // Store unclamped total for async readback (capacity expansion decisions)
        indirect_dispatch.unclamped_total = total;

        // Overflow detection: compare against capacity
        bool overflow = (total > params.global_sort_capacity) && (params.global_sort_capacity > 0u);
        indirect_dispatch.overflow_flag = overflow ? 1u : 0u;

        // Clamp element_count to capacity for downstream GPU stages
        uint clamped_total = overflow ? params.global_sort_capacity : total;
        indirect_dispatch.element_count = clamped_total;

        // Populate dispatch groups based on clamped count
        uint groups_x = (clamped_total + GS_PREFIX_LOCAL_SIZE - 1u) / GS_PREFIX_LOCAL_SIZE;
        indirect_dispatch.dispatch_xyz[0] = max(groups_x, 1u);
        indirect_dispatch.dispatch_xyz[1] = 1u;
        indirect_dispatch.dispatch_xyz[2] = 1u;
    }
}
#endif // GS_TILE_PREFIX_PASS_3
