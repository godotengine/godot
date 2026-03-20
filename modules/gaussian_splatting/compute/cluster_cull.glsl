#[compute]

#version 450

#VERSION_DEFINES

// Cluster-level coarse culling shader (LiteGS-style)
// Tests cluster AABBs against frustum to reject entire groups of splats early.
// Visible clusters set their bit in the visibility mask, enabling indirect dispatch
// for the fine per-splat culling pass.

// Platform compatibility macros (atomics, memory barriers, etc.)
// Note: platform_compat.glsl may not be required for basic cluster culling,
// but kept for consistency with other compute shaders in this module.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Cluster AABB data - 32 bytes per cluster
struct ClusterAABB {
    vec3 min_bounds;
    uint splat_start;     // First splat index in this cluster
    vec3 max_bounds;
    uint splat_count;     // Number of splats in this cluster
};

layout(set = 0, binding = 0, std430) readonly buffer ClusterBuffer {
    ClusterAABB clusters[];
} cluster_buffer;

// Output: bitmask of visible clusters (1 bit per cluster)
// For 32K clusters, this is 1KB
layout(set = 0, binding = 1, std430) writeonly buffer ClusterVisibility {
    uint visible_mask[];
} visibility_output;

// Output: indirect dispatch args for fine culling pass
// Format: [group_count_x, group_count_y, group_count_z, visible_splat_count]
layout(set = 0, binding = 2, std430) buffer IndirectDispatch {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
    uint visible_cluster_count;
    uint visible_splat_count;
    uint clusters_culled;
} indirect;

// Visible cluster indices for fine pass (prefix-sum compacted)
layout(set = 0, binding = 3, std430) writeonly buffer VisibleClusters {
    uint indices[];
} visible_clusters;

layout(set = 0, binding = 4, std140) uniform ClusterCullParams {
    mat4 view_matrix;
    mat4 proj_matrix;
    vec4 frustum_planes[6];
    vec3 camera_position;
    float frustum_plane_slack;
    uint total_clusters;
    uint fine_cull_workgroup_size;  // Typically 256
    uint pad0;
    uint pad1;
} params;

// Shared memory for subgroup-accelerated atomic compaction
shared uint local_visible_count;
shared uint local_visible_splats;
shared uint global_write_offset;

// Precomputed absolute normals for frustum planes (shared per workgroup)
// Optimization: abs(normal) is constant per frame, compute once per workgroup
// Saves 6 vec3 abs operations per cluster in aabb_frustum_visible
shared vec3 frustum_abs_normals[6];

// AABB-frustum intersection test (conservative)
bool aabb_frustum_visible(vec3 aabb_min, vec3 aabb_max) {
    // Expand AABB by slack factor for conservative culling
    vec3 center = (aabb_min + aabb_max) * 0.5;
    vec3 half_extent = (aabb_max - aabb_min) * 0.5 * max(params.frustum_plane_slack, 1.0);

    for (int i = 0; i < 6; i++) {
        vec4 plane = params.frustum_planes[i];
        vec3 normal = plane.xyz;

        // Compute effective radius along plane normal
        // Note: abs(normal) precomputed in frustum_abs_normals
        float effective_radius = dot(frustum_abs_normals[i], half_extent);

        // Distance from center to plane
        float distance = dot(normal, center) - plane.w;

        // Cull if entirely outside (positive side)
        if (distance > effective_radius) {
            return false;
        }
    }
    return true;
}

// Cluster culling entry point; writes visibility decisions for the active workgroup.
void main() {
    uint cluster_idx = gl_GlobalInvocationID.x;

    // Initialize shared memory on first thread
    if (gl_LocalInvocationIndex == 0) {
        for (int i = 0; i < 6; i++) {
            frustum_abs_normals[i] = abs(params.frustum_planes[i].xyz);
        }
        local_visible_count = 0u;
        local_visible_splats = 0u;
    }
    barrier();

    bool is_visible = false;
    uint splat_count = 0u;

    if (cluster_idx < params.total_clusters) {
        ClusterAABB cluster = cluster_buffer.clusters[cluster_idx];
        splat_count = cluster.splat_count;

        if (splat_count > 0u) {
            is_visible = aabb_frustum_visible(cluster.min_bounds, cluster.max_bounds);
        }
    }

    // Count visible clusters and splats in this workgroup using atomics
    uint local_offset = 0u;
    if (is_visible) {
        local_offset = atomicAdd(local_visible_count, 1u);
        atomicAdd(local_visible_splats, splat_count);
    } else if (cluster_idx < params.total_clusters) {
        atomicAdd(indirect.clusters_culled, 1u);
    }

    barrier();

    // First thread acquires global write offset
    if (gl_LocalInvocationIndex == 0 && local_visible_count > 0u) {
        global_write_offset = atomicAdd(indirect.visible_cluster_count, local_visible_count);
        atomicAdd(indirect.visible_splat_count, local_visible_splats);
    }

    barrier();

    // Write visible cluster index
    if (is_visible) {
        uint write_idx = global_write_offset + local_offset;
        visible_clusters.indices[write_idx] = cluster_idx;

        // Set visibility bit
        uint word_idx = cluster_idx / 32u;
        uint bit_idx = cluster_idx % 32u;
        atomicOr(visibility_output.visible_mask[word_idx], 1u << bit_idx);
    }

    // Last workgroup updates indirect dispatch args
    // We use a memory barrier and check if all clusters processed
    memoryBarrierBuffer();
    barrier();

    if (gl_LocalInvocationIndex == 0) {
        // Check if this is the last workgroup
        uint processed = atomicAdd(indirect.dispatch_z, 1u) + 1u;
        uint total_workgroups = (params.total_clusters + 63u) / 64u;

        if (processed == total_workgroups) {
            // Compute dispatch size for fine culling
            uint visible_splats = indirect.visible_splat_count;
            uint workgroups_needed = (visible_splats + params.fine_cull_workgroup_size - 1u) / params.fine_cull_workgroup_size;
            indirect.dispatch_x = max(workgroups_needed, 1u);
            indirect.dispatch_y = 1u;
            // dispatch_z was used as counter, reset to 1
            indirect.dispatch_z = 1u;
        }
    }
}
