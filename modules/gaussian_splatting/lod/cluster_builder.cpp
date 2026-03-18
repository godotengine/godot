#include "cluster_builder.h"
#include "core/os/os.h"
#include "core/math/math_funcs.h"
#include "../logger/gs_logger.h"
#include <algorithm>
#include <cstring>

namespace GaussianSplatting {

ClusterBuilder::ClusterBuilder() {
}

ClusterBuilder::~ClusterBuilder() {
}

uint32_t ClusterBuilder::expand_bits(uint32_t v) const {
    // Expand 10-bit integer to 30 bits with 2-bit gaps for Morton interleaving
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

uint32_t ClusterBuilder::compute_morton_code(const Vector3 &p_position, const AABB &p_bounds) const {
    // Normalize position to [0, 1] within bounds
    Vector3 normalized = (p_position - p_bounds.position) / p_bounds.size;

    // Clamp to valid range
    normalized.x = CLAMP(normalized.x, 0.0f, 1.0f);
    normalized.y = CLAMP(normalized.y, 0.0f, 1.0f);
    normalized.z = CLAMP(normalized.z, 0.0f, 1.0f);

    // Convert to 10-bit integers (0-1023)
    uint32_t x = static_cast<uint32_t>(normalized.x * 1023.0f);
    uint32_t y = static_cast<uint32_t>(normalized.y * 1023.0f);
    uint32_t z = static_cast<uint32_t>(normalized.z * 1023.0f);

    // Interleave bits: Z-order curve
    return (expand_bits(z) << 2) | (expand_bits(y) << 1) | expand_bits(x);
}

AABB ClusterBuilder::compute_cluster_bounds(
    const LocalVector<Gaussian> &p_gaussians,
    const LocalVector<uint32_t> &p_sorted_order,
    uint32_t p_start,
    uint32_t p_count,
    float &r_max_radius
) const {
    AABB bounds;
    r_max_radius = 0.0f;

    if (p_count == 0) {
        return bounds;
    }

    // Initialize with first splat
    uint32_t first_idx = p_sorted_order[p_start];
    const Gaussian &first = p_gaussians[first_idx];
    float first_radius = MAX(MAX(Math::abs(first.scale.x), Math::abs(first.scale.y)), Math::abs(first.scale.z)) * 3.0f;
    r_max_radius = first_radius;

    Vector3 half_extent(first_radius, first_radius, first_radius);
    bounds = AABB(first.position - half_extent, half_extent * 2.0f);

    // Expand bounds to include all splats
    for (uint32_t i = 1; i < p_count; i++) {
        uint32_t idx = p_sorted_order[p_start + i];
        const Gaussian &g = p_gaussians[idx];

        float radius = MAX(MAX(Math::abs(g.scale.x), Math::abs(g.scale.y)), Math::abs(g.scale.z)) * 3.0f;
        r_max_radius = MAX(r_max_radius, radius);

        Vector3 splat_half(radius, radius, radius);
        AABB splat_bounds(g.position - splat_half, splat_half * 2.0f);
        bounds = bounds.merge(splat_bounds);
    }

    return bounds;
}

ClusterBuildResult ClusterBuilder::build_clusters(
    const LocalVector<Gaussian> &p_gaussians,
    const ClusterBuildParams &p_params
) {
    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    ClusterBuildResult result;
    result.original_splat_count = p_gaussians.size();

    if (p_gaussians.is_empty()) {
        result.build_time_ms = 0.0f;
        return result;
    }

    // Compute scene bounds
    AABB scene_bounds;
    for (uint32_t i = 0; i < p_gaussians.size(); i++) {
        const Vector3 &pos = p_gaussians[i].position;
        if (i == 0) {
            scene_bounds.position = pos;
            scene_bounds.size = Vector3(0.001f, 0.001f, 0.001f);
        } else {
            scene_bounds.expand_to(pos);
        }
    }

    // Ensure non-zero size
    if (scene_bounds.size.x < 0.001f) scene_bounds.size.x = 0.001f;
    if (scene_bounds.size.y < 0.001f) scene_bounds.size.y = 0.001f;
    if (scene_bounds.size.z < 0.001f) scene_bounds.size.z = 0.001f;

    // Compute Morton codes for all splats
    LocalVector<std::pair<uint32_t, uint32_t>> morton_pairs; // (morton_code, original_index)
    morton_pairs.resize(p_gaussians.size());

    if (p_params.use_morton_order) {
        for (uint32_t i = 0; i < p_gaussians.size(); i++) {
            uint32_t morton = compute_morton_code(p_gaussians[i].position, scene_bounds);
            morton_pairs[i] = std::make_pair(morton, i);
        }

        // Sort by Morton code
        std::sort(
            morton_pairs.ptr(),
            morton_pairs.ptr() + morton_pairs.size(),
            [](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b) {
                return a.first < b.first;
            }
        );
    } else {
        // No Morton ordering - use original order
        for (uint32_t i = 0; i < p_gaussians.size(); i++) {
            morton_pairs[i] = std::make_pair(i, i);
        }
    }

    // Build sorted order
    result.sorted_splat_order.resize(p_gaussians.size());
    for (uint32_t i = 0; i < p_gaussians.size(); i++) {
        result.sorted_splat_order[i] = morton_pairs[i].second;
    }

    // Group splats into clusters
    uint32_t target_size = CLAMP(p_params.target_cluster_size, p_params.min_cluster_size, p_params.max_cluster_size);
    uint32_t num_clusters = (p_gaussians.size() + target_size - 1) / target_size;

    result.clusters.resize(num_clusters);
    result.splat_to_cluster.resize(p_gaussians.size());

    uint32_t splat_offset = 0;
    for (uint32_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
        uint32_t remaining = p_gaussians.size() - splat_offset;
        uint32_t count = MIN(target_size, remaining);

        // Handle last cluster - ensure minimum size
        if (cluster_idx == num_clusters - 1) {
            count = remaining;
        } else if (remaining - count < p_params.min_cluster_size) {
            // Merge small remainder into this cluster
            count = remaining;
        }

        SplatCluster &cluster = result.clusters.write[cluster_idx];
        cluster.splat_start = splat_offset;
        cluster.splat_count = count;

        // Compute bounds
        float max_radius = 0.0f;
        cluster.bounds = compute_cluster_bounds(p_gaussians, result.sorted_splat_order, splat_offset, count, max_radius);
        cluster.max_splat_radius = max_radius;
        cluster.center = cluster.bounds.get_center();
        cluster.radius = cluster.bounds.get_longest_axis_size() * 0.5f;

        // Compute importance sum
        if (p_params.compute_importance) {
            cluster.importance_sum = 0.0f;
            for (uint32_t i = 0; i < count; i++) {
                uint32_t idx = result.sorted_splat_order[splat_offset + i];
                const Gaussian &g = p_gaussians[idx];
                float opacity = CLAMP(g.opacity, 0.0f, 1.0f);
                float scale_max = MAX(MAX(Math::abs(g.scale.x), Math::abs(g.scale.y)), Math::abs(g.scale.z));
                cluster.importance_sum += opacity * scale_max;
            }
        }

        // Update splat-to-cluster mapping
        for (uint32_t i = 0; i < count; i++) {
            uint32_t original_idx = result.sorted_splat_order[splat_offset + i];
            result.splat_to_cluster[original_idx] = cluster_idx;
        }

        splat_offset += count;

        // Check if we've processed all splats
        if (splat_offset >= p_gaussians.size()) {
            result.clusters.resize(cluster_idx + 1);
            break;
        }
    }

    result.build_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;

    GS_LOG_RENDERER_INFO(vformat("[ClusterBuilder] Built %d clusters from %d splats (%.2f ms, avg %.1f splats/cluster)",
        result.clusters.size(), p_gaussians.size(), result.build_time_ms,
        float(p_gaussians.size()) / MAX(1, result.clusters.size())));

    return result;
}

ClusterBuildResult ClusterBuilder::update_clusters_incremental(
    const LocalVector<Gaussian> &p_gaussians,
    const LocalVector<uint32_t> &p_changed_indices,
    const ClusterBuildResult &p_previous
) {
    // For now, fall back to full rebuild if more than 10% of splats changed
    // A proper incremental update would:
    // 1. Identify affected clusters
    // 2. Recompute only those cluster bounds
    // 3. Potentially split/merge clusters if sizes change significantly

    if (p_previous.clusters.is_empty() ||
        p_changed_indices.size() > p_gaussians.size() / 10 ||
        p_gaussians.size() != p_previous.original_splat_count) {
        // Full rebuild needed
        return build_clusters(p_gaussians, ClusterBuildParams());
    }

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    // Copy previous result
    ClusterBuildResult result;
    result.clusters = p_previous.clusters;
    result.splat_to_cluster = p_previous.splat_to_cluster;
    result.sorted_splat_order = p_previous.sorted_splat_order;
    result.original_splat_count = p_gaussians.size();

    // Find affected clusters
    LocalVector<bool> cluster_dirty;
    cluster_dirty.resize(result.clusters.size());
    memset(cluster_dirty.ptr(), 0, cluster_dirty.size() * sizeof(bool));

    for (uint32_t changed_idx : p_changed_indices) {
        if (changed_idx < result.splat_to_cluster.size()) {
            uint32_t cluster_idx = result.splat_to_cluster[changed_idx];
            if (cluster_idx < cluster_dirty.size()) {
                cluster_dirty[cluster_idx] = true;
            }
        }
    }

    // Recompute bounds for dirty clusters
    for (uint32_t cluster_idx = 0; cluster_idx < result.clusters.size(); cluster_idx++) {
        if (!cluster_dirty[cluster_idx]) {
            continue;
        }

        SplatCluster &cluster = result.clusters.write[cluster_idx];
        float max_radius = 0.0f;
        cluster.bounds = compute_cluster_bounds(
            p_gaussians,
            result.sorted_splat_order,
            cluster.splat_start,
            cluster.splat_count,
            max_radius
        );
        cluster.max_splat_radius = max_radius;
        cluster.center = cluster.bounds.get_center();
        cluster.radius = cluster.bounds.get_longest_axis_size() * 0.5f;

        // Recompute importance
        cluster.importance_sum = 0.0f;
        for (uint32_t i = 0; i < cluster.splat_count; i++) {
            uint32_t idx = result.sorted_splat_order[cluster.splat_start + i];
            const Gaussian &g = p_gaussians[idx];
            float opacity = CLAMP(g.opacity, 0.0f, 1.0f);
            float scale_max = MAX(MAX(Math::abs(g.scale.x), Math::abs(g.scale.y)), Math::abs(g.scale.z));
            cluster.importance_sum += opacity * scale_max;
        }
    }

    result.build_time_ms = (OS::get_singleton()->get_ticks_usec() - start_time) / 1000.0f;

    return result;
}

Vector<uint8_t> ClusterBuilder::pack_for_gpu(const Vector<SplatCluster> &p_clusters) const {
    // GPU format per cluster (32 bytes):
    //   vec3 min_bounds (12 bytes)
    //   uint splat_start (4 bytes)
    //   vec3 max_bounds (12 bytes)
    //   uint splat_count (4 bytes)

    const size_t cluster_stride = 32;
    Vector<uint8_t> buffer;
    buffer.resize(p_clusters.size() * cluster_stride);

    uint8_t *ptr = buffer.ptrw();

    for (int i = 0; i < p_clusters.size(); i++) {
        const SplatCluster &cluster = p_clusters[i];
        uint8_t *cluster_ptr = ptr + i * cluster_stride;

        // min_bounds (vec3)
        Vector3 min_b = cluster.bounds.position;
        memcpy(cluster_ptr + 0, &min_b.x, sizeof(float));
        memcpy(cluster_ptr + 4, &min_b.y, sizeof(float));
        memcpy(cluster_ptr + 8, &min_b.z, sizeof(float));

        // splat_start (uint)
        memcpy(cluster_ptr + 12, &cluster.splat_start, sizeof(uint32_t));

        // max_bounds (vec3)
        Vector3 max_b = cluster.bounds.position + cluster.bounds.size;
        memcpy(cluster_ptr + 16, &max_b.x, sizeof(float));
        memcpy(cluster_ptr + 20, &max_b.y, sizeof(float));
        memcpy(cluster_ptr + 24, &max_b.z, sizeof(float));

        // splat_count (uint)
        memcpy(cluster_ptr + 28, &cluster.splat_count, sizeof(uint32_t));
    }

    return buffer;
}

} // namespace GaussianSplatting
