#include "hierarchical_splat_structure.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/templates/local_vector.h"
#include "../logger/gs_logger.h"
#include <algorithm>
#include <functional>

namespace {

static inline bool frustum_intersects_aabb(const GaussianSplatting::Frustum &p_frustum, const AABB &p_aabb, float p_padding) {
    AABB expanded = p_padding > 0.0f ? p_aabb.grow(p_padding) : p_aabb;
    RendererSceneCull::InstanceBounds bounds(expanded);
    return bounds.in_frustum(p_frustum);
}

static inline Vector3 get_closest_point_on_aabb(const AABB &p_aabb, const Vector3 &p_point) {
    const Vector3 min = p_aabb.position;
    const Vector3 max = p_aabb.position + p_aabb.size;
    return Vector3(
        std::clamp(p_point.x, min.x, max.x),
        std::clamp(p_point.y, min.y, max.y),
        std::clamp(p_point.z, min.z, max.z));
}

} // namespace

namespace GaussianSplatting {

HierarchicalSplatStructure::HierarchicalSplatStructure()
    : total_splats(0), nodes_created(0), build_time_us(0), thread_pool(nullptr) {
}

HierarchicalSplatStructure::~HierarchicalSplatStructure() {
    if (thread_pool) {
        // Clean up thread pool
    }
}

void HierarchicalSplatStructure::build_hierarchy(
    const Vector<GaussianData>& splats,
    const BuildParams& params) {

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();

    // Clear existing hierarchy
    root.reset();
    splat_data.clear();
    nodes_created = 0;

    if (splats.is_empty()) {
        return;
    }

    total_splats = splats.size();
    splat_data.resize(total_splats);

    // Convert splat data to internal format and compute bounds
    AABB total_bounds;
    for (uint32_t i = 0; i < total_splats; i++) {
        const GaussianData& g = splats[i];
        SplatInfo& info = splat_data.write[i];

        info.index = i;
        info.position = g.position;
        info.radius = g.compute_radius();  // Compute from covariance
        info.color = g.color;
        info.opacity = g.color.a;
        info.importance = 1.0f;  // Will be computed later

        // Expand bounds - AABB takes (position, size), not (min, max)
        Vector3 half_size(info.radius, info.radius, info.radius);
        AABB splat_bounds(info.position - half_size, half_size * 2.0f);
        if (i == 0) {
            total_bounds = splat_bounds;
        } else {
            total_bounds = total_bounds.merge(splat_bounds);
        }
    }

    // Create root node
    root = std::make_unique<OctreeNode>();
    root->bounds = total_bounds;
    root->splat_start = 0;
    root->splat_count = total_splats;
    root->depth = 0;
    nodes_created++;

    // Build recursively or in parallel
    if (params.parallel_build && total_splats > 10000) {
        // Parallel build for large datasets
        Vector<OctreeNode*> current_level;
        current_level.push_back(root.get());

        for (uint32_t depth = 0; depth < params.max_depth; depth++) {
            parallel_build_level(depth, current_level, splat_data, params);

            // Collect next level nodes
            Vector<OctreeNode*> next_level;
            for (OctreeNode* node : current_level) {
                if (!node->is_leaf()) {
                    for (auto& child : node->children) {
                        if (child) {
                            next_level.push_back(child.get());
                        }
                    }
                }
            }

            if (next_level.is_empty()) {
                break;
            }
            current_level = next_level;
        }
    } else {
        // Sequential build for smaller datasets
        build_node_recursive(root.get(), splat_data, 0, total_splats, 0, params);
    }

    // Compute statistics for all nodes
    compute_node_statistics(root.get(), splat_data);

    build_time_us = OS::get_singleton()->get_ticks_usec() - start_time;

    GS_LOG_RENDERER_INFO(vformat("Octree built: %d nodes, %d splats, %.2f ms",
            nodes_created.load(), total_splats, build_time_us.load() / 1000.0));
}

void HierarchicalSplatStructure::build_node_recursive(
    OctreeNode* node,
    Vector<SplatInfo>& splats,
    uint32_t start,
    uint32_t count,
    uint32_t depth,
    const BuildParams& params) {

    node->splat_start = start;
    node->splat_count = count;
    node->depth = depth;

    // Check termination conditions
    if (depth >= params.max_depth ||
        count <= params.min_splats_per_node ||
        node->bounds.get_longest_axis_size() < params.size_threshold) {
        // Leaf node - compute statistics
        float sum_size = 0.0f;
        float max_size = 0.0f;
        float importance_sum = 0.0f;

        for (uint32_t i = start; i < start + count; i++) {
            const SplatInfo& info = splats[i];
            sum_size += info.radius;
            max_size = MAX(max_size, info.radius);
            importance_sum += info.importance;
        }

        node->avg_size = count > 0 ? sum_size / count : 0.0f;
        node->max_size = max_size;
        node->importance_sum = importance_sum;

        return;
    }

    // Subdivide node into 8 children
    Vector3 center = node->bounds.get_center();
    Vector3 half_size = node->bounds.get_size() * 0.5f;

    // Create child bounds
    AABB child_bounds[8];
    for (uint32_t i = 0; i < 8; i++) {
        Vector3 offset;
        offset.x = (i & 1) ? 0.0f : -half_size.x;
        offset.y = (i & 2) ? 0.0f : -half_size.y;
        offset.z = (i & 4) ? 0.0f : -half_size.z;

        Vector3 child_min = center + offset;
        // AABB takes (position, size), not (min, max)
        child_bounds[i] = AABB(child_min, half_size);
    }

    // Partition splats into children
    LocalVector<uint32_t> child_indices[8];
    for (uint32_t i = start; i < start + count; i++) {
        const SplatInfo& info = splats[i];
        uint32_t child_idx = node->get_child_index(info.position);
        child_indices[child_idx].push_back(i);
    }

    // Reorder splats array to group by child
    Vector<SplatInfo> temp_splats;
    temp_splats.resize(count);
    uint32_t current_offset = start;

    for (uint32_t child = 0; child < 8; child++) {
        if (child_indices[child].size() > 0) {
            // Copy splats for this child
            for (uint32_t idx : child_indices[child]) {
                temp_splats.write[current_offset - start] = splats[idx];
                current_offset++;
            }
        }
    }

    // Copy back to main array
    for (uint32_t i = 0; i < count; i++) {
        splats.write[start + i] = temp_splats[i];
    }

    // Create and build child nodes
    current_offset = start;
    for (uint32_t child = 0; child < 8; child++) {
        uint32_t child_count = child_indices[child].size();
        if (child_count > 0) {
            node->children[child] = std::make_unique<OctreeNode>();
            node->children[child]->bounds = child_bounds[child];
            nodes_created++;

            build_node_recursive(
                node->children[child].get(),
                splats,
                current_offset,
                child_count,
                depth + 1,
                params
            );

            current_offset += child_count;
        }
    }

    // Update node statistics from children
    float sum_size = 0.0f;
    float max_size = 0.0f;
    float importance_sum = 0.0f;
    uint32_t child_count_sum = 0;

    for (auto& child : node->children) {
        if (child) {
            sum_size += child->avg_size * child->splat_count;
            max_size = MAX(max_size, child->max_size);
            importance_sum += child->importance_sum;
            child_count_sum += child->splat_count;
        }
    }

    node->avg_size = child_count_sum > 0 ? sum_size / child_count_sum : 0.0f;
    node->max_size = max_size;
    node->importance_sum = importance_sum;
}

void HierarchicalSplatStructure::compute_node_statistics(
    OctreeNode* node,
    const Vector<SplatInfo>& splats) {

    if (!node) return;

    // Compute statistics for this node
    if (node->splat_count > 0) {
        float volume = node->bounds.get_volume();
        node->stats.density = volume > 0.0f ? node->splat_count / volume : 0.0f;

        // Compute average color and opacity
        Color sum_color;
        float sum_opacity = 0.0f;
        Vector3 mean_pos;

        for (uint32_t i = node->splat_start; i < node->splat_start + node->splat_count; i++) {
            const SplatInfo& info = splats[i];
            sum_color += info.color;
            sum_opacity += info.opacity;
            mean_pos += info.position;
        }

        if (node->splat_count > 0) {
            node->stats.avg_color = sum_color / float(node->splat_count);
            node->stats.avg_opacity = sum_opacity / node->splat_count;
            mean_pos /= float(node->splat_count);
        }

        // Compute spatial variance
        float variance = 0.0f;
        for (uint32_t i = node->splat_start; i < node->splat_start + node->splat_count; i++) {
            const SplatInfo& info = splats[i];
            Vector3 diff = info.position - mean_pos;
            variance += diff.length_squared();
        }
        node->stats.variance = node->splat_count > 0 ? variance / node->splat_count : 0.0f;
    }

    // Recursively compute for children
    for (auto& child : node->children) {
        if (child) {
            compute_node_statistics(child.get(), splats);
        }
    }
}

HierarchicalSplatStructure::QueryResult HierarchicalSplatStructure::query_visible_splats(
    const Frustum& frustum,
    const Vector3& camera_pos,
    float lod_bias,
    uint32_t max_splats) {

    QueryResult result;
    result.total_splats = total_splats;

    if (!root || total_splats == 0) {
        result.culled_percentage = 0.0f;
        return result;
    }

    // Reserve space for results
    result.visible_indices.reserve(MIN(max_splats, total_splats));
    result.lod_weights.reserve(MIN(max_splats, total_splats));

    // Perform hierarchical culling
    cull_hierarchical(root.get(), frustum, camera_pos, result, 1.0f, lod_bias);

    // Sort by distance if we exceeded max_splats
    if (result.visible_indices.size() > max_splats) {
        // Create distance pairs
        LocalVector<std::pair<float, uint32_t>> distance_pairs;
        distance_pairs.resize(result.visible_indices.size());

        for (uint32_t i = 0; i < result.visible_indices.size(); i++) {
            uint32_t splat_idx = result.visible_indices[i];
            float dist = (splat_data[splat_idx].position - camera_pos).length();
            distance_pairs[i] = {dist, splat_idx};
        }

        // Partial sort to get closest max_splats
        std::partial_sort(
            distance_pairs.ptr(),
            distance_pairs.ptr() + max_splats,
            distance_pairs.ptr() + distance_pairs.size(),
            [](const auto& a, const auto& b) { return a.first < b.first; }
        );

        // Update result with closest splats
        result.visible_indices.clear();
        result.lod_weights.clear();
        for (uint32_t i = 0; i < max_splats; i++) {
            result.visible_indices.push_back(distance_pairs[i].second);

            // Compute LOD weight based on distance
            float dist = distance_pairs[i].first;
            float weight = 1.0f;
            if (dist > 50.0f) weight = 0.75f;
            if (dist > 100.0f) weight = 0.5f;
            if (dist > 200.0f) weight = 0.25f;
            result.lod_weights.push_back(weight);
        }
    }

    if (result.visible_indices.size() != result.lod_weights.size()) {
        const uint32_t safe_size = MIN(result.visible_indices.size(), result.lod_weights.size());
        ERR_PRINT(vformat(
            "Hierarchical LOD query cardinality mismatch (indices=%d, weights=%d). Truncating to %d.",
            result.visible_indices.size(),
            result.lod_weights.size(),
            safe_size));
        result.visible_indices.resize(safe_size);
        result.lod_weights.resize(safe_size);
    }

    result.culled_percentage = 100.0f * (1.0f - float(result.visible_indices.size()) / total_splats);

    return result;
}

void HierarchicalSplatStructure::cull_hierarchical(
    const OctreeNode* node,
    const Frustum& frustum,
    const Vector3& camera_pos,
    QueryResult& result,
    float parent_importance,
    float lod_bias) {

    if (!node || node->splat_count == 0) {
        return;
    }

    result.nodes_visited++;

    // Check frustum intersection
    // Expand node bounds by the maximum splat radius in this subtree so we don't drop nodes whose
    // splats extend into the frustum even if their centers (and node AABB) are slightly outside.
    if (!frustum_intersects_aabb(frustum, node->bounds, node->max_size)) {
        result.nodes_frustum_culled++;
        result.splats_frustum_culled += node->splat_count;
        return;  // Entire subtree culled
    }

    // Compute distance to node
    Vector3 closest_point = get_closest_point_on_aabb(node->bounds, camera_pos);
    float distance = (closest_point - camera_pos).length();

    // Compute LOD level for this node
    uint32_t lod_level = select_lod_level(node, camera_pos, lod_bias);

    // Update LOD statistics
    switch (lod_level) {
        case 0: result.lod_stats.lod0_count += node->splat_count; break;
        case 1: result.lod_stats.lod1_count += node->splat_count; break;
        case 2: result.lod_stats.lod2_count += node->splat_count; break;
        case 3: result.lod_stats.lod3_count += node->splat_count; break;
    }

    // Check if we should use this node's LOD or recurse to children
    bool use_node_lod = false;

    // Criteria for using node LOD:
    // 1. Node is a leaf
    // 2. Distance is far enough for this LOD level
    // 3. Node importance is below threshold
    if (node->is_leaf()) {
        use_node_lod = true;
    } else if (lod_level >= 2) {
        // Use merged representation for distant nodes
        use_node_lod = true;
    } else if (distance > 100.0f && node->importance_sum < 0.5f * node->splat_count) {
        // Low importance distant nodes
        use_node_lod = true;
    }

    if (use_node_lod) {
        // Add splats from this node with LOD weights
        const uint32_t visible_start = result.visible_indices.size();
        get_lod_splats(node, lod_level, result.visible_indices);
        const uint32_t emitted_count = result.visible_indices.size() - visible_start;

        // Add LOD weights
        float lod_weight = 1.0f / (1.0f + lod_level);
        for (uint32_t i = 0; i < emitted_count; i++) {
            result.lod_weights.push_back(lod_weight);
        }
    } else {
        // Recurse to children
        for (const auto& child : node->children) {
            if (child) {
                cull_hierarchical(
                    child.get(),
                    frustum,
                    camera_pos,
                    result,
                    parent_importance * 0.8f,  // Reduce importance with depth
                    lod_bias
                );
            }
        }
    }
}

uint32_t HierarchicalSplatStructure::select_lod_level(
    const OctreeNode* node,
    const Vector3& camera_pos,
    float lod_bias) const {

    // Compute distance to node center
    Vector3 node_center = node->bounds.get_center();
    float distance = (node_center - camera_pos).length();

    // Compute screen space size estimate
    float node_size = node->bounds.get_longest_axis_size();
    float screen_size = node_size / (distance + 1.0f);  // Rough approximation

    // Apply LOD bias in the same direction as adaptive quality control:
    // lower bias => more aggressive culling (lower effective screen size).
    const float effective_lod_bias = MAX(lod_bias, 0.0f);
    screen_size *= effective_lod_bias;

    // Select LOD level based on screen size
    if (screen_size > 0.1f) {
        return 0;  // Full detail
    } else if (screen_size > 0.05f) {
        return 1;  // 50% detail
    } else if (screen_size > 0.02f) {
        return 2;  // 25% detail
    } else {
        return 3;  // Minimal detail
    }
}

void HierarchicalSplatStructure::get_lod_splats(
    const OctreeNode* node,
    uint32_t lod_level,
    LocalVector<uint32_t>& indices) {

    if (lod_level == 0) {
        // Full detail - add all splats
        for (uint32_t i = node->splat_start; i < node->splat_start + node->splat_count; i++) {
            indices.push_back(splat_data[i].index);
        }
    } else {
        // Reduced detail - sample splats based on importance
        uint32_t target_count = node->splat_count;

        // Reduce count based on LOD level
        if (lod_level == 1) target_count = target_count / 2;
        else if (lod_level == 2) target_count = target_count / 4;
        else if (lod_level == 3) target_count = MAX(1u, target_count / 10);

        // Sample splats uniformly (can be improved with importance sampling)
        uint32_t step = MAX(1u, node->splat_count / target_count);
        for (uint32_t i = node->splat_start; i < node->splat_start + node->splat_count; i += step) {
            indices.push_back(splat_data[i].index);
        }
    }
}

float HierarchicalSplatStructure::calculate_importance(
    const SplatInfo& splat,
    const OctreeNode* node) const {

    // Importance based on:
    // 1. Size relative to node
    // 2. Opacity
    // 3. Color contrast with neighbors

    float size_importance = splat.radius / (node->avg_size + 0.001f);
    float opacity_importance = splat.opacity;

    // Color contrast (simplified - compare with node average)
    Vector3 splat_rgb(splat.color.r, splat.color.g, splat.color.b);
    Vector3 avg_rgb(node->stats.avg_color.r, node->stats.avg_color.g, node->stats.avg_color.b);
    float color_diff = (splat_rgb - avg_rgb).length();
    float color_importance = MIN(1.0f, color_diff);

    // Combined importance
    return (size_importance + opacity_importance + color_importance) / 3.0f;
}

void HierarchicalSplatStructure::parallel_build_level(
    uint32_t level,
    Vector<OctreeNode*>& nodes_to_process,
    Vector<SplatInfo>& splats,
    const BuildParams& params) {

    // Process nodes in parallel
    // Note: In production, use Godot's WorkerThreadPool
    for (OctreeNode* node : nodes_to_process) {
        if (node->splat_count > params.min_splats_per_node &&
            level < params.max_depth) {

            // Subdivide this node
            // (Similar to build_node_recursive but for one level only)
            // This would be parallelized in production
        }
    }
}

HierarchicalSplatStructure::TreeStats HierarchicalSplatStructure::get_statistics() const {
    TreeStats stats = {};

    if (!root) {
        return stats;
    }

    // Count nodes recursively
    std::function<void(const OctreeNode*, uint32_t)> count_nodes =
        [&](const OctreeNode* node, uint32_t depth) {
        if (!node) return;

        stats.total_nodes++;
        stats.max_depth_reached = MAX(stats.max_depth_reached, depth);

        if (node->is_leaf()) {
            stats.leaf_nodes++;
            stats.avg_splats_per_leaf += node->splat_count;
        } else {
            for (const auto& child : node->children) {
                count_nodes(child.get(), depth + 1);
            }
        }
    };

    count_nodes(root.get(), 0);

    if (stats.leaf_nodes > 0) {
        stats.avg_splats_per_leaf /= stats.leaf_nodes;
    }

    // Estimate memory usage
    stats.memory_usage = stats.total_nodes * sizeof(OctreeNode);
    stats.memory_usage += splat_data.size() * sizeof(SplatInfo);

    return stats;
}

} // namespace GaussianSplatting
