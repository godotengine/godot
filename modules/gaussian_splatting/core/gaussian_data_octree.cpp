/**
 * @file gaussian_data_octree.cpp
 * @brief Companion .cpp for gaussian_data.h containing octree construction
 *        and spatial query methods (build_octree, _subdivide_octree_node,
 *        query_octree, gather_frustum_indices).
 */

#include "gaussian_data.h"
#include "core/templates/hash_set.h"
#include "core/templates/sort_array.h"
#include "../logger/gs_logger.h"

namespace {

bool _sphere_intersects_planes(const Vector3 &p_center, float p_radius, const Vector<Plane> &p_planes) {
    if (p_planes.is_empty()) {
        return true;
    }

    for (int i = 0; i < p_planes.size(); i++) {
        if (p_planes[i].distance_to(p_center) > p_radius) {
            return false;
        }
    }

    return true;
}

bool _aabb_intersects_planes(const AABB &p_aabb, const Vector<Plane> &p_planes) {
    if (p_planes.is_empty()) {
        return true;
    }

    for (int i = 0; i < p_planes.size(); i++) {
        const Plane &plane = p_planes[i];
        Vector3 positive = p_aabb.position;

        if (plane.normal.x >= 0.0f) {
            positive.x += p_aabb.size.x;
        }
        if (plane.normal.y >= 0.0f) {
            positive.y += p_aabb.size.y;
        }
        if (plane.normal.z >= 0.0f) {
            positive.z += p_aabb.size.z;
        }

        if (plane.distance_to(positive) > 0.0f) {
            return false;
        }
    }

    return true;
}

} // namespace

void GaussianData::build_octree(int p_max_depth, uint32_t p_min_gaussians) {
    ERR_FAIL_COND(p_max_depth < 1 || p_max_depth > 16);
    ERR_FAIL_COND(p_min_gaussians == 0);

    // Clear existing octree
    octree.clear();

    if (gaussians.is_empty()) {
        return;
    }

    // Get overall bounds
    AABB total_bounds = get_aabb();
    if (total_bounds.size == Vector3()) {
        return; // Degenerate bounds
    }

    // Create root node
    OctreeNode root;
    root.bounds = total_bounds;
    root.level = 0;

    // Add all indices to root initially
    root.indices.resize(gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        root.indices[i] = i;
    }

    // Initialize children as invalid
    for (int i = 0; i < 8; i++) {
        root.children[i] = 0xFFFFFFFF;  // Invalid marker for uint32_t
    }

    octree.push_back(root);

    // Simple octree - subdivide based on configured thresholds
    _subdivide_octree_node(0, p_max_depth, p_min_gaussians);

    GS_LOG_INFO_DEFAULT("[Octree] Built with " + itos(octree.size()) + " nodes, max depth " + itos(p_max_depth) +
            ", min splats per leaf " + itos(p_min_gaussians));
}

void GaussianData::_subdivide_octree_node(uint32_t node_idx, int max_depth, uint32_t min_gaussians) {
    // Copy node data to avoid invalidation issues
    OctreeNode node = octree[node_idx];

    // Stop if max depth reached or too few gaussians
    if (node.level >= max_depth || node.indices.size() <= min_gaussians) {
        return;
    }

    // Calculate the largest Gaussian radius in this node
    float max_gaussian_radius = 0.0f;
    for (uint32_t idx : node.indices) {
        const Gaussian &g = gaussians[idx];
        float max_scale = MAX(MAX(g.scale.x, g.scale.y), g.scale.z);
        float radius = max_scale * 3.0f; // 3-sigma coverage
        max_gaussian_radius = MAX(max_gaussian_radius, radius);
    }

    // Stop subdivision if node is smaller than the largest Gaussian
    // This prevents pathological subdivision for large Gaussians
    float node_min_dimension = MIN(MIN(node.bounds.size.x, node.bounds.size.y), node.bounds.size.z);
    if (node_min_dimension <= max_gaussian_radius * 2.0f) {
        return; // Node too small relative to Gaussian size
    }

    // Calculate child bounds
    Vector3 center = node.bounds.get_center();
    Vector3 half_size = node.bounds.size * 0.5f;

    // Store child indices to update parent later
    uint32_t child_indices[8];  // Changed from uint8_t to match OctreeNode
    for (int i = 0; i < 8; i++) {
        child_indices[i] = 0xFFFFFFFF;  // Invalid marker for uint32_t
    }

    // Track if we created any children
    bool has_children = false;
    LocalVector<uint32_t> indices_moved_to_children; // Track which indices actually went to children

    // Create 8 children
    for (int i = 0; i < 8; i++) {
        // Calculate child bounds based on octant
        // Octant encoding: bit 0 = X, bit 1 = Y, bit 2 = Z
        Vector3 child_min = node.bounds.position;
        child_min.x += (i & 1) ? half_size.x : 0;
        child_min.y += (i & 2) ? half_size.y : 0;
        child_min.z += (i & 4) ? half_size.z : 0;

        OctreeNode child;
        child.bounds = AABB(child_min, half_size);
        child.level = node.level + 1;

        // Partition gaussians into this child
        // Check if Gaussian overlaps child bounds (considering scale)
        for (uint32_t idx : node.indices) {
            const Gaussian &g = gaussians[idx];

            // Create bounding box for Gaussian considering its scale
            float max_scale = MAX(MAX(g.scale.x, g.scale.y), g.scale.z);
            float radius = max_scale * 3.0f; // 3-sigma coverage
            AABB gaussian_bounds(g.position - Vector3(radius, radius, radius),
                                Vector3(radius * 2, radius * 2, radius * 2));

            // Check if Gaussian bounds overlap with child bounds
            if (child.bounds.intersects(gaussian_bounds)) {
                child.indices.push_back(idx);
            }
        }

        // Only create child if it has content and provides meaningful reduction
        if (!child.indices.is_empty()) {
            // Skip child creation if it would contain nearly all parent's Gaussians
            // This prevents exponential expansion for large Gaussians
            if (child.indices.size() >= node.indices.size() * 0.95f) {
                // More than 95% of parent's Gaussians - subdivision not helpful
                // Don't track these indices as moved since we're not creating the child
                continue;
            }

            // Track which indices are actually moving to a child
            for (uint32_t idx : child.indices) {
                indices_moved_to_children.push_back(idx);
            }

            uint32_t child_idx = octree.size();
            // Initialize all children as invalid
            for (int j = 0; j < 8; j++) {
                child.children[j] = 0xFFFFFFFF;  // Invalid marker for uint32_t
            }
            octree.push_back(child);
            child_indices[i] = child_idx;
            has_children = true;

            // Recursively subdivide
            _subdivide_octree_node(child_idx, max_depth, min_gaussians);
        }
    }

    // Update parent node with child indices
    for (int i = 0; i < 8; i++) {
        octree[node_idx].children[i] = child_indices[i];
    }

    // Clear only the indices that were actually moved to children
    // Keep any indices that were skipped due to the 95% threshold
    if (has_children) {
        // Build a set of moved indices for efficient lookup
        HashSet<uint32_t> moved_set;
        for (uint32_t idx : indices_moved_to_children) {
            moved_set.insert(idx);
        }

        // Keep only indices that weren't moved to children
        LocalVector<uint32_t> remaining_indices;
        for (uint32_t idx : octree[node_idx].indices) {
            if (!moved_set.has(idx)) {
                remaining_indices.push_back(idx);
            }
        }

        octree[node_idx].indices = remaining_indices;
    }
}

TypedArray<int> GaussianData::query_octree(const AABB &p_bounds) const {
    TypedArray<int> results;

    // Early exit if no octree or empty
    if (octree.is_empty() || gaussians.is_empty()) {
        return results;
    }

    // Use a stack for iterative traversal (avoid recursion overhead)
    LocalVector<uint32_t> node_stack;
    node_stack.push_back(0); // Start with root

    // Track unique indices (Gaussian might be in multiple nodes)
    HashSet<uint32_t> unique_indices;

    while (!node_stack.is_empty()) {
        uint32_t node_idx = node_stack[node_stack.size() - 1];
        node_stack.resize(node_stack.size() - 1);

        const OctreeNode &node = octree[node_idx];

        // Early termination: skip if node doesn't overlap query bounds
        if (!node.bounds.intersects(p_bounds)) {
            continue;
        }

        // If leaf node (has indices), check Gaussians
        if (!node.indices.is_empty()) {
            for (uint32_t idx : node.indices) {
                const Gaussian &g = gaussians[idx];

                // Check if Gaussian overlaps query bounds (considering scale)
                float max_scale = MAX(MAX(g.scale.x, g.scale.y), g.scale.z);
                float radius = max_scale * 3.0f; // 3-sigma coverage
                AABB gaussian_bounds(g.position - Vector3(radius, radius, radius),
                                    Vector3(radius * 2, radius * 2, radius * 2));

                if (gaussian_bounds.intersects(p_bounds)) {
                    unique_indices.insert(idx);
                }
            }
        }

        // If internal node, add children to stack
        bool has_children = false;
        for (int i = 0; i < 8; i++) {
            if (node.children[i] != 0xFFFFFFFF) {  // Check for valid child (uint32_t)
                has_children = true;
                // Only add child if it might overlap
                const OctreeNode &child = octree[node.children[i]];
                if (child.bounds.intersects(p_bounds)) {
                    node_stack.push_back(node.children[i]);
                }
            }
        }

        // Safety check: internal nodes shouldn't have indices if they have children
        // This is just for debugging, can be removed in production
        if (has_children && !node.indices.is_empty()) {
            GS_LOG_WARN_DEFAULT("Octree node has both children and indices at level " + itos(node.level));
        }
    }

    // Convert unique indices to array
    for (uint32_t idx : unique_indices) {
        results.append((int)idx);
    }

    return results;
}

void GaussianData::gather_frustum_indices(const Vector<Plane> &p_planes, LocalVector<uint32_t> &r_indices) const {
    r_indices.clear();

    if (gaussians.is_empty()) {
        return;
    }

    if (p_planes.is_empty()) {
        r_indices.resize(gaussians.size());
        for (uint32_t i = 0; i < gaussians.size(); i++) {
            r_indices[i] = i;
        }
        return;
    }

    HashSet<uint32_t> unique_indices;
    LocalVector<uint32_t> collected;

    auto evaluate_gaussian = [&](uint32_t p_index) {
        if (unique_indices.has(p_index)) {
            return;
        }

        const Gaussian &g = gaussians[p_index];
        float max_scale = MAX(MAX(g.scale.x, g.scale.y), g.scale.z);
        if (max_scale <= 0.0f) {
            max_scale = 1.0f;
        }
        float radius = max_scale * 3.0f;

        if (_sphere_intersects_planes(g.position, radius, p_planes)) {
            unique_indices.insert(p_index);
            collected.push_back(p_index);
        }
    };

    if (!octree.is_empty()) {
        LocalVector<uint32_t> stack;
        stack.push_back(0);

        while (!stack.is_empty()) {
            uint32_t node_idx = stack[stack.size() - 1];
            stack.resize(stack.size() - 1);

            const OctreeNode &node = octree[node_idx];
            if (!_aabb_intersects_planes(node.bounds, p_planes)) {
                continue;
            }

            if (!node.indices.is_empty()) {
                for (uint32_t idx : node.indices) {
                    evaluate_gaussian(idx);
                }
            }

            for (int i = 0; i < 8; i++) {
                if (node.children[i] != 0xFFFFFFFF) {
                    stack.push_back(node.children[i]);
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < gaussians.size(); i++) {
            evaluate_gaussian(i);
        }
    }

    if (collected.is_empty()) {
        return;
    }

    SortArray<uint32_t> sorter;
    sorter.sort(collected.ptr(), collected.size());

    r_indices = collected;
}
