#ifndef HIERARCHICAL_SPLAT_STRUCTURE_H
#define HIERARCHICAL_SPLAT_STRUCTURE_H

#include "core/math/aabb.h"
#include "core/math/vector3.h"
#include "core/templates/vector.h"
#include "core/templates/local_vector.h"
#include "../core/gaussian_data.h"
#include "servers/rendering/renderer_scene_cull.h"
#include <memory>
#include <array>
#include <atomic>

namespace GaussianSplatting {

using Frustum = RendererSceneCull::Frustum;

class HierarchicalSplatStructure {
public:
    struct SplatInfo {
        uint32_t index;
        Vector3 position;
        float radius;
        float importance;
        Color color;
        float opacity;
    };

    struct OctreeNode {
        AABB bounds;
        uint32_t splat_start;
        uint32_t splat_count;
        float avg_size;         // Average splat size in this node
        float max_size;         // Maximum splat size for culling
        float importance_sum;   // Sum of importance values
        uint32_t depth;         // Depth in octree

        // Child nodes (nullptr if leaf)
        std::array<std::unique_ptr<OctreeNode>, 8> children;

        // Statistics for LOD decisions
        struct Stats {
            float density;      // Splats per unit volume
            float variance;     // Spatial variance of splats
            Color avg_color;    // Average color for LOD merging
            float avg_opacity;  // Average opacity
        } stats;

        OctreeNode() : splat_start(0), splat_count(0), avg_size(0.0f),
                      max_size(0.0f), importance_sum(0.0f), depth(0) {}

        bool is_leaf() const {
            for (const auto &child : children) {
                if (child) {
                    return false;
                }
            }
            return true;
        }
        uint32_t get_child_index(const Vector3& pos) const;
    };

    struct QueryResult {
        LocalVector<uint32_t> visible_indices;
        LocalVector<float> lod_weights;    // One weight per visible index; consumed as a coarse candidate weight by GPUCuller.
        uint32_t total_splats;
        float culled_percentage;
        uint32_t nodes_visited = 0;
        uint32_t nodes_frustum_culled = 0;
        uint32_t splats_frustum_culled = 0;

        struct LODStats {
            uint32_t lod0_count = 0;
            uint32_t lod1_count = 0;
            uint32_t lod2_count = 0;
            uint32_t lod3_count = 0;
        } lod_stats;
    };

    struct BuildParams {
        uint32_t max_depth = 8;            // Maximum octree depth
        uint32_t min_splats_per_node = 16; // Minimum splats to subdivide
        float size_threshold = 0.01f;      // Minimum node size
        bool compute_importance = true;    // Consume source GaussianData::importance for LOD decisions; otherwise treat splats as equally important.
        bool parallel_build = false;       // Requests parallel build, but the live path still uses the supported sequential builder.
    };

public:
    HierarchicalSplatStructure();
    ~HierarchicalSplatStructure();

    // Build a coarse CPU hierarchy for live candidate generation. GPUCuller still performs the
    // final per-splat frustum, distance, and budget filtering on the query output.
    void build_hierarchy(const Vector<GaussianData>& splats, const BuildParams& params);
    void build_hierarchy(const Vector<GaussianData>& splats) { build_hierarchy(splats, BuildParams()); }

    // Rebuild specific nodes (for dynamic updates)
    void update_node(OctreeNode* node, const Vector<GaussianData>& splats);

    // Query coarse visible candidates. The returned visible_indices and lod_weights vectors are
    // kept aligned so downstream culling can consume them as paired lists.
    QueryResult query_visible_splats(
        const Frustum& frustum,
        const Vector3& camera_pos,
        float lod_bias = 1.0f,
        uint32_t max_splats = 500000
    );

    // Hierarchical culling with early termination
    void cull_hierarchical(
        const OctreeNode* node,
        const Frustum& frustum,
        const Vector3& camera_pos,
        QueryResult& result,
        float parent_importance = 1.0f,
        float lod_bias = 1.0f
    );

    // LOD selection based on distance and importance
    uint32_t select_lod_level(
        const OctreeNode* node,
        const Vector3& camera_pos,
        float lod_bias
    ) const;

    // Get appropriate splat indices for a LOD level
    void get_lod_splats(
        const OctreeNode* node,
        uint32_t lod_level,
        LocalVector<uint32_t>& indices
    );

    // Statistics and debugging
    struct TreeStats {
        uint32_t total_nodes;
        uint32_t leaf_nodes;
        uint32_t max_depth_reached;
        float avg_splats_per_leaf;
        uint64_t memory_usage;
    };

    TreeStats get_statistics() const;
    void debug_draw_octree(uint32_t max_depth = 3) const;

    // Accessors
    const OctreeNode* get_root() const { return root.get(); }
    uint32_t get_total_splats() const { return total_splats; }
    AABB get_bounds() const { return root ? root->bounds : AABB(); }

private:
    // Build helpers
    void build_node_recursive(
        OctreeNode* node,
        Vector<SplatInfo>& splats,
        uint32_t start,
        uint32_t count,
        uint32_t depth,
        const BuildParams& params
    );

    void compute_node_statistics(OctreeNode* node, const Vector<SplatInfo>& splats);

    // Importance calculation
    float calculate_importance(const SplatInfo& splat, const OctreeNode* node) const;

    // Memory management
    void clear_node(OctreeNode* node);

private:
    std::unique_ptr<OctreeNode> root;
    Vector<SplatInfo> splat_data;        // Cached splat information
    uint32_t total_splats;

    // Build statistics
    std::atomic<uint32_t> nodes_created;
    std::atomic<uint64_t> build_time_us;

    // Thread pool for parallel operations
    class ThreadPool* thread_pool;
};

// Inline implementations for performance
inline uint32_t HierarchicalSplatStructure::OctreeNode::get_child_index(const Vector3& pos) const {
    Vector3 center = bounds.get_center();
    uint32_t index = 0;
    if (pos.x > center.x) index |= 1;
    if (pos.y > center.y) index |= 2;
    if (pos.z > center.z) index |= 4;
    return index;
}

} // namespace GaussianSplatting

#endif // HIERARCHICAL_SPLAT_STRUCTURE_H
