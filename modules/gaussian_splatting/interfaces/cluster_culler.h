#ifndef GS_CLUSTER_CULLER_H
#define GS_CLUSTER_CULLER_H

#include "culler_interfaces.h"
#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"
#include "../lod/cluster_builder.h"
#include <memory>

namespace GaussianSplatting {
class ClusterBuilder;
struct ClusterBuildResult;
struct SplatCluster;
}

/**
 * @struct ClusterCullConfig
 * @brief Configuration for cluster-level coarse culling.
 */
struct ClusterCullConfig {
    bool enabled = true;                      ///< Enable cluster-level culling
    uint32_t target_cluster_size = 128;       ///< Target splats per cluster (32-256)
    uint32_t min_cluster_size = 32;           ///< Minimum splats per cluster
    uint32_t max_cluster_size = 256;          ///< Maximum splats per cluster
    float frustum_slack = 2.0f;               ///< Slack factor for conservative frustum tests
    bool use_morton_order = true;             ///< Use Morton (Z-order) curve for spatial locality
    bool use_indirect_dispatch = true;        ///< Use indirect dispatch for fine pass
    uint32_t rebuild_threshold_percent = 10;  ///< Rebuild clusters if >N% splats changed
};

/**
 * @struct ClusterCullStats
 * @brief Statistics from cluster-level culling.
 */
struct ClusterCullStats {
    uint32_t total_clusters = 0;
    uint32_t visible_clusters = 0;
    uint32_t culled_clusters = 0;
    uint32_t visible_splats = 0;
    uint32_t culled_splats = 0;
    float cluster_cull_time_ms = 0.0f;
    float fine_cull_time_ms = 0.0f;
    float cluster_cull_ratio = 0.0f;  ///< Percentage of splats skipped via cluster cull
};

/**
 * @class ClusterCuller
 * @brief GPU-accelerated two-level hierarchical culler.
 *
 * ClusterCuller implements LiteGS-style coarse culling:
 * 1. Group splats into clusters of 32-256 (Morton-ordered for locality)
 * 2. GPU tests cluster AABBs against frustum (coarse pass)
 * 3. Only visible clusters proceed to per-splat culling (fine pass)
 *
 * For a scene with 1M splats and 128 splats/cluster:
 * - Traditional: 1M frustum tests
 * - With clusters: ~8K cluster tests + visible splat tests
 * - If 50% culled at cluster level: 8K + 500K = 508K tests (49% reduction)
 *
 * The coarse pass uses indirect dispatch to skip culled clusters entirely,
 * avoiding wasted GPU threads on invisible geometry.
 */
class ClusterCuller : public RefCounted {
    GDCLASS(ClusterCuller, RefCounted);

public:
    ClusterCuller();
    ~ClusterCuller();

    /**
     * @brief Initializes GPU resources for cluster culling.
     * @param p_device RenderingDevice to use.
     * @return OK on success, error code on failure.
     */
    Error initialize(RenderingDevice *p_device);

    /**
     * @brief Releases all GPU resources.
     */
    void shutdown();

    /**
     * @brief Checks if the culler is ready to use.
     */
    bool is_ready() const;

    /**
     * @brief Builds or rebuilds clusters from Gaussian data.
     * @param p_gaussians Source Gaussian data.
     * @param p_force_rebuild Force full rebuild even if data seems unchanged.
     * @return True if clusters were rebuilt.
     */
    bool build_clusters(const LocalVector<Gaussian> &p_gaussians, bool p_force_rebuild = false);

    /**
     * @brief Uploads cluster data to GPU buffers.
     * @return True on success.
     */
    bool upload_clusters_to_gpu();

    /**
     * @brief Performs coarse cluster-level culling.
     * @param p_params Culling parameters (frustum planes, camera, etc.)
     * @return True if culling completed successfully.
     *
     * After this call, get_visible_cluster_mask() returns which clusters passed.
     * Use get_indirect_dispatch_buffer() for indirect dispatch of fine pass.
     */
    bool cull_clusters(const CullParams &p_params);

    /**
     * @brief Gets the visibility mask buffer RID.
     * @return RID of the visibility mask buffer (1 bit per cluster).
     */
    RID get_visible_cluster_mask() const { return cluster_visibility_buffer; }

    /**
     * @brief Gets the indirect dispatch buffer for fine culling.
     * @return RID of the indirect dispatch args buffer.
     */
    RID get_indirect_dispatch_buffer() const { return indirect_dispatch_buffer; }

    /**
     * @brief Gets the list of visible cluster indices.
     * @return RID of the visible cluster indices buffer.
     */
    RID get_visible_cluster_indices() const { return visible_cluster_buffer; }

    /**
     * @brief Gets the sorted splat order buffer.
     * @return RID of the buffer mapping sorted index -> original splat index.
     *
     * The fine culling pass should iterate splats in this order to match
     * the cluster structure.
     */
    RID get_sorted_splat_order_buffer() const { return sorted_order_buffer; }

    /**
     * @brief Gets statistics from the last cull operation.
     */
    ClusterCullStats get_stats() const { return last_stats; }

    /**
     * @brief Gets the most recent compute infrastructure error (if any).
     */
    String get_last_compute_error() const { return last_compute_error; }

    /**
     * @brief Gets the number of clusters.
     */
    uint32_t get_cluster_count() const { return cluster_count; }

    /**
     * @brief Gets the cluster configuration.
     */
    ClusterCullConfig &get_config() { return config; }
    const ClusterCullConfig &get_config() const { return config; }

    /**
     * @brief Invalidates cluster data, forcing rebuild on next build_clusters().
     */
    void invalidate();

protected:
    static void _bind_methods();

private:
    void _ensure_shader(RenderingDevice *p_device);
    void _ensure_buffers(RenderingDevice *p_device, uint32_t p_cluster_count);
    void _release_resources();
    void _on_stats_readback(const Vector<uint8_t> &p_data);

private:
    RenderingDevice *rd = nullptr;
    RenderingDevice *resource_device = nullptr;

    // Shader resources
    RID cluster_cull_shader;
    RID cluster_cull_shader_version;
    RID cluster_cull_pipeline;

    // Buffers
    RID cluster_buffer;              // ClusterAABB data (32 bytes/cluster)
    RID cluster_visibility_buffer;   // Visibility bitmask (1 bit/cluster)
    RID visible_cluster_buffer;      // Compacted visible cluster indices
    RID indirect_dispatch_buffer;    // Indirect dispatch args for fine pass
    RID param_buffer;                // Uniform buffer for cull params
    RID sorted_order_buffer;         // Sorted splat indices

    // Capacity tracking
    uint32_t buffer_cluster_capacity = 0;
    uint32_t buffer_splat_capacity = 0;
    uint32_t cluster_count = 0;

    // Cluster builder
    std::unique_ptr<GaussianSplatting::ClusterBuilder> cluster_builder;
    GaussianSplatting::ClusterBuildResult cluster_data;
    bool clusters_dirty = true;
    uint32_t last_splat_count = 0;

    // Configuration
    ClusterCullConfig config;

    // Statistics
    ClusterCullStats last_stats;
    String last_compute_error;
    bool initialized = false;

    // Async readback state
    struct AsyncStatsState {
        bool pending = false;
        uint64_t request_id = 0;
        uint64_t generation = 0;
    };
    AsyncStatsState async_stats;
};

#endif // GS_CLUSTER_CULLER_H
